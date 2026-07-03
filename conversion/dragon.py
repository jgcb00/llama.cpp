"""Dragon 7A1B converter.

Hybrid Mamba3-MIMO (M layers) + Differential-TPA-V2 attention (V layers) + MoE,
with a geodesic-rotation residual on every block.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

import gguf
from . import ModelBase, TextModel, logger


@ModelBase.register("DragonForCausalLM")
class DragonModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DRAGON

    # Block-letter convention: 'M' = Mamba3-MIMO mixer, 'V' = Differential-TPA-V2 mixer.
    SSM_LAYER_TYPES = {"M"}
    ATTN_LAYER_TYPES = {"V"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The Dragon config exposes `layers_config` as a per-block letter string.
        layers_cfg: str = self.hparams.get("layers_config", "")
        if not layers_cfg or len(layers_cfg) != self.block_count:
            raise ValueError(
                f"layers_config must be a string of length block_count ({self.block_count}); got {layers_cfg!r}"
            )
        self.layer_types = layers_cfg
        self._ssm_layers = [i for i, c in enumerate(layers_cfg) if c in self.SSM_LAYER_TYPES]
        self._attn_layers = [i for i, c in enumerate(layers_cfg) if c in self.ATTN_LAYER_TYPES]
        unknown = sorted({c for c in layers_cfg} - self.SSM_LAYER_TYPES - self.ATTN_LAYER_TYPES)
        if unknown:
            raise ValueError(f"Dragon converter only handles M and V layers; got unknown: {unknown}")

        # Cache a few derived quantities for set_gguf_parameters / modify_tensors.
        self._head_dim       = int(self.hparams["head_dim"])
        self._n_head         = int(self.hparams["num_attention_heads"])
        self._n_sig          = int(self.hparams.get("num_signal_heads_diff") or self._n_head // 2)
        self._n_noise        = self._n_head - self._n_sig
        self._tpa_rank       = int(self.hparams["tpa_rank"])
        self._mimo_dim       = int(self.hparams["mamba_mimo_dim"])
        self._mamba_headdim  = int(self.hparams["mamba_headdim"])
        self._mamba_d_state  = int(self.hparams["mamba_d_state"])
        self._mamba_ngroups  = int(self.hparams["mamba_ngroups"])
        self._d_inner        = 2 * int(self.hparams["hidden_size"])     # expand_factor=2
        self._nheads_ssm     = self._d_inner // self._mamba_headdim     # 3072 / 64 = 48
        # rope_fraction = 0.5 in the Mamba3MimoFast __init__ (line 1863 of modeling_dragon.py)
        split = int(self._mamba_d_state * 0.5)
        if split % 2 != 0:
            split -= 1
        self._num_rope_angles = split // 2

        self._zero_centered = bool(self.hparams.get("zero_centered_gamma", False))
        self._geodesic      = bool(self.hparams.get("geodesic_update", False))
        if not self._geodesic:
            raise NotImplementedError("Dragon converter currently only handles geodesic_update=true configs")

    # ------------------------------------------------------------------
    # GGUF parameters
    # ------------------------------------------------------------------

    def set_vocab(self):
        # tokenizer_class == "Qwen2Tokenizer" → GPT-2-style BPE.
        # transformers≥5 expects `extra_special_tokens` to be a dict; the Dragon
        # checkpoint stores it as a list. Patch a tokenizer copy in a temp dir
        # rather than mutating the source checkpoint.
        import json, shutil, tempfile
        from pathlib import Path

        cfg_path = self.dir_model / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text())
        if isinstance(cfg.get("extra_special_tokens"), list):
            tmp = Path(tempfile.mkdtemp(prefix="dragon_vocab_"))
            # Only copy the small files AutoTokenizer needs.
            for name in ("tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"):
                src = self.dir_model / name
                if src.exists():
                    shutil.copy(src, tmp / name)
            cfg["extra_special_tokens"] = {t: t for t in cfg["extra_special_tokens"]}
            (tmp / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
            # Re-point the loader at the patched copy for this one call.
            original = self.dir_model
            try:
                self.dir_model = tmp
                self._set_vocab_gpt2()
            finally:
                self.dir_model = original
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        # --- standard fields the base class would set ---
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(int(self.hparams["max_position_embeddings"]))
        self.gguf_writer.add_embedding_length(int(self.hparams["hidden_size"]))
        # `intermediate_size` in the Dragon config is unused at inference (the MoE has its own sizes).
        # Some llama.cpp consumers still want a value; set it to 0 to make it clear there's no dense FFN.
        self.gguf_writer.add_feed_forward_length(0)
        self.gguf_writer.add_head_count(self._n_head)
        # KV heads = num_noise_heads (V-layer GQA fan-out); for M layers there is no attention K/V cache.
        self.gguf_writer.add_head_count_kv(self._n_noise)
        self.gguf_writer.add_key_length(self._head_dim)
        self.gguf_writer.add_value_length(self._head_dim)
        self.gguf_writer.add_layer_norm_rms_eps(float(self.hparams["norm_epsilon"]))
        self.gguf_writer.add_file_type(self.ftype)

        # --- attention ---
        # `slw_wsize` (= 2400 here) is the per-V-layer sliding window. `sliding_window_size`
        # (= 65536) is a legacy / max-context field, NOT the active mask window.
        self.gguf_writer.add_sliding_window(int(self.hparams["slw_wsize"]))
        if (cap := float(self.hparams.get("softcap_attn", 0.0))) > 0.0:
            self.gguf_writer.add_attn_logit_softcapping(cap)

        # --- MoE ---
        self.gguf_writer.add_expert_count(int(self.hparams["moe_num_routed_experts"]))
        self.gguf_writer.add_expert_used_count(int(self.hparams["moe_num_active_experts"]))
        self.gguf_writer.add_expert_feed_forward_length(int(self.hparams["moe_routed_intermediate_size"]))
        if (shared_ff := int(self.hparams.get("moe_shared_intermediate_size", 0))) > 0:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_ff)
            self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_weights_scale(float(self.hparams["moe_routed_scaling_factor"]))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        if (latent := int(self.hparams.get("moe_routed_input_dim", 0))) > 0:
            self.gguf_writer.add_moe_latent_size(latent)

        # --- SSM (Mamba3-MIMO). Note: no causal conv (mamba3_remove_conv=true). ---
        self.gguf_writer.add_ssm_inner_size(self._d_inner)
        self.gguf_writer.add_ssm_state_size(self._mamba_d_state)
        self.gguf_writer.add_ssm_group_count(self._mamba_ngroups)
        self.gguf_writer.add_ssm_time_step_rank(self._mamba_headdim)  # repurposed: per-head value dim for M layer

        # --- Dragon-specific (carry the strictly-novel knobs as arch.* keys) ---
        # NOTE: we use raw add_uint32/add_string for fields that don't have a typed setter.
        kvw = self.gguf_writer
        arch = "dragon"
        kvw.add_uint32(f"{arch}.mamba_mimo_dim",       self._mimo_dim)
        kvw.add_uint32(f"{arch}.mamba_headdim",        self._mamba_headdim)
        kvw.add_uint32(f"{arch}.num_signal_heads",     self._n_sig)
        kvw.add_uint32(f"{arch}.num_noise_heads",      self._n_noise)
        kvw.add_uint32(f"{arch}.tpa_rank",             self._tpa_rank)
        kvw.add_uint32(f"{arch}.num_rope_angles",      self._num_rope_angles)
        kvw.add_string(f"{arch}.layer_types",          self.layer_types)
        # zero-centered γ on RMSNorm: in the original code, `y = rms(x) * (1+w)`.
        # We fold (1+w) at conversion time (see modify_tensors), so runtime can treat
        # the stored weight as a plain RMSNorm gamma. Still record the source flag.
        kvw.add_bool(f"{arch}.zero_centered_gamma",    self._zero_centered)
        # Block-level gate (V layers): silu(gate_proj(x) + gate_bias).
        gate_bias = 1.15 if bool(self.hparams.get("zero_centered_gate", False)) else 0.0
        kvw.add_float32(f"{arch}.gate_bias",           gate_bias)
        kvw.add_bool(f"{arch}.gate_attn",              bool(self.hparams.get("gate_attn", False)))
        kvw.add_bool(f"{arch}.gate_gdn",               bool(self.hparams.get("gate_gdn", False)))
        kvw.add_bool(f"{arch}.geodesic_update",        self._geodesic)
        kvw.add_bool(f"{arch}.final_norm",             bool(self.hparams.get("final_norm", False)))

    # ------------------------------------------------------------------
    # Per-tensor surgery
    # ------------------------------------------------------------------

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # 1. Drop tensors that don't carry weights we need.
        #    `prosres_scalar` is a registered-buffer in DragonGeodesicNorm that
        #    isn't actually used in the forward pass.
        if name.endswith(".prosres_scalar"):
            return

        # 2. Fold zero-centered γ into the stored RMSNorm weight: the model
        #    computes `y = rms(x) * (1 + w_stored)`, so we write `1 + w_stored`
        #    and let runtime treat it as plain RMSNorm γ.
        #    Affected tensors: B_norm, C_norm (M layer), q_norm, k_norm (V layer).
        if self._zero_centered and name.endswith(".norm.weight") and ".mixer." in name:
            data_torch = (1.0 + data_torch.to(torch.float32)).to(data_torch.dtype)

        # 3. Geodesic scalars are stored as 0-d tensors in the checkpoint, which
        #    GGUF doesn't accept; reshape to a 1-element 1-D tensor so they
        #    survive the writer.
        if (
            name.endswith(".geodesic_mixer.scale")
            or name.endswith(".geodesic_mixer.bias")
            or name.endswith(".geodesic_mlp.scale")
            or name.endswith(".geodesic_mlp.bias")
        ):
            data_torch = data_torch.reshape(1)

        # 4. The MoE router weight is stored fp32 in the checkpoint; keep it.
        #    (Base class will pick a quant for non-norm/non-bias tensors;
        #    add an override here if quantizing the router becomes problematic.)

        yield from super().modify_tensors(data_torch, name, bid)
