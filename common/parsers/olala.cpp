#include "parsers.h"

// Olala (Dragon 7A1B) - named channels with XML-wrapped JSON tool calls:
//   <|im_start|>assistant
//     <|channel_start|>analysis<|content|>{reasoning}<|channel_end|>
//     <|channel_start|>final<|content|>{answer}<|channel_end|>
//     [<|channel_start|>tools<|content|><toolcalls><call id="ID"><name>fn</name><arguments>{json}</arguments></call>...</toolcalls><|channel_end|>]
//   <|im_end|>
// The generation prompt leaves the analysis channel open, or closes it when reasoning_effort is "none"/"disabled".
// Semantics follow the vLLM reference parsers (jgcb00/olala parsers/olala/): the tools channel never leaks into content.
common_chat_params common_chat_params_init_olala(const common_chat_template &          tmpl,
                                                 const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string IM_START      = "<|im_start|>";
    const std::string IM_END        = "<|im_end|>";
    const std::string CH_START      = "<|channel_start|>";
    const std::string CH_END        = "<|channel_end|>";
    const std::string CONTENT       = "<|content|>";
    const std::string ASSISTANT     = IM_START + "assistant";
    const std::string ANALYSIS_OPEN = CH_START + "analysis" + CONTENT;
    const std::string FINAL_OPEN    = CH_START + "final" + CONTENT;
    const std::string TOOLS_OPEN    = CH_START + "tools" + CONTENT;
    const std::string CALLS_OPEN    = "<toolcalls>";
    const std::string CALLS_CLOSE   = "</toolcalls>";

    // the template has no enable_thinking, only reasoning_effort. the server maps OAI reasoning_effort "none" to enable_thinking=false
    std::optional<json> additional_context;
    if (!inputs.enable_thinking) {
        additional_context = json{ { "reasoning_effort", "none" } };
    }

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, additional_context);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, std::nullopt, std::nullopt, additional_context);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    data.thinking_start_tag = ANALYSIS_OPEN;
    data.thinking_end_tags  = { CH_END };

    // only the markers are special tokens. channel names and XML tags are plain text
    data.preserved_tokens = {
        CH_START,
        CH_END,
        CONTENT,
        IM_START,
        IM_END,
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, ASSISTANT },
        { COMMON_CHAT_ROLE_USER,      IM_START + "user" },
        { COMMON_CHAT_ROLE_TOOL,      IM_START + "tool" },
        { COMMON_CHAT_ROLE_SYSTEM,    IM_START + "system" },
        { COMMON_CHAT_ROLE_SYSTEM,    IM_START + "developer" },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = ASSISTANT + ANALYSIS_OPEN + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += CH_END + FINAL_OPEN + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto tools_enabled       = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || tools_enabled;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.optional(p.literal(ASSISTANT));

        // with reasoning extraction off, the analysis channel stays inline in the content
        auto analysis_body = p.until_one_of({ CH_END, FINAL_OPEN, TOOLS_OPEN });
        common_peg_parser analysis = p.eps();
        if (extract_reasoning) {
            analysis = p.optional(p.literal(ANALYSIS_OPEN) + p.space() + p.reasoning(analysis_body) +
                                  p.optional(p.literal(CH_END)));
        } else {
            analysis = p.optional(p.content(p.literal(ANALYSIS_OPEN) + analysis_body + p.optional(p.literal(CH_END))));
        }

        // the final channel ends at its closer, or at the tools opener / EOS if the closer is missing
        common_peg_parser final_body = has_response_format
            ? p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema))
            : p.content(p.until_one_of({ CH_END, TOOLS_OPEN, IM_END }));
        auto final_channel = p.literal(FINAL_OPEN) + p.space() + final_body + p.optional(p.literal(CH_END));

        auto trailer = p.space() + p.optional(p.literal(IM_END)) + p.end();

        // drop channels after the answer (stray tools channel, repeated final/tools pair)
        auto extra_channels = p.zero_or_more(p.space() + p.literal(CH_START) + p.until_one_of({ CH_END, IM_END }) +
                                             p.optional(p.literal(CH_END)));

        if (has_response_format) {
            return start + analysis + p.space() + final_channel + trailer;
        }

        if (!tools_enabled) {
            return start + analysis + p.space() + p.optional(final_channel) + extra_channels + trailer;
        }

        //   <call id="ID"><name>NAME</name><arguments>{json}</arguments></call>
        auto tool_choices = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto   schema   = common_chat_tool_parameters(function);

            // name and </name> are atomic, so a name that is a prefix of another tool name is never streamed
            auto call = p.tool(
                p.tool_open(p.literal("<call id=\"") + p.tool_id(p.until("\"")) + p.literal("\">")) + p.space() +
                p.atomic(p.literal("<name>") + p.tool_name(p.literal(name)) + p.literal("</name>")) + p.space() +
                p.literal("<arguments>") + p.space() +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)) + p.space() +
                p.literal("</arguments>") + p.space() +
                p.tool_close(p.literal("</call>")));

            tool_choices |= p.rule("olala-tool-" + name, call);
        });

        auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
        auto calls     = p.repeat(tool_choices + p.space(), 1, max_calls);

        // the channel closer is in the trigger rule, or else the lazy grammar rejects it
        auto tools_channel = p.trigger_rule("olala-tool-calls",
            p.literal(TOOLS_OPEN) + p.space() + p.literal(CALLS_OPEN) + p.space() + calls +
            p.optional(p.literal(CALLS_CLOSE)) + p.optional(p.literal(CH_END)));

        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            return start + analysis + p.space() + p.optional(final_channel) + p.space() + tools_channel + trailer;
        }
        return start + analysis + p.space() + p.optional(final_channel) + p.space() + p.optional(tools_channel) +
               extra_channels + trailer;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, TOOLS_OPEN },
        };
    }

    return data;
}
