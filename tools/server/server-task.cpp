#include "server-task.h"
#include "server-cache-plan-authority.h"
#include "server-cache-destruction-quote.h"

#include "../../common/common-cache-plan-estimate.h"

#include "build-info.h"
#include "server-cache-authority.h"
#include "server-cache-retention-proof.h"
#include "server-vbr-artifact-store.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <limits>
#include <cmath>
#include <new>
#include <thread>
#include <tuple>

using json = nlohmann::ordered_json;

json server_task_result_cache_control::to_json() {
    return server_cache_control_json(operation, result);
}

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    common_chat_msg new_msg;
    try {
        new_msg = common_chat_parse(
            generated_text,
            is_partial,
            chat_parser_params);
    } catch (const std::exception & e) {
        // A parse failure of a malformed generation must never take down the caller: the PEG
        // parser throws on a FINAL parse it cannot match (and on a partial parse that fails at
        // position 0), which aborted llama-cli on an uncaught exception and would fail a fully
        // generated server request. Degrade to the raw text as plain content — and skip the
        // incremental diff machinery entirely: the raw fallback is not prefix-consistent with
        // the earlier partial parses, and string_diff throws (by design) on non-prefix updates.
        SRV_WRN("chat parse failed (%s) — falling back to raw content\n", e.what());
        chat_msg         = {};
        chat_msg.role    = "assistant";
        chat_msg.content = generated_text;
        chat_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        return chat_msg;
    }
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//

// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    if (kv_bpv >= 0.0) {
        base["kv_bpv"] = kv_bpv;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    if (!cache_receipt.is_null()) {
        res["cache_receipt"] = cache_receipt;
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (timings.prompt_n >= 0) {
        server_sent_events.back().at("data").push_back({"timings", timings.to_json()});
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (timings.prompt_n >= 0) {
            data.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            data.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "n_draft_tokens_total",            n_draft_tokens_total },
        { "n_draft_accepted_total",          n_draft_accepted_total },
        { "n_draft_verif_steps_total",       n_draft_verif_steps_total },
        { "n_accepted_per_pos_total",        n_accepted_per_pos_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

json server_task_result_cache_capture::to_json() {
    const char * consistency_name = "_count";
    switch (consistency) {
        case server_cache_capture_consistency::unavailable:
            consistency_name = "unavailable";
            break;
        case server_cache_capture_consistency::capture_exact:
            consistency_name = "capture_exact";
            break;
        case server_cache_capture_consistency::_count:
            break;
    }
    return json {
        { "status", server_vbr_artifact_capture_status_name(status) },
        { "consistency", consistency_name },
        { "reference", reference },
        { "controllers", controllers },
        { "units", units },
        { "companions", companions },
        { "payload_bytes", payload_bytes },
        { "stash_bytes", stash_bytes },
        { "companion_bytes", companion_bytes },
        { "chunks", chunks },
        { "backpressure_waits", backpressure_waits },
        { "event_completions", event_completions },
        { "synchronous_fallbacks", synchronous_fallbacks },
        { "dedup", dedup },
    };
}

const char * server_cache_import_consistency_name(
        server_cache_import_consistency consistency) noexcept {
    switch (consistency) {
        case server_cache_import_consistency::unavailable: return "unavailable";
        case server_cache_import_consistency::capture_exact: return "capture_exact";
        case server_cache_import_consistency::live_rebased: return "live_rebased";
        case server_cache_import_consistency::_count: break;
    }
    return "_count";
}

json server_task_result_cache_import::to_json() {
    const char * consistency_name =
        server_cache_import_consistency_name(consistency);
    return json {
        { "status", server_vbr_artifact_import_status_name(status) },
        { "validation_status",
          vbr_manifest_validation_status_name(validation_status) },
        { "stage_status", vbr_adopt_stage_status_name(stage_status) },
        { "downward_reserve_status",
          vbr_downward_reserve_status_name(downward_reserve_status) },
        { "adopt_status", vbr_adopt_status_name(adopt_status) },
        { "phase", adopt_attempted
              ? json(vbr_adopt_phase_name(phase)) : json(nullptr) },
        { "downward_subphase",
          adopt_attempted
              ? json(vbr_downward_adopt_subphase_name(downward_subphase))
              : json(nullptr) },
        { "downward_edge",
          downward_edge == UINT32_MAX ? json(nullptr) : json(downward_edge) },
        { "decision", vbr_import_decision_name(decision) },
        { "consistency", consistency_name },
        { "units", units },
        { "companions", companions },
        { "payload_bytes", payload_bytes },
        { "companion_bytes", companion_bytes },
    };
}

json server_task_result_cache_plan_preflight::to_json() {
    return server_cache_plan_preflight_json(view);
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
server_prompt_cache::server_prompt_cache(
        int32_t limit_size_mib,
        size_t limit_tokens) {
    limit_size = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens = limit_tokens;
}

bool server_prompt_cache::enable_retention_shadow() noexcept {
    if (!retention_shadow_rows) {
        retention_shadow_rows.reset(new (std::nothrow)
            server_prompt_cache_shadow_row[
                SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES]);
    }
    if (!retention_shadow_rows || !retention_df2_capacity_authority) {
        return bool(retention_shadow_rows);
    }
    if (retention_shadow_artifacts && retention_shadow_lineages) {
        return true;
    }
    std::unique_ptr<server_prompt_cache_shadow_artifact_slot[]> artifacts(
        new (std::nothrow) server_prompt_cache_shadow_artifact_slot[
            SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY]);
    std::unique_ptr<server_prompt_cache_shadow_lineage_slot[]> lineages(
        new (std::nothrow) server_prompt_cache_shadow_lineage_slot[
            SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY]);
    if (!artifacts || !lineages) {
        return false;
    }
    retention_shadow_artifacts = std::move(artifacts);
    retention_shadow_lineages = std::move(lineages);
    return true;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache::iterator server_prompt_cache::find_state_exact(
        const server_tokens & tokens,
        const std::string & adapter_config_key) noexcept {
    return std::find_if(states.begin(), states.end(), [&](const auto & state) {
        // Identity-scoped [I6]: token equality under another adapter is not a
        // durable copy. Equal length closes the recurrent/hybrid prefix hole.
        return state.adapter_config_key == adapter_config_key &&
               state.prompt.tokens.size() == tokens.size() &&
               state.prompt.tokens.get_common_prefix(tokens) == tokens.size();
    });
}

server_prompt_cache::const_iterator server_prompt_cache::find_state_exact(
        const server_tokens & tokens,
        const std::string & adapter_config_key) const noexcept {
    return const_cast<server_prompt_cache *>(this)->find_state_exact(
        tokens, adapter_config_key);
}

bool server_prompt_cache::contains(
        const server_tokens & tokens,
        const std::string & adapter_config_key) const {
    return find_state_exact(tokens, adapter_config_key) != states.end();
}

void server_prompt_cache::cache_plan_begin_inventory() noexcept {
    cache_plan_next_source_id = 0;
    for (auto & state : states) {
        state.cache_plan_source_id = -1;
    }
}

bool server_prompt_cache::cache_plan_get_source_id(
        server_prompt_cache_state & state,
        int32_t & source_id) noexcept {
    return server_cache_plan_assign_source_id(
        state.cache_plan_source_id, cache_plan_next_source_id, source_id);
}

std::list<server_prompt_cache_state> server_prompt_cache::stage(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft, std::string adapter_config_key) {
    // calculate checkpoints size to see if it will fit with the prompt. This prices the
    // invalidate-first COPY made below (checkpoint copies drop their generation-record shadow),
    // so admission and eviction stay byte-identical to pre-shadow accounting.
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // this state can't be cached at all; report failure (the caller keeps the live slot)
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return {};
    }

    // Allocate the entry as a DETACHED single-node list, entirely outside `states`. Every allocation
    // that can throw (the list node, the state vectors, the token clone, the checkpoint copy) is
    // performed here; on any failure we return an empty list and leave the cache completely
    // untouched — no eviction, no limit reduction for a save that did not happen [I7]. publish()
    // then splices this node in without allocating.
    std::list<server_prompt_cache_state> staged;
    try {
        staged.emplace_back();
        auto & entry = staged.back();

        entry.data.main.resize(state_size_tgt);
        entry.data.drft.resize(state_size_dft);
        entry.prompt.tokens      = prompt.tokens.clone();
        entry.prompt.checkpoints = prompt.checkpoints;
        entry.prompt.sequence_epoch = prompt.sequence_epoch;
        entry.adapter_config_key = std::move(adapter_config_key);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());
        return {};
    }

    return staged;
}

bool server_prompt_cache::payload_bytes(
        const server_prompt_cache_state & st,
        uint64_t & snapshot_bytes,
        uint64_t & checkpoint_bytes,
        uint64_t & accelerator_bytes) noexcept {
    snapshot_bytes    = uint64_t(st.data.size());
    checkpoint_bytes  = 0;
    accelerator_bytes = 0;
    const auto add_checked = [](uint64_t & acc, size_t value) {
        if (uint64_t(value) > std::numeric_limits<uint64_t>::max() - acc) {
            return false;
        }
        acc += uint64_t(value);
        return true;
    };
    for (const auto & ckpt : st.prompt.checkpoints) {
        if (!add_checked(checkpoint_bytes, ckpt.data_tgt.size()) ||
            !add_checked(checkpoint_bytes, ckpt.data_dft.size()) ||
            !add_checked(accelerator_bytes, ckpt.accel.size())) {
            snapshot_bytes = checkpoint_bytes = accelerator_bytes = 0;
            return false;
        }
    }
    return true;
}

bool server_prompt_cache::payload_leaves(
        server_prompt_cache_state & st,
        std::array<server_prompt_cache_payload_leaf, 3> & leaves) noexcept {
    leaves = {{
        {
            llama_cache_acct_category::full_snapshot_payload,
            0,
            &st.acct_op_snapshot,
        },
        {
            llama_cache_acct_category::checkpoint_state_payload,
            0,
            &st.acct_op_ckpt,
        },
        {
            llama_cache_acct_category::typed_accelerator_payload,
            0,
            &st.acct_op_accel,
        },
    }};
    uint64_t snapshot_bytes = 0;
    uint64_t checkpoint_bytes = 0;
    uint64_t accelerator_bytes = 0;
    if (!payload_bytes(
            st, snapshot_bytes, checkpoint_bytes, accelerator_bytes)) {
        return false;
    }
    leaves[0].bytes = snapshot_bytes;
    leaves[1].bytes = checkpoint_bytes;
    leaves[2].bytes = accelerator_bytes;
    return true;
}

// C0 shadow producer [P2]: one accounting transaction per charged leaf category of a published
// entry, at the publication boundary (the splice into `states`), released when the entry leaves
// `states` on any path. Aggregate entry size is a provider grouping and is NEVER charged — the
// leaves below are mutually exclusive so their sum cannot double-count. The fill-failure abort
// mapping (stage() → abort) lands with F's real artifact transaction. The ledger is
// non-throwing by contract, so no accounting failure can escape into the shipped cache path;
// the `acct_unavailable` fault seam proves that invariance in the gate.
void server_prompt_cache::acct_charge_entry(server_prompt_cache_state & st) {
    if (!acct) {
        return;
    }
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);

    // checked sums: an overflowing observation latches the leaf unavailable instead of
    // charging a fabricated value (the shipped path is untouched either way)
    std::array<server_prompt_cache_payload_leaf, 3> leaves;
    const bool sums_ok = payload_leaves(st, leaves);

    if (!sums_ok || server_fault("acct_unavailable")) { // [P2 gate] shipped-path invariance seam
        for (const auto & leaf : leaves) {
            server_cache_acct_mark_shadow_unavailable(
                *acct, leaf.category, domain,
                llama_cache_acct_producer::host_cache);
        }
        return;
    }

    for (const auto & leaf : leaves) {
        *leaf.operation = server_cache_acct_charge_shadow(
            *acct, leaf.category, domain,
            llama_cache_acct_producer::host_cache, {},
            leaf.bytes, leaf.bytes);
    }
}

void server_prompt_cache::acct_release_entry(server_prompt_cache_state & st) {
    if (!acct) {
        return;
    }
    for (const auto op : st.release_ops()) {
        if (op) {
            acct->release(op);
        }
    }
    st.acct_op_snapshot = {};
    st.acct_op_ckpt = {};
    st.acct_op_accel = {};
}

bool server_cache_lease_build_identity(
        const std::string & execution_identity,
        const std::string & adapter_identity,
        const server_tokens & tokens,
        int64_t coverage_tokens,
        server_cache_lease_identity & out) {
    if (execution_identity.empty() ||
        adapter_identity.empty() ||
        coverage_tokens < 0) {
        return false;
    }
    out.execution_identity = execution_identity;
    out.adapter_config_identity = adapter_identity;
    return tokens.media_content_identity(
               coverage_tokens, out.media_content_identity) &&
           out.valid();
}

static bool server_prompt_retention_exact_scope(
        const server_prompt & prompt,
        const std::string & adapter_config_key,
        int64_t coverage_tokens,
        std::string & out) noexcept {
    out.clear();
    if (coverage_tokens < 0 ||
        uint64_t(coverage_tokens) > prompt.tokens.size()) {
        return false;
    }
    std::string media_identity;
    if (!prompt.tokens.media_content_identity(
            coverage_tokens, media_identity)) {
        return false;
    }
    try {
        const uint64_t adapter_size = adapter_config_key.size();
        const uint64_t media_size = media_identity.size();
        size_t total = sizeof(adapter_size);
        if (adapter_size > SIZE_MAX - total) {
            return false;
        }
        total += size_t(adapter_size);
        if (sizeof(media_size) > SIZE_MAX - total) {
            return false;
        }
        total += sizeof(media_size);
        if (media_size > SIZE_MAX - total) {
            return false;
        }
        total += size_t(media_size);
        out.reserve(total);
        out.append(
            reinterpret_cast<const char *>(&adapter_size),
            sizeof(adapter_size));
        out.append(adapter_config_key);
        out.append(
            reinterpret_cast<const char *>(&media_size),
            sizeof(media_size));
        out.append(media_identity);
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_prompt_retention_publish_exact_prefix(
        server_retention_sidecar_store & retention,
        const server_retention_instance_key & key,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens) noexcept {
    if (!retention.prefix_tracking_enabled()) {
        return true;
    }
    std::string scope;
    if (!server_prompt_retention_exact_scope(
            prompt, adapter_identity, coverage_tokens, scope)) {
        return retention.publish_prefix(
            key, {}, prompt.tokens.retention_token_ids());
    }
    return retention.publish_prefix(
        key, scope, prompt.tokens.retention_token_ids());
}

static void server_prompt_cache_mirror_prefix(
        server_prompt_cache & cache,
        const server_retention_instance_key & key,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens) noexcept {
    if (cache.retention_obs) {
        (void) server_prompt_retention_publish_exact_prefix(
            *cache.retention_obs, key, prompt, adapter_identity,
            coverage_tokens);
    }
}

static void server_prompt_cache_mirror_lease(
        server_prompt_cache & cache,
        bool mirrored,
        const server_cache_lease_subject * source,
        const server_cache_lease_subject & destination,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens) {
    if (!mirrored || !cache.lease_obs) {
        return;
    }
    server_cache_lease_identity identity;
    if (destination.valid() &&
        cache.lease_execution_identity &&
        server_cache_lease_build_identity(
            *cache.lease_execution_identity, adapter_identity,
            prompt.tokens, coverage_tokens, identity)) {
        if (source) {
            (void) cache.lease_obs->artifact_cloned(
                *source, destination, identity);
        } else {
            (void) cache.lease_obs->artifact_rebound(
                destination.artifact, identity);
        }
    } else {
        cache.lease_obs->artifact_identity_unavailable(destination);
    }
}

static void server_prompt_cache_mirror_artifact_clone(
        server_prompt_cache & cache,
        const server_retention_instance_key & source_key,
        common_retention_artifact_kind source_kind,
        int32_t source_slot,
        const server_retention_instance_key & destination_key,
        common_retention_artifact_kind destination_kind,
        int32_t destination_slot,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens) {
    if (!cache.retention_obs) {
        return;
    }

    const bool cloned = cache.retention_obs->clone(
        source_key, destination_key);
    if (cloned &&
        destination_kind != common_retention_artifact_kind::checkpoint) {
        server_prompt_cache_mirror_prefix(
            cache, destination_key, prompt, adapter_identity,
            coverage_tokens);
    }
    const server_cache_lease_subject source {
        cache.retention_obs->artifact_id(source_key),
        source_kind,
        source_slot,
    };
    const server_cache_lease_subject destination {
        cache.retention_obs->artifact_id(destination_key),
        destination_kind,
        destination_slot,
    };
    server_prompt_cache_mirror_lease(
        cache, cloned, &source, destination, prompt,
        adapter_identity, coverage_tokens);
}

static server_cache_destruction_admission server_prompt_cache_observe_drop(
        server_prompt_cache & cache,
        const server_prompt_cache_state & state,
        server_cache_destruction_reason reason) noexcept {
    if (!cache.destruction_obs) {
        return {};
    }

    server_cache_destruction_request request;
    request.cls    = server_cache_destruction_class::host_artifact_drop;
    request.reason = reason;
    request.add_target(
        server_cache_destruction_target_kind::host_artifact,
        -1,
        cache.retention_obs ? cache.retention_obs->artifact_id(
            server_retention_instance_key::for_host_entry(&state))
            : llama_cache_acct_artifact_id{});

    const auto ops = state.release_ops();
    const llama_cache_acct_category categories[] = {
        llama_cache_acct_category::full_snapshot_payload,
        llama_cache_acct_category::checkpoint_state_payload,
        llama_cache_acct_category::typed_accelerator_payload,
    };
    for (size_t i = 0; i < std::size(ops); ++i) {
        llama_cache_acct_release_preview preview;
        const bool known = cache.acct && ops[i] &&
            cache.acct->preview_release(ops[i], preview);
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated }) {
            server_cache_destruction_yield value;
            value.category = known ? preview.category : categories[i];
            value.measure  = measure;
            value.domain_known = known;
            if (known) {
                value.domain = preview.domain;
                value.value = measure == llama_cache_acct_measure::logical_payload
                    ? preview.logical_payload
                    : preview.resident_allocated;
            }
            request.add_yield(value);
        }
    }
    return server_cache_retention_admit(cache.destruction_obs, request);
}

struct server_prompt_cache_retirement_manifest {
    server_retention_instance_key host;
    std::vector<server_retention_instance_key> checkpoints;
};

static bool server_prompt_cache_capture_retirement(
        server_prompt_cache & cache,
        server_prompt_cache::iterator it,
        server_prompt_cache_retirement_manifest & manifest) noexcept {
    try {
        manifest = {};
        if (!cache.retention_obs) {
            return true;
        }
        manifest.host =
            server_retention_instance_key::for_host_entry(&*it);
        manifest.checkpoints.reserve(it->prompt.checkpoints.size());
        for (auto & checkpoint : it->prompt.checkpoints) {
            manifest.checkpoints.push_back(
                server_retention_instance_key::for_checkpoint(
                    -1, &checkpoint));
        }
        return true;
    } catch (...) {
        manifest = {};
        return false;
    }
}

static void server_prompt_cache_retire_manifest(
        server_prompt_cache & cache,
        const server_prompt_cache_retirement_manifest & manifest) noexcept {
    if (!cache.retention_obs) {
        return;
    }
    cache.retention_obs->retire(manifest.host);
    for (const auto & checkpoint : manifest.checkpoints) {
        cache.retention_obs->retire(checkpoint);
    }
}

static void server_prompt_cache_retire_entry(
        server_prompt_cache & cache,
        server_prompt_cache::iterator it) noexcept {
    // Retention retirement is a post-capability finalizer on authoritative
    // paths because it releases the sidecar provenance op. The legacy wrapper
    // invokes it at its historical pre-erase position.
    if (!cache.retention_obs) {
        return;
    }
    cache.retention_obs->retire(
        server_retention_instance_key::for_host_entry(&*it));
    for (auto & checkpoint : it->prompt.checkpoints) {
        cache.retention_obs->retire(
            server_retention_instance_key::for_checkpoint(
                -1, &checkpoint));
    }
}

static server_prompt_cache::iterator server_prompt_cache_destroy_entry_impl(
        server_prompt_cache & cache,
        server_prompt_cache::iterator it) {
    GGML_ASSERT(it->recovery_pins == 0);
    return cache.states.erase(it);
}

server_prompt_cache::iterator server_prompt_cache::destroy_entry(
        iterator it,
        server_cache_destruction_reason reason) {
    return destroy_entry_impl(it, reason, states.end());
}

using server_cache_checkpoint_iterator =
    server_cache_checkpoint_authority_context::checkpoint_iterator;

void server_cache_checkpoint_ring_changed(
        server_cache_checkpoint_authority_context & context) noexcept {
    context.attempts.ring_changed();
    context.seam_heuristic = nullptr;
    context.thinning_refusal =
        common_cache_plan_destruction_reason::none;
    context.floor_refusal =
        common_cache_plan_destruction_reason::mandatory_anchor;
}

bool server_cache_checkpoint_thinning_attempt_begin(
        server_cache_checkpoint_authority_context & context,
        bool capacity_mode) noexcept {
    return context.attempts.begin(
        capacity_mode
            ? server_cache_checkpoint_attempt_lane::capacity_thinning
            : server_cache_checkpoint_attempt_lane::optional_thinning);
}

bool server_cache_checkpoint_refusal_state_changed(
        server_cache_checkpoint_authority_context & context,
        common_cache_plan_destruction_reason reason,
        bool publication_skip) noexcept {
    return context.attempts.refusal_changed(reason, publication_skip);
}

server_cache_destruction_admission server_cache_checkpoint_observe_drop(
        const server_cache_checkpoint_authority_context & context,
        server_cache_destruction_reason reason,
        llama_cache_acct_artifact_id artifact) noexcept {
    server_cache_destruction_request request;
    request.cls = server_cache_destruction_class::checkpoint_drop;
    request.reason = reason;
    request.add_target(
        server_cache_destruction_target_kind::checkpoint_ring,
        context.slot_id, artifact);
    request.add_yield(
        llama_cache_acct_category::checkpoint_state_payload);
    return server_cache_retention_admit(context.destruction, request);
}

namespace {

bool build_checkpoint_destruction_artifact(
        const server_cache_checkpoint_authority_context & context,
        server_cache_checkpoint_iterator checkpoint,
        server_cache_destruction_artifact & out) noexcept {
    out = {};
    try {
        if (!context.retention || !context.leases ||
            checkpoint == context.checkpoints.end()) {
            return false;
        }
        const auto key = server_retention_instance_key::for_checkpoint(
            context.slot_id, &*checkpoint);
        server_retention_checkpoint_inventory inventory;
        server_retention_candidate catalog;
        if (!context.retention->checkpoint_inventory(key, inventory) ||
            !inventory.identity_known || !inventory.release_owned ||
            !context.retention->candidate_for_instance(key, catalog) ||
            catalog.artifact_id.v == 0 ||
            catalog.record.kind !=
                common_retention_artifact_kind::checkpoint ||
            catalog.release_ops.empty()) {
            return false;
        }
        out.candidate.artifact_id = catalog.artifact_id;
        out.candidate.record = catalog.record;
        out.candidate.lineage = catalog.lineage;
        out.candidate.availability = catalog.avail;
        out.candidate.release_ops = catalog.release_ops;
        out.candidate.identity_known = true;
        out.candidate.lease = inventory.lease;
        out.kind = common_retention_artifact_kind::checkpoint;
        out.owner_slot = context.slot_id;
        out.pool = catalog.record.stamp.pool;
        out.mandatory_anchor =
            catalog.record.stamp.mandatory_anchor;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

void emit_checkpoint_destruction(
        const server_cache_checkpoint_authority_context & context,
        const common_cache_plan_destruction_receipt & receipt,
        uint64_t projected_bytes,
        uint64_t price_us,
        uint32_t weight_milli,
        uint32_t ordinal) noexcept {
    if (!context.debug_observability) {
        return;
    }
    try {
        json payload = server_cache_destruction_receipt_json(
            receipt, projected_bytes, "checkpoint_drop");
        payload["price_us"] = price_us;
        payload["retention_weight_milli"] = weight_milli;
        payload["rank_ordinal"] = ordinal;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n",
                payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb checkpoint ownership.
    }
}

bool checkpoint_drop_certified(
        server_cache_checkpoint_authority_context & context,
        server_cache_checkpoint_iterator victim,
        server_cache_checkpoint_iterator recovery,
        server_cache_destruction_reason reason,
        uint64_t price_us,
        uint32_t weight_milli,
        uint32_t ordinal,
        server_cache_checkpoint_iterator & next) noexcept {
    if (!context.authority || !context.retention || !context.destruction ||
        victim == context.checkpoints.end() ||
        recovery == context.checkpoints.end() || victim == recovery) {
        return false;
    }
    auto & authority = *context.authority;
    const uint64_t sequence = ++authority.destruction_quote_sequence;
    const auto refuse = [&](common_cache_plan_destruction_receipt * existing,
                            common_cache_plan_destruction_reason why) {
        context.thinning_refusal = why;
        if (!server_cache_checkpoint_refusal_state_changed(context, why)) {
            return;
        }
        common_cache_plan_destruction_receipt receipt = existing
            ? std::move(*existing)
            : common_cache_plan_destruction_receipt{};
        receipt.state = common_cache_plan_destruction_state::refused;
        receipt.reason = why;
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::checkpoint_member_drop);
        receipt.admission_sequence = sequence;
        authority.observe_host_destruction(receipt, true);
        context.destruction->note_checkpoint_thin_refused();
        emit_checkpoint_destruction(context,
            receipt, 0, price_us, weight_milli, ordinal);
    };

    server_cache_destruction_artifact victim_artifact;
    server_cache_destruction_artifact recovery_artifact;
    if (!build_checkpoint_destruction_artifact(context,
            victim, victim_artifact) ||
        !build_checkpoint_destruction_artifact(context,
            recovery, recovery_artifact)) {
        refuse(nullptr,
               common_cache_plan_destruction_reason::manifest_incomplete);
        return false;
    }
    const auto recovery_key =
        server_retention_instance_key::for_checkpoint(context.slot_id, &*recovery);
    auto pin = context.retention->acquire_recovery_pin(recovery_key);
    if (!pin.valid() || !pin.binds_exact(
            recovery_artifact.candidate.artifact_id,
            recovery_artifact.candidate.release_ops)) {
        refuse(nullptr,
               common_cache_plan_destruction_reason::recovery_unavailable);
        return false;
    }

    const auto preview = [&](const auto & ops, uint64_t serial,
                             auto & released) {
        return authority.ledger.preview_release_set(
            ops, serial, released);
    };
    const auto project = [&](const auto & released, auto & domains) {
        return authority.project_release(released, domains);
    };
    const uint64_t accounting_serial = authority.ledger.serial();
    auto quote = server_cache_destruction_quote_single_artifact(
        victim_artifact,
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::checkpoint_member_drop),
        accounting_serial, sequence,
        preview, project);
    if (quote.receipt.state !=
            common_cache_plan_destruction_state::quoted) {
        const auto why = quote.receipt.reason;
        refuse(&quote.receipt, why);
        return false;
    }
    authority.observe_host_destruction(quote.receipt, false);
    std::vector<server_cache_destruction_artifact> current;
    try {
        current.push_back(std::move(victim_artifact));
    } catch (...) {
        refuse(&quote.receipt,
               common_cache_plan_destruction_reason::internal_fault);
        return false;
    }
    auto prepared = server_cache_prepare_release_set(
        quote, current, authority.ledger, authority.ledger.serial(),
        project, std::move(pin));
    if (prepared.status !=
            server_cache_prepare_release_status::prepared) {
        refuse(&quote.receipt, prepared.reason);
        return false;
    }
    uint64_t projected_bytes = 0;
    for (const auto & row : quote.projected_domains) {
        if (row.projected_release_bytes.state !=
                llama_cache_acct_known::known ||
            row.projected_release_bytes.value >
                std::numeric_limits<uint64_t>::max() - projected_bytes) {
            refuse(&quote.receipt,
                   common_cache_plan_destruction_reason::
                       accounting_unavailable);
            return false;
        }
        projected_bytes += row.projected_release_bytes.value;
    }
    quote.receipt.displaced_fate =
        common_cache_plan_displaced_fate::exact_replay_recipe;
    quote.receipt.recovery_citation =
        common_cache_plan_recovery_citation::resolved;
    quote.receipt.recovery_source_artifact_id =
        recovery_artifact.candidate.artifact_id;
    quote.receipt.recovery_source_manifest_digest =
        server_cache_destruction_recovery_source_digest(
            recovery_artifact.candidate.artifact_id,
            recovery_artifact.candidate.release_ops);
    quote.receipt.state =
        common_cache_plan_destruction_state::certified;
    authority.observe_host_destruction(quote.receipt, true);
    emit_checkpoint_destruction(context,
        quote.receipt, projected_bytes,
        price_us, weight_milli, ordinal);

    const auto victim_key =
        server_retention_instance_key::for_checkpoint(context.slot_id, &*victim);
    const auto admission = server_cache_checkpoint_observe_drop(context,
        reason, current.front().candidate.artifact_id);
    const std::thread::id scheduler_owner = std::this_thread::get_id();
    GGML_ASSERT(context.raw_owner && context.raw_drop);
    next = context.raw_drop(
        context.raw_owner, victim, std::next(victim));
    // The typed raw_drop adapter is pinned to the slot's X-macro _impl door;
    // that door only advances the ring latch and erases this list node. The
    // node destructor frees checkpoint-owned vectors and shadow metadata and
    // cannot write C, so no ledger producer can interleave before commit.
    GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
    server_cache_recovery_pin retained_pin;
    const auto committed = prepared.capability.commit(retained_pin);
    GGML_ASSERT(committed ==
                common_cache_plan_destruction_reason::none);
    context.retention->retire_after_committed_release(victim_key);
    quote.receipt.state =
        common_cache_plan_destruction_state::executed;
    quote.receipt.actual_accounting_serial =
        authority.ledger.serial();
    authority.observe_host_destruction(quote.receipt, false);
    emit_checkpoint_destruction(context,
        quote.receipt, projected_bytes,
        price_us, weight_milli, ordinal);
    context.destruction->note_checkpoint_thin_executed(
        admission.sequence, projected_bytes);
    return true;
}

} // namespace

bool server_cache_checkpoint_thin_priced(
        server_cache_checkpoint_authority_context & context,
        int checkpoint_task_id,
        uint64_t max_replay_tokens,
        const common_prompt_checkpoint * seam_heuristic,
        bool capacity_mode,
        bool attempt_claimed) noexcept {
    if (!context.authority || !context.retention || !context.leases ||
        context.checkpoints.size() < 2) {
        return false;
    }
    if (!attempt_claimed &&
        !server_cache_checkpoint_thinning_attempt_begin(context, capacity_mode)) {
        return false;
    }
    context.thinning_refusal =
        common_cache_plan_destruction_reason::none;
    context.leases->lifecycle_point();
    struct local_candidate {
        server_cache_checkpoint_iterator victim;
        server_cache_checkpoint_iterator recovery;
        server_cache_checkpoint_trade_input price;
    };
    struct member_inventory {
        server_cache_checkpoint_iterator member;
        server_retention_checkpoint_inventory catalog;
        bool found = false;
    };
    std::vector<local_candidate> local;
    std::vector<server_cache_checkpoint_trade_input> prices;
    try {
        local.reserve(context.checkpoints.size());
        prices.reserve(context.checkpoints.size());
        std::vector<member_inventory> inventory;
        inventory.reserve(context.checkpoints.size());
        for (auto it = context.checkpoints.begin();
             it != context.checkpoints.end(); ++it) {
            member_inventory member;
            member.member = it;
            member.found = context.retention->checkpoint_inventory(
                server_retention_instance_key::for_checkpoint(context.slot_id, &*it),
                member.catalog);
            inventory.push_back(std::move(member));
        }

        size_t previous_index = 0;
        for (size_t index = 1; index < inventory.size(); ++index) {
            auto it = inventory[index].member;
            auto previous = inventory[previous_index].member;
            const bool close = it->n_tokens >= previous->n_tokens &&
                uint64_t(it->n_tokens - previous->n_tokens) <=
                    max_replay_tokens;
            if ((!capacity_mode && !close) ||
                it->id_task == checkpoint_task_id) {
                previous_index = index;
                continue;
            }

            local_candidate candidate;
            candidate.victim = it;
            candidate.recovery = previous;
            candidate.price.ordinal = uint32_t(index);
            candidate.price.recovery_ordinal =
                uint32_t(previous_index);
            candidate.price.payload_bytes = it->size();
            candidate.price.replay_tokens =
                it->n_tokens >= previous->n_tokens
                    ? uint64_t(it->n_tokens - previous->n_tokens)
                    : UINT64_MAX;
            candidate.price.seam_heuristic_protected =
                seam_heuristic == &*it;
            const bool same_replay_lineage =
                server_cache_checkpoint_bounded_replay(
                    *previous, *it, max_replay_tokens);
            candidate.price.recovery_available =
                same_replay_lineage &&
                inventory[previous_index].found &&
                inventory[previous_index].catalog.identity_known &&
                inventory[previous_index].catalog.release_owned;
            const auto & victim_catalog = inventory[index].catalog;
            if (inventory[index].found &&
                victim_catalog.identity_known &&
                victim_catalog.release_owned) {
                candidate.price.artifact =
                    victim_catalog.artifact_id;
                candidate.price.stable_id =
                    victim_catalog.stable_id;
                candidate.price.identity_known = true;
                candidate.price.mandatory_anchor =
                    victim_catalog.mandatory_anchor ||
                    victim_catalog.recovery_pinned;
                candidate.price.hard_leased = server_cache_lease_is_hard(
                    victim_catalog.lease);
                uint32_t weight = 0;
                GGML_ASSERT(server_cache_retention_weight_milli(
                    victim_catalog.lease.cls ==
                        server_cache_lease_class::soft,
                    context.main_family,
                    SERVER_CACHE_HOST_WEIGHT_SCALE, weight));
                candidate.price.weight_milli = weight;
            }
            local.push_back(std::move(candidate));
            prices.push_back(local.back().price);
            if (!close) {
                previous_index = index;
            }
        }
    } catch (...) {
        return false;
    }
    if (local.empty()) {
        return false;
    }

    const auto * calib = common_cache_plan_calib_find(
        context.authority->calibration_profile);
    while (!local.empty()) {
        const auto plan = server_cache_plan_checkpoint_thinning(
            prices, calib);
        if (!plan.selected) {
            context.thinning_refusal = plan.reason;
            if (!server_cache_checkpoint_refusal_state_changed(context, plan.reason)) {
                return false;
            }
            common_cache_plan_destruction_receipt receipt;
            receipt.state = common_cache_plan_destruction_state::refused;
            receipt.reason = plan.reason;
            receipt.effects = common_cache_plan_destruction_effect_bit(
                common_cache_plan_destruction_effect::
                    checkpoint_member_drop);
            receipt.admission_sequence =
                ++context.authority->destruction_quote_sequence;
            context.authority->observe_host_destruction(receipt, true);
            if (context.destruction) {
                context.destruction->note_checkpoint_thin_refused();
                if (plan.protection !=
                        server_cache_checkpoint_protection::none) {
                    switch (plan.protection) {
                        case server_cache_checkpoint_protection::
                                 seam_heuristic:
                            context.destruction->
                                note_checkpoint_thin_heuristic_refused();
                            break;
                        case server_cache_checkpoint_protection::
                                 mandatory_anchor:
                            context.destruction->
                                note_checkpoint_thin_mandatory_refused();
                            break;
                        case server_cache_checkpoint_protection::
                                 hard_lease:
                            context.destruction->
                                note_checkpoint_thin_hard_lease_refused();
                            break;
                        case server_cache_checkpoint_protection::none:
                        case server_cache_checkpoint_protection::_count:
                            break;
                    }
                }
            }
            emit_checkpoint_destruction(context,
                receipt, 0, 0,
                SERVER_CACHE_HOST_WEIGHT_SCALE, UINT32_MAX);
            return false;
        }
        const auto chosen = std::find_if(
            local.begin(), local.end(), [&](const auto & candidate) {
                return candidate.price.ordinal == plan.ordinal;
            });
        if (chosen == local.end()) {
            return false;
        }
        const auto chosen_index = size_t(chosen - local.begin());
        server_cache_checkpoint_iterator next;
        if (checkpoint_drop_certified(context,
                chosen->victim, chosen->recovery,
                capacity_mode
                    ? server_cache_destruction_reason::checkpoint_capacity
                    : server_cache_destruction_reason::checkpoint_thin,
                plan.price_us, plan.weight_milli,
                plan.ordinal, next)) {
            return true;
        }
        local.erase(chosen);
        prices.erase(prices.begin() + chosen_index);
    }
    return false;
}

bool server_cache_checkpoint_capacity_floor(
        server_cache_checkpoint_authority_context & context,
        int checkpoint_task_id,
        const common_prompt_checkpoint * seam_heuristic,
        server_cache_checkpoint_iterator & victim,
        common_cache_plan_destruction_reason & refusal) noexcept {
    victim = context.checkpoints.end();
    refusal = context.floor_refusal;
    if (!context.attempts.begin(
            server_cache_checkpoint_attempt_lane::capacity_floor)) {
        return false;
    }
    refusal = common_cache_plan_destruction_reason::mandatory_anchor;
    if (context.leases) {
        context.leases->lifecycle_point();
    }
    std::vector<server_cache_checkpoint_floor_input> inputs;
    std::vector<server_cache_checkpoint_iterator> members;
    try {
        inputs.reserve(context.checkpoints.size());
        members.reserve(context.checkpoints.size());
        uint32_t ordinal = 0;
        for (auto it = context.checkpoints.begin();
             it != context.checkpoints.end(); ++it, ++ordinal) {
            server_cache_checkpoint_floor_input input;
            input.ordinal = ordinal;
            const auto key =
                server_retention_instance_key::for_checkpoint(context.slot_id, &*it);
            server_retention_checkpoint_inventory catalog;
            const bool catalog_found = context.retention &&
                context.retention->checkpoint_inventory(key, catalog);
            input.recovery_pinned = catalog_found &&
                catalog.recovery_pinned;
            if (it->id_task == checkpoint_task_id ||
                input.recovery_pinned) {
                input.protection =
                    server_cache_checkpoint_protection::mandatory_anchor;
            } else if (seam_heuristic == &*it) {
                input.protection =
                    server_cache_checkpoint_protection::seam_heuristic;
            }
            if (catalog_found) {
                if (catalog.mandatory_anchor) {
                    input.protection =
                        server_cache_checkpoint_protection::
                            mandatory_anchor;
                }
                if (catalog.identity_known &&
                    server_cache_lease_is_hard(catalog.lease)) {
                    input.protection =
                        server_cache_checkpoint_protection::hard_lease;
                }
            }
            inputs.push_back(input);
            members.push_back(it);
        }
    } catch (...) {
        refusal = common_cache_plan_destruction_reason::internal_fault;
        context.floor_refusal = refusal;
        return false;
    }
    const auto plan = server_cache_plan_checkpoint_capacity_floor(inputs);
    refusal = plan.reason;
    context.floor_refusal = refusal;
    if (!plan.selected || plan.ordinal >= members.size()) {
        return false;
    }
    victim = members[plan.ordinal];
    return true;
}

void server_cache_checkpoint_publication_skipped(
        server_cache_checkpoint_authority_context & context,
        common_cache_plan_destruction_reason reason) noexcept {
    if (!context.authority ||
        !server_cache_checkpoint_refusal_state_changed(context, reason, true)) {
        return;
    }
    common_cache_plan_destruction_receipt receipt;
    receipt.state = common_cache_plan_destruction_state::refused;
    receipt.reason = reason;
    receipt.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::checkpoint_member_drop);
    receipt.admission_sequence =
        ++context.authority->destruction_quote_sequence;
    context.authority->observe_host_destruction(receipt, true);
    if (context.destruction) {
        context.destruction->note_checkpoint_publication_skip();
    }
    emit_checkpoint_destruction(context,
        receipt, 0, 0, SERVER_CACHE_HOST_WEIGHT_SCALE, UINT32_MAX);
}


namespace {

bool checkpoint_payload_equal(
        const common_prompt_checkpoint & a,
        const common_prompt_checkpoint & b) noexcept {
    return a.n_tokens == b.n_tokens &&
           a.id_task == b.id_task &&
           a.pos_min == b.pos_min &&
           a.pos_max == b.pos_max &&
           a.checkpoint_epoch == b.checkpoint_epoch &&
           a.checkpoint_epoch_swa == b.checkpoint_epoch_swa &&
           a.computation_frontier == b.computation_frontier &&
           a.data_tgt == b.data_tgt &&
           a.data_dft == b.data_dft &&
           a.accel.ring == b.accel.ring &&
           a.accel.spec == b.accel.spec;
}

bool build_host_destruction_artifact(
        server_prompt_cache & cache,
        server_prompt_cache_state & state,
        server_cache_destruction_artifact & out) noexcept {
    out = {};
    try {
        if (!cache.retention_obs || !cache.lease_obs ||
            !cache.lease_execution_identity) {
            return false;
        }
        server_retention_candidate catalog;
        const auto key = server_retention_instance_key::for_host_entry(&state);
        if (!cache.retention_obs->candidate_for_instance(key, catalog) ||
            catalog.artifact_id.v == 0 ||
            catalog.record.kind != common_retention_artifact_kind::host_entry) {
            return false;
        }
        server_cache_lease_identity identity;
        if (!server_cache_lease_build_identity(
                *cache.lease_execution_identity,
                state.adapter_config_key,
                state.prompt.tokens,
                state.prompt.n_tokens(),
                identity)) {
            return false;
        }
        out.candidate.artifact_id = catalog.artifact_id;
        out.candidate.record = catalog.record;
        out.candidate.lineage = catalog.lineage;
        out.candidate.availability = catalog.avail;
        out.candidate.lease = cache.lease_obs->inspect(
            catalog.artifact_id, identity);
        out.candidate.identity_known = true;
        out.candidate.release_ops.reserve(3);
        for (const auto op : state.release_ops()) {
            if (!op) {
                return false;
            }
            out.candidate.release_ops.push_back(op);
        }
        out.kind = common_retention_artifact_kind::host_entry;
        out.host_source_id = state.cache_plan_source_id;
        out.pool = catalog.record.stamp.pool;
        out.mandatory_anchor = catalog.record.stamp.mandatory_anchor;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

void release_host_recovery_pin(void * context) noexcept {
    auto * state = static_cast<server_prompt_cache_state *>(context);
    GGML_ASSERT(state && state->recovery_pins > 0);
    state->recovery_pins--;
}

bool build_host_recovery_source(
        server_prompt_cache & cache,
        server_prompt_cache_state & state,
        std::vector<llama_cache_acct_artifact_id> & artifacts,
        std::vector<llama_cache_acct_op_id> & ops) noexcept {
    artifacts.clear();
    ops.clear();
    try {
        if (!cache.retention_obs) {
            return false;
        }
        server_retention_candidate catalog;
        const auto key = server_retention_instance_key::for_host_entry(&state);
        if (!cache.retention_obs->candidate_for_instance(key, catalog) ||
            catalog.artifact_id.v == 0 ||
            catalog.record.kind != common_retention_artifact_kind::host_entry ||
            catalog.avail != server_retention_candidate_availability::available) {
            return false;
        }
        artifacts.push_back(catalog.artifact_id);
        ops.reserve(3);
        for (const auto op : state.release_ops()) {
            if (!op) {
                return false;
            }
            ops.push_back(op);
        }
        std::sort(ops.begin(), ops.end());
        ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
        return ops.size() == 3;
    } catch (...) {
        artifacts.clear();
        ops.clear();
        return false;
    }
}

server_cache_recovery_pin acquire_host_recovery_pin(
        server_prompt_cache_state & state,
        std::vector<llama_cache_acct_artifact_id> artifacts,
        std::vector<llama_cache_acct_op_id> ops) noexcept {
    if (state.recovery_pins == std::numeric_limits<uint32_t>::max()) {
        return {};
    }
    state.recovery_pins++;
    auto pin = server_cache_recovery_pin::acquire(
        &state,
        release_host_recovery_pin,
        std::move(artifacts),
        std::move(ops));
    if (!pin.valid()) {
        state.recovery_pins--;
    }
    return pin;
}

struct host_trade_ranking {
    bool price_known = false;
    uint64_t price_us = 0;
    uint32_t weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
    uint32_t ordinal = 0;
    int32_t source_id = -1;
    llama_cache_acct_artifact_id artifact_id;
    bool zero_destruction_known = false;
    bool zero_destruction = false;
    bool zero_destruction_tie_break = false;
    common_cache_family_role family_role = common_cache_family_role::_count;
};

struct host_destruction_certification {
    bool ready = false;
    server_cache_prepared_release_capability capability;
    server_cache_recovery_pin pin;
    common_cache_plan_destruction_quote quote;
    server_prompt_cache_retirement_manifest retirement;
    uint64_t projected_bytes = 0;
};

void server_prompt_cache_observe_host_destruction(
        server_prompt_cache & cache,
        const common_cache_plan_destruction_receipt & receipt,
        bool observe_classification,
        uint64_t projected_bytes,
        const host_trade_ranking * ranking = nullptr) noexcept {
    cache.publish_authority->observe_host_destruction(
        receipt, observe_classification);
    if (!cache.debug_observability) {
        return;
    }
    try {
        const json unavailable = common_cache_acct_known_name(
            llama_cache_acct_known::unavailable);
        json payload = server_cache_destruction_receipt_json(
            receipt, projected_bytes);
        payload["price_us"] = ranking && ranking->price_known
            ? json(ranking->price_us) : unavailable;
        payload["retention_weight_milli"] = ranking
            ? json(ranking->weight_milli) : unavailable;
        payload["rank_ordinal"] = ranking
            ? json(ranking->ordinal) : unavailable;
        payload["victim_source_id"] = ranking && ranking->source_id >= 0
            ? json(ranking->source_id) : unavailable;
        payload["victim_artifact_id"] = ranking &&
                ranking->artifact_id.v != 0
            ? json(ranking->artifact_id.v) : unavailable;
        payload["zero_destruction"] = ranking &&
                ranking->zero_destruction_known
            ? json(ranking->zero_destruction) : unavailable;
        payload["zero_destruction_tie_break"] = ranking
            ? json(ranking->zero_destruction_tie_break) : json(false);
        payload["declared_family_role"] = ranking &&
                ranking->family_role < common_cache_family_role::_count
            ? json(ranking->family_role == common_cache_family_role::main
                ? "main" : ranking->family_role ==
                    common_cache_family_role::branch
                    ? "branch" : "background")
            : json(nullptr);
        payload["legacy_fallbacks"] = cache.destruction_obs
            ? cache.destruction_obs->host_trade_legacy_fallbacks : uint64_t(0);
        payload["df2_executed"] = cache.destruction_obs
            ? cache.destruction_obs->host_trade_df2_executed : uint64_t(0);
        payload["publication_skips"] = cache.destruction_obs
            ? cache.destruction_obs->host_trade_publication_skips : uint64_t(0);
        cache.debug_destruction_emissions++;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n", payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb maintenance or destruction.
    }
}

llama_cache_acct_artifact_id host_entry_artifact_id(
        const server_prompt_cache & cache,
        const server_prompt_cache_state & state) noexcept {
    return cache.retention_obs
        ? cache.retention_obs->artifact_id(
              server_retention_instance_key::for_host_entry(&state))
        : llama_cache_acct_artifact_id {};
}

void emit_recovery_pin_excluded(
        server_prompt_cache & cache,
        const server_prompt_cache_state & state) noexcept {
    if (!cache.debug_observability) {
        return;
    }
    try {
        const auto artifact = host_entry_artifact_id(cache, state);
        common_cache_plan_destruction_receipt receipt;
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
        json payload = server_cache_destruction_receipt_json(receipt, 0);
        payload["evidence_event"] = "recovery_pin_excluded";
        payload["recovery_pin_excluded"] = {
            { "artifact_id", artifact.v },
            { "source_id", state.cache_plan_source_id },
            { "pin_count", state.recovery_pins },
        };
        payload["floor_outcome"] = "pending";
        cache.debug_recovery_pin_exclusions++;
        cache.debug_last_recovery_pin_excluded = artifact;
        cache.debug_destruction_emissions++;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n", payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb victim selection or pressure.
    }
}

void emit_host_pressure_floor_outcome(
        server_prompt_cache & cache,
        const char * outcome,
        llama_cache_acct_artifact_id victim_artifact,
        int32_t victim_source_id) noexcept {
    if (!cache.debug_observability) {
        return;
    }
    try {
        common_cache_plan_destruction_receipt receipt;
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
        json payload = server_cache_destruction_receipt_json(receipt, 0);
        payload["evidence_event"] = "floor_outcome";
        payload["recovery_pin_excluded"] = nullptr;
        payload["floor_outcome"] = outcome;
        payload["floor_victim_artifact_id"] = victim_artifact.v != 0
            ? json(victim_artifact.v)
            : json(common_cache_acct_known_name(
                  llama_cache_acct_known::unavailable));
        payload["floor_victim_source_id"] = victim_source_id >= 0
            ? json(victim_source_id)
            : json(common_cache_acct_known_name(
                  llama_cache_acct_known::unavailable));
        cache.debug_host_pressure_floor_outcomes++;
        cache.debug_destruction_emissions++;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n", payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb the already-chosen terminal.
    }
}

host_destruction_certification certify_host_destruction(
        server_prompt_cache & cache,
        server_prompt_cache::iterator victim_state,
        server_prompt_cache::iterator survivor_state,
        uint64_t admission_sequence,
        bool allow_authorized_recovery,
        bool count_redundant_refusals,
        const host_trade_ranking * ranking = nullptr) noexcept {
    host_destruction_certification out;
    auto & authority = *cache.publish_authority;

    const auto refuse_initial = [&](common_cache_plan_destruction_reason reason) {
        out.quote = {};
        auto & receipt = out.quote.receipt;
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
        receipt.state = common_cache_plan_destruction_state::refused;
        receipt.reason = reason;
        receipt.admission_sequence = admission_sequence;
        server_prompt_cache_observe_host_destruction(
            cache, receipt, true, 0, ranking);
        if (cache.destruction_obs && count_redundant_refusals) {
            cache.destruction_obs->note_redundant_host_refused(
                admission_sequence);
        }
    };
    const auto refuse_certified = [&](common_cache_plan_destruction_reason reason) {
        auto & receipt = out.quote.receipt;
        receipt.state = common_cache_plan_destruction_state::refused;
        receipt.reason = reason;
        server_prompt_cache_observe_host_destruction(
            cache, receipt, true, 0, ranking);
        if (cache.destruction_obs && count_redundant_refusals) {
            cache.destruction_obs->note_redundant_host_refused(
                admission_sequence);
        }
    };

    server_cache_destruction_artifact victim;
    std::vector<llama_cache_acct_artifact_id> recovery_ids;
    std::vector<llama_cache_acct_op_id> recovery_ops;
    llama_cache_acct_artifact_id recovery_artifact;
    common_cache_plan_displaced_fate recovery_fate =
        common_cache_plan_displaced_fate::unavailable;
    // D-A2 taxonomy: a named survivor that is not an exact three-payload
    // duplicate is recovery_unavailable even when the victim's artifact
    // manifest is independently incomplete. Redundancy is the outer proof.
    if (survivor_state != cache.states.end() &&
        !server_prompt_cache::exactly_redundant(
            *victim_state, *survivor_state)) {
        refuse_initial(
            common_cache_plan_destruction_reason::recovery_unavailable);
        return out;
    }
    if (!server_prompt_cache_capture_retirement(
            cache, victim_state, out.retirement)) {
        refuse_initial(common_cache_plan_destruction_reason::manifest_incomplete);
        return out;
    }
    if (!build_host_destruction_artifact(cache, *victim_state, victim)) {
        // Every host-consumption/redundancy effect needs its own by-host
        // catalog contribution. A partial displacement-only union is not
        // certifiable.
        refuse_initial(common_cache_plan_destruction_reason::manifest_incomplete);
        return out;
    }
    if (survivor_state != cache.states.end()) {
        if (!build_host_recovery_source(
                cache, *survivor_state, recovery_ids, recovery_ops)) {
            refuse_initial(
                common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }
        recovery_artifact = recovery_ids.front();
        out.pin = acquire_host_recovery_pin(
            *survivor_state, recovery_ids, recovery_ops);
        recovery_fate = common_cache_plan_displaced_fate::exact_duplicate;
    } else if (allow_authorized_recovery && authority.host_recovery) {
        server_cache_host_recovery_evidence evidence;
        if (!authority.host_recovery(
                authority.host_recovery_context, *victim_state, evidence) ||
            evidence.artifact.v == 0 || evidence.ops.empty() ||
            !evidence.pin.valid() ||
            !evidence.pin.binds_exact(evidence.artifact, evidence.ops) ||
            evidence.fate !=
                common_cache_plan_displaced_fate::retained_sealed_artifact) {
            refuse_initial(
                common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }
        recovery_artifact = evidence.artifact;
        recovery_ops = std::move(evidence.ops);
        try {
            recovery_ids.push_back(recovery_artifact);
        } catch (...) {
            refuse_initial(
                common_cache_plan_destruction_reason::internal_fault);
            return out;
        }
        out.pin = std::move(evidence.pin);
        recovery_fate = evidence.fate;
    } else {
        refuse_initial(common_cache_plan_destruction_reason::recovery_unavailable);
        return out;
    }

    try {
        if (!out.pin.valid()) {
            refuse_initial(common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }

        const server_cache_destruction_preview_callback preview =
            [&](const auto & ops, uint64_t serial, auto & released) {
                return cache.acct->preview_release_set(ops, serial, released);
            };
        const server_cache_destruction_projection_callback project =
            [&](const auto & released, auto & domains) {
                return authority.project_release(released, domains);
            };
        const uint64_t accounting_serial = cache.acct->serial();
        out.quote = server_cache_destruction_quote_redundant_host(
            victim,
            accounting_serial,
            admission_sequence,
            preview,
            project);
        server_prompt_cache_observe_host_destruction(
            cache,
            out.quote.receipt,
            out.quote.receipt.state !=
                common_cache_plan_destruction_state::quoted,
            0,
            ranking);
        if (out.quote.receipt.state !=
                common_cache_plan_destruction_state::quoted) {
            if (cache.destruction_obs && count_redundant_refusals) {
                cache.destruction_obs->note_redundant_host_refused(
                    admission_sequence);
            }
            return out;
        }

        std::vector<server_cache_destruction_artifact> current = {
            std::move(victim),
        };
        auto prepared = server_cache_prepare_release_set(
            out.quote,
            current,
            *cache.acct,
            cache.acct->serial(),
            project,
            std::move(out.pin));
        if (prepared.status !=
                server_cache_prepare_release_status::prepared) {
            refuse_certified(prepared.reason);
            return out;
        }
        if (!common_cache_plan_projected_release_bytes(
                out.quote.projected_domains, out.projected_bytes)) {
            refuse_certified(
                common_cache_plan_destruction_reason::accounting_unavailable);
            return out;
        }

        // The exact-duplicate fate and resolved citation become claims only
        // after every refusal conjunct, fresh effect check, and disjoint pin
        // has succeeded. Schema 6 retains the source identity after the pin
        // itself closes.
        server_cache_destruction_certify_receipt(
            out.quote.receipt, recovery_fate,
            recovery_artifact, recovery_ops);
        server_prompt_cache_observe_host_destruction(
            cache, out.quote.receipt, true, out.projected_bytes, ranking);
        out.capability = std::move(prepared.capability);
        out.ready = true;
        return out;
    } catch (...) {
        out.pin = {};
        if (out.quote.receipt.union_effect_digest.valid()) {
            refuse_certified(common_cache_plan_destruction_reason::internal_fault);
        } else {
            refuse_initial(common_cache_plan_destruction_reason::internal_fault);
        }
        return out;
    }
}

void commit_certified_host_destruction(
        server_prompt_cache & cache,
        host_destruction_certification & certified,
        const std::thread::id & scheduler_owner,
        const host_trade_ranking * ranking = nullptr) noexcept {
    GGML_ASSERT(certified.ready);
    GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
    const auto release_status =
        certified.capability.commit(certified.pin);
    GGML_ASSERT(release_status ==
                common_cache_plan_destruction_reason::none);
    server_prompt_cache_retire_manifest(cache, certified.retirement);
    certified.quote.receipt.state =
        common_cache_plan_destruction_state::executed;
    certified.quote.receipt.actual_accounting_serial =
        cache.acct->serial();
    server_prompt_cache_observe_host_destruction(
        cache,
        certified.quote.receipt,
        false,
        certified.projected_bytes,
        ranking);
    // The cited recovery source was held through accounting commit and
    // receipt publication; this local destruction dependency now closes.
    certified.pin = {};
}

struct host_trade_candidate {
    server_prompt_cache::iterator victim;
    server_prompt_cache::iterator recovery;
    host_trade_ranking ranking;
    bool attempted = false;
    bool lease_known = false;
    bool main_family = false;
    bool soft_leased = false;
    bool hard_leased = false;
};

uint64_t server_prompt_cache_shadow_hash(uint64_t value) noexcept {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

server_prompt_cache::iterator find_exact_host_recovery(
        server_prompt_cache & cache,
        server_prompt_cache::iterator victim) noexcept {
    for (auto it = cache.states.begin(); it != cache.states.end(); ++it) {
        if (it != victim && server_prompt_cache::exactly_redundant(
                *victim, *it)) {
            return it;
        }
    }
    return cache.states.end();
}

bool host_trade_price(
        server_prompt_cache & cache,
        server_prompt_cache::iterator victim,
        uint32_t ordinal,
        const common_cache_plan_calib * calib,
        host_trade_candidate & out) noexcept {
    out = {};
    out.victim = victim;
    out.ranking.ordinal = ordinal;
    out.ranking.source_id = victim->cache_plan_source_id;
    out.ranking.family_role = victim->cache_family.declared()
        ? victim->cache_family.role : common_cache_family_role::_count;
    out.main_family = victim->main_family;
    try {
        auto & authority = *cache.publish_authority;
        server_cache_destruction_artifact artifact;
        if (!build_host_destruction_artifact(cache, *victim, artifact)) {
            return false;
        }
        out.ranking.artifact_id = artifact.candidate.artifact_id;
        if (artifact.candidate.lease.state !=
                server_cache_lease_eval_state::known) {
            return false;
        }
        out.lease_known = true;
        out.soft_leased = artifact.candidate.lease.cls ==
            server_cache_lease_class::soft;
        out.hard_leased = server_cache_lease_is_hard(
            artifact.candidate.lease);
        if (!calib) {
            return false;
        }

        out.recovery = find_exact_host_recovery(cache, victim);
        out.ranking.zero_destruction_known = true;
        out.ranking.zero_destruction = out.recovery != cache.states.end();
        if (out.hard_leased) {
            return false;
        }

        uint32_t additional_weight = SERVER_CACHE_HOST_WEIGHT_SCALE;
        if (common_cache_family_allows_additional_weight(
                victim->cache_family) && authority.host_retention_weight) {
            if (!authority.host_retention_weight(
                    authority.host_retention_weight_context,
                    *victim, additional_weight) ||
                additional_weight == 0) {
                return false;
            }
        }

        uint32_t weight = 0;
        uint64_t price = 0;
        if (!server_cache_host_retention_price_us(
                *calib, victim->size(), out.soft_leased,
                out.main_family, weight, price, additional_weight)) {
            return false;
        }
        out.ranking.weight_milli = weight;
        out.ranking.price_us = price;
        out.ranking.price_known = true;
        return true;
    } catch (...) {
        return false;
    }
}

struct host_trade_df2_projection {
    bool complete = false;
    uint64_t candidate_count = 0;
    llama_cache_acct_artifact_id artifact;
    uint64_t lineage_id = 0;
    common_retention_pool pool = common_retention_pool::attention;
    uint64_t lost_work = 0;
    uint64_t resource = 0;
};

// Allocation-free singleton projection for DF2's synchronous authority seam.
// Host artifacts currently own independent three-leaf payload allocations;
// a zero singleton yield is therefore not executable here and remains on the
// legacy floor. The broader counterfactual projector retains compound support
// for debug/model-free analysis.
host_trade_df2_projection project_host_trade_df2(
        server_prompt_cache & cache,
        server_prompt_cache::iterator incoming,
        const std::vector<host_trade_candidate> & candidates,
        server_prompt_cache_shadow_row * rows,
        server_prompt_cache_shadow_artifact_slot * artifacts,
        server_prompt_cache_shadow_lineage_slot * lineages) noexcept {
    host_trade_df2_projection result;
    if (!rows || !artifacts || !lineages || !cache.retention_obs || !cache.acct ||
        candidates.size() > SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES) {
        return result;
    }

    static_assert((SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY &
                   (SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY - 1)) == 0);
    constexpr size_t index_mask =
        SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY - 1;
    std::fill_n(artifacts, SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY,
        server_prompt_cache_shadow_artifact_slot {});
    std::fill_n(lineages, SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY,
        server_prompt_cache_shadow_lineage_slot {});

    struct fill_context {
        server_prompt_cache_shadow_row * rows = nullptr;
        server_prompt_cache_shadow_artifact_slot * artifacts = nullptr;
        server_prompt_cache_shadow_lineage_slot * lineages = nullptr;
        size_t size = 0;
    } fill { rows, artifacts, lineages, 0 };
    const auto fill_value = [](void * opaque,
            const server_retention_value_snapshot & value) noexcept {
        auto & context = *static_cast<fill_context *>(opaque);
        if (context.size == SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES ||
            !value.artifact_id.v || !value.stamp.lineage_id) {
            return false;
        }
        context.rows[context.size] = {
            value.artifact_id,
            value.instance_key,
            value.kind,
            value.stamp,
            value.lineage,
            value.external_shared_coverage_tokens,
            0,
            false,
            false,
        };

        constexpr size_t mask =
            SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY - 1;
        size_t artifact_slot = size_t(server_prompt_cache_shadow_hash(
            value.artifact_id.v)) & mask;
        size_t probes = 0;
        while (context.artifacts[artifact_slot].artifact_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (context.artifacts[artifact_slot].artifact_id ==
                    value.artifact_id.v) {
                return false;
            }
            artifact_slot = (artifact_slot + 1) & mask;
        }
        if (context.artifacts[artifact_slot].artifact_id) {
            return false;
        }
        context.artifacts[artifact_slot] = {
            value.artifact_id.v, uint32_t(context.size) };

        const uint64_t lineage_key = value.stamp.lineage_id ^
            (uint64_t(uint8_t(value.stamp.pool)) << 56);
        size_t lineage_slot = size_t(server_prompt_cache_shadow_hash(
            lineage_key)) & mask;
        probes = 0;
        while (context.lineages[lineage_slot].lineage_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (context.lineages[lineage_slot].lineage_id ==
                    value.stamp.lineage_id &&
                context.lineages[lineage_slot].pool == value.stamp.pool) {
                break;
            }
            lineage_slot = (lineage_slot + 1) & mask;
        }
        auto & lineage = context.lineages[lineage_slot];
        if (lineage.lineage_id &&
            (lineage.lineage_id != value.stamp.lineage_id ||
             lineage.pool != value.stamp.pool)) {
            return false;
        }
        if (!lineage.lineage_id) {
            lineage.lineage_id = value.stamp.lineage_id;
            lineage.pool = value.stamp.pool;
        }
        const uint64_t coverage = value.stamp.coverage_tokens;
        if (coverage > lineage.maximum_coverage) {
            lineage.second_coverage = lineage.maximum_coverage;
            lineage.maximum_coverage = coverage;
            lineage.maximum_count = 1;
        } else if (coverage == lineage.maximum_coverage) {
            lineage.maximum_count++;
        } else {
            lineage.second_coverage = std::max(
                lineage.second_coverage, coverage);
        }
        context.size++;
        return true;
    };
    const auto inventory = cache.retention_obs->value_snapshots(
        &fill, fill_value);
    if (inventory.status !=
            server_retention_value_snapshot_status::complete ||
        inventory.size != fill.size || fill.size == 0) {
        return result;
    }

    auto * begin = rows;
    auto * end = rows + fill.size;
    const auto find_artifact = [&](llama_cache_acct_artifact_id id) {
        if (!id.v) {
            return end;
        }
        size_t slot = size_t(server_prompt_cache_shadow_hash(id.v)) &
            index_mask;
        size_t probes = 0;
        while (artifacts[slot].artifact_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (artifacts[slot].artifact_id == id.v) {
                return begin + artifacts[slot].row_index;
            }
            slot = (slot + 1) & index_mask;
        }
        return end;
    };

    // The priced inventory already covers every ordinary physical host
    // entry. Join those artifacts directly instead of repeating a sidecar
    // association lookup for every state. Incoming publications and
    // recovery-pinned entries are deliberately absent from that inventory;
    // join only those exceptional retained providers afterward.
    for (const auto & candidate : candidates) {
        auto * row = find_artifact(candidate.ranking.artifact_id);
        if (!candidate.ranking.artifact_id.v || row == end ||
            row->kind != common_retention_artifact_kind::host_entry ||
            row->backing_known || row->releasable ||
            !candidate.lease_known) {
            return {};
        }
        row->backing_known = true;
        row->releasable = !candidate.hard_leased &&
            candidate.victim != incoming &&
            candidate.victim->recovery_pins == 0;
        if (!row->releasable) {
            continue;
        }
        result.candidate_count++;
        // Published host entries own three independent accounting leaves.
        // Their immutable payload sizes are the exact resident bytes charged
        // at publication; the selected victim is still re-previewed and
        // serial-certified immediately before destruction.
        uint64_t snapshot_bytes = 0;
        uint64_t checkpoint_bytes = 0;
        uint64_t accelerator_bytes = 0;
        if (!server_prompt_cache::payload_bytes(
                *candidate.victim, snapshot_bytes,
                checkpoint_bytes, accelerator_bytes) ||
            checkpoint_bytes > UINT64_MAX - snapshot_bytes ||
            accelerator_bytes >
                UINT64_MAX - snapshot_bytes - checkpoint_bytes) {
            return {};
        }
        row->resource =
            snapshot_bytes + checkpoint_bytes + accelerator_bytes;
    }
    for (auto state = cache.states.begin(); state != cache.states.end();
            ++state) {
        if (state != incoming && state->recovery_pins == 0) {
            continue;
        }
        const auto artifact = cache.retention_obs->artifact_id(
            server_retention_instance_key::for_host_entry(&*state));
        auto * row = find_artifact(artifact);
        if (!artifact.v || row == end ||
            row->kind != common_retention_artifact_kind::host_entry ||
            row->backing_known) {
            return {};
        }
        row->backing_known = true;
    }
    for (const auto * row = begin; row != end; ++row) {
        if (row->kind == common_retention_artifact_kind::host_entry &&
            !row->backing_known) {
            return {};
        }
    }

    const auto find_lineage = [&](common_retention_pool pool,
                                  uint64_t lineage_id) {
        if (!lineage_id) {
            return static_cast<server_prompt_cache_shadow_lineage_slot *>(nullptr);
        }
        const uint64_t key = lineage_id ^
            (uint64_t(uint8_t(pool)) << 56);
        size_t slot = size_t(server_prompt_cache_shadow_hash(key)) &
            index_mask;
        size_t probes = 0;
        while (lineages[slot].lineage_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (lineages[slot].lineage_id == lineage_id &&
                lineages[slot].pool == pool) {
                return &lineages[slot];
            }
            slot = (slot + 1) & index_mask;
        }
        return static_cast<server_prompt_cache_shadow_lineage_slot *>(nullptr);
    };

    bool have_best = false;
    common_retention_shadow_value best;
    const uint64_t competition_epoch =
        cache.retention_obs->competition_epoch_value();
    for (const auto * row = begin; row != end; ++row) {
        if (!row->releasable || row->resource == 0) {
            continue;
        }
        auto * lineage = find_lineage(
            row->stamp.pool, row->stamp.lineage_id);
        GGML_ASSERT(lineage != nullptr);
        uint64_t retained = row->external_shared_coverage_tokens;
        retained = std::max(
            retained,
            row->stamp.coverage_tokens == lineage->maximum_coverage &&
                    lineage->maximum_count == 1
                ? lineage->second_coverage : lineage->maximum_coverage);
        const uint64_t lost_work = row->stamp.coverage_tokens > retained
            ? row->stamp.coverage_tokens - retained : 0;
        common_retention_shadow_value quote;
        if (!common_retention_shadow_quote(
                row->lineage, competition_epoch,
                lost_work, row->resource, row->stamp.recency_ordinal,
                {}, quote)) {
            return {};
        }
        const int comparison = have_best
            ? common_retention_shadow_compare(quote, best) : -1;
        if (!have_best || comparison < 0 || (comparison == 0 &&
                std::tie(row->stamp.pool, row->stamp.lineage_id,
                         row->artifact_id.v) <
                std::tie(result.pool, result.lineage_id,
                         result.artifact.v))) {
            have_best = true;
            best = quote;
            result.artifact = row->artifact_id;
            result.lineage_id = row->stamp.lineage_id;
            result.pool = row->stamp.pool;
            result.lost_work = lost_work;
            result.resource = row->resource;
        }
    }
    result.complete = have_best;
    return result;
}

void observe_host_trade_refusal(
        server_prompt_cache & cache,
        uint64_t admission_sequence,
        common_cache_plan_destruction_reason reason,
        const host_trade_ranking * ranking = nullptr) noexcept {
    common_cache_plan_destruction_receipt receipt;
    receipt.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::
            different_host_source_consumption);
    receipt.state = common_cache_plan_destruction_state::refused;
    receipt.reason = reason;
    receipt.admission_sequence = admission_sequence;
    server_prompt_cache_observe_host_destruction(
        cache, receipt, true, 0, ranking);
}

} // namespace

bool server_prompt_cache::acquire_durable_recovery(
        const server_tokens & tokens,
        const std::string & adapter_config_key,
        llama_cache_acct_artifact_id & artifact,
        std::vector<llama_cache_acct_op_id> & ops,
        server_cache_recovery_pin & pin) noexcept {
    return acquire_durable_recovery(
        find_state_exact(tokens, adapter_config_key), artifact, ops, pin);
}

bool server_prompt_cache::acquire_durable_recovery(
        iterator state,
        llama_cache_acct_artifact_id & artifact,
        std::vector<llama_cache_acct_op_id> & ops,
        server_cache_recovery_pin & pin) noexcept {
    artifact = {};
    ops.clear();
    pin = {};
    try {
        if (state == states.end()) {
            return false;
        }
        std::vector<llama_cache_acct_artifact_id> artifacts;
        if (!build_host_recovery_source(
                *this, *state, artifacts, ops) || artifacts.size() != 1) {
            return false;
        }
        artifact = artifacts.front();
        pin = acquire_host_recovery_pin(
            *state, std::move(artifacts), ops);
        if (!pin.valid() || !pin.binds_exact(artifact, ops)) {
            artifact = {};
            ops.clear();
            pin = {};
            return false;
        }
        return true;
    } catch (...) {
        artifact = {};
        ops.clear();
        pin = {};
    }
    return false;
}

server_cache_durable_fallback_proof
server_prompt_cache_host_fallback_proof(
        server_prompt_cache & cache,
        const server_cache_control_selector & selector) noexcept {
    if (selector.kind != server_cache_control_subject_kind::host_snapshot ||
        selector.retention_key.kind !=
            common_retention_artifact_kind::host_entry) {
        return {};
    }
    auto state = std::find_if(
        cache.states.begin(), cache.states.end(), [&](const auto & value) {
            return &value == reinterpret_cast<const server_prompt_cache_state *>(
                selector.retention_key.instance);
        });
    if (state == cache.states.end()) {
        return {};
    }
    llama_cache_acct_artifact_id artifact;
    std::vector<llama_cache_acct_op_id> ops;
    server_cache_recovery_pin pin;
    if (!cache.acquire_durable_recovery(state, artifact, ops, pin)) {
        return {};
    }
    return server_cache_retention_fallback_proof(std::move(pin));
}

bool server_prompt_cache::exactly_redundant(
        const server_prompt_cache_state & victim,
        const server_prompt_cache_state & survivor) noexcept {
    try {
        if (&victim == &survivor ||
            victim.adapter_config_key != survivor.adapter_config_key ||
            victim.prompt.sequence_epoch != survivor.prompt.sequence_epoch ||
            victim.prompt.n_tokens() > survivor.prompt.n_tokens() ||
            victim.prompt.tokens.get_common_prefix(survivor.prompt.tokens) !=
                size_t(victim.prompt.n_tokens()) ||
            victim.data.main != survivor.data.main ||
            victim.data.drft != survivor.data.drft ||
            victim.prompt.checkpoints.size() !=
                survivor.prompt.checkpoints.size()) {
            return false;
        }
        std::string victim_media;
        std::string survivor_media;
        if (!victim.prompt.tokens.media_content_identity(
                victim.prompt.n_tokens(), victim_media) ||
            !survivor.prompt.tokens.media_content_identity(
                victim.prompt.n_tokens(), survivor_media) ||
            victim_media != survivor_media) {
            return false;
        }
        auto a = victim.prompt.checkpoints.begin();
        auto b = survivor.prompt.checkpoints.begin();
        for (; a != victim.prompt.checkpoints.end(); ++a, ++b) {
            if (!checkpoint_payload_equal(*a, *b)) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

void server_prompt_cache::observe_retention_pressure_choice(
        server_cache_destruction_reason reason,
        iterator incoming,
        iterator incumbent,
        bool competition_wave_valid) noexcept {
    if (!retention_obs) {
        return;
    }
    const auto increment = [](uint64_t & value) noexcept {
        if (value != UINT64_MAX) {
            value++;
        }
    };
    increment(retention_shadow.choices);
    auto & event = retention_shadow.last;
    event = {};
    event.reason = reason;
    event.competition_epoch = retention_obs->competition_epoch_value();

    const auto unavailable = [&]() noexcept {
        increment(retention_shadow.unavailable);
        if (debug_observability) {
            SRV_INF(
                "CACHE_RETENTION_SHADOW status=unavailable reason=%u "
                "epoch=%" PRIu64 " candidates=%" PRIu64 "\n",
                unsigned(reason), event.competition_epoch,
                event.candidate_count);
        }
    };

    struct fill_context {
        server_prompt_cache_shadow_row * rows = nullptr;
        size_t size = 0;
    };
    const auto fill_value = [](void * opaque,
            const server_retention_value_snapshot & value) noexcept {
        auto & context = *static_cast<fill_context *>(opaque);
        if (context.size == SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES) {
            return false;
        }
        context.rows[context.size++] = {
            value.artifact_id,
            value.instance_key,
            value.kind,
            value.stamp,
            value.lineage,
            value.external_shared_coverage_tokens,
            0,
            false,
            false,
        };
        return true;
    };

    // Lifecycle shadow uses the same immutable catalog, evaluated leases and
    // serial-bound accounting release preview as the certified authority. It
    // remains counterfactual: the already-selected lifecycle victim is the
    // incumbent and no projection result can authorize mutation in DF1.
    if (publish_authority) {
        if (!competition_wave_valid || !retention_shadow_rows || !acct ||
            !lease_obs || !lease_execution_identity ||
            reason != server_cache_destruction_reason::host_capacity ||
            states.size() > SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES ||
            incumbent == states.end()) {
            unavailable();
            return;
        }
        try {
            fill_context fill { retention_shadow_rows.get(), 0 };
            const auto inventory = retention_obs->value_snapshots(
                &fill, fill_value);
            if (inventory.status !=
                    server_retention_value_snapshot_status::complete ||
                inventory.size != fill.size || fill.size == 0) {
                unavailable();
                return;
            }
            auto * begin = retention_shadow_rows.get();
            auto * end = begin + fill.size;
            std::sort(begin, end, [](const auto & a, const auto & b) {
                return a.artifact_id.v < b.artifact_id.v;
            });
            const auto find_artifact = [&](llama_cache_acct_artifact_id id) {
                const auto found = std::lower_bound(
                    begin, end, id.v, [](const auto & row, uint64_t value) {
                        return row.artifact_id.v < value;
                    });
                return found != end && found->artifact_id == id ? found : end;
            };

            std::vector<server_cache_yield_candidate> candidates;
            candidates.reserve(fill.size);
            for (auto * row = begin; row != end; ++row) {
                server_retention_candidate catalog;
                if (!retention_obs->candidate_for_instance(
                        row->instance_key, catalog) ||
                    catalog.artifact_id != row->artifact_id ||
                    catalog.record.kind != row->kind ||
                    catalog.record.stamp.stable_id != row->stamp.stable_id ||
                    catalog.record.stamp.lineage_id !=
                        row->stamp.lineage_id ||
                    catalog.lineage != row->lineage) {
                    unavailable();
                    return;
                }
                server_cache_yield_candidate candidate;
                candidate.artifact_id = catalog.artifact_id;
                candidate.record = catalog.record;
                candidate.lineage = catalog.lineage;
                candidate.availability =
                    server_retention_candidate_availability::
                        in_flight_mutation;
                candidate.lease = {
                    server_cache_lease_eval_state::known,
                    server_cache_lease_class::none,
                    server_cache_lease_eligibility::eligible,
                };
                candidate.identity_known = true;
                candidate.external_shared_coverage_tokens =
                    row->external_shared_coverage_tokens;
                candidates.push_back(std::move(candidate));
            }

            uint64_t n_candidates = 0;
            for (auto it = states.begin(); it != states.end(); ++it) {
                server_cache_destruction_artifact artifact;
                if (!build_host_destruction_artifact(*this, *it, artifact)) {
                    unavailable();
                    return;
                }
                auto candidate = std::move(artifact.candidate);
                auto * row = find_artifact(candidate.artifact_id);
                if (row == end || row->kind !=
                        common_retention_artifact_kind::host_entry ||
                    row->backing_known) {
                    unavailable();
                    return;
                }
                row->backing_known = true;
                candidate.external_shared_coverage_tokens =
                    row->external_shared_coverage_tokens;
                if (it == incoming || it->recovery_pins != 0) {
                    candidate.availability =
                        server_retention_candidate_availability::
                            in_flight_mutation;
                }
                if (candidate.availability ==
                        server_retention_candidate_availability::available &&
                    candidate.lease.eligibility ==
                        server_cache_lease_eligibility::eligible &&
                    !server_cache_lease_is_hard(candidate.lease) &&
                    !candidate.release_ops.empty()) {
                    n_candidates++;
                }
                candidates[size_t(row - begin)] = std::move(candidate);
            }
            for (const auto * row = begin; row != end; ++row) {
                if (row->kind == common_retention_artifact_kind::host_entry &&
                    !row->backing_known) {
                    unavailable();
                    return;
                }
            }
            event.candidate_count = n_candidates;
            event.incumbent_artifact = host_entry_artifact_id(
                *this, *incumbent);
            if (!event.incumbent_artifact.v) {
                unavailable();
                return;
            }

            const uint64_t accounting_serial = acct->serial();
            const auto host_domain =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::pageable_host);
            const auto projection = server_retention_shadow_project(
                candidates,
                event.competition_epoch,
                host_domain,
                accounting_serial,
                [this](const std::vector<llama_cache_acct_op_id> & ops,
                       uint64_t serial,
                       llama_cache_acct_release_set_preview & out) {
                    return acct->preview_release_set(ops, serial, out);
                });
            if (!projection.complete || projection.alternatives.empty() ||
                projection.alternatives.front().artifact_ids.size() != 1) {
                unavailable();
                return;
            }
            const auto & alternative = projection.alternatives.front();
            event.proposed_artifact = alternative.artifact_ids.front();
            event.proposed_lineage = alternative.lineage_id;
            event.proposed_pool = alternative.pool;
            event.proposed_lost_work = alternative.lost_work_units;
            event.proposed_resource = alternative.value.marginal_resource;
            event.status = server_prompt_cache_shadow_status::complete;
            event.agrees =
                event.incumbent_artifact == event.proposed_artifact;
            increment(retention_shadow.complete);
            increment(event.agrees
                ? retention_shadow.agreements
                : retention_shadow.disagreements);
            if (debug_observability) {
                SRV_INF(
                    "CACHE_RETENTION_SHADOW status=complete reason=%u "
                    "epoch=%" PRIu64 " candidates=%" PRIu64
                    " incumbent=%" PRIu64 " proposed=%" PRIu64
                    " agrees=%s lost_work=%" PRIu64
                    " resource=%" PRIu64 "\n",
                    unsigned(reason), event.competition_epoch,
                    event.candidate_count, event.incumbent_artifact.v,
                    event.proposed_artifact.v,
                    event.agrees ? "true" : "false",
                    event.proposed_lost_work, event.proposed_resource);
            }
        } catch (...) {
            unavailable();
        }
        return;
    }

    // The lifecycle route returned above after lowering its evaluated lease
    // and accounting evidence. This branch observes only the ordinary
    // fixed-cache FIFO route; never mix the two evidence shapes.
    if (!competition_wave_valid ||
        !retention_shadow_rows || incumbent == states.end() ||
        states.size() > SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES) {
        unavailable();
        return;
    }

    fill_context fill { retention_shadow_rows.get(), 0 };
    const auto inventory = retention_obs->value_snapshots(
        &fill, fill_value);
    if (inventory.status !=
            server_retention_value_snapshot_status::complete ||
        inventory.size != fill.size) {
        unavailable();
        return;
    }
    const size_t n_rows = fill.size;
    if (n_rows == 0) {
        unavailable();
        return;
    }

    auto * begin = retention_shadow_rows.get();
    auto * end = begin + n_rows;
    std::sort(begin, end, [](const auto & a, const auto & b) {
        return a.artifact_id.v < b.artifact_id.v;
    });
    const auto find_artifact = [&](llama_cache_acct_artifact_id id) {
        const auto found = std::lower_bound(
            begin, end, id.v, [](const auto & row, uint64_t value) {
                return row.artifact_id.v < value;
            });
        return found != end && found->artifact_id == id ? found : end;
    };
    uint64_t n_candidates = 0;
    for (auto it = states.begin(); it != states.end(); ++it) {
        const auto artifact = retention_obs->artifact_id(
            server_retention_instance_key::for_host_entry(&*it));
        auto * row = find_artifact(artifact);
        const uint64_t resource =
            reason == server_cache_destruction_reason::host_token_limit
                ? uint64_t(it->prompt.n_tokens())
                : uint64_t(it->size());
        if (!artifact.v || row == end ||
            row->kind != common_retention_artifact_kind::host_entry ||
            row->backing_known || resource == 0) {
            unavailable();
            return;
        }
        row->backing_known = true;
        row->resource = resource;
        row->releasable = it != incoming && it->recovery_pins == 0;
        if (row->releasable) {
            n_candidates++;
        }
    }
    for (const auto * row = begin; row != end; ++row) {
        if (row->kind == common_retention_artifact_kind::host_entry &&
            !row->backing_known) {
            unavailable();
            return;
        }
    }
    event.candidate_count = n_candidates;
    const auto incumbent_artifact = retention_obs->artifact_id(
        server_retention_instance_key::for_host_entry(&*incumbent));
    const auto * incumbent_row = find_artifact(incumbent_artifact);
    if (!incumbent_artifact.v || incumbent_row == end ||
        incumbent_row->kind != common_retention_artifact_kind::host_entry ||
        !incumbent_row->backing_known || !incumbent_row->releasable) {
        unavailable();
        return;
    }
    event.incumbent_artifact = incumbent_artifact;
    event.incumbent_lineage = incumbent_row->lineage.lineage_id;

    std::sort(begin, end, [](const auto & a, const auto & b) {
        return std::tie(
                   a.stamp.pool, a.stamp.lineage_id,
                   a.stamp.coverage_tokens, a.stamp.recency_ordinal,
                   a.artifact_id.v) <
               std::tie(
                   b.stamp.pool, b.stamp.lineage_id,
                   b.stamp.coverage_tokens, b.stamp.recency_ordinal,
                   b.artifact_id.v);
    });

    bool have_proposed = false;
    common_retention_shadow_value proposed_value;
    for (size_t first = 0; first < n_rows;) {
        size_t last = first + 1;
        while (last < n_rows &&
               begin[last].stamp.pool == begin[first].stamp.pool &&
               begin[last].stamp.lineage_id ==
                   begin[first].stamp.lineage_id) {
            if (begin[last].lineage != begin[first].lineage) {
                unavailable();
                return;
            }
            last++;
        }

        uint64_t maximum = 0;
        uint64_t second = 0;
        size_t n_maximum = 0;
        for (size_t i = first; i < last; ++i) {
            const uint64_t coverage = begin[i].stamp.coverage_tokens;
            if (coverage > maximum) {
                second = maximum;
                maximum = coverage;
                n_maximum = 1;
            } else if (coverage == maximum) {
                n_maximum++;
            } else if (coverage > second) {
                second = coverage;
            }
        }

        for (size_t i = first; i < last; ++i) {
            if (!begin[i].releasable) {
                continue;
            }
            const uint64_t coverage = begin[i].stamp.coverage_tokens;
            if (begin[i].external_shared_coverage_tokens > coverage) {
                unavailable();
                return;
            }
            const uint64_t retained = std::max(
                second, begin[i].external_shared_coverage_tokens);
            const uint64_t lost_work =
                coverage == maximum && n_maximum == 1
                    ? coverage - retained : 0;
            common_retention_shadow_value quote;
            if (!common_retention_shadow_quote(
                    begin[i].lineage, event.competition_epoch,
                    lost_work, begin[i].resource,
                    begin[i].stamp.recency_ordinal, {}, quote)) {
                unavailable();
                return;
            }
            const bool lower = !have_proposed ||
                common_retention_shadow_compare(
                    quote, proposed_value) < 0;
            const bool tied = have_proposed &&
                common_retention_shadow_compare(
                    quote, proposed_value) == 0;
            if (lower || (tied &&
                    std::tie(begin[i].stamp.pool,
                             begin[i].stamp.lineage_id,
                             begin[i].artifact_id.v) <
                    std::tie(event.proposed_pool,
                             event.proposed_lineage,
                             event.proposed_artifact.v))) {
                have_proposed = true;
                proposed_value = quote;
                event.proposed_artifact = begin[i].artifact_id;
                event.proposed_lineage = begin[i].stamp.lineage_id;
                event.proposed_pool = begin[i].stamp.pool;
                event.proposed_lost_work = lost_work;
                event.proposed_resource = begin[i].resource;
            }
        }
        first = last;
    }

    if (!have_proposed) {
        unavailable();
        return;
    }
    event.status = server_prompt_cache_shadow_status::complete;
    event.agrees = event.incumbent_artifact == event.proposed_artifact;
    increment(retention_shadow.complete);
    if (event.agrees) {
        increment(retention_shadow.agreements);
    } else {
        increment(retention_shadow.disagreements);
    }
    if (debug_observability) {
        SRV_INF(
            "CACHE_RETENTION_SHADOW status=complete reason=%u "
            "epoch=%" PRIu64 " candidates=%" PRIu64
            " incumbent=%" PRIu64 " proposed=%" PRIu64
            " agrees=%s lost_work=%" PRIu64
            " resource=%" PRIu64 "\n",
            unsigned(reason), event.competition_epoch,
            event.candidate_count, event.incumbent_artifact.v,
            event.proposed_artifact.v,
            event.agrees ? "true" : "false",
            event.proposed_lost_work, event.proposed_resource);
    }
}

bool server_prompt_cache::destroy_priced_host_entry(
        server_cache_destruction_reason reason,
        iterator incoming,
        iterator & legacy_floor,
        common_cache_plan_destruction_reason & floor_reason,
        bool & recovery_pin_excluded,
        bool competition_wave_valid,
        bool & observe_retention_shadow) {
    legacy_floor = states.end();
    floor_reason = common_cache_plan_destruction_reason::capacity_refused;
    recovery_pin_excluded = false;
    if (!publish_authority ||
        (reason != server_cache_destruction_reason::host_capacity &&
         reason != server_cache_destruction_reason::host_token_limit)) {
        return false;
    }
    if (!acct || !retention_obs || !lease_obs || !lease_execution_identity) {
        floor_reason = common_cache_plan_destruction_reason::lease_unavailable;
        if (destruction_obs) {
            destruction_obs->note_host_trade_substrate_fault();
            destruction_obs->host_trade_legacy_fallbacks++;
        }
        if (!host_trade_substrate_warned) {
            host_trade_substrate_warned = true;
            SRV_WRN("%s\n",
                    "host retention pricing unavailable: lifecycle lease/accounting substrate is incomplete");
        }
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (it == incoming) {
                continue;
            }
            if (it->recovery_pins != 0) {
                recovery_pin_excluded = true;
                emit_recovery_pin_excluded(*this, *it);
                continue;
            }
            legacy_floor = it;
            break;
        }
        const uint64_t sequence =
            ++publish_authority->destruction_quote_sequence;
        observe_host_trade_refusal(*this, sequence, floor_reason);
        return false;
    }

    // D-A3 is an execution-time lease boundary. Expire first, then inspect
    // each immutable host artifact once for pricing. Soft protection raises
    // price; only a hard lease makes a candidate ineligible. If every priced
    // candidate fails certification, the caller deliberately executes the
    // historical FIFO victim so the user's configured bound remains real.
    lease_obs->lifecycle_point();
    const auto * calib = common_cache_plan_calib_find(
        publish_authority->calibration_profile);
    std::vector<host_trade_candidate> candidates;
    try {
        candidates.reserve(states.size());
        uint32_t ordinal = 0;
        for (auto it = states.begin(); it != states.end(); ++it, ++ordinal) {
            if (it == incoming) {
                continue;
            }
            if (it->recovery_pins != 0) {
                recovery_pin_excluded = true;
                emit_recovery_pin_excluded(*this, *it);
                continue;
            }
            host_trade_candidate candidate;
            (void) host_trade_price(
                *this, it, ordinal, calib, candidate);
            if (candidate.hard_leased) {
                candidate.attempted = true;
                const uint64_t quote_sequence =
                    ++publish_authority->destruction_quote_sequence;
                observe_host_trade_refusal(
                    *this,
                    quote_sequence,
                    common_cache_plan_destruction_reason::hard_lease_blocked,
                    &candidate.ranking);
                if (destruction_obs) {
                    destruction_obs->note_host_trade_veto();
                }
            }
            candidates.push_back(std::move(candidate));
        }
    } catch (...) {
        if (destruction_obs) {
            destruction_obs->host_trade_legacy_fallbacks++;
        }
        return false;
    }

    const auto stable_key = [](const host_trade_candidate & candidate) {
        // Keep B's planner-key shape explicit even though this inventory has
        // only the host provider; later mixed-provider trades retain ordering.
        return std::make_tuple(
            uint8_t(common_cache_plan_provider::host_cache_entry),
            candidate.victim->cache_plan_source_id,
            candidate.ranking.ordinal);
    };
    while (true) {
        uint64_t minimum = std::numeric_limits<uint64_t>::max();
        for (const auto & candidate : candidates) {
            if (!candidate.attempted && candidate.ranking.price_known) {
                minimum = std::min(minimum, candidate.ranking.price_us);
            }
        }
        if (minimum == std::numeric_limits<uint64_t>::max()) {
            break;
        }
        const long double floor = std::max<long double>(
            (long double) minimum * COMMON_CACHE_PLAN_TIE_REL_FLOOR,
            COMMON_CACHE_PLAN_TIE_ABS_FLOOR_US);
        host_trade_candidate * chosen = nullptr;
        bool saw_zero = false;
        bool saw_destructive = false;
        for (auto & candidate : candidates) {
            if (candidate.attempted || !candidate.ranking.price_known ||
                (long double) candidate.ranking.price_us >
                    (long double) minimum + floor) {
                continue;
            }
            saw_zero |= candidate.ranking.zero_destruction;
            saw_destructive |= !candidate.ranking.zero_destruction;
            if (!chosen ||
                std::make_tuple(!candidate.ranking.zero_destruction,
                                stable_key(candidate)) <
                std::make_tuple(!chosen->ranking.zero_destruction,
                                stable_key(*chosen))) {
                chosen = &candidate;
            }
        }
        const bool mixed_destruction_tie = saw_zero && saw_destructive;
        GGML_ASSERT(chosen != nullptr);
        chosen->attempted = true;
        chosen->ranking.zero_destruction_tie_break =
            mixed_destruction_tie && chosen->ranking.zero_destruction;

        const uint64_t quote_sequence =
            ++publish_authority->destruction_quote_sequence;
        auto certified = certify_host_destruction(
            *this,
            chosen->victim,
            chosen->recovery,
            quote_sequence,
            true,
            false,
            &chosen->ranking);
        if (!certified.ready) {
            if (destruction_obs) {
                if (certified.quote.receipt.reason ==
                        common_cache_plan_destruction_reason::
                            hard_lease_blocked) {
                    destruction_obs->note_host_trade_veto();
                } else {
                    destruction_obs->note_host_trade_refused();
                }
            }
            if (certified.quote.receipt.reason ==
                    common_cache_plan_destruction_reason::
                        hard_lease_blocked) {
                chosen->hard_leased = true;
            } else if (certified.quote.receipt.reason ==
                    common_cache_plan_destruction_reason::
                        lease_unavailable) {
                chosen->lease_known = false;
            }
            continue;
        }

        if (destruction_obs) {
            destruction_obs->host_trade_attempted++;
        }

        const auto admission = server_prompt_cache_observe_drop(
            *this, *chosen->victim, reason);
        const std::thread::id scheduler_owner = std::this_thread::get_id();
        if (observe_retention_shadow) {
            observe_retention_pressure_choice(
                reason, incoming, chosen->victim, competition_wave_valid);
        }
        SRV_WRN(
            " - removing priced host entry source_id=%d (size = %.3f MiB)\n",
            chosen->victim->cache_plan_source_id,
            chosen->victim->size() / (1024.0 * 1024.0));
        server_prompt_cache_destroy_entry_impl(*this, chosen->victim);
        // D-A3 uses the same no-interleaving terminal as D-A2. Pricing and all
        // fallible recovery work completed before the physical erase; the raw
        // list mutation has no callback/C writer, and capability commit is the
        // immediately following operation on update_slots' owner thread.
        commit_certified_host_destruction(
            *this, certified, scheduler_owner, &chosen->ranking);
        if (destruction_obs) {
            destruction_obs->note_host_trade_executed(
                admission.sequence,
                certified.projected_bytes,
                chosen->main_family,
                chosen->soft_leased,
                chosen->ranking.zero_destruction_tie_break);
        }
        if (recovery_pin_excluded) {
            emit_host_pressure_floor_outcome(
                *this, "priced_evicted", chosen->ranking.artifact_id,
                chosen->ranking.source_id);
        }
        return true;
    }

    // Candidates without a fitted/complete price never join a partial
    // optimum. Emit one typed refusal per skipped victim, then retain the
    // exact historical FIFO terminal. No new request is refused merely
    // because D-A evidence is incomplete.
    for (auto & candidate : candidates) {
        if (candidate.attempted || candidate.ranking.price_known) {
            continue;
        }
        candidate.attempted = true;
        const uint64_t quote_sequence =
            ++publish_authority->destruction_quote_sequence;
        observe_host_trade_refusal(
            *this,
            quote_sequence,
            common_cache_plan_destruction_reason::capacity_refused,
            &candidate.ranking);
        if (destruction_obs) {
            destruction_obs->note_host_trade_unpriced();
        }
    }
    for (const auto & candidate : candidates) {
        if (candidate.lease_known && !candidate.hard_leased &&
            candidate.victim->recovery_pins == 0) {
            legacy_floor = candidate.victim;
            break;
        }
    }
    if (legacy_floor == states.end() && std::any_of(
            candidates.begin(), candidates.end(), [](const auto & candidate) {
                return candidate.hard_leased;
            })) {
        floor_reason =
            common_cache_plan_destruction_reason::hard_lease_blocked;
    }

    // First DF2 execution ratchet: only replace the lawful lifecycle
    // host-capacity fallback. The calibrated/certified ladder above, hard
    // leases, pins, incoming publication, and token pressure remain exactly
    // where they were. Reproject on every victim; record only the first
    // decision in a multi-removal competition wave.
    if (retention_df2_capacity_authority &&
        reason == server_cache_destruction_reason::host_capacity &&
        legacy_floor != states.end()) {
        const auto projection = competition_wave_valid
            ? project_host_trade_df2(
                  *this, incoming, candidates, retention_shadow_rows.get(),
                  retention_shadow_artifacts.get(),
                  retention_shadow_lineages.get())
            : host_trade_df2_projection {};
        const auto proposed = projection.artifact;
        if (observe_retention_shadow) {
            const auto increment = [](uint64_t & value) noexcept {
                if (value != UINT64_MAX) {
                    value++;
                }
            };
            increment(retention_shadow.choices);
            auto & event = retention_shadow.last;
            event = {};
            event.reason = reason;
            event.competition_epoch =
                retention_obs->competition_epoch_value();
            event.candidate_count = projection.candidate_count;
            event.incumbent_artifact = host_entry_artifact_id(
                *this, *legacy_floor);
            if (projection.complete && proposed.v != 0 &&
                event.incumbent_artifact.v != 0) {
                event.proposed_artifact = proposed;
                event.proposed_lineage = projection.lineage_id;
                event.proposed_pool = projection.pool;
                event.proposed_lost_work = projection.lost_work;
                event.proposed_resource = projection.resource;
                event.status = server_prompt_cache_shadow_status::complete;
                event.agrees = event.incumbent_artifact == proposed;
                increment(retention_shadow.complete);
                increment(event.agrees
                    ? retention_shadow.agreements
                    : retention_shadow.disagreements);
                if (debug_observability) {
                    SRV_INF(
                        "CACHE_RETENTION_SHADOW status=complete reason=%u "
                        "epoch=%" PRIu64 " candidates=%" PRIu64
                        " incumbent=%" PRIu64 " proposed=%" PRIu64
                        " agrees=%s lost_work=%" PRIu64
                        " resource=%" PRIu64 "\n",
                        unsigned(reason), event.competition_epoch,
                        event.candidate_count,
                        event.incumbent_artifact.v,
                        event.proposed_artifact.v,
                        event.agrees ? "true" : "false",
                        event.proposed_lost_work,
                        event.proposed_resource);
                }
            } else {
                increment(retention_shadow.unavailable);
                if (debug_observability) {
                    SRV_INF(
                        "CACHE_RETENTION_SHADOW status=unavailable "
                        "reason=%u epoch=%" PRIu64
                        " candidates=%" PRIu64 "\n",
                        unsigned(reason), event.competition_epoch,
                        event.candidate_count);
                }
            }
        }
        observe_retention_shadow = false;
        if (proposed.v != 0) {
            const auto selected = std::find_if(
                candidates.begin(), candidates.end(), [&](const auto & value) {
                    return value.ranking.artifact_id == proposed &&
                        value.lease_known && !value.hard_leased &&
                        value.victim != incoming &&
                        value.victim->recovery_pins == 0;
            });
            if (selected != candidates.end()) {
                const int32_t source_id =
                    selected->victim->cache_plan_source_id;
                const size_t victim_bytes = selected->victim->size();
                if (destroy_df2_entry(selected->victim, reason)) {
                    if (destruction_obs) {
                        destruction_obs->host_trade_df2_executed++;
                    }
                    SRV_WRN(
                        " - removing DF2 host entry source_id=%d (size = %.3f MiB)\n",
                        source_id, victim_bytes / (1024.0 * 1024.0));
                    return true;
                }
            }
        }
    }
    if (destruction_obs) {
        destruction_obs->host_trade_legacy_fallbacks++;
    }
    return false;
}

bool server_prompt_cache::evict_front_under_pressure(
        server_cache_destruction_reason reason,
        iterator incoming,
        bool competition_wave_valid,
        bool observe_retention_shadow) {
    GGML_ASSERT(!states.empty());
    iterator legacy_floor = states.end();
    common_cache_plan_destruction_reason floor_reason =
        common_cache_plan_destruction_reason::capacity_refused;
    bool recovery_pin_excluded = false;
    if (destroy_priced_host_entry(
            reason, incoming, legacy_floor, floor_reason,
            recovery_pin_excluded, competition_wave_valid,
            observe_retention_shadow)) {
        return true;
    }

    // Lifecycle-off is the byte-identical historical FIFO floor. Once
    // lifecycle authority exists, the floor skips recovery pins and admits
    // only entries whose inspected lease is known non-hard.
    if (!publish_authority) {
        legacy_floor = states.begin();
    }
    if (legacy_floor != states.end()) {
        const auto floor_artifact = host_entry_artifact_id(
            *this, *legacy_floor);
        const int32_t floor_source_id = legacy_floor->cache_plan_source_id;
        if (observe_retention_shadow) {
            observe_retention_pressure_choice(
                reason, incoming, legacy_floor, competition_wave_valid);
        }
        SRV_WRN(
            " - removing fallback host entry source_id=%d (size = %.3f MiB)\n",
            legacy_floor->cache_plan_source_id,
            legacy_floor->size() / (1024.0 * 1024.0));
        destroy_entry(legacy_floor, reason);
        if (recovery_pin_excluded) {
            emit_host_pressure_floor_outcome(
                *this, "legacy_evicted", floor_artifact, floor_source_id);
        }
        return true;
    }

    // Hard leases are proof-backed guarantees. If no unpinned, known-nonhard
    // retained entry exists, publication—not an existing protected entry—is
    // the refused operation. Public maintenance without an incoming save
    // simply leaves the configured pressure visible for a later retry.
    if (destruction_obs && incoming != states.end()) {
        destruction_obs->note_host_trade_publication_skip();
    }
    if (publish_authority) {
        const uint64_t sequence =
            ++publish_authority->destruction_quote_sequence;
        observe_host_trade_refusal(*this, sequence, floor_reason);
    }
    if (incoming != states.end()) {
        if (observe_retention_shadow) {
            observe_retention_pressure_choice(
                reason, incoming, incoming, competition_wave_valid);
        }
        destroy_entry(incoming, reason);
    }
    if (recovery_pin_excluded) {
        emit_host_pressure_floor_outcome(
            *this, "publication_skipped", {}, -1);
    }
    return false;
}

bool server_prompt_cache::destroy_df2_entry(
        iterator it,
        server_cache_destruction_reason reason) {
    if (!publish_authority || !acct || it == states.end()) {
        return false;
    }

    server_prompt_cache_retirement_manifest retirement;
    if (!server_prompt_cache_capture_retirement(*this, it, retirement)) {
        return false;
    }
    std::vector<llama_cache_acct_op_id> ops;
    try {
        const auto release_ops = it->release_ops();
        ops.reserve(release_ops.size());
        for (const auto op : release_ops) {
            if (!op) {
                return false;
            }
            ops.push_back(op);
        }
    } catch (...) {
        return false;
    }

    const uint64_t serial = acct->serial();
    auto prepared = llama_cache_prepare_release_set(*acct, ops, serial);
    if (!prepared.ready()) {
        return false;
    }
    const auto admission =
        server_prompt_cache_observe_drop(*this, *it, reason);
    if (acct->serial() != prepared.accounting_serial()) {
        return false;
    }

    const std::thread::id scheduler_owner = std::this_thread::get_id();
    server_prompt_cache_destroy_entry_impl(*this, it);
    GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
    const auto release_status = prepared.commit();
    GGML_ASSERT(release_status ==
                llama_cache_conditional_release_status::released);
    server_prompt_cache_retire_manifest(*this, retirement);
    if (destruction_obs) {
        destruction_obs->note_prepared_release(admission.sequence, true);
    }
    return true;
}

server_prompt_cache::iterator server_prompt_cache::destroy_entry_impl(
        iterator it,
        server_cache_destruction_reason reason,
        iterator recovery) {
    const auto admission = server_prompt_cache_observe_drop(*this, *it, reason);
    // This pass-through owns exactly one accounting terminal outside the raw
    // physical primitive. Victim ordering belongs to the caller (historical
    // lifecycle floor or the gated DF2 authority); either route executes the
    // same exact terminal through a freshly prepared capability.
    const auto release_ops = it->release_ops();
    const std::thread::id scheduler_owner = std::this_thread::get_id();

    llama_cache_prepared_release_set prepared;
    server_prompt_cache_retirement_manifest retirement;
    // The legacy-fallback manifest is captured independently from D-A2's
    // certified manifest: a refused exact-redundancy proof must still execute
    // the historical retirement terminal. Both are read-only snapshots; only
    // the selected terminal retires them after the physical erase.
    const bool retirement_ready = publish_authority && acct &&
        server_prompt_cache_capture_retirement(*this, it, retirement);
    bool capability_ready = false;
    host_destruction_certification redundant;
    if (publish_authority && acct &&
        reason == server_cache_destruction_reason::host_dedup &&
        recovery != states.end()) {
        redundant = certify_host_destruction(
            *this, it, recovery, admission.sequence, false, true);
    }
    if (publish_authority && acct) {
        std::vector<llama_cache_acct_op_id> ops;
        bool setup_ok = retirement_ready;
        try {
            ops.reserve(release_ops.size());
            for (const auto op : release_ops) {
                if (op) {
                    ops.push_back(op);
                }
            }
        } catch (...) {
            setup_ok = false;
        }

        if (!redundant.ready && setup_ok && !ops.empty()) {
            const uint64_t serial = acct->serial();
            prepared = llama_cache_prepare_release_set(
                *acct, ops, serial);
            capability_ready = prepared.ready();
        }
    }

    if (!capability_ready && !redundant.ready) {
        server_prompt_cache_retire_entry(*this, it);
    }
    auto next = server_prompt_cache_destroy_entry_impl(*this, it);
    if (redundant.ready) {
        // D-A2 certify→mutate→commit boundary. Like D-A1, publication and
        // dedup run synchronously on update_slots. The physical list erase has
        // no callback or C producer; the immediate capability commit is the
        // only ledger terminal, so no ledger write can interleave. The
        // recovery pin remains live across both operations and prevents the
        // cited survivor from entering this raw primitive.
        commit_certified_host_destruction(
            *this, redundant, scheduler_owner);
        if (destruction_obs) {
            destruction_obs->note_redundant_host_executed(
                admission.sequence, redundant.projected_bytes);
        }
    } else if (capability_ready) {
        GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
        // D-A1 prepare→mutate→commit boundary. destroy_entry() is called
        // synchronously by update_slots-owned prompt-cache publication/load
        // maintenance. The raw erase only destroys value storage: it has no
        // callback and no C producer. The very next operation commits the
        // prepared release, so no scheduler-owned ledger write can interleave.
        // A future asynchronous producer invalidates this proof and must add
        // a real claim/lock before this authority remains enabled.
        // The same-frame assertion above is a machine-checked refactor
        // tripwire: the CMake contract scan deliberately string-matches it.
        const auto release_status = prepared.commit();
        GGML_ASSERT(release_status ==
                    llama_cache_conditional_release_status::released);
        server_prompt_cache_retire_manifest(*this, retirement);
        if (destruction_obs) {
            destruction_obs->note_prepared_release(
                admission.sequence, true);
        }
    } else if (acct) {
        // Preparation is fail-closed with respect to the new capability, but
        // D-A1 does not yet own host victim selection: retain the bounded
        // legacy FIFO/dedup behavior and its exactly-one release terminal.
        for (const auto op : release_ops) {
            if (op) {
                (void) acct->release(op);
            }
        }
        if (publish_authority && destruction_obs) {
            destruction_obs->note_prepared_release(
                admission.sequence, false);
        }
    }
    return next;
}

// Release symmetry for whole-cache destruction/replacement (model reload swaps the cache
// object while the observer ledger survives): every charged entry releases before the
// container dies, or the next snapshot would carry phantom host-cache bytes.
void server_prompt_cache::clear_accounting() {
    if (!acct && !destruction_obs && !retention_obs) {
        return;
    }
    for (auto & st : states) {
        server_prompt_cache_observe_drop(
            *this, st, server_cache_destruction_reason::host_shutdown);
        if (acct) {
            acct_release_entry(st);
        }
        if (retention_obs) {
            retention_obs->retire(
                server_retention_instance_key::for_host_entry(&st));
            for (auto & checkpoint : st.prompt.checkpoints) {
                retention_obs->retire(
                    server_retention_instance_key::for_checkpoint(
                        -1, &checkpoint));
            }
        }
    }
}

bool server_prompt_cache::publish(
        std::list<server_prompt_cache_state> entry,
        const server_prompt * source_prompt,
        int32_t source_slot,
        iterator * published) {
    if (published) {
        *published = states.end();
    }
    if (entry.empty()) {
        return false;
    }

    // F0b authority boundary: the detached entry is complete, but no shipped cache state has
    // changed yet. Refusal drops only this detached node; the live slot remains the sole copy.
    // The callback commits all accounting leaves before returning true, and states.splice below
    // is allocation-free/noexcept, so accounting can never lag a published entry.
    if (publish_authority &&
        !publish_authority->admit_host_entry(entry.front())) {
        return false;
    }

    // Splice the pre-allocated node in FIRST (no allocation, no throw) so the new entry is durably
    // linked before any potentially-throwing comparison below. Then remove cached prompts of the
    // SAME adapter identity fully contained in the new (larger) prompt [I6] -- a contained entry
    // under a different adapter config is a distinct valid state and is kept. If a comparison throws
    // (OOM) mid-loop, the new entry is already safely in `states`; at worst a few obsolete entries
    // remain (benign, FIFO-evicted later) -- never a lost node or partial corruption.
    states.splice(states.end(), entry);
    const auto self = std::prev(states.end());

    if (acct && !publish_authority) {
        acct_charge_entry(*self);
    }
    if (retention_obs && source_prompt && source_slot >= 0) {
        const auto source_key =
            server_retention_instance_key::for_slot(source_slot);
        const auto destination_key =
            server_retention_instance_key::for_host_entry(&*self);
        server_prompt_cache_mirror_artifact_clone(
            *this,
            source_key, common_retention_artifact_kind::live_slot,
            source_slot,
            destination_key, common_retention_artifact_kind::host_entry,
            -1,
            self->prompt, self->adapter_config_key,
            self->prompt.n_tokens());
        auto source_checkpoint = source_prompt->checkpoints.begin();
        auto host_checkpoint = self->prompt.checkpoints.begin();
        for (; source_checkpoint != source_prompt->checkpoints.end() &&
               host_checkpoint != self->prompt.checkpoints.end();
               ++source_checkpoint, ++host_checkpoint) {
            const auto source_checkpoint_key =
                server_retention_instance_key::for_checkpoint(
                    source_slot, &*source_checkpoint);
            const auto host_checkpoint_key =
                server_retention_instance_key::for_checkpoint(
                    -1, &*host_checkpoint);
            server_prompt_cache_mirror_artifact_clone(
                *this,
                source_checkpoint_key,
                common_retention_artifact_kind::checkpoint,
                source_slot,
                host_checkpoint_key,
                common_retention_artifact_kind::checkpoint,
                -1,
                self->prompt, self->adapter_config_key,
                host_checkpoint->n_tokens);
        }
    }

    for (auto it = states.begin(); it != states.end();) {
        if (it != self && it->adapter_config_key == self->adapter_config_key) {
            const int len = it->prompt.tokens.get_common_prefix(self->prompt.tokens);
            if (len == (int) it->prompt.tokens.size()) {
                // A D-A recovery citation outlives its destruction commit
                // through the dependent B execution. Dedup is another victim
                // enumerator: it must leave the cited physical host node in
                // place rather than reaching the raw eraser's invariant
                // assert while that non-policy pin is live.
                if (it->recovery_pins != 0) {
                    ++it;
                    continue;
                }
                SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);
                it = destroy_entry_impl(
                    it, server_cache_destruction_reason::host_dedup,
                    self);
                continue;
            }
        }
        ++it;
    }

    // enforce the cache limits through the single canonical eviction primitive. The entry's bytes
    // are already committed, so a local make-room loop would prevent no memory spike and, being
    // size-only, would skip the token limit. update() enforces both and evicts oldest-first,
    // preserving the just-added entry.
    if (!update_impl(self)) {
        return false;
    }
    if (published) {
        *published = self;
    }
    return true;
}

bool server_prompt_cache::prepare_restore_delivery(
        iterator source,
        server_prompt_cache_restore_delivery & delivery) const noexcept {
    delivery = {};
    delivery.cache_family = source->cache_family;
    if (!publish_authority) {
        return true;
    }
    try {
        if (server_fault("load_clone_fail")) {
            return false;
        }
        delivery.prompt = source->prompt.clone();
        delivery.retains_source = true;
        return true;
    } catch (...) {
        delivery = {};
        return false;
    }
}

static void server_prompt_cache_mirror_restore_retention(
        server_prompt_cache & cache,
        server_prompt_cache::iterator source,
        const server_prompt & destination,
        int32_t id_slot,
        bool retained_source,
        bool continues_lineage) {
    if (!cache.retention_obs) {
        return;
    }
    const auto host_key =
        server_retention_instance_key::for_host_entry(&*source);
    const auto live_key =
        server_retention_instance_key::for_slot(id_slot);
    if (continues_lineage) {
        server_prompt_cache_mirror_artifact_clone(
            cache,
            host_key, common_retention_artifact_kind::host_entry, -1,
            live_key, common_retention_artifact_kind::live_slot, id_slot,
            destination, source->adapter_config_key,
            destination.n_tokens());
    } else {
        const bool branched = cache.retention_obs->branch(
            host_key, live_key, nullptr, true);
        if (branched) {
            server_prompt_cache_mirror_prefix(
                cache, live_key, destination,
                source->adapter_config_key, destination.n_tokens());
        }
    }
    // Selection does not count as reuse. Carry the immutable host source in a
    // scheduler-consumed transition receipt so successful launch credits it
    // once before admitting a divergent destination; every unlaunched path
    // abandons the receipt and provisional branch.
    (void) cache.retention_obs->prepare_for_launch(host_key, live_key);

    const auto admit_restored_checkpoints = [&]() {
        if (!cache.publish_authority || !cache.retention_obs) {
            return;
        }
        try {
            std::vector<server_retention_instance_key> keys;
            std::vector<server_cache_live_checkpoint_admission> batch;
            keys.reserve(destination.checkpoints.size());
            batch.reserve(destination.checkpoints.size());
            for (const auto & checkpoint : destination.checkpoints) {
                const auto key =
                    server_retention_instance_key::for_checkpoint(
                        id_slot, &checkpoint);
                llama_cache_acct_artifact_id artifact;
                if (!cache.retention_obs->checkpoint_admission_artifact(
                        key, artifact)) {
                    continue;
                }
                const uint64_t accelerator_bytes = checkpoint.accel.size();
                server_cache_live_checkpoint_admission member;
                member.artifact = artifact;
                member.checkpoint_bytes =
                    checkpoint.size() >= accelerator_bytes
                        ? checkpoint.size() - accelerator_bytes
                        : 0;
                member.accelerator_bytes = accelerator_bytes;
                keys.push_back(key);
                batch.push_back(std::move(member));
            }
            if (batch.empty()) {
                return;
            }
            if (!cache.publish_authority->admit_live_checkpoints(batch)) {
                SRV_WRN(
                    "restored checkpoint ownership batch admission failed; "
                    "%zu members remain fail-closed\n", batch.size());
                return;
            }
            GGML_ASSERT(keys.size() == batch.size());
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!cache.retention_obs->attach_release_ops(
                        keys[i], std::move(batch[i].committed))) {
                    SRV_WRN("%s\n",
                        "restored checkpoint ownership attach failed; member remains fail-closed");
                }
            }
        } catch (...) {
            SRV_WRN("%s\n",
                "restored checkpoint ownership batch setup failed; ring remains fail-closed");
        }
    };

    if (!retained_source) {
        // std::list's equal allocator move transfers the original nodes, so
        // destination checkpoint addresses are the historical host keys.
        for (const auto & checkpoint : destination.checkpoints) {
            const auto host_checkpoint =
                server_retention_instance_key::for_checkpoint(
                    -1, &checkpoint);
            const auto live_checkpoint =
                server_retention_instance_key::for_checkpoint(
                    id_slot, &checkpoint);
            const bool rebound = continues_lineage
                ? cache.retention_obs->rebind(
                    host_checkpoint, live_checkpoint)
                : cache.retention_obs->branch(
                    host_checkpoint, live_checkpoint, &live_key);
            const auto artifact =
                cache.retention_obs->artifact_id(live_checkpoint);
            const server_cache_lease_subject checkpoint_destination {
                artifact,
                common_retention_artifact_kind::checkpoint,
                id_slot,
            };
            server_prompt_cache_mirror_lease(
                cache, rebound, nullptr, checkpoint_destination,
                destination, source->adapter_config_key,
                checkpoint.n_tokens);
        }
        admit_restored_checkpoints();
        return;
    }

    auto source_checkpoint = source->prompt.checkpoints.begin();
    auto destination_checkpoint = destination.checkpoints.begin();
    for (; source_checkpoint != source->prompt.checkpoints.end() &&
           destination_checkpoint != destination.checkpoints.end();
           ++source_checkpoint, ++destination_checkpoint) {
        const auto source_key =
            server_retention_instance_key::for_checkpoint(
                -1, &*source_checkpoint);
        const auto destination_key =
            server_retention_instance_key::for_checkpoint(
                id_slot, &*destination_checkpoint);
        if (continues_lineage) {
            server_prompt_cache_mirror_artifact_clone(
                cache,
                source_key, common_retention_artifact_kind::checkpoint, -1,
                destination_key, common_retention_artifact_kind::checkpoint,
                id_slot,
                destination, source->adapter_config_key,
                destination_checkpoint->n_tokens);
        } else {
            (void) cache.retention_obs->branch(
                source_key, destination_key, &live_key);
        }
    }
    GGML_ASSERT(source_checkpoint == source->prompt.checkpoints.end());
    GGML_ASSERT(destination_checkpoint == destination.checkpoints.end());
    admit_restored_checkpoints();
}

void server_prompt_cache::commit_restore_delivery(
        iterator source,
        server_prompt_cache_restore_delivery && delivery,
        server_prompt & destination,
        int32_t id_slot,
        int32_t debug_source_id,
        uint64_t reused_prefix_tokens,
        bool continues_lineage) {
    if (delivery.retains_source) {
        GGML_ASSERT(publish_authority != nullptr);
        destination = std::move(delivery.prompt);
        server_prompt_cache_mirror_restore_retention(
            *this, source, destination, id_slot, true,
            continues_lineage);
        if (retention_obs && reused_prefix_tokens == 0) {
            retention_obs->abandon_prepared_launch(
                server_retention_instance_key::for_slot(id_slot));
        }
        if (destruction_obs) {
            destruction_obs->note_host_restore(true);
        }
        if (debug_observability) {
            debug_lifecycle_emissions++;
            SRV_INF(
                "CACHE_HOST_LIFECYCLE {\"mode\":\"non_consuming\","
                "\"source_id\":%d,\"host_entries\":%zu,"
                "\"host_bytes\":%zu,\"retained_restores\":%" PRIu64 "}\n",
                debug_source_id, states.size(), size(),
                destruction_obs
                    ? destruction_obs->host_restores_retained
                    : uint64_t(0));
        }
        return;
    }

    // Lifecycle-off is the historical move/rebind/erase terminal verbatim.
    destination = std::move(source->prompt);
    server_prompt_cache_mirror_restore_retention(
        *this, source, destination, id_slot, false,
        continues_lineage);
    if (retention_obs && reused_prefix_tokens == 0) {
        retention_obs->abandon_prepared_launch(
            server_retention_instance_key::for_slot(id_slot));
    }
    if (destruction_obs) {
        destruction_obs->note_host_restore(false);
    }
    destroy_entry(
        source, server_cache_destruction_reason::host_consumed_restore);
}

// The observed/unobserved split is a compile-time instantiation (F8/B-a): with the observer
// off, load() runs the pre-B0 candidate loop with zero observer branches. Single source —
// every `if constexpr (Observed)` block vanishes from the <false> instantiation.
template <bool Observed>
bool server_prompt_cache::load_impl(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, const std::string & adapter_config_key, common_cache_plan_record * rec, int32_t required_source_id, common_cache_family_binding * restored_family) {
    if constexpr (!Observed) {
        (void) rec;
        (void) required_source_id;
    }
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // observer tallies [B-a]: these exist only in the observed instantiation, and only carry
    // values this selection computes anyway
    [[maybe_unused]] int32_t obs_source_best = -1;
    [[maybe_unused]] int     obs_lcp_sel  = 0;
    int reuse_lcp_best = 0;

    // find the most similar cached prompt, that would also preserve the most context.
    // Observer transport [A2, noexcept]: ONE row per visited entry, keyed by its
    // request-local immutable source id; every evaluated survivor starts as a cost loser and
    // the shipped winner is promoted to accepted after the scan. find_or_add returning
    // nullptr = inventory overflow — the provider's state latches and rows stop, the
    // shipped scan is untouched.
    for (auto it = states.begin(); it != states.end(); ++it) {
        [[maybe_unused]] common_cache_plan_candidate * row = nullptr;
        [[maybe_unused]] int32_t obs_source = -1;
        if constexpr (Observed) {
            if (cache_plan_get_source_id(*it, obs_source)) {
                // Required-provider authority evaluated every host row before
                // mutation. Save-before-load may deduplicate the list, but a
                // surviving node keeps its request-local id; skip non-selected
                // states before their O(context) token LCP.
                if (required_source_id >= 0 &&
                    obs_source != required_source_id) {
                    continue;
                }
                row = rec->find_or_add(
                    common_cache_plan_provider::host_cache_entry,
                    obs_source, COMMON_CACHE_PLAN_PHASE_HOST_SCAN,
                    rec->id_slot, rec->selection);
            } else {
                rec->inventory_states[size_t(
                    common_cache_plan_provider::host_cache_entry)] =
                    common_cache_plan_inventory_state::overflowed;
                if (required_source_id >= 0) {
                    continue;
                }
            }
        }

        int lcp_cur = 0;
        if constexpr (Observed) {
            lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
            server_cache_plan_apply_host(row, server_cache_plan_evaluate_host(
                !it->data.main.empty(),
                it->adapter_config_key == adapter_config_key,
                lcp_cur, tokens_new.size(), it->prompt.tokens.size(),
                it->data.size()));
        }

        // never select a structurally-empty entry [I7/I10]: a size-0 main would "restore" as a
        // false success (0 == 0) and leave the slot on empty state, then continue as if a prefix
        // were present -> pos_min == -1 with n_past > 0 abort. The transactional save/load here
        // never produces an empty-main entry, but guard the selector so a stray one is inert.
        if (it->data.main.empty()) {
            continue;
        }

        // never serve state built under a different adapter configuration [I6]: token-LCP alone
        // would hand adapter A's KV to a base/adapter-B request. Live-slot rebinds are caught at
        // launch, but a host-cache entry is restored here during prefill, after that check.
        if (it->adapter_config_key != adapter_config_key) {
            continue;
        }

        if constexpr (!Observed) {
            // Preserve the pre-B-A shipped instantiation: the potentially O(n)
            // LCP scan occurs only after both O(1) reject guards.
            lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
        }

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if constexpr (Observed) {
            if (required_source_id >= 0) {
                // B-A1 exact-provider authority: the planner already selected
                // this complete host plan. Preserve all structural/identity
                // guards above, but do not re-run the legacy two-axis choice.
                it_best = it;
                f_keep_best = f_keep_cur;
                f_sim_best = f_sim_cur;
                obs_source_best = obs_source;
                obs_lcp_sel = lcp_cur;
                reuse_lcp_best = lcp_cur;
                continue;
            }
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
            reuse_lcp_best = lcp_cur;
            if constexpr (Observed) {
                obs_source_best = obs_source;
                obs_lcp_sel  = lcp_cur; // the winner's exact LCP, from the shipped computation
            }
        }
    }

    if constexpr (Observed) {
        // the scan visits every entry (no short-circuit): the declared domain is complete
        // even when it is empty
        rec->note_inventory_complete(common_cache_plan_provider::host_cache_entry);
        if (it_best != states.end() && obs_source_best >= 0) {
            auto * win = rec->find_or_add(common_cache_plan_provider::host_cache_entry,
                                          obs_source_best, COMMON_CACHE_PLAN_PHASE_HOST_SCAN,
                                          rec->id_slot, rec->selection);
            if (win) {
                win->accept(); // shipped winner: promote over the scan-time cost-loser default
                win->lcp_tokens    = llama_cache_acct_value::measured((uint64_t) obs_lcp_sel);
                // bytes the restore actually installs (main+draft state) — NOT entry
                // size(), which also sums every retained checkpoint (verify-r1 finding 3)
                win->payload_bytes = llama_cache_acct_value::measured((uint64_t) it_best->data.size());
                rec->select(common_cache_plan_provider::host_cache_entry, win);
            }
        }
    }

    if (it_best == states.end()) {
        if constexpr (Observed) {
            if (required_source_id >= 0) {
                return false;
            }
        }
        // nothing better than the slot's current state; leave the slot as-is
        return true;
    }

    SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    // D-A1 stages the immutable source copy before either main or draft
    // context is touched. Allocation failure therefore leaves the source and
    // both target contexts at the caller-owned pre-restore boundary.
    server_prompt_cache_restore_delivery delivery;
    if (!prepare_restore_delivery(it_best, delivery)) {
        SRV_ERR("%s\n", "failed to stage non-consuming host restore");
        return false;
    }

    // Source bytes remain immutable throughout both restores. Lifecycle mode
    // keeps the entry after success too; legacy mode consumes it only after
    // BOTH sides succeed. On any failure the source remains fully intact and
    // the caller resets both target sequences, never leaving a half-restore.
    {
        const size_t size_tgt = it_best->data.main.size();
        size_t n_tgt = llama_state_seq_set_data_ext(ctx_tgt, it_best->data.main.data(), size_tgt, id_slot, 0);
        if (server_fault("load_fail")) { n_tgt = size_tgt > 0 ? size_tgt - 1 : 0; } // [P0 gate]
        if (n_tgt != size_tgt) {
            SRV_ERR("failed to restore target state (%zu != %zu bytes)\n", n_tgt, size_tgt);
            if constexpr (Observed) {
                // the accepted row was ELIGIBILITY; the attempted restore failed short —
                // note_reject demotes the disposition and records the structural reason
                if (auto * sel = rec->selected_row(common_cache_plan_provider::host_cache_entry)) {
                    sel->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
                }
            }
            return false;
        }
    }

    if (ctx_dft && !it_best->data.drft.empty()) {
        const size_t size_dft = it_best->data.drft.size();
        const size_t n_dft = llama_state_seq_set_data_ext(ctx_dft, it_best->data.drft.data(), size_dft, id_slot, 0);
        if (n_dft != size_dft) {
            SRV_WRN("failed to restore draft state (%zu != %zu bytes)\n", n_dft, size_dft);
            if constexpr (Observed) {
                if (auto * sel = rec->selected_row(common_cache_plan_provider::host_cache_entry)) {
                    sel->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
                }
            }
            return false;
        }
    }

    // Both sides restored: atomically select the lifecycle retain terminal or
    // the historical move+release+erase terminal.
    if constexpr (Observed) {
        if (auto * sel = rec->selected_row(common_cache_plan_provider::host_cache_entry)) {
            sel->delivered = true; // recorded at the delivery point, never inferred [B0]
        }
    }
    if (restored_family) {
        *restored_family = delivery.cache_family;
    }
    commit_restore_delivery(
        it_best, std::move(delivery), prompt, id_slot, obs_source_best,
        uint64_t(std::max(reuse_lcp_best, 0)),
        reuse_lcp_best == int(it_best->prompt.tokens.size()));

    return true;
}

template bool server_prompt_cache::load_impl<false>(server_prompt &, const server_tokens &, llama_context *, llama_context *, int32_t, const std::string &, common_cache_plan_record *, int32_t, common_cache_family_binding *);
template bool server_prompt_cache::load_impl<true>(server_prompt &, const server_tokens &, llama_context *, llama_context *, int32_t, const std::string &, common_cache_plan_record *, int32_t, common_cache_family_binding *);

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, const std::string & adapter_config_key, common_cache_plan_record * rec, int32_t required_source_id, common_cache_family_binding * restored_family) {
    GGML_ASSERT(rec != nullptr || required_source_id < 0);
    // one dispatch outside every loop: the off path is the pre-B0 loop [F8/B-a]
    return rec != nullptr
        ? load_impl<true>(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot, adapter_config_key, rec, required_source_id, restored_family)
        : load_impl<false>(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot, adapter_config_key, nullptr, required_source_id, restored_family);
}

void server_prompt_cache::update() {
    (void) update_impl(states.end());
}

bool server_prompt_cache::update_impl(iterator incoming) {
    bool pressure_wave_started = false;
    bool competition_wave_valid = true;
    bool retention_shadow_observed = false;
    size_t cache_bytes = 0;
    size_t cache_tokens = 0;
    const auto measure_cache = [&]() noexcept {
        cache_bytes = 0;
        cache_tokens = 0;
        for (const auto & state : states) {
            cache_bytes += state.size();
            cache_tokens += state.prompt.n_tokens();
        }
    };
    measure_cache();
    const auto begin_pressure_wave = [&]() noexcept {
        if (pressure_wave_started) {
            return;
        }
        pressure_wave_started = true;
        if (!retention_obs) {
            return;
        }
        if (retention_shadow.pressure_waves != UINT64_MAX) {
            retention_shadow.pressure_waves++;
        }
        competition_wave_valid =
            retention_obs->begin_competition_wave();
    };
    if (limit_size > 0) {
        while (!states.empty() && cache_bytes > limit_size) {
            begin_pressure_wave();
            SRV_WRN(" - cache size limit reached (size = %.3f MiB)\n",
                    cache_bytes / (1024.0 * 1024.0));

            const bool observe_shadow = !retention_shadow_observed;
            if (!evict_front_under_pressure(
                    server_cache_destruction_reason::host_capacity,
                    incoming, competition_wave_valid, observe_shadow)) {
                return false;
            }
            retention_shadow_observed |= observe_shadow;
            measure_cache();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(
        1.0f, float(cache_bytes) / std::max<size_t>(1, cache_tokens));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && cache_tokens > limit_tokens_cur) {
            begin_pressure_wave();
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur,
                    cache_bytes / (1024.0 * 1024.0));

            const bool observe_shadow = !retention_shadow_observed;
            if (!evict_front_under_pressure(
                    server_cache_destruction_reason::host_token_limit,
                    incoming, competition_wave_valid, observe_shadow)) {
                return false;
            }
            retention_shadow_observed |= observe_shadow;
            measure_cache();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), cache_bytes / (1024.0 * 1024.0),
            limit_size / (1024.0 * 1024.0), limit_tokens,
            limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
    return true;
}
