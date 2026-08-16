#pragma once

#include "common.h"
#include "common-cache-family.h"
#include "common-cache-plan.h" // B0 shadow observer row + C0 ledger types [P2]
#include "llama.h"
#include "server-cache-lifecycle.h"
#include "server-cache-lease.h"
#include "server-cache-plan-preflight.h"
#include "server-cache-control.h"
#include "server-retention-sidecar.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_set>
#include <list>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

using json = nlohmann::ordered_json;

bool server_cache_lease_build_identity(
    const std::string & execution_identity,
    const std::string & adapter_identity,
    const server_tokens & tokens,
    int64_t coverage_tokens,
    server_cache_lease_identity & out);

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_CACHE_CAPTURE,
    SERVER_TASK_TYPE_CACHE_IMPORT,
    SERVER_TASK_TYPE_CACHE_PLAN_PREFLIGHT,
    SERVER_TASK_TYPE_CACHE_HOLDER_CREATE,
    SERVER_TASK_TYPE_CACHE_HOLDER_CLOSE,
    SERVER_TASK_TYPE_CACHE_HOLDER_REATTACH,
    SERVER_TASK_TYPE_CACHE_FAMILY_REGISTER,
    SERVER_TASK_TYPE_CACHE_FAMILY_BIND,
    SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE,
    SERVER_TASK_TYPE_CACHE_LEASE_INSPECT,
    SERVER_TASK_TYPE_CACHE_LEASE_RENEW,
    SERVER_TASK_TYPE_CACHE_LEASE_RELEASE,
    SERVER_TASK_TYPE_CACHE_CONTROL_EVENTS,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params);

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // Optional E1 declared-family policy input. E1.2 supplies only this
    // opaque token; scheduler launch resolves the strong binding locally, so
    // an HTTP worker cannot inject a policy value into a task.
    server_cache_control_token cache_family_binding_token;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_CACHE_CAPTURE / SERVER_TASK_TYPE_CACHE_IMPORT
    struct cache_capture_action {
        int id_slot = -1;
        std::string tenant_key;
    };
    cache_capture_action cache_capture;

    struct cache_import_action {
        int id_slot = -1;
        std::string tenant_key;
        std::string reference;
    };
    cache_import_action cache_import;

    // E1.1a scheduler-internal control task. E1.2 is the only unit allowed to
    // construct this from HTTP; until then production has no caller.
    std::shared_ptr<const server_cache_control_request> cache_control;

    // E1.2 wire selectors carry semantic inputs only. The scheduler resolves
    // these into exact retention keys before invoking the E1.1a authority.
    struct cache_control_semantic_selector {
        int32_t slot_id = -1;
        std::shared_ptr<const server_tokens> tokens;
        std::map<int, float> lora;
    } cache_control_subject, cache_control_fallback;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    static server_task cache_control_task(
            server_cache_control_operation operation) {
        GGML_ASSERT(operation < server_cache_control_operation::_count);
        return server_task(static_cast<server_task_type>(
            SERVER_TASK_TYPE_CACHE_HOLDER_CREATE + int(operation)));
    }

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot
        copy.cache_family_binding_token = cache_family_binding_token;

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }
};

struct result_timings {
    int32_t cache_n = -1;

    int32_t prompt_n = -1;
    double prompt_ms = 0.0;
    double prompt_per_token_ms = 0.0;
    double prompt_per_second = 0.0;

    int32_t predicted_n = -1;
    double predicted_ms = 0.0;
    double predicted_per_token_ms = 0.0;
    double predicted_per_second = 0.0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;

    // effective bits/value of the attention KV cache at its current tensor types (moves at
    // runtime under dynamic VBR); emitted only when >= 0
    double kv_bpv = -1.0;

    json to_json() const;
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    result_timings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    // cache receipt (§7.7): serialized JSON attached verbatim when enabled;
    // empty = no receipt. Built in send_final_response from slot state.
    json cache_receipt;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    result_timings timings;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

struct server_task_result_metrics : server_task_result {
    int n_idle_slots;
    int n_processing_slots;
    int n_tasks_deferred;
    int64_t t_start;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    uint64_t n_draft_tokens_total      = 0;
    uint64_t n_draft_accepted_total    = 0;
    uint64_t n_draft_verif_steps_total = 0;
    std::vector<uint64_t> n_accepted_per_pos_total;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

enum class server_vbr_artifact_capture_status : uint8_t;
enum class server_vbr_artifact_import_status : uint8_t;
enum class vbr_manifest_validation_status : uint8_t;
enum class vbr_adopt_stage_status : uint8_t;
enum class vbr_downward_reserve_status : uint8_t;
enum class vbr_adopt_status : uint8_t;
enum class vbr_adopt_phase : uint8_t;
enum class vbr_downward_adopt_subphase : uint8_t;
enum class vbr_import_decision : uint8_t;

enum class server_cache_capture_consistency : uint8_t {
    unavailable = 0,
    capture_exact,
    _count,
};

struct server_task_result_cache_capture : server_task_result {
    server_vbr_artifact_capture_status status {};
    server_cache_capture_consistency consistency =
        server_cache_capture_consistency::unavailable;
    std::string reference;
    uint32_t controllers = 0;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t stash_bytes = 0;
    uint64_t companion_bytes = 0;
    uint64_t chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    bool dedup = false;

    virtual json to_json() override;
};

enum class server_cache_import_consistency : uint8_t {
    unavailable = 0,
    capture_exact,
    live_rebased,
    _count,
};

const char * server_cache_import_consistency_name(
    server_cache_import_consistency consistency) noexcept;

struct server_task_result_cache_import : server_task_result {
    server_vbr_artifact_import_status status {};
    vbr_manifest_validation_status validation_status {};
    vbr_adopt_stage_status stage_status {};
    vbr_downward_reserve_status downward_reserve_status {};
    vbr_adopt_status adopt_status {};
    bool adopt_attempted = false;
    vbr_adopt_phase phase {};
    vbr_downward_adopt_subphase downward_subphase {};
    uint32_t downward_edge = UINT32_MAX;
    vbr_import_decision decision {};
    server_cache_import_consistency consistency =
        server_cache_import_consistency::unavailable;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t companion_bytes = 0;

    virtual json to_json() override;
};

// Internal E0.1 scheduler result. E0.2 adds the deliberately redacted wire
// serializer; until then no route can serialize this task result.
struct server_task_result_cache_plan_preflight : server_task_result {
    server_cache_plan_preflight_view view;

    virtual json to_json() override;
};

struct server_task_result_cache_control : server_task_result {
    server_cache_control_operation operation =
        server_cache_control_operation::_count;
    server_cache_control_result result;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    // Server-local lineage for the computation-frontier migration [WS-4].
    // It moves with host-cache/child-slot prompt clones and resets whenever
    // the prompt ledger is structurally cleared.
    uint64_t sequence_epoch = 0;

    void clear() {
        tokens.clear();
        checkpoints.clear();
        sequence_epoch = 0;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            sequence_epoch,
        };
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    // canonical identity of the adapter configuration this state was computed under [I6]; a load is
    // only served from an entry whose key matches the requesting slot's current adapter config
    std::string adapter_config_key;

    // C0 shadow accounting op ids, one per charged leaf category (zero id = not charged):
    // the publish boundary, released when the entry leaves `states` [P2]
    llama_cache_acct_op_id acct_op_snapshot;
    llama_cache_acct_op_id acct_op_ckpt;
    llama_cache_acct_op_id acct_op_accel;

    // D-A2's non-policy recovery guard. List nodes are stable; authoritative
    // redundant eviction increments this before prepare and the raw eraser
    // refuses to destroy the cited survivor until capability close.
    uint32_t recovery_pins = 0;

    // Automatic pre-E1 family signal. A save sourced from a parent/main slot
    // receives the provisional D-A3 retention weight; child-task saves do not.
    // E1 declared identity replaces this heuristic rather than stacking with it.
    bool main_family = false;
    common_cache_family_binding cache_family;

    std::array<llama_cache_acct_op_id, 3> release_ops() const noexcept {
        return { acct_op_snapshot, acct_op_ckpt, acct_op_accel };
    }

    // Request-local observer identity. It lives on the list node so save-time
    // dedup/splice preserves surviving identities and an allocator-reused
    // address can never inherit a consumed entry's source id.
    int32_t cache_plan_source_id = -1;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

inline void server_prompt_cache_apply_family(
        server_prompt_cache_state & state,
        common_cache_family_binding binding,
        bool automatic_main_family) noexcept {
    state.cache_family = binding;
    state.main_family = common_cache_family_main_family(
        binding, automatic_main_family);
}

struct server_cache_authority;
class server_cache_recovery_pin;

struct server_prompt_cache_payload_leaf {
    llama_cache_acct_category category =
        llama_cache_acct_category::full_snapshot_payload;
    uint64_t bytes = 0;
    llama_cache_acct_op_id * operation = nullptr;
};

// Move-only storage transaction staged before llama_state_seq_set_data_ext().
// In lifecycle mode it owns an immutable copy of the host prompt/checkpoints;
// the successful delivery moves this copy into the live slot and retains the
// source node. In legacy mode it stays empty and commit consumes the source.
// Move-only is currently derived from server_prompt/server_tokens; preserve
// that intent explicitly if server_tokens ever becomes copyable.
struct server_prompt_cache_restore_delivery {
    server_prompt prompt;
    common_cache_family_binding cache_family;
    bool retains_source = false;
};

constexpr size_t SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES =
    SERVER_RETENTION_MAX_CANDIDATES;

enum class server_prompt_cache_shadow_status : uint8_t {
    unavailable = 0,
    complete,
};

struct server_prompt_cache_shadow_event {
    server_prompt_cache_shadow_status status =
        server_prompt_cache_shadow_status::unavailable;
    server_cache_destruction_reason reason =
        server_cache_destruction_reason::host_capacity;
    uint64_t competition_epoch = 0;
    uint64_t candidate_count = 0;
    llama_cache_acct_artifact_id incumbent_artifact;
    llama_cache_acct_artifact_id proposed_artifact;
    uint64_t incumbent_lineage = 0;
    uint64_t proposed_lineage = 0;
    common_retention_pool proposed_pool = common_retention_pool::attention;
    uint64_t proposed_lost_work = 0;
    uint64_t proposed_resource = 0;
    bool agrees = false;
};

struct server_prompt_cache_shadow_snapshot {
    uint64_t pressure_waves = 0;
    uint64_t choices = 0;
    uint64_t complete = 0;
    uint64_t unavailable = 0;
    uint64_t agreements = 0;
    uint64_t disagreements = 0;
    server_prompt_cache_shadow_event last;
};

struct server_prompt_cache_shadow_row {
    llama_cache_acct_artifact_id artifact_id;
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    common_retention_stamp stamp;
    common_retention_lineage_record lineage;
    uint64_t resource = 0;
    bool backing_known = false;
    bool releasable = false;
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens);

    std::list<server_prompt_cache_state> states;
    using iterator = std::list<server_prompt_cache_state>::iterator;
    using const_iterator = std::list<server_prompt_cache_state>::const_iterator;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    int32_t cache_plan_next_source_id = 0;

    size_t size() const;

    size_t n_tokens() const;

    // true if a token-identical entry with the SAME adapter identity is already fully cached, i.e.
    // the state is durable and the live slot may be safely cleared without saving again [I6/I7].
    bool contains(const server_tokens & tokens, const std::string & adapter_config_key) const;

    // Resolve the exact durable host state used by prompt_save's durability
    // predicate and pin its three-payload accounting source. D-A5 calls this
    // after the same-flow save and before preparing live-slot destruction.
    bool acquire_durable_recovery(
            const server_tokens & tokens,
            const std::string & adapter_config_key,
            llama_cache_acct_artifact_id & artifact,
            std::vector<llama_cache_acct_op_id> & ops,
            server_cache_recovery_pin & pin) noexcept;

    bool acquire_durable_recovery(
            iterator state,
            llama_cache_acct_artifact_id & artifact,
            std::vector<llama_cache_acct_op_id> & ops,
            server_cache_recovery_pin & pin) noexcept;

    void cache_plan_begin_inventory() noexcept;
    bool cache_plan_get_source_id(
        server_prompt_cache_state & state,
        int32_t & source_id) noexcept;

    // Transactional save is a stage -> fill -> publish sequence [I7]. stage() allocates a DETACHED
    // single-node list WITHOUT touching `states`; any allocation failure there leaves the cache
    // completely untouched (no eviction, no limit change). The caller fills + validates the state
    // bytes; publish() then removes now-obsolete entries and splices the completed node in (no
    // allocation, no throw). A failed fill drops the staged node — never a poisoned/half-filled
    // published entry, never an eviction that bought nothing. Under lifecycle hard-lease pressure,
    // publish() may also return false after removing only its just-spliced incoming node; every
    // previously retained hard-leased/recovery-pinned entry remains untouched.
    std::list<server_prompt_cache_state> stage(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft, std::string adapter_config_key);
    bool publish(
            std::list<server_prompt_cache_state> entry,
            const server_prompt * source_prompt = nullptr,
            int32_t source_slot = -1,
            iterator * published = nullptr);

    // `obs` is the B0 shadow observer row for the host_cache_entry candidate (nullptr = observer
    // off). It only receives values this selection already computes — never a re-scan [B-a].
    // Dispatches ONCE to an unobserved or observed instantiation, so the disabled path's
    // candidate loop is the pre-B0 loop with zero observer branches.
    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, const std::string & adapter_config_key, common_cache_plan_record * rec = nullptr, int32_t required_source_id = -1, common_cache_family_binding * restored_family = nullptr);

    template <bool Observed>
    bool load_impl(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, const std::string & adapter_config_key, common_cache_plan_record * rec, int32_t required_source_id, common_cache_family_binding * restored_family);

    // D-A1's two-phase immutable host restore. prepare() runs before either
    // target is touched; commit() is called only after main+draft restore.
    // Public only so the model-free server cache test can pin the storage
    // transaction without constructing a llama_context.
    bool prepare_restore_delivery(
            iterator source,
            server_prompt_cache_restore_delivery & delivery) const noexcept;
    void commit_restore_delivery(
            iterator source,
            server_prompt_cache_restore_delivery && delivery,
            server_prompt & destination,
            int32_t id_slot,
            int32_t debug_source_id = -1,
            uint64_t reused_prefix_tokens = 0,
            bool continues_lineage = true);

    void update();

    iterator destroy_entry(
            iterator it,
            server_cache_destruction_reason reason);

    // Exact D-A2 proof over snapshot, checkpoint-ring, and typed accelerator
    // payloads. Token coverage is necessary but never sufficient.
    static bool exactly_redundant(
            const server_prompt_cache_state & victim,
            const server_prompt_cache_state & survivor) noexcept;

    // C0 ledger (nullptr = off). Debug-only retains shadow semantics; lifecycle publication
    // and explicit host erasure use its reservation/prepared-release authority. Every path
    // that removes an entry from `states` releases its ops, including whole-cache replacement,
    // or the surviving ledger would carry phantom bytes. It outlives this cache by member order.
    llama_cache_acct_ledger * acct = nullptr;
    // F0b/D-A1 lifecycle authority. Null keeps the consuming legacy path. When present, it
    // gates publication, makes restore non-consuming, and prepares explicit-eviction releases.
    server_cache_authority * publish_authority = nullptr;
    server_cache_destruction_observer * destruction_obs = nullptr;
    server_retention_sidecar_store * retention_obs = nullptr;
    server_cache_lease_table * lease_obs = nullptr;
    const std::string * lease_execution_identity = nullptr;
    // Explicit emission gate. An observed load also exists under B authority,
    // so rec != nullptr is not evidence that --cache-debug was enabled.
    bool debug_observability = false;
    uint64_t debug_lifecycle_emissions = 0;
    uint64_t debug_destruction_emissions = 0;
    uint64_t debug_recovery_pin_exclusions = 0;
    uint64_t debug_host_pressure_floor_outcomes = 0;
    llama_cache_acct_artifact_id debug_last_recovery_pin_excluded;
    bool host_trade_substrate_warned = false;

    ~server_prompt_cache() {
        clear_accounting();
    }

    void clear_accounting();
    void acct_charge_entry(server_prompt_cache_state & st);
    void acct_release_entry(server_prompt_cache_state & st);

    static bool payload_bytes(
            const server_prompt_cache_state & st,
            uint64_t & snapshot_bytes,
            uint64_t & checkpoint_bytes,
            uint64_t & accelerator_bytes) noexcept;
    static bool payload_leaves(
            server_prompt_cache_state & st,
            std::array<server_prompt_cache_payload_leaf, 3> & leaves) noexcept;

    bool enable_retention_shadow() noexcept;

    server_prompt_cache_shadow_snapshot retention_shadow_snapshot() const noexcept {
        return retention_shadow;
    }

private:
    iterator find_state_exact(
        const server_tokens & tokens,
        const std::string & adapter_config_key) noexcept;
    const_iterator find_state_exact(
        const server_tokens & tokens,
        const std::string & adapter_config_key) const noexcept;
    bool destroy_priced_host_entry(
            server_cache_destruction_reason reason,
            iterator incoming,
            iterator & legacy_floor,
            common_cache_plan_destruction_reason & floor_reason,
            bool & recovery_pin_excluded,
            bool competition_wave_valid);
    bool evict_front_under_pressure(
            server_cache_destruction_reason reason,
            iterator incoming,
            bool competition_wave_valid);
    bool update_impl(iterator incoming);
    void observe_retention_pressure_choice(
            server_cache_destruction_reason reason,
            iterator incoming,
            iterator incumbent,
            bool competition_wave_valid) noexcept;
    iterator destroy_entry_impl(
            iterator it,
            server_cache_destruction_reason reason,
            iterator recovery);

    std::unique_ptr<server_prompt_cache_shadow_row[]> retention_shadow_rows;
    server_prompt_cache_shadow_snapshot retention_shadow;
};

// E1.1a proof adapter over the same list-node recovery counter consulted by
// every host victim selector and by the raw eraser assertion. The semantic
// selector is resolved against the current list before the pin is acquired.
server_cache_durable_fallback_proof
server_prompt_cache_host_fallback_proof(
    server_prompt_cache & cache,
    const server_cache_control_selector & selector) noexcept;

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
