#pragma once

#include "common-cache-plan.h"

#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// This is an internal scheduler result only. The public adapter owns the JSON schema;
// none of these values is a reservation, capability, or replayable claim.
enum class server_cache_plan_preflight_status : uint8_t {
    ok = 0,
    no_target,
    internal_fault,
    _count,
};

// Closed  expected-path vocabulary. New public semantics require an  wire
// schema revision rather than a synonym or intermediate state here.
enum class server_cache_plan_preflight_expected_path : uint8_t {
    legacy = 0,
    planner_if_still_current,
    conditional_on_destruction_certification,
    _count,
};

enum class server_cache_plan_preflight_target_relation : uint8_t {
    unavailable = 0,
    forced_slot,
    same_as_legacy,
    retarget,
    _count,
};

enum class server_cache_plan_preflight_cache_hit : uint8_t {
    unavailable = 0,
    miss,
    partial,
    full,
    _count,
};

struct server_cache_plan_preflight_miss_reason {
    common_cache_plan_provider provider =
        common_cache_plan_provider::cold_replay;
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_NONE;
    uint32_t count = 0;
};

struct server_cache_plan_preflight_destruction {
    common_cache_plan_destruction_state state =
        common_cache_plan_destruction_state::failed;
    common_cache_plan_destruction_reason reason =
        common_cache_plan_destruction_reason::manifest_incomplete;
    common_cache_plan_destruction_effect_set effects = 0;
    common_cache_plan_destruction_lease_verdict protection =
        common_cache_plan_destruction_lease_verdict::unavailable;
    common_cache_plan_displaced_fate displaced_fate =
        common_cache_plan_displaced_fate::unavailable;
    common_cache_plan_recovery_citation recovery =
        common_cache_plan_recovery_citation::unavailable;
    llama_cache_acct_value projected_release_bytes;
    llama_cache_acct_value estimated_destruction_us;
};

struct server_cache_plan_preflight_view {
    server_cache_plan_preflight_status status =
        server_cache_plan_preflight_status::internal_fault;
    server_cache_plan_preflight_expected_path expected_path =
        server_cache_plan_preflight_expected_path::legacy;
    common_cache_plan_planner_status planner_status =
        common_cache_plan_planner_status::not_attempted;
    common_cache_plan_authority_level configured_level =
        common_cache_plan_authority_level::off;
    common_cache_plan_selection selection_tier =
        common_cache_plan_selection::none;
    common_cache_plan_authority_fallback fallback_reason =
        common_cache_plan_authority_fallback::none;
    common_cache_plan_provider provider =
        common_cache_plan_provider::cold_replay;
    bool provider_available = false;
    server_cache_plan_preflight_target_relation target_relation =
        server_cache_plan_preflight_target_relation::unavailable;
    server_cache_plan_preflight_cache_hit cache_hit =
        server_cache_plan_preflight_cache_hit::unavailable;
    llama_cache_acct_value prompt_tokens;
    llama_cache_acct_value predicted_reuse_tokens;
    llama_cache_acct_value predicted_replay_tokens;
    llama_cache_acct_value predicted_restore_bytes;
    llama_cache_acct_value predicted_ttft_us;
    uint32_t estimator_version = 0;
    std::array<llama_cache_acct_cost_term,
               size_t(llama_cache_acct_cost_kind::_count)> cost_terms =
        common_cache_plan_default_cost_terms();
    server_cache_plan_preflight_destruction destruction;
    std::vector<server_cache_plan_preflight_miss_reason> miss_reasons;
};

// Public v1 projection. This is deliberately independent of the schema-7
// debug serializer: identities, digests, accounting rows, ordinals, serials,
// leases, and recovery-source handles have no representation here.
nlohmann::ordered_json server_cache_plan_preflight_json(
    const server_cache_plan_preflight_view & view);

// Exposure remains opt-in and single-principal. The existing API-key
// middleware authenticates zero/one configured key;  refuses configurations
// where that middleware represents multiple principals.
bool server_cache_plan_preflight_exposure_allowed(
    const std::string & hostname,
    size_t api_key_count) noexcept;

bool server_cache_plan_preflight_request_field_allowed(
    std::string_view field) noexcept;

server_cache_plan_preflight_expected_path
server_cache_plan_preflight_derive_expected_path(
    const common_cache_plan_record & rec,
    bool planner_inputs_current = true) noexcept;

bool server_cache_plan_preflight_build_view(
    const common_cache_plan_record & rec,
    int32_t legacy_target_slot_id,
    bool planner_inputs_current,
    server_cache_plan_preflight_view & out) noexcept;
