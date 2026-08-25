#pragma once

#include "common-cache-plan.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

enum class server_cache_plan_execution_kind : uint8_t {
    legacy = 0,
    live_replay,
    host_restore,
    checkpoint_restore,
    host_checkpoint_restore,
    cold_replay,
    _count,
};

// Process-local execution capability. It contains only request-local inventory
// ordinals/source ids, never a pointer into a cache container. The server
// revalidates those ids immediately before the first mutation.
struct server_cache_plan_execution {
    server_cache_plan_execution_kind kind =
        server_cache_plan_execution_kind::legacy;
    int32_t target = -1;
    int32_t host_source_id = -1;
    int32_t checkpoint_source_id = -1;

    constexpr bool authoritative() const noexcept {
        return kind != server_cache_plan_execution_kind::legacy;
    }

    constexpr bool restores_checkpoint() const noexcept {
        return kind == server_cache_plan_execution_kind::checkpoint_restore ||
               kind == server_cache_plan_execution_kind::host_checkpoint_restore;
    }

    void clear() noexcept { *this = {}; }
};

static_assert(uint8_t(common_cache_plan_selection::none) ==
              uint8_t(common_cache_plan_authority_level::off));
static_assert(uint8_t(common_cache_plan_selection::by_id) ==
              uint8_t(common_cache_plan_authority_level::by_id));
static_assert(uint8_t(common_cache_plan_selection::similarity) ==
              uint8_t(common_cache_plan_authority_level::similarity));
static_assert(uint8_t(common_cache_plan_selection::route_home) ==
              uint8_t(common_cache_plan_authority_level::route_home));
static_assert(uint8_t(common_cache_plan_selection::lru) ==
              uint8_t(common_cache_plan_authority_level::lru));
static_assert(uint8_t(common_cache_plan_selection::_count) ==
              uint8_t(common_cache_plan_authority_level::_count));

constexpr common_cache_plan_authority_level server_cache_plan_level_of(
        common_cache_plan_selection selection) noexcept {
    return static_cast<common_cache_plan_authority_level>(selection);
}

// Highest behavior-changing ratchet implemented in this tree. LRU is the final
// declared level; its configured value keeps every earlier prefix tier enabled.
constexpr common_cache_plan_authority_level
    SERVER_CACHE_PLAN_IMPLEMENTED_AUTHORITY_LEVEL =
        common_cache_plan_authority_level::lru;

constexpr bool server_cache_plan_level_enabled(
        common_cache_plan_authority_level configured,
        common_cache_plan_authority_level decision) noexcept {
    return configured >= decision &&
           SERVER_CACHE_PLAN_IMPLEMENTED_AUTHORITY_LEVEL >= decision;
}

constexpr bool server_cache_plan_candidate_prequalified(
        const common_cache_plan_record & rec) noexcept {
    return rec.authority_prequalified &&
           rec.planner_status == common_cache_plan_planner_status::ok;
}

constexpr bool server_cache_plan_selection_admits_retarget(
        common_cache_plan_authority_level configured,
        common_cache_plan_selection selection) noexcept {
    const auto decision = server_cache_plan_level_of(selection);
    return decision > common_cache_plan_authority_level::by_id &&
           decision < common_cache_plan_authority_level::_count &&
           server_cache_plan_level_enabled(configured, decision);
}

constexpr bool server_cache_plan_shadow_choice_valid(
        const common_cache_plan_record & rec) noexcept {
    return rec.shadow_choice >= 0 &&
           uint32_t(rec.shadow_choice) < rec.n_inventory;
}

// Revalidate inventory currency only when authority changes the physical
// target. Same-target executions are revalidated by their provider-specific
// seam; in particular an empty LRU target has no live row to revalidate.
constexpr bool server_cache_plan_retarget_currency_required(
        bool selection_admits_retarget,
        int32_t planned_target_slot_id,
        int32_t legacy_target_slot_id) noexcept {
    return selection_admits_retarget &&
           planned_target_slot_id != legacy_target_slot_id;
}

// Cold authority is executable only against a still-construction-empty target.
// Keep this predicate shared with the model-free seam regression.
constexpr bool server_cache_plan_cold_target_current(
        const server_cache_plan_execution & execution,
        bool prompt_empty,
        bool checkpoints_empty) noexcept {
    return execution.kind != server_cache_plan_execution_kind::cold_replay ||
           (prompt_empty && checkpoints_empty);
}

// Single planned-target derivation for both authorization and the server's
// process-local slot lookup. -1 is malformed/incomplete evidence.
constexpr int32_t server_cache_plan_planned_target(
        const common_cache_plan_record & rec,
        common_cache_plan_authority_level configured,
        int32_t legacy_target_slot_id) noexcept {
    if (!server_cache_plan_selection_admits_retarget(
            configured, rec.selection)) {
        return legacy_target_slot_id;
    }
    if (!server_cache_plan_shadow_choice_valid(rec)) {
        return -1;
    }
    const auto & selected = rec.inventory[size_t(rec.shadow_choice)];
    return common_cache_plan_origin_in_domain(
               selected.origin_tier, rec.selection)
        ? selected.target_slot_id : -1;
}

// One compiled classification door shared by planner authority and
// destruction-quote assembly. A nonzero bit is precisely a destruction certificate
// the envelope must refuse until its ratchet is enabled.
int32_t server_cache_plan_host_source(
    const common_cache_plan_record & rec,
    int32_t candidate) noexcept;

constexpr common_cache_plan_destruction_effect_set
    SERVER_CACHE_LIVE_DISPLACEMENT_EFFECTS =
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::cross_target_displacement) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                destructive_similarity_retarget) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                same_target_cold_replacement);

constexpr common_cache_plan_destruction_effect_set
server_cache_plan_nonconsuming_host_effects(bool lifecycle) noexcept {
    return lifecycle
        ? common_cache_plan_destruction_effect_bit(
              common_cache_plan_destruction_effect::
                  different_host_source_consumption)
        : 0;
}

common_cache_plan_destruction_effect_set server_cache_destruction_effects_for(
    const common_cache_plan_record & rec,
    int32_t candidate,
    int32_t legacy_candidate,
    common_cache_plan_destruction_effect_set permitted_effects = 0) noexcept;

// Pre-mutation decision substrate. It is process-local and contains no
// shipped cache state. Authority is graduated through the parallel selection
// and configured-level order pinned above.
struct server_cache_plan_authority {
    common_cache_plan_authority_level configured_level =
        common_cache_plan_authority_level::off;
    common_cache_plan_authority_counters counters;
    std::string calibration_profile;

    explicit server_cache_plan_authority(
        common_cache_plan_authority_level level) noexcept : configured_level(level) {}

    // Runs the existing cost planner against the complete, target-qualified
    // pre-mutation inventory. Capability is sampled on both sides of the call;
    // drift refuses qualification rather than changing the shipped path.
    void plan_before_mutation(
        common_cache_plan_record & rec,
        uint64_t capability_before,
        uint64_t capability_after) noexcept;

    void fail_closed(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason =
            common_cache_plan_authority_fallback::internal_fault) noexcept;

    // Authorize one complete plan at the record's selected tier. The argument
    // names the legacy-selected target: by_id remains bound to it, while later
    // tiers may name a different target only inside the non-destructive safety
    // envelope. Any planner refusal or malformed plan returns the legacy
    // directive and records a typed fallback before mutation.
    server_cache_plan_execution authorize(
        common_cache_plan_record & rec,
        int32_t legacy_target_slot_id,
        bool host_lookup_enabled = true,
        bool target_identity_matches = true,
        common_cache_plan_destruction_effect_set permitted_effects = 0) noexcept;

    // Capability drift discovered after planning but before mutation. Preserve
    // the planner verdict for agreement telemetry while reverting execution to
    // the untouched legacy path.
    void fallback_legacy(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason) noexcept;

    // Execution has completed. The schema-v5 receipt keeps the counterfactual
    // legacy plan under authority and always records the plan that really ran.
    void finalize_execution(common_cache_plan_record & rec) noexcept;
};

// Counterfactual forced-slot legacy provider sequence over the complete
// pre-mutation inventory. This is telemetry identity only; it never executes.
int32_t server_cache_plan_legacy_candidate(
    const common_cache_plan_record & rec,
    int32_t target_slot_id,
    bool host_lookup_enabled = true) noexcept;

bool server_cache_plan_execution_from_candidate(
    const common_cache_plan_record & rec,
    int32_t candidate,
    int32_t target_slot_id,
    server_cache_plan_execution & out) noexcept;

constexpr int32_t server_cache_plan_checkpoint_ordinal_from_source_id(
    int32_t source_id,
    int32_t host_source_id) noexcept;

// A selected slot is armed before launch so the existing mutation seams can
// consume the directive. The recovery source outlives a successful
// displacement through the dependent B execution, but every exit that does
// not launch must disarm all three process-local pieces together; otherwise a
// later request could inherit a stale directive/record or over-retain the
// displaced state's durable copy.
template<class PlanPtr, class RecoveryPin>
void server_cache_plan_disarm_unlaunched(
        server_cache_plan_execution & execution,
        PlanPtr & plan,
        RecoveryPin & recovery_pin) noexcept {
    execution.clear();
    plan.reset();
    recovery_pin = {};
}

// Coverage recovery is a shipped correctness seam, not a cost-planner input.
// Until the planner prices SWA/recurrent frontiers, an authoritative non-checkpoint
// plan reaching this condition must demote and run the legacy recovery block.
constexpr bool server_cache_plan_requires_coverage_recovery(
        const server_cache_plan_execution & execution,
        int64_t pos_min,
        int64_t pos_min_threshold) noexcept {
    return execution.authoritative() && !execution.restores_checkpoint() &&
           pos_min >= pos_min_threshold;
}

bool server_cache_plan_demote_for_coverage_recovery(
    server_cache_plan_authority & authority,
    common_cache_plan_record & rec,
    server_cache_plan_execution & execution,
    int64_t pos_min,
    int64_t pos_min_threshold) noexcept;

// Dynamic-VBR low-LCP reset is another shipped correctness recovery that can
// supersede a previously authorized plan. The exact outcome is known only
// after idle reclaim and the second ownership sample; a completed reset must
// therefore demote before the execution receipt is finalized.
bool server_cache_plan_demote_for_vbr_low_lcp_reset(
    server_cache_plan_authority & authority,
    common_cache_plan_record & rec,
    server_cache_plan_execution & execution,
    bool reset_applied) noexcept;

constexpr bool server_cache_plan_live_replay_lost_to_logits(
        const server_cache_plan_execution & execution,
        int64_t n_past_after_decrement) noexcept {
    return n_past_after_decrement == 0 &&
           execution.kind == server_cache_plan_execution_kind::live_replay;
}

// Translate a planned checkpoint only after the live seam has revalidated its
// eligibility. A false result means "use the legacy iterator", never "cold".
constexpr bool server_cache_plan_checkpoint_override_ordinal(
        const server_cache_plan_execution & execution,
        size_t checkpoint_count,
        bool eligible,
        int32_t & ordinal) noexcept {
    if (!execution.restores_checkpoint() || !eligible) {
        ordinal = -1;
        return false;
    }
    const int32_t host_source =
        execution.kind ==
                server_cache_plan_execution_kind::host_checkpoint_restore
            ? execution.host_source_id : -1;
    ordinal = server_cache_plan_checkpoint_ordinal_from_source_id(
        execution.checkpoint_source_id, host_source);
    return ordinal >= 0 && size_t(ordinal) < checkpoint_count;
}

bool server_cache_plan_revalidate_checkpoint_execution(
    server_cache_plan_authority & authority,
    common_cache_plan_record & rec,
    server_cache_plan_execution & execution,
    size_t checkpoint_count,
    bool eligible,
    int32_t & ordinal) noexcept;

// Stable, allocation-free capability fold used at the immediately-pre-mutation
// revalidation seam. Callers fold exactly the state that made candidates usable.
uint64_t server_cache_plan_capability_fold(
    uint64_t hash,
    uint64_t value) noexcept;

constexpr int32_t SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE = 1000000;
constexpr int32_t SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE = 10000;
constexpr int32_t SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID =
    (INT32_MAX - SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE -
     (SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE - 1)) /
    SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE;

constexpr int32_t server_cache_plan_host_checkpoint_source_id(
        int32_t host_source_id,
        int32_t checkpoint_ordinal = 0) noexcept {
    return host_source_id < 0 ||
           host_source_id > SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID ||
           checkpoint_ordinal < 0 ||
           checkpoint_ordinal >= SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE
        ? -1
        : SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE +
          host_source_id*SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE +
          checkpoint_ordinal;
}

// Decode a checkpoint source id to the inventory's stable forward ordinal.
// A host-qualified source must remain inside that host's namespace.
constexpr int32_t server_cache_plan_checkpoint_ordinal_from_source_id(
        int32_t source_id,
        int32_t host_source_id = -1) noexcept {
    if (source_id < 0) {
        return -1;
    }
    if (host_source_id < 0) {
        return source_id < SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE
            ? source_id : -1;
    }
    const int32_t base = server_cache_plan_host_checkpoint_source_id(
        host_source_id, 0);
    if (base < 0 || source_id < base) {
        return -1;
    }
    const int32_t ordinal = source_id - base;
    return ordinal < SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE
        ? ordinal : -1;
}

constexpr int32_t server_cache_plan_checkpoint_reverse_position_from_source_id(
        size_t checkpoint_count,
        int32_t source_id,
        int32_t host_source_id = -1) noexcept {
    const int32_t ordinal = server_cache_plan_checkpoint_ordinal_from_source_id(
        source_id, host_source_id);
    return ordinal < 0 || size_t(ordinal) >= checkpoint_count
        ? -1 : int32_t(checkpoint_count - 1 - size_t(ordinal));
}

// Observer-only request-local host-state identity. The id is stored on the
// list node, so surviving nodes remain stable across save dedup/splice and an
// allocator-reused address cannot inherit a consumed source id.
bool server_cache_plan_assign_source_id(
    int32_t & instance_source_id,
    int32_t & next_source_id,
    int32_t & source_id) noexcept;

// Checkpoint containers are enumerated forward by the authority inventory but
// reverse by the shipped selector. Translate reverse visit position back to
// the forward ordinal used by both live and host-composed inventory rows.
constexpr int32_t server_cache_plan_checkpoint_source_id_from_reverse(
    size_t checkpoint_count,
    uint32_t reverse_ordinal,
    int32_t host_source_id = -1) noexcept {
    if (checkpoint_count == 0 || reverse_ordinal >= checkpoint_count) {
        return -1;
    }
    const size_t forward = checkpoint_count - 1 - reverse_ordinal;
    if (forward >= size_t(SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE)) {
        return -1;
    }
    return host_source_id >= 0
        ? server_cache_plan_host_checkpoint_source_id(
              host_source_id, int32_t(forward))
        : int32_t(forward);
}

constexpr bool server_cache_plan_viable(
        common_cache_plan_reason reason) noexcept {
    return reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
}

struct server_cache_plan_live_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    float sim = 0.0f;
    float f_keep = 0.0f;
};

server_cache_plan_live_evaluation server_cache_plan_evaluate_live(
    bool busy,
    bool has_payload,
    uint64_t lcp_tokens,
    uint64_t prompt_tokens,
    uint64_t source_tokens = 0) noexcept;

void server_cache_plan_apply_live(
    common_cache_plan_candidate * row,
    const server_cache_plan_live_evaluation & evaluation) noexcept;

struct server_cache_plan_host_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    uint64_t payload_bytes = 0;
    float sim = 0.0f;
    float f_keep = 0.0f;
};

server_cache_plan_host_evaluation server_cache_plan_evaluate_host(
    bool payload_present,
    bool identity_matches,
    uint64_t lcp_tokens,
    uint64_t prompt_tokens,
    uint64_t source_tokens,
    uint64_t payload_bytes) noexcept;

void server_cache_plan_apply_host(
    common_cache_plan_candidate * row,
    const server_cache_plan_host_evaluation & evaluation) noexcept;

struct server_cache_plan_checkpoint_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    uint64_t payload_bytes = 0;
};

server_cache_plan_checkpoint_evaluation server_cache_plan_evaluate_checkpoint(
    bool payload_present,
    bool frontier_current,
    bool recurrent,
    bool checkpoint_lineage_matches,
    int64_t pos_min,
    int64_t pos_max,
    int64_t next_position,
    int64_t min_position_threshold,
    uint64_t payload_bytes) noexcept;

void server_cache_plan_apply_checkpoint(
    common_cache_plan_candidate * row,
    const server_cache_plan_checkpoint_evaluation & evaluation) noexcept;
