#pragma once

#include "../src/llama-cache-accounting.h" // staging API precedent: fit.cpp / speculative.cpp

#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// common-cache-plan.h — prompt-cache decision record, schema version 7.
//
// §7.7 decision records + §7.5 shadow-planner inventory: the ONE closed plan-reason enum
// shared by server and tests, the orthogonal candidate disposition, the closed provider
// inventory (today's real candidates only), and the multi-stage per-request record.
// SHADOW-ONLY: the record observes the shipped selection path; `slot.cache_status` and the
// live four-tier logic remain authoritative and untouched. Everything here is inert unless
// the --cache-debug observer is enabled, and the disabled branch performs strictly zero
// observer work.
//
// In schema v2, the four per-provider summary rows were replaced by
// a bounded per-entry CANDIDATE INVENTORY — one row per candidate instance the shipped
// selectors actually visited (every evaluated live slot across the three slot loops, every
// scanned host entry, every visited checkpoint sibling), merged across selector phases by
// (provider, request-local source id). The declared candidate domain is exactly this
// shipped-visited set (observers never rescan); per-provider inventory-state markers
// record truncation (shipped short-circuit) and overflow. Cost terms move from the record
// to each row; shadow choice / tie set are planner outputs, typed-unavailable until the
// planner fills them. Candidate observation transport is noexcept by construction: fixed
// capacity in the record, append-or-mark-overflowed, no allocation in selector hooks.
// v3 embeds accounting schema v2. v4 adds the shadow yield projection: selected
// artifacts and exact union-level projected domain values, plus an explicitly
// not_observed actual-yield slot reserved for authoritative mutation. v5 adds the
// authority receipt and target-qualified candidate identity without changing
// the established meaning of `shadow_choice` (the planner counterfactual). v6 adds
// the shadow destruction quote/receipt; accounting remains schema 2. v7 names the
// host payload representation independently of the provider so fixed state and sealed
// VBR artifacts remain distinguishable throughout the control plane.

constexpr uint32_t COMMON_CACHE_PLAN_SCHEMA_VERSION = 7;

// Explicit record→embedded-accounting compatibility table. A C schema bump cannot compile
// under the current record version until this table and the record version move together.
constexpr uint32_t common_cache_plan_accounting_schema(uint32_t record_schema) {
    return (record_schema == 3 || record_schema == 4 || record_schema == 5 ||
            record_schema == 6 || record_schema == 7) ? 2 :
           (record_schema == 1 || record_schema == 2 ? 1 : 0);
}
static_assert(common_cache_plan_accounting_schema(COMMON_CACHE_PLAN_SCHEMA_VERSION) ==
              LLAMA_CACHE_ACCT_SCHEMA_VERSION);

// Bounded inventory capacity fixed in the record. No
// fixed bound can cover the unconstrained slot x host-state x checkpoint product. A failed
// append therefore latches typed saturation, immediately stops the authority scan, and
// makes planner qualification unavailable; record emission and the shipped path continue.
constexpr size_t COMMON_CACHE_PLAN_MAX_CANDIDATES = 96;

// Bounded component references for composed candidate plans (host entry + checkpoint
// continuation today). A chain row references its components by inventory ordinal.
constexpr size_t COMMON_CACHE_PLAN_MAX_COMPONENTS = 2;

// Reserved source ids (the merge key is (provider, source_id); real sources are >= 0):
// an AGGREGATE row carries a provider-level classification when no entries were scanned;
// a CHAIN row is the derived composed plan over selected component rows.
constexpr int32_t COMMON_CACHE_PLAN_SOURCE_AGGREGATE = -1;
constexpr int32_t COMMON_CACHE_PLAN_SOURCE_CHAIN     = -2;

// Fixed §7.7 precedence encoded in the VALUES: validity before economics, band order
// identity(100) → structural(200) → generation/lineage(300) → domain/tier(400) → budget(500)
// → cost(600). The first failing check in that order IS the reason; out-of-order observation
// keeps the lowest value (see note_reject). Values are stable and append-only within a band
// for this schema version.
//
// ONE X-macro list mechanically binds membership, wire value, and name-table spelling:
// adding a member is one line, and an omission anywhere is a compile failure (the name
// switch stays a real switch, so -Wswitch exhaustiveness is preserved). Each name string
// below is its ONLY spelling in the tree — CI extracts this list and bans replicas.
#define COMMON_CACHE_PLAN_REASON_LIST(X) \
    X(NONE,                          "none",                          0)   \
    /* identity */ \
    X(MODEL_IDENTITY_MISMATCH,       "model_identity_mismatch",       100) \
    X(EXECUTION_IDENTITY_MISMATCH,   "execution_identity_mismatch",   101) \
    X(ADAPTER_CONFIG_MISMATCH,       "adapter_config_mismatch",       102) \
    X(MEDIA_CONTENT_MISMATCH,        "media_content_mismatch",        103) \
    X(TOKENIZER_TEMPLATE_MISMATCH,   "tokenizer_template_mismatch",   104) \
    X(PREFIX_TOKEN_DIGEST_MISMATCH,  "prefix_token_digest_mismatch",  105) \
    /* structural */ \
    X(PROVIDER_UNAVAILABLE,          "provider_unavailable",          200) \
    X(PROVIDER_BUSY,                 "provider_busy",                 201) \
    X(FRONTIER_INVALID,              "frontier_invalid",              202) \
    X(COVERAGE_INSUFFICIENT,         "coverage_insufficient",         203) \
    X(PAYLOAD_EMPTY,                 "payload_empty",                 204) \
    X(PAYLOAD_INCOMPLETE,            "payload_incomplete",            205) \
    X(PAYLOAD_SHORT,                 "payload_short",                 206) \
    X(CHECKSUM_MISMATCH,             "checksum_mismatch",             207) \
    X(PAYLOAD_VERSION_UNSUPPORTED,   "payload_version_unsupported",   208) \
    X(COMPONENT_SHAPE_MISMATCH,      "component_shape_mismatch",      209) \
    X(ACCELERATOR_UNRESTORABLE,      "accelerator_unrestorable",      210) \
    /* generation / lineage — the A-track evaluator stays the ONE authority; its closed
       category/reason/tombstone/refinement ride as subfields, never as mirrored values */ \
    X(REPRESENTATION_EPOCH_CHANGED,  "representation_epoch_changed",  300) \
    X(SEQUENCE_EPOCH_CHANGED,        "sequence_epoch_changed",        301) \
    X(GENERATION_NOT_ELIGIBLE,       "generation_not_eligible",       302) \
    /* domain / tier */ \
    X(STORAGE_DOMAIN_MISMATCH,       "storage_domain_mismatch",       400) \
    X(REPRESENTATION_TIER_UNSUPPORTED, "representation_tier_unsupported", 401) \
    X(KV_TYPE_MISMATCH,              "kv_type_mismatch",              402) \
    X(SHARD_TOPOLOGY_MISMATCH,       "shard_topology_mismatch",       403) \
    X(RECOVERABILITY_UNSUPPORTED,    "recoverability_unsupported",    404) \
    /* budget */ \
    X(PERSISTENT_BUDGET_EXCEEDED,    "persistent_budget_exceeded",    500) \
    X(WORKSPACE_BUDGET_EXCEEDED,     "workspace_budget_exceeded",     501) \
    X(MANDATORY_ANCHOR_OVERFLOW,     "mandatory_anchor_overflow",     502) \
    /* cost — the only rejection of a VALID candidate (it lost the economics) */ \
    X(COST_NOT_MINIMAL,              "cost_not_minimal",              600)

enum common_cache_plan_reason : uint16_t {
#define COMMON_CACHE_PLAN_REASON_ENUM_MEMBER(sym, name, val) COMMON_CACHE_PLAN_REASON_##sym = val,
    COMMON_CACHE_PLAN_REASON_LIST(COMMON_CACHE_PLAN_REASON_ENUM_MEMBER)
#undef COMMON_CACHE_PLAN_REASON_ENUM_MEMBER
    // Closed-set sentinel: one past the last member.
    COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL,
};

// Every member, generated from the same list — cannot drift from the enum.
constexpr common_cache_plan_reason common_cache_plan_reason_all[] = {
#define COMMON_CACHE_PLAN_REASON_ARRAY_MEMBER(sym, name, val) COMMON_CACHE_PLAN_REASON_##sym,
    COMMON_CACHE_PLAN_REASON_LIST(COMMON_CACHE_PLAN_REASON_ARRAY_MEMBER)
#undef COMMON_CACHE_PLAN_REASON_ARRAY_MEMBER
};

constexpr size_t COMMON_CACHE_PLAN_REASON_MEMBER_COUNT =
    sizeof(common_cache_plan_reason_all) / sizeof(common_cache_plan_reason_all[0]);
static_assert(uint16_t(COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL) == 601,
              "sentinel is one past cost_not_minimal for schema version 1");

// band = value / 100
constexpr uint16_t common_cache_plan_reason_band(common_cache_plan_reason r) {
    return uint16_t(r) / 100;
}

constexpr bool common_cache_plan_reasons_monotone() {
    for (size_t i = 1; i < COMMON_CACHE_PLAN_REASON_MEMBER_COUNT; i++) {
        if (uint16_t(common_cache_plan_reason_all[i]) <=
            uint16_t(common_cache_plan_reason_all[i - 1])) {
            return false;
        }
    }
    return true;
}
static_assert(common_cache_plan_reasons_monotone(),
              "plan-reason values must be strictly increasing (bands encode precedence)");
// exact band starts are schema-stable wire values; a move is a breaking change
static_assert(COMMON_CACHE_PLAN_REASON_MODEL_IDENTITY_MISMATCH      == 100 &&
              COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE         == 200 &&
              COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED == 300 &&
              COMMON_CACHE_PLAN_REASON_STORAGE_DOMAIN_MISMATCH      == 400 &&
              COMMON_CACHE_PLAN_REASON_PERSISTENT_BUDGET_EXCEEDED   == 500 &&
              COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL             == 600,
              "band starts pinned at 100/200/300/400/500/600");

// Orthogonal candidate disposition: a valid loser is not an invalid candidate.
enum class common_cache_plan_disposition : uint8_t {
    accepted = 0,
    rejected_invalid,
    valid_not_chosen_cost,
    unavailable,
    _count,
};

// Closed provider inventory — today's real request-level candidates ONLY. `seq_cp` is a
// capability (copy_state_to's internal primitive), reported as a constant on the emitted
// record, not a candidate; device/disk/remote tiers and the parked rolling tape enter only
// when their providers exist.
enum class common_cache_plan_provider : uint8_t {
    live_slot = 0,
    live_context_checkpoint,
    host_cache_entry,
    cold_replay,
    _count,
};

// Orthogonal to provider. A host_cache_entry can be either the historical fixed state
// image or an immutable VBR artifact; live/checkpoint/cold rows carry `unavailable`.
enum class common_cache_plan_payload_kind : uint8_t {
    unavailable = 0,
    fixed_state,
    vbr_artifact,
    _count,
};

const char * common_cache_plan_payload_kind_name(
    common_cache_plan_payload_kind kind) noexcept;

constexpr bool common_cache_plan_provider_is_live(
        common_cache_plan_provider provider) noexcept {
    return provider == common_cache_plan_provider::live_slot ||
           provider == common_cache_plan_provider::live_context_checkpoint;
}

enum class common_cache_plan_outcome : uint8_t {
    unknown = 0,        // the typed not-finalized state
    restored,
    restore_failed_fell_back_cold,
    cold,
    _count,
};

// How the shipped path picked the slot before cache-plan preflight.
enum class common_cache_plan_selection : uint8_t {
    none = 0,
    by_id,
    similarity,
    route_home,
    lru,
    _count,
};

constexpr bool common_cache_plan_strict_similarity(
        double similarity,
        double threshold) noexcept {
    return threshold != 0.0 && similarity > threshold;
}

// Graduated authority domain: each decision tier can consume candidates first
// admitted by that tier or an earlier one. `none` is never a production plan
// origin and therefore fails closed.
constexpr bool common_cache_plan_origin_in_domain(
        common_cache_plan_selection origin,
        common_cache_plan_selection decision) noexcept {
    return origin != common_cache_plan_selection::none &&
           decision != common_cache_plan_selection::none &&
           uint8_t(origin) <= uint8_t(decision);
}

// Schema-v5 authority vocabulary. The configured level is graduated: each
// level includes every earlier tier. The receipt distinguishes the legacy
// counterfactual, planner result, and actually executed complete plan without
// changing schema-v4 `shadow_choice` semantics.
constexpr uint32_t COMMON_CACHE_PLAN_AUTHORITY_POLICY_VERSION = 1;

enum class common_cache_plan_authority_level : uint8_t {
    off = 0,
    by_id,
    similarity,
    route_home,
    lru,
    _count,
};

enum class common_cache_plan_authority_state : uint8_t {
    shadow = 0,
    authoritative,
    fallback_legacy,
    _count,
};

enum class common_cache_plan_authority_fallback : uint8_t {
    none = 0,
    tier_not_enabled,
    no_profile,
    profile_unfitted,
    invalid_calibration,
    incomplete_evidence,
    stale_capability,
    destruction_authority_required,
    budget_or_lease_unavailable,
    destruction_not_certified,
    internal_fault,
    _count,
};

struct common_cache_plan_authority_receipt {
    uint32_t policy_version = COMMON_CACHE_PLAN_AUTHORITY_POLICY_VERSION;
    common_cache_plan_authority_level configured_level =
        common_cache_plan_authority_level::off;
    common_cache_plan_selection legacy_tier = common_cache_plan_selection::none;
    common_cache_plan_selection decision_tier = common_cache_plan_selection::none;
    common_cache_plan_authority_state state =
        common_cache_plan_authority_state::shadow;
    int32_t legacy_plan_candidate = -1;
    int32_t planner_plan_candidate = -1;
    int32_t executed_plan_candidate = -1;
    common_cache_plan_authority_fallback fallback_reason =
        common_cache_plan_authority_fallback::none;
    bool disagreed = false;
};

// Process-local, fixed-size authority telemetry. JSON remains debug-gated; this
// POD is the bounded receipt/counter surface used by the independent
// debug-or-authority substrate. Shadow mode exercises it from the existing observer only.
struct common_cache_plan_authority_counters {
    std::array<uint64_t, size_t(common_cache_plan_selection::_count)> observed{};
    std::array<uint64_t, size_t(common_cache_plan_selection::_count)> authority_eligible{};
    std::array<uint64_t, size_t(common_cache_plan_selection::_count)> authority_executed{};
    std::array<uint64_t, size_t(common_cache_plan_selection::_count)> agree{};
    std::array<uint64_t, size_t(common_cache_plan_selection::_count)> disagree{};
    std::array<uint64_t, size_t(common_cache_plan_selection::_count)> fallback_legacy{};
    std::array<uint64_t, size_t(common_cache_plan_authority_fallback::_count)> fallback_reason{};
    common_cache_plan_authority_receipt last_receipt;
    bool has_receipt = false;

    // `qualified` is the dual-run eligibility result, independent of whether
    // authority actually executed. Eligibility is indexed by decision_tier;
    // observed/agreement remain indexed by the legacy tier.
    void observe(const common_cache_plan_authority_receipt & receipt,
                 bool qualified = false) noexcept;
};

// Cache-destruction evidence. These are wire-layer mirrors of the
// server-only lifecycle vocabulary; common/ must not depend on tools/server.
// The maintenance receipt freezes the resolved recovery-source
// citation used by certified and executed redundant-host evictions.
constexpr uint32_t COMMON_CACHE_PLAN_DESTRUCTION_POLICY_VERSION = 1;

struct common_cache_plan_yield_domain;

enum class common_cache_plan_destruction_state : uint8_t {
    not_required = 0,
    quoted,
    certified,
    executed,
    refused,
    failed,
    _count,
};

enum class common_cache_plan_destruction_reason : uint8_t {
    none = 0,
    lifecycle_disabled,
    manifest_incomplete,
    identity_unavailable,
    mandatory_anchor,
    lease_unavailable,
    hard_lease_blocked,
    accounting_unavailable,
    effect_drift,
    release_evidence_unavailable,
    recovery_unavailable,
    profile_unfitted,
    capacity_refused,
    mutation_failed,
    internal_fault,
    _count,
};

enum class common_cache_plan_destruction_effect : uint8_t {
    none = 0,
    cross_target_displacement,
    destructive_similarity_retarget,
    same_target_cold_replacement,
    // Physical host-artifact retirement. A restore uses the consumption
    // spelling; the maintenance receipt distinguishes certified redundant
    // eviction with displaced_fate=exact_duplicate and a resolved citation.
    different_host_source_consumption,
    // One independently accounted member of a live checkpoint ring.
    checkpoint_member_drop,
    _count,
};

enum class common_cache_plan_destruction_class : uint8_t {
    slot_drop = 0,
    live_range_drop,
    host_artifact_drop,
    checkpoint_drop,
    token_ledger_truncate,
    mandatory_recovery_reset,
    _count,
};

enum class common_cache_plan_destruction_physical_reason : uint8_t {
    slot_rebind = 0,
    idle_reclaim,
    prompt_trim,
    cache_capacity,
    cache_update,
    prompt_clear,
    checkpoint_replace,
    mandatory_recovery,
    _count,
};

enum class common_cache_plan_destruction_lease_verdict : uint8_t {
    unavailable = 0,
    unleased,
    soft_leased,
    hard_leased,
    mandatory_recovery,
    _count,
};

enum class common_cache_plan_displaced_fate : uint8_t {
    unavailable = 0,
    retained_live,
    retained_host,
    retained_sealed_artifact,
    exact_duplicate,
    exact_replay_recipe,
    destroyed_by_policy,
    _count,
};

enum class common_cache_plan_recovery_citation : uint8_t {
    unavailable = 0,
    resolved,
    prospective,
    _count,
};

enum class common_cache_plan_destruction_comparison : uint8_t {
    not_compared = 0,
    matched,
    differed,
    ds6_insufficient_yield,
    ds6_unsupported_required,
    ds6_unavailable,
    _count,
};

using common_cache_plan_destruction_effect_set = uint32_t;

constexpr common_cache_plan_destruction_effect_set
common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect effect) noexcept {
    return effect > common_cache_plan_destruction_effect::none &&
           effect < common_cache_plan_destruction_effect::_count
        ? common_cache_plan_destruction_effect_set(1U) << uint8_t(effect)
        : 0;
}

constexpr bool common_cache_plan_destruction_effect_has(
        common_cache_plan_destruction_effect_set effects,
        common_cache_plan_destruction_effect effect) noexcept {
    return (effects & common_cache_plan_destruction_effect_bit(effect)) != 0;
}

constexpr common_cache_plan_destruction_class
common_cache_plan_destruction_class_for_effect(
        common_cache_plan_destruction_effect effect) noexcept {
    switch (effect) {
        case common_cache_plan_destruction_effect::
                 different_host_source_consumption:
            return common_cache_plan_destruction_class::host_artifact_drop;
        case common_cache_plan_destruction_effect::checkpoint_member_drop:
            return common_cache_plan_destruction_class::checkpoint_drop;
        case common_cache_plan_destruction_effect::none:
        case common_cache_plan_destruction_effect::cross_target_displacement:
        case common_cache_plan_destruction_effect::
                 destructive_similarity_retarget:
        case common_cache_plan_destruction_effect::
                 same_target_cold_replacement:
        case common_cache_plan_destruction_effect::_count:
            return common_cache_plan_destruction_class::slot_drop;
    }
    return common_cache_plan_destruction_class::slot_drop;
}

constexpr common_cache_plan_destruction_physical_reason
common_cache_plan_destruction_physical_reason_for_effect(
        common_cache_plan_destruction_effect effect) noexcept {
    switch (effect) {
        case common_cache_plan_destruction_effect::
                 different_host_source_consumption:
            return common_cache_plan_destruction_physical_reason::cache_update;
        case common_cache_plan_destruction_effect::checkpoint_member_drop:
            return common_cache_plan_destruction_physical_reason::
                checkpoint_replace;
        case common_cache_plan_destruction_effect::none:
        case common_cache_plan_destruction_effect::cross_target_displacement:
        case common_cache_plan_destruction_effect::
                 destructive_similarity_retarget:
        case common_cache_plan_destruction_effect::
                 same_target_cold_replacement:
        case common_cache_plan_destruction_effect::_count:
            return common_cache_plan_destruction_physical_reason::slot_rebind;
    }
    return common_cache_plan_destruction_physical_reason::slot_rebind;
}

struct common_cache_plan_destruction_manifest_digest_tag;
struct common_cache_plan_destruction_effect_digest_tag;
struct common_cache_plan_destruction_recovery_digest_tag;
using common_cache_plan_destruction_manifest_digest =
    llama_cache_acct_digest<common_cache_plan_destruction_manifest_digest_tag>;
using common_cache_plan_destruction_effect_digest =
    llama_cache_acct_digest<common_cache_plan_destruction_effect_digest_tag>;
using common_cache_plan_destruction_recovery_digest =
    llama_cache_acct_digest<common_cache_plan_destruction_recovery_digest_tag>;

struct common_cache_plan_destruction_receipt {
    uint32_t policy_version = COMMON_CACHE_PLAN_DESTRUCTION_POLICY_VERSION;
    common_cache_plan_destruction_state state =
        common_cache_plan_destruction_state::not_required;
    common_cache_plan_destruction_reason reason =
        common_cache_plan_destruction_reason::none;
    // A candidate may displace a live target and consume a distinct host
    // artifact in the same execution. Every bit needs its own certificate;
    // manifest/effect digests bind the merged physical union exactly once.
    common_cache_plan_destruction_effect_set effects = 0;
    common_cache_plan_destruction_lease_verdict lease_verdict =
        common_cache_plan_destruction_lease_verdict::unavailable;
    common_cache_plan_displaced_fate displaced_fate =
        common_cache_plan_displaced_fate::unavailable;
    common_cache_plan_recovery_citation recovery_citation =
        common_cache_plan_recovery_citation::unavailable;
    common_cache_plan_destruction_comparison post_finalize_comparison =
        common_cache_plan_destruction_comparison::not_compared;
    int32_t plan_candidate = -1;
    uint64_t admission_sequence = 0;
    uint64_t quote_duration_us = 0;
    // Evidence serial sampled with the quote. The projected yield record stays
    // joined to the final record's accounting serial instead of borrowing this.
    uint64_t quote_accounting_serial = 0;
    uint64_t actual_accounting_serial = 0;
    common_cache_plan_destruction_manifest_digest manifest_digest;
    common_cache_plan_destruction_effect_digest union_effect_digest;
    std::vector<llama_cache_acct_artifact_id> selected_attention;
    std::vector<llama_cache_acct_artifact_id> selected_recurrent;
    // Schema-v6 recovery-source detail retained by schema v7. A resolved citation is legal without
    // these fields: recovery can resolve a logical/durable source that
    // has no artifact pin. When it resolves a concrete protected source,
    // the artifact plus a tagged digest over its canonical op set identifies
    // that survivor without leaking process-local C handles onto the wire.
    // Prospective and unavailable citations also leave both fields unavailable.
    llama_cache_acct_artifact_id recovery_source_artifact_id;
    common_cache_plan_destruction_recovery_digest recovery_source_manifest_digest;
    // Representation owned by the selected host candidate. Live/checkpoint-only
    // receipts retain `unavailable`, which serializes as JSON null.
    common_cache_plan_payload_kind payload_kind =
        common_cache_plan_payload_kind::unavailable;
};

// Process-local quote pre-image. Domain bytes are projected into the existing
// schema-v4 yield table only for the selected destructive plan, so schema 7
// does not grow a second predicted/actual byte vocabulary.
struct common_cache_plan_destruction_quote {
    common_cache_plan_destruction_receipt receipt;
    std::vector<common_cache_plan_yield_domain> projected_domains;
};

struct common_cache_plan_destruction_counters {
    static constexpr size_t n_tiers = size_t(common_cache_plan_selection::_count);
    static constexpr size_t n_classes = size_t(common_cache_plan_destruction_class::_count);
    static constexpr size_t n_reasons = size_t(common_cache_plan_destruction_reason::_count);
    static constexpr size_t n_verdicts = size_t(common_cache_plan_destruction_lease_verdict::_count);
    static constexpr size_t n_fates = size_t(common_cache_plan_displaced_fate::_count);
    std::array<std::array<uint64_t, n_classes>, n_tiers> quoted{};
    std::array<std::array<uint64_t, n_classes>, n_tiers> certified{};
    std::array<std::array<uint64_t, n_classes>, n_tiers> executed{};
    std::array<std::array<uint64_t, n_reasons>, n_tiers> refused{};
    std::array<std::array<uint64_t, n_verdicts>, n_tiers> lease_verdict{};
    std::array<std::array<uint64_t, n_classes>, n_tiers> actual_yield_unavailable{};
    std::array<std::array<uint64_t, n_fates>, n_tiers> recovery_outcome{};
    uint64_t quote_memo_hits = 0;
    uint64_t quote_memo_misses = 0;
    uint64_t quote_samples = 0;
    uint64_t quote_duration_us_total = 0;
    uint64_t quote_duration_us_max = 0;
    common_cache_plan_destruction_receipt last_receipt;
    bool has_receipt = false;

    void observe(common_cache_plan_selection tier,
                 const common_cache_plan_destruction_receipt & receipt,
                 bool observe_classification = true) noexcept;
};

// Which authoritative shipped scan observed a candidate (bitmask on the row). A physical
// candidate visited by several phases keeps ONE row (merge key = provider + source id);
// each phase ORs its bit and adds only the scalars that phase computed.
enum common_cache_plan_phase : uint8_t {
    COMMON_CACHE_PLAN_PHASE_BY_ID      = 1 << 0,
    COMMON_CACHE_PLAN_PHASE_SIMILARITY = 1 << 1,
    COMMON_CACHE_PLAN_PHASE_ROUTE_HOME = 1 << 2,
    COMMON_CACHE_PLAN_PHASE_LRU        = 1 << 3,
    COMMON_CACHE_PLAN_PHASE_HOST_SCAN  = 1 << 4,
    COMMON_CACHE_PLAN_PHASE_CKPT_SCAN  = 1 << 5,
    COMMON_CACHE_PLAN_PHASE_CHAIN      = 1 << 6,   // derived composed-plan row
};

constexpr common_cache_plan_selection common_cache_plan_origin_for_phase(uint8_t phase) noexcept {
    return (phase & COMMON_CACHE_PLAN_PHASE_BY_ID)      ? common_cache_plan_selection::by_id :
           (phase & COMMON_CACHE_PLAN_PHASE_SIMILARITY) ? common_cache_plan_selection::similarity :
           (phase & COMMON_CACHE_PLAN_PHASE_ROUTE_HOME) ? common_cache_plan_selection::route_home :
           (phase & COMMON_CACHE_PLAN_PHASE_LRU)        ? common_cache_plan_selection::lru :
                                                          common_cache_plan_selection::none;
}

// Per-provider observed-inventory completeness over the DECLARED domain (= the shipped-
// visited set). `truncated_by_shipped_short_circuit` marks scans the shipped path cut off
// (checkpoint reverse find_if): entries beyond it are outside the domain, and such a record
// is scoped evidence only, never full-inventory absorption evidence.
// `overflowed` = the fixed inventory filled; shadow choice is then unavailable, never an
// optimum over a partial set.
enum class common_cache_plan_inventory_state : uint8_t {
    unobserved = 0,
    complete,
    truncated_by_shipped_short_circuit,
    overflowed,
    _count,
};

// Closed planner-attempt status: every finalized record says exactly
// what the planner did, and an ordinary refusal is countable without conflating "no fitted
// profile exists" with an internal fault.
enum class common_cache_plan_planner_status : uint8_t {
    not_attempted = 0,      // record finalized before the planner stage (should not emit)
    ok,                     // matched profile, complete evidence, shadow choice computed
    no_profile,             // the server composed no calibration profile
    profile_unfitted,       // profile composed but no fitted table entry exists
    invalid_calibration,    // profile mismatch, unreviewed version, or non-finite/negative coefficients
    incomplete_evidence,    // overflow / unresolved candidate / missing scalars — never a partial optimum
    internal_fault,         // exception inside the planner boundary
    _count,
};

// Schema-v4 yield projection. The selected artifact rows and projected domain
// values describe the shadow planner's selected UNION. They are never measured yield:
// Shadow projection does not execute an eviction, so the actual side stays explicitly not_observed
// until an authoritative post-mutation measurement is available.
enum class common_cache_plan_yield_status : uint8_t {
    fits = 0,
    insufficient_yield,
    unsupported_required,
    unavailable,
    _count,
};

enum class common_cache_plan_yield_plan_state : uint8_t {
    not_required = 0,
    planned,
    unavailable,
    _count,
};

enum class common_cache_plan_yield_actual_state : uint8_t {
    not_observed = 0,
    measured,
    unavailable,
    _count,
};

struct common_cache_plan_yield_domain {
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_value current_resident_bytes;
    llama_cache_acct_value fit_before_bytes;
    llama_cache_acct_value projected_release_bytes;
    llama_cache_acct_value projected_reserve_bytes;
    llama_cache_acct_value projected_after_bytes;
};

struct common_cache_plan_actual_yield_domain {
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_value before_bytes;
    llama_cache_acct_value released_bytes;
    llama_cache_acct_value after_bytes;
};

bool common_cache_plan_projected_release_bytes(
    const std::vector<common_cache_plan_yield_domain> & domains,
    uint64_t & total) noexcept;

struct common_cache_plan_yield_record {
    common_cache_plan_yield_status status =
        common_cache_plan_yield_status::unavailable;
    common_cache_plan_yield_plan_state plan_state =
        common_cache_plan_yield_plan_state::unavailable;
    common_cache_plan_yield_actual_state actual_state =
        common_cache_plan_yield_actual_state::not_observed;
    uint32_t yield_policy_version = 0;
    uint64_t accounting_serial = 0;
    std::vector<llama_cache_acct_artifact_id> selected_attention;
    std::vector<llama_cache_acct_artifact_id> selected_recurrent;
    std::vector<llama_cache_acct_artifact_id> unsupported;
    std::vector<common_cache_plan_yield_domain> projected_domains;
    std::vector<common_cache_plan_actual_yield_domain> actual_domains;
};

void common_cache_plan_fill_actual_yield(
    common_cache_plan_yield_record & yield,
    const std::vector<common_cache_plan_yield_domain> & projected,
    const std::vector<common_cache_plan_yield_domain> & observed_after) noexcept;

// Opaque identity evidence (§7.7 redaction: digests of already-computed keys/strings, never
// raw values). An identity the server has not computed stays typed unknown — never a
// fabricated digest.
struct common_cache_plan_identity_evidence {
    llama_cache_acct_value model_digest;
    llama_cache_acct_value execution_digest;
    llama_cache_acct_value adapter_config_digest;
    llama_cache_acct_value media_content_digest;
    llama_cache_acct_value tokenizer_template_digest;
    llama_cache_acct_value prefix_token_digest;
};

// §7.5 cost-term slots: one per kind with its canonical raw unit, unavailable until an
// estimator fills them (a default array would collapse to five "restore" slots).
constexpr std::array<llama_cache_acct_cost_term, size_t(llama_cache_acct_cost_kind::_count)>
common_cache_plan_default_cost_terms() {
    std::array<llama_cache_acct_cost_term,
               size_t(llama_cache_acct_cost_kind::_count)> terms{};
    for (size_t i = 0; i < terms.size(); i++) {
        terms[i].kind     = llama_cache_acct_cost_kind(i);
        terms[i].raw_unit = llama_cache_acct_cost_kind_unit(terms[i].kind);
    }
    return terms;
}

// One candidate-plan row: a candidate instance the shipped path actually visited (or a
// derived chain over such instances). Membership in the inventory IS presence — no row, no
// observation, never a vacuous verdict. `delivered` = this candidate actually applied state
// to the slot — recorded as data at the delivery site, never inferred.
// Trivially copyable and written only through noexcept record methods.
struct common_cache_plan_candidate {
    common_cache_plan_provider provider = common_cache_plan_provider::cold_replay;
    common_cache_plan_payload_kind payload_kind =
        common_cache_plan_payload_kind::unavailable;
    // Schema-v5 executable-plan identity. The target is part of the merge key: the same
    // provider/source offered to two physical slots is two distinct plans. origin_tier is
    // the legacy/planner tier that introduced the plan, independent of the record's
    // eventually executed `selection`.
    int32_t target_slot_id = -1;
    common_cache_plan_selection origin_tier = common_cache_plan_selection::none;
    // request-local source identity: live slot id, host-entry scan ordinal, checkpoint scan
    // ordinal; -1 for cold_replay and derived rows
    int32_t source_id   = -1;
    uint8_t phases_seen = 0;    // OR of common_cache_plan_phase bits

    bool delivered = false;

    // Root feasibility: a row whose state was only reachable through
    // a delivered base component (e.g. a checkpoint exposed by a host restore) is EVIDENCE
    // and a chain component, but not a standalone plan — it never enters the root optimum.
    bool component_only = false;

    // Process-local composition identity for a host-dependent checkpoint. This is the
    // source_id of the exact host entry whose restore exposed the checkpoint; -1 for
    // standalone rows. It is deliberately not serialized: the wire identity remains the
    // stable component ordinals of the resulting chain.
    int32_t dependent_host_source_id = -1;

    common_cache_plan_disposition disposition = common_cache_plan_disposition::unavailable;
    common_cache_plan_reason      reason      = COMMON_CACHE_PLAN_REASON_NONE;

    // Diagnostics transported from the shipped path's own computation (never re-derived;
    // a scalar the visiting loop did not compute stays typed unknown)
    llama_cache_acct_value lcp_tokens;
    llama_cache_acct_value payload_bytes;
    llama_cache_acct_value t_last_used_us;                  // LRU loop recency
    double sim        = 0; bool sim_known          = false;
    double f_keep     = 0; bool f_keep_known       = false;
    bool spec_capable = false; bool spec_capable_known = false;
    // live_context_checkpoint: rows the short-circuiting shipped scan actually VISITED
    // (not the container size) + how many it rejected for a changed representation epoch
    uint32_t siblings_rejected_epoch = 0;
    uint32_t siblings_scanned        = 0;

    // composed plan: ordinals of this chain's component rows (-1 = unused slot); only rows
    // with the CHAIN phase bit use these
    std::array<int32_t, COMMON_CACHE_PLAN_MAX_COMPONENTS> component_ids = {-1, -1};

    // per-candidate §7.5 economics: filled by the B estimator inside the planner boundary;
    // typed-unavailable in transport
    std::array<llama_cache_acct_cost_term,
               size_t(llama_cache_acct_cost_kind::_count)> cost_terms =
        common_cache_plan_default_cost_terms();
    llama_cache_acct_value predicted_total_us;

    bool is_chain() const noexcept { return (phases_seen & COMMON_CACHE_PLAN_PHASE_CHAIN) != 0; }

    bool viable() const noexcept {
        return disposition == common_cache_plan_disposition::accepted ||
               disposition == common_cache_plan_disposition::valid_not_chosen_cost;
    }

    // the shipped winner's promotion over the scan-time cost-loser default — one invariant
    // pair, so no site can reset the disposition but forget the reason
    void accept() noexcept {
        disposition = common_cache_plan_disposition::accepted;
        reason      = COMMON_CACHE_PLAN_REASON_NONE;
    }

    // First-failing-check discipline for out-of-order observation: the lowest value (earliest
    // precedence band) is THE reason; later/higher failures do not overwrite it.
    void note_reject(common_cache_plan_reason r) noexcept {
        if (r == COMMON_CACHE_PLAN_REASON_NONE) {
            return;
        }
        if (reason == COMMON_CACHE_PLAN_REASON_NONE || uint16_t(r) < uint16_t(reason)) {
            reason = r;
        }
        // disposition follows the retained (earliest-band) reason, not the arrival order
        disposition = (reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL)
            ? common_cache_plan_disposition::valid_not_chosen_cost
            : common_cache_plan_disposition::rejected_invalid;
    }
};

// Multi-stage per-request record. The shipped path selects and mutates across three stages
// (slot routing → host-cache load → context-checkpoint selection); candidate rows
// accumulate at each stage and the record is finalized exactly once (outcome flips off
// `unknown`), after the actual restore/cold path and measured TTFT are known. Fields no
// shipped computation produced remain typed unknown/unavailable. Yield data is
// populated later at finalize inside its own observer-only boundary.
//
// Transport contract: this base record exists independently
// of planner outputs; the inventory is FIXED-CAPACITY in the record, so every selector hook
// is allocation-free and noexcept — append either succeeds into reserved storage or latches
// the provider's inventory state `overflowed` without touching the shipped loop. The
// planner (estimation, composed-plan construction, tie set, shadow choice) runs later,
// inside its own boundary in finalize, and its failure clears planner outputs only: the base
// record is always emitted.
struct common_cache_plan_record {
    uint32_t schema_version = COMMON_CACHE_PLAN_SCHEMA_VERSION;

    int64_t id_task = -1;   // request/decision id
    int32_t id_slot = -1;

    common_cache_plan_selection selection = common_cache_plan_selection::none;
    double sim_best_any = 0; bool sim_best_any_known = false;

    common_cache_plan_identity_evidence identity;

    // Stable calibration-profile id ({model class}/{hardware class}/{batch regime});
    // empty = no matching profile (typed unknown on the wire) — estimators then refuse.
    // Set once at record creation (inside the creation try), never from selector hooks.
    std::string calibration_profile;

    // ---- candidate inventory (declared domain = shipped-visited set) ----
    std::array<common_cache_plan_candidate, COMMON_CACHE_PLAN_MAX_CANDIDATES> inventory{};
    uint32_t n_inventory = 0;
    std::array<common_cache_plan_inventory_state,
               size_t(common_cache_plan_provider::_count)> inventory_states{};   // unobserved

    // the shipped path's selected row per provider (inventory ordinal, -1 = none) — delivery
    // marking, revocation, and the delivered chain operate on selected rows
    std::array<int32_t, size_t(common_cache_plan_provider::_count)> selected = {-1, -1, -1, -1};

    common_cache_plan_provider chosen  = common_cache_plan_provider::cold_replay;
    common_cache_plan_outcome  outcome = common_cache_plan_outcome::unknown; // != unknown ⇔ finalized

    // the complete shipped plan as a candidate ordinal: the chain row
    // when the delivery was composed, else the terminal provider's selected row. `chosen`
    // stays the outcome summary; offline agreement runs against THIS ordinal.
    int32_t shipped_plan_candidate = -1;

    // a derived plan (chain) could not be recorded at capacity: the
    // plan set is incomplete even though every provider inventory looks intact — the
    // planner must refuse.
    bool derived_plans_incomplete = false;

    bool inventory_saturated() const noexcept {
        if (derived_plans_incomplete) {
            return true;
        }
        for (const auto state : inventory_states) {
            if (state == common_cache_plan_inventory_state::overflowed) {
                return true;
            }
        }
        return false;
    }

    // Closed planner-attempt outcome, set at finalize.
    common_cache_plan_planner_status planner_status = common_cache_plan_planner_status::not_attempted;

    // Schema-v5 three-plan authority receipt. In shadow mode it is always off:
    // legacy/executed name the shipped plan and planner names `shadow_choice`.
    // Planner evidence is a pre-mutation counterfactual. Agreement stats
    // therefore include known structural noise: save-before-load can introduce
    // a fresh host entry absent from the planner inventory; the legacy
    // counterfactual does not model the seam's pos_min/SWA/recurrent coverage
    // recovery and can over-claim checkpoint reuse. In particular, live-context
    // replay cannot beat a checkpoint that legacy would restore: below the
    // coverage threshold legacy replays, while at/above it replay is invalid;
    // replay-wins differences between those rows are receipt-only noise. The
    // counterfactual can also over-claim
    // live replay when an adapter rebind makes the shipped path cold; host-composed
    // checkpoints are evaluated optimistically until the post-restore frontier
    // exists; and a flipped frontier ratchet can select by logical-next-position
    // while the inventory still records legacy physical coverage. Ratchet gates
    // must separate these classes rather than treating every disagreement as an
    // economic-policy miss. Route-home cross-target disagreements are also
    // receipt-only before destruction authority: dynamic VBR has no host cache to preserve the
    // displaced target, and schema 5 carries no priced retention loss. A
    // BOS-only apparent home therefore stays legacy rather than being treated
    // as a free retarget. Shared-system-prefix fleets add a second safe noise
    // class: candidates inside the planner tie floor use its stable slot-key
    // ordering, which can differ from legacy strict-max-LCP selection. The
    // resulting cross-target plan is refused as destruction_authority_required;
    // ratchet reads must separate that churn from economic disagreements.
    // At LRU, budget_or_lease_unavailable also names missing certified eviction
    // evidence for cross-target or cold-replacement shapes: schema 5 has no
    // eviction_evidence_unavailable spelling, and expanding the frozen fallback
    // vocabulary would itself be a wire change. Consuming a different retained
    // host source remains destruction_authority_required at every tier.
    // Schema 5 has no separate execution_failed fallback;
    // a genuine post-authorization restore failure is recorded as internal_fault
    // rather than silently extending the frozen vocabulary.
    common_cache_plan_authority_receipt authority;

    // Schema-v6 destruction receipt. `destruction_quotes` is process-local staging:
    // every destructive candidate is quoted before minimization, then the
    // winning candidate's quote is projected into `destruction` for the wire.
    common_cache_plan_destruction_receipt destruction;
    std::vector<common_cache_plan_destruction_quote> destruction_quotes;
    // Process-local selection join. Quotes exist only for destructive rows;
    // this preserves the exact legacy reference needed to distinguish a
    // non-destructive selected row from a missing destructive quote.
    int32_t destruction_legacy_plan_candidate = -1;

    // Process-local planner staging state; deliberately not serialized. A
    // precomputed planner result survives legacy mutation/finalization, while
    // the receipt remains the schema-v5 wire surface.
    bool planner_precomputed = false;
    bool authority_prequalified = false;
    bool authority_inventory_complete = false;

    // measured actuals (never estimates)
    llama_cache_acct_value n_prompt_tokens;
    llama_cache_acct_value n_reused_tokens;
    llama_cache_acct_value n_replayed_tokens;
    llama_cache_acct_value ttft_us;

    // a restore was attempted and failed (drives restore_failed_fell_back_cold at finalize
    // when nothing else delivered)
    bool restore_attempt_failed = false;

    // ---- planner outputs (B chooser; unavailable until it runs, cleared on planner fault) ----
    int32_t  shadow_choice = -1;                       // inventory ordinal; -1 = unavailable
    std::array<int32_t, COMMON_CACHE_PLAN_MAX_CANDIDATES> shadow_tie_set = {};   // valid [0, n_shadow_ties)
    uint32_t n_shadow_ties = 0;

    // Yield is a separate observer projection, not a planner output.
    common_cache_plan_yield_record yield;

    // Accounting snapshot, meaningful once outcome != unknown.
    llama_cache_acct_snapshot acct;

    // Find the row for (target_slot_id, provider, source_id) or append one — the cross-phase merge point:
    // one row per physical candidate, each visiting phase ORs its bit and adds its scalars.
    // noexcept by construction: fixed storage, linear scan over n_inventory (O(visited)).
    // nullptr = capacity exhausted; the provider's inventory latches `overflowed`, planner
    // completeness dies, the caller (a shipped-path hook) just skips.
    common_cache_plan_candidate * find_or_add(common_cache_plan_provider provider,
                                              int32_t source_id, uint8_t phase_bit,
                                              int32_t target_slot_id = -1,
                                              common_cache_plan_selection origin_tier =
                                                  common_cache_plan_selection::none) noexcept {
        for (uint32_t i = 0; i < n_inventory; i++) {
            if (inventory[i].target_slot_id == target_slot_id &&
                inventory[i].provider == provider && inventory[i].source_id == source_id) {
                inventory[i].phases_seen |= phase_bit;
                // first-writer-wins: the tier that INTRODUCED the row keeps
                // attribution; later phases only add phase bits
                if (origin_tier != common_cache_plan_selection::none &&
                    inventory[i].origin_tier == common_cache_plan_selection::none) {
                    inventory[i].origin_tier = origin_tier;
                }
                return &inventory[i];
            }
        }
        if (n_inventory >= COMMON_CACHE_PLAN_MAX_CANDIDATES) {
            inventory_states[size_t(provider)] = common_cache_plan_inventory_state::overflowed;
            return nullptr;
        }
        auto & c = inventory[n_inventory++];
        c.target_slot_id = target_slot_id;
        c.origin_tier    = origin_tier;
        c.provider       = provider;
        c.source_id      = source_id;
        c.phases_seen    = phase_bit;
        // first observation of this provider upgrades unobserved → complete; truncation and
        // overflow are latched explicitly by the hooks that detect them and never downgrade
        auto & st = inventory_states[size_t(provider)];
        if (st == common_cache_plan_inventory_state::unobserved) {
            st = common_cache_plan_inventory_state::complete;
        }
        return &c;
    }

    // provider-level inventory verdicts from the hooks that KNOW the scan's shape. A
    // completed scan with zero rows is still an observation (complete, empty domain);
    // truncation records a shipped short-circuit; overflow (set by find_or_add) never
    // downgrades. noexcept.
    void note_inventory_complete(common_cache_plan_provider provider) noexcept {
        auto & st = inventory_states[size_t(provider)];
        if (st == common_cache_plan_inventory_state::unobserved) {
            st = common_cache_plan_inventory_state::complete;
        }
    }
    void note_inventory_truncated(common_cache_plan_provider provider) noexcept {
        auto & st = inventory_states[size_t(provider)];
        if (st != common_cache_plan_inventory_state::overflowed) {
            st = common_cache_plan_inventory_state::truncated_by_shipped_short_circuit;
        }
    }

    // the shipped path's selected candidate for a provider (nullptr = none selected)
    common_cache_plan_candidate * selected_row(common_cache_plan_provider provider) noexcept {
        const int32_t i = selected[size_t(provider)];
        return i >= 0 && uint32_t(i) < n_inventory ? &inventory[size_t(i)] : nullptr;
    }

    // Append a derived CHAIN row (composed plan over component ordinals). Distinct from
    // find_or_add: a failed append is silently skipped WITHOUT latching any provider's
    // inventory state — a derived-row capacity miss must never poison a real provider's
    // completeness (that would make the estimator refuse the whole record). noexcept.
    common_cache_plan_candidate * add_chain(common_cache_plan_provider base_provider,
                                            int32_t comp0, int32_t comp1) noexcept {
        if (n_inventory >= COMMON_CACHE_PLAN_MAX_CANDIDATES) {
            derived_plans_incomplete = true; // Incomplete plan set; the planner refuses it.
            return nullptr;
        }
        auto & c = inventory[n_inventory++];
        c.provider          = base_provider;
        c.source_id         = COMMON_CACHE_PLAN_SOURCE_CHAIN;
        c.phases_seen       = COMMON_CACHE_PLAN_PHASE_CHAIN;
        c.component_ids[0]  = comp0;
        c.component_ids[1]  = comp1;
        if (comp0 >= 0 && uint32_t(comp0) < n_inventory - 1) {
            c.target_slot_id = inventory[size_t(comp0)].target_slot_id;
            c.origin_tier = inventory[size_t(comp0)].origin_tier;
            c.payload_kind = inventory[size_t(comp0)].payload_kind;
        }
        if (comp1 >= 0 && uint32_t(comp1) < n_inventory - 1) {
            const auto & rhs = inventory[size_t(comp1)];
            if (c.target_slot_id != rhs.target_slot_id) {
                c.target_slot_id = -1;
            }
            if (c.origin_tier != rhs.origin_tier) {
                c.origin_tier = common_cache_plan_selection::none;
            }
        }
        return &c;
    }

    const common_cache_plan_candidate * find_chain(
            common_cache_plan_provider base_provider,
            int32_t comp0,
            int32_t comp1) const noexcept {
        for (uint32_t i = 0; i < n_inventory; ++i) {
            const auto & candidate = inventory[i];
            if (!candidate.is_chain() || candidate.provider != base_provider ||
                candidate.component_ids[0] != comp0 ||
                candidate.component_ids[1] != comp1) {
                continue;
            }
            bool exact = true;
            for (size_t j = 2; j < candidate.component_ids.size(); ++j) {
                exact = exact && candidate.component_ids[j] == -1;
            }
            if (exact) {
                return &candidate;
            }
        }
        return nullptr;
    }

    common_cache_plan_candidate * find_chain(
            common_cache_plan_provider base_provider,
            int32_t comp0,
            int32_t comp1) noexcept {
        return const_cast<common_cache_plan_candidate *>(
            static_cast<const common_cache_plan_record &>(*this).find_chain(
                base_provider, comp0, comp1));
    }
    void select(common_cache_plan_provider provider, const common_cache_plan_candidate * c) noexcept {
        selected[size_t(provider)] = c ? int32_t(c - inventory.data()) : -1;
    }

    // A fallback that invalidates already-installed provider state revokes its deliveries:
    // failure/fallback dominates historical delivery — cold is a FINAL-STATE fact, and a
    // restore whose bytes were later discarded did not deliver. noexcept.
    void revoke_deliveries() noexcept {
        for (uint32_t i = 0; i < n_inventory; i++) {
            if (inventory[i].provider != common_cache_plan_provider::cold_replay) {
                inventory[i].delivered = false;
            }
        }
    }

    // Planner-fault cleanup: clear every planner output and leave the observed evidence intact.
    void clear_planner_outputs() noexcept {
        shadow_choice = -1;
        n_shadow_ties = 0;
        for (uint32_t i = 0; i < n_inventory; i++) {
            inventory[i].cost_terms         = common_cache_plan_default_cost_terms();
            inventory[i].predicted_total_us = {};
        }
    }
};

// Exhaustive name tables (presentation layer; switch-based so -Wswitch enforces coverage).
// The accounting enums are named here too: src/llama-cache-accounting.h stays policy- and
// string-free, and these are the only spellings — CI bans replicas.
const char * common_cache_plan_reason_name(common_cache_plan_reason r);
const char * common_cache_plan_disposition_name(common_cache_plan_disposition d);
const char * common_cache_plan_provider_name(common_cache_plan_provider p);
const char * common_cache_plan_outcome_name(common_cache_plan_outcome o);
const char * common_cache_plan_selection_name(common_cache_plan_selection s);
const char * common_cache_plan_authority_level_name(common_cache_plan_authority_level level);
common_cache_plan_authority_level common_cache_plan_authority_level_parse(
    const std::string & value);
const char * common_cache_plan_authority_state_name(common_cache_plan_authority_state state);
const char * common_cache_plan_authority_fallback_name(common_cache_plan_authority_fallback reason);
const char * common_cache_plan_destruction_state_name(common_cache_plan_destruction_state state);
const char * common_cache_plan_destruction_reason_name(common_cache_plan_destruction_reason reason);
const char * common_cache_plan_destruction_effect_name(common_cache_plan_destruction_effect effect);
const char * common_cache_plan_destruction_class_name(common_cache_plan_destruction_class action);
const char * common_cache_plan_destruction_physical_reason_name(
    common_cache_plan_destruction_physical_reason reason);
const char * common_cache_plan_destruction_lease_verdict_name(
    common_cache_plan_destruction_lease_verdict verdict);
const char * common_cache_plan_displaced_fate_name(common_cache_plan_displaced_fate fate);
const char * common_cache_plan_recovery_citation_name(common_cache_plan_recovery_citation citation);
const char * common_cache_plan_destruction_comparison_name(
    common_cache_plan_destruction_comparison comparison);
// Populate the shadow/off receipt after the planner attempt. This is observation
// only: it does not alter selection, delivery, shadow choice, or any shipped state.
void common_cache_plan_finalize_shadow_authority(common_cache_plan_record & rec) noexcept;
void common_cache_plan_derive_shadow_authority(
    common_cache_plan_record & rec,
    common_cache_plan_authority_level configured_level,
    common_cache_plan_authority_fallback fallback_reason) noexcept;
// Finalize-time chain composition (one testable implementation; the server calls this
// after `chosen`/`selected`/deliveries settle). Sets shipped_plan_candidate to the
// complete shipped plan: selected[chosen], upgraded to the delivered host→checkpoint
// chain on composed deliveries. When the host entry delivered, EVERY checkpoint sibling
// becomes component-only (the scanned list arrived with the host restore) and each valid
// sibling gains its true complete plan as a cost-loser chain; a chain dropped at capacity
// latches derived_plans_incomplete, and a composed delivery whose own chain could not be
// recorded reports NO shipped-plan ordinal (-1) rather than the infeasible bare row.
void common_cache_plan_compose_chains(common_cache_plan_record & rec);

const char * common_cache_plan_inventory_state_name(common_cache_plan_inventory_state s);
const char * common_cache_plan_planner_status_name(common_cache_plan_planner_status s);
const char * common_cache_plan_yield_status_name(common_cache_plan_yield_status s);
const char * common_cache_plan_yield_plan_state_name(common_cache_plan_yield_plan_state s);
const char * common_cache_plan_yield_actual_state_name(common_cache_plan_yield_actual_state s);

const char * common_cache_acct_category_name(llama_cache_acct_category c);
const char * common_cache_acct_residency_name(llama_cache_acct_residency r);
const char * common_cache_acct_domain_kind_name(llama_cache_acct_domain_kind k);
const char * common_cache_acct_producer_name(llama_cache_acct_producer p);
const char * common_cache_acct_measure_name(llama_cache_acct_measure m);
const char * common_cache_acct_known_name(llama_cache_acct_known k);
const char * common_cache_acct_unit_name(llama_cache_acct_unit u);
const char * common_cache_acct_cost_kind_name(llama_cache_acct_cost_kind k);

// One typed-known-value JSON shape shared by CACHE_PLAN and process-local
// observer siblings such as CACHE_BUDGET.
nlohmann::ordered_json common_cache_plan_value_json(
        const llama_cache_acct_value & value);

// One JSON shape for both cache-plan surfaces: the --cache-debug log and /slots.cache_plan.
// Identities stay opaque by construction: no prompt bytes, no raw adapter/media identities —
// only closed-enum names, counts, and sizes. Only present rows and non-unknown accounting
// cells are emitted.
nlohmann::ordered_json common_cache_plan_record_json(const common_cache_plan_record & rec);
