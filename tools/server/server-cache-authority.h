#pragma once

#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"
#include "server-cache-yield.h"
#include "server-retention-sidecar.h"
#include "../../common/common-cache-plan.h"
#include "../../common/common-cache-plan-estimate.h"
#include "../../common/common-cache-family.h"
#include "../../src/llama-cache-authority.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <list>
#include <vector>

struct server_prompt_cache_state;
struct common_prompt_checkpoint;
struct server_cache_authority;

struct server_cache_live_checkpoint_admission {
    llama_cache_acct_artifact_id artifact;
    const common_prompt_checkpoint * checkpoint = nullptr;
    std::vector<llama_cache_acct_op_id> committed;
};

// Process-local retention-policy seams. The weight callback is deliberately a
// dimensionless fixed-point multiplier: fitted restore cost remains the
// economic base, while policy can replace the provisional
// automatic weight without changing the victim ladder. A weight callback that
// returns false, or returns true with weight_milli == 0, refuses pricing for
// that victim (fail-closed); zero never means free-to-evict. The recovery callback
// is invoked only for an already-authorized durable source; it may never
// enumerate or widen tenant authorization.
constexpr uint32_t SERVER_CACHE_HOST_WEIGHT_SCALE = 1000;
constexpr uint32_t SERVER_CACHE_HOST_SOFT_LEASE_WEIGHT = 2000;
constexpr uint32_t SERVER_CACHE_HOST_MAIN_FAMILY_WEIGHT = 2000;

inline bool server_cache_multiply_retention_weight(
        uint32_t & weight_milli,
        uint32_t factor_milli) noexcept {
    if (factor_milli == 0) {
        return false;
    }
    const uint64_t weighted =
        (uint64_t(weight_milli) * factor_milli +
         SERVER_CACHE_HOST_WEIGHT_SCALE - 1) /
        SERVER_CACHE_HOST_WEIGHT_SCALE;
    if (weighted > UINT32_MAX) {
        return false;
    }
    weight_milli = uint32_t(weighted);
    return true;
}

bool server_cache_weighted_price_us(
    long double base_us,
    uint32_t weight_milli,
    uint64_t & out) noexcept;

bool server_cache_retention_weight_milli(
    bool soft_leased,
    bool main_family,
    uint32_t additional_weight_milli,
    uint32_t & weight_milli) noexcept;

bool server_cache_host_retention_price_us(
    const common_cache_plan_calib & calib,
    uint64_t bytes,
    bool soft_leased,
    bool main_family,
    uint32_t & weight_milli,
    uint64_t & price_us,
    uint32_t additional_weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE) noexcept;

enum class server_cache_checkpoint_protection : uint8_t {
    none = 0,
    seam_heuristic,
    mandatory_anchor,
    hard_lease,
    _count,
};

struct server_cache_checkpoint_trade_input {
    uint32_t ordinal = 0;
    uint32_t recovery_ordinal = UINT32_MAX;
    llama_cache_acct_artifact_id artifact;
    uint64_t stable_id = 0;
    uint64_t payload_bytes = 0;
    uint64_t replay_tokens = 0;
    uint32_t weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
    bool identity_known = false;
    bool recovery_available = false;
    bool seam_heuristic_protected = false;
    bool mandatory_anchor = false;
    bool hard_leased = false;
};

struct server_cache_checkpoint_trade_plan {
    bool selected = false;
    uint32_t ordinal = UINT32_MAX;
    uint32_t recovery_ordinal = UINT32_MAX;
    uint64_t price_us = 0;
    uint64_t stable_id = 0;
    uint32_t weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
    server_cache_checkpoint_protection protection =
        server_cache_checkpoint_protection::none;
    common_cache_plan_destruction_reason reason =
        common_cache_plan_destruction_reason::recovery_unavailable;
};

// Policy-free checkpoint-member optimum. Inputs already encode the
// ownership/recovery relation; the pure chooser refuses incomplete evidence,
// protects the best-effort seam heuristic and the mandatory/hard rows, and
// uses the fitted replay-plus-restore formula with the same fixed-point
// retention weights as host eviction. Bounded same-lineage replay, not this heuristic,
// is the correctness guarantee for a selected thinning.
server_cache_checkpoint_trade_plan server_cache_plan_checkpoint_thinning(
    const std::vector<server_cache_checkpoint_trade_input> & candidates,
    const common_cache_plan_calib * calib) noexcept;

// A later checkpoint may be omitted after an optional thinning refusal only
// when the retained predecessor is a same-lineage recovery point within the
// configured marginal replay bound.
bool server_cache_checkpoint_bounded_replay(
    const common_prompt_checkpoint & recovery,
    const common_prompt_checkpoint & later,
    uint64_t max_replay_tokens) noexcept;

// An exact recurrent-checkpoint restore may remove only the live attention suffix after the
// installed frontier. Checkpoints wholly before that suffix still cite identical attention rows;
// rebase those matching the pre-trim lineage so dedup/thinning do not copy a replacement image.
size_t server_cache_checkpoint_rebase_preserved_suffix(
    std::list<common_prompt_checkpoint> & checkpoints,
    const llama_memory_vbr_state_data & before,
    const llama_memory_vbr_state_data & after,
    llama_pos suffix_begin) noexcept;

struct server_cache_checkpoint_floor_input {
    uint32_t ordinal = 0;
    server_cache_checkpoint_protection protection =
        server_cache_checkpoint_protection::none;
    bool recovery_pinned = false;
};

struct server_cache_checkpoint_floor_plan {
    bool selected = false;
    uint32_t ordinal = UINT32_MAX;
    common_cache_plan_destruction_reason reason =
        common_cache_plan_destruction_reason::mandatory_anchor;
};

// Capacity's legacy-order floor. Heuristic members remain eligible when every
// unprotected member is gone; hard/mandatory/current-task/pinned members never
// are. No selection means the incoming checkpoint publication must be skipped.
server_cache_checkpoint_floor_plan server_cache_plan_checkpoint_capacity_floor(
    const std::vector<server_cache_checkpoint_floor_input> & candidates) noexcept;

// A protected/fail-closed ring has no new evidence until its membership
// changes.  Keep the three expensive creation-time policy passes independently
// latched so an unchanged ring pays one integer comparison per pass rather than
// rebuilding identities, leases, and quotes for every attempted publication.
// A committed member erase or publication advances the generation and re-arms
// every lane.
enum class server_cache_checkpoint_attempt_lane : uint8_t {
    optional_thinning = 0,
    capacity_thinning,
    capacity_floor,
    _count,
};

class server_cache_checkpoint_attempt_latch {
public:
    bool begin(server_cache_checkpoint_attempt_lane lane) noexcept {
        const size_t index = size_t(lane);
        if (index >= attempted_generation_.size() ||
            attempted_generation_[index] == generation_) {
            return false;
        }
        attempted_generation_[index] = generation_;
        return true;
    }

    bool refusal_changed(
            common_cache_plan_destruction_reason reason,
            bool publication_skip = false) noexcept {
        const size_t lane = publication_skip ? 1 : 0;
        if (refusal_generation_[lane] == generation_ &&
            refusal_reason_[lane] == reason) {
            return false;
        }
        refusal_generation_[lane] = generation_;
        refusal_reason_[lane] = reason;
        return true;
    }

    void ring_changed() noexcept {
        generation_++;
        if (generation_ == 0) {
            generation_ = 1;
            attempted_generation_ = {};
            refusal_generation_ = {};
        }
    }

    uint64_t generation() const noexcept {
        return generation_;
    }

private:
    uint64_t generation_ = 1;
    std::array<uint64_t,
        size_t(server_cache_checkpoint_attempt_lane::_count)>
        attempted_generation_ = {};
    std::array<uint64_t, 2> refusal_generation_ = {};
    std::array<common_cache_plan_destruction_reason, 2> refusal_reason_ = {
        common_cache_plan_destruction_reason::none,
        common_cache_plan_destruction_reason::none,
    };
};

// Checkpoint ownership is prompt-cache authority work even though its
// physical list belongs to a live slot. The slot supplies this narrow view and
// retains only the raw X-macro eraser plus thin adapters; policy, quoting,
// capability preparation, evidence, and ranking live beside the
// prompt-cache authority orchestration.
struct server_cache_checkpoint_authority_context {
    using checkpoint_list = std::list<common_prompt_checkpoint>;
    using checkpoint_iterator = checkpoint_list::iterator;
    using checkpoint_drop_fn = checkpoint_iterator (*)(
        void * owner,
        checkpoint_iterator first,
        checkpoint_iterator last);

    int32_t slot_id = -1;
    checkpoint_list & checkpoints;
    server_cache_authority * authority = nullptr;
    server_retention_sidecar_store * retention = nullptr;
    server_cache_destruction_observer * destruction = nullptr;
    server_cache_lease_table * leases = nullptr;
    server_cache_checkpoint_attempt_latch & attempts;
    const common_prompt_checkpoint *& seam_heuristic;
    common_cache_plan_destruction_reason & thinning_refusal;
    common_cache_plan_destruction_reason & floor_refusal;
    bool main_family = false;
    common_cache_family_binding cache_family;
    bool debug_observability = false;
    void * raw_owner = nullptr;
    checkpoint_drop_fn raw_drop = nullptr;
};

void server_cache_checkpoint_ring_changed(
    server_cache_checkpoint_authority_context & context) noexcept;

bool server_cache_checkpoint_thinning_attempt_begin(
    server_cache_checkpoint_authority_context & context,
    bool capacity_mode) noexcept;

bool server_cache_checkpoint_refusal_state_changed(
    server_cache_checkpoint_authority_context & context,
    common_cache_plan_destruction_reason reason,
    bool publication_skip = false) noexcept;

server_cache_destruction_admission server_cache_checkpoint_observe_drop(
    const server_cache_checkpoint_authority_context & context,
    server_cache_destruction_reason reason,
    llama_cache_acct_artifact_id artifact = {}) noexcept;

bool server_cache_checkpoint_thin_priced(
    server_cache_checkpoint_authority_context & context,
    int checkpoint_task_id,
    uint64_t max_replay_tokens,
    const common_prompt_checkpoint * seam_heuristic,
    bool capacity_mode,
    bool attempt_claimed = false) noexcept;

bool server_cache_checkpoint_capacity_floor(
    server_cache_checkpoint_authority_context & context,
    int checkpoint_task_id,
    const common_prompt_checkpoint * seam_heuristic,
    server_cache_checkpoint_authority_context::checkpoint_iterator & victim,
    common_cache_plan_destruction_reason & refusal) noexcept;

void server_cache_checkpoint_publication_skipped(
    server_cache_checkpoint_authority_context & context,
    common_cache_plan_destruction_reason reason) noexcept;

using server_cache_host_retention_weight_fn = bool (*)(
    void * context,
    const server_prompt_cache_state & victim,
    uint32_t & weight_milli) noexcept;

struct server_cache_host_recovery_evidence {
    llama_cache_acct_artifact_id artifact;
    std::vector<llama_cache_acct_op_id> ops;
    server_cache_recovery_pin pin;
    common_cache_plan_displaced_fate fate =
        common_cache_plan_displaced_fate::unavailable;
};

using server_cache_host_recovery_fn = bool (*)(
    void * context,
    const server_prompt_cache_state & victim,
    server_cache_host_recovery_evidence & out) noexcept;

// Prompt-cache authority substrate. The debug observer is only a serialization layer over this
// independently-owned state; --cache-lifecycle can therefore enforce accounting with debug off.
// Member order is lifetime order: retention releases lease memberships and accounting operations,
// so the ledger and leases must outlive it.
struct server_cache_authority {
    struct device_binding {
        ggml_backend_dev_t               device = nullptr;
        llama_cache_acct_resource_domain domain;
    };

    llama_cache_acct_ledger ledger;
    server_cache_lease_table leases;
    server_retention_sidecar_store retention;
    server_cache_destruction_observer destruction;
    llama_cache_budget_coordinator budget;
    server_cache_yield_result last_yield;
    common_cache_plan_destruction_counters destruction_counters;

    // Immutable bridge from load-time placement to ledger-local device domains.
    std::vector<device_binding> live_device_domains;
    // Fixed-at-reserve-time compute rows. Physical capacity is sampled at observation/admission.
    std::vector<llama_cache_budget_device_input> budget_devices;
    llama_cache_budget_config budget_config;

    uint64_t admission_retries   = 0;
    uint64_t admission_refusals  = 0;
    uint64_t admission_commits   = 0;
    uint64_t admission_rollbacks = 0;
    uint64_t destruction_quote_sequence = 0;
    std::string calibration_profile;
    void * host_retention_weight_context = nullptr;
    server_cache_host_retention_weight_fn host_retention_weight = nullptr;
    void * host_recovery_context = nullptr;
    server_cache_host_recovery_fn host_recovery = nullptr;
    bool configured = true;
    bool summary_emitted = false;

    // Construct a point-in-time budget input. pending_host_bytes are already allocated in the
    // detached host-cache node, so they are added back to the sampled CPU free-memory headroom.
    bool sample_budget(
            llama_cache_budget_config & config,
            uint64_t pending_host_bytes = 0) noexcept;

    // Lower one exact accounting union into the capacity domains that its
    // release would affect. Every destruction class shares this projection door.
    bool project_release(
            const llama_cache_acct_release_set_preview & release,
            std::vector<common_cache_plan_yield_domain> & out) noexcept;

    // Re-sample the affected accounting domains after a committed release.
    // Actual yield is derived from this post-mutation observation, never
    // relabeled from the quote-time projection.
    bool observe_release_domains(
            const std::vector<common_cache_plan_yield_domain> & projected,
            std::vector<common_cache_plan_yield_domain> & out) noexcept;

    // The cache plan's first authoritative producer: admit, stage, and commit all host-entry payload leaves as one
    // all-or-nothing server transaction. Publication itself remains the caller's no-throw splice.
    bool admit_host_entry(server_prompt_cache_state & entry) noexcept;

    // Charge independently owned live-checkpoint payloads. The caller
    // publishes sidecar identities first, then attaches these exact operations
    // in the same scheduler turn before any planner can observe the members.
    // The batch is one all-or-nothing reservation transaction; host-entry
    // checkpoint copies never call this door.
    bool admit_live_checkpoints(
        std::vector<server_cache_live_checkpoint_admission> & batch) noexcept;

    // Single-member creation adapter. Restore paths must use the batch door so
    // one ring incurs one budget sample and one reservation transaction.
    bool admit_live_checkpoint(
        llama_cache_acct_artifact_id artifact,
        const common_prompt_checkpoint & checkpoint,
        std::vector<llama_cache_acct_op_id> & committed) noexcept;

    // Bounded process-local receipt publication for destruction work that
    // occurs during host-cache maintenance rather than one B request record.
    void observe_host_destruction(
        common_cache_plan_destruction_receipt receipt,
        bool observe_classification = true) noexcept;
};
