#pragma once

#include "server-cache-yield.h"
#include "server-cache-plan-authority.h"
#include "../../src/llama-cache-authority.h"

#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <thread>
#include <vector>

struct server_cache_destruction_artifact {
    server_cache_yield_candidate candidate;
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    int32_t owner_slot = -1;
    int32_t host_source_id = -1;
    common_retention_pool pool = common_retention_pool::attention;
    bool mandatory_anchor = false;
    // Fixed pooled KV retains its physical allocation after sequence removal;
    // this explicit bit distinguishes a certified zero release from missing
    // transactional ownership evidence.
    bool fixed_pool_logical_ownership = false;
};

using server_cache_destruction_preview_callback = std::function<bool(
    const std::vector<llama_cache_acct_op_id> &,
    uint64_t,
    llama_cache_acct_release_set_preview &)>;

using server_cache_destruction_projection_callback = std::function<bool(
    const llama_cache_acct_release_set_preview &,
    std::vector<common_cache_plan_yield_domain> &)>;

struct server_cache_destruction_quote_options {
    bool lifecycle_available = false;
    common_cache_plan_recovery_citation recovery_citation =
        common_cache_plan_recovery_citation::unavailable;
    uint64_t admission_sequence = 0;
    common_cache_plan_destruction_effect_set permitted_effects = 0;
    // Preflight may inspect an explicitly unminted receipt. Ordinary callers retain
    // quote_all's fail-closed sequence-zero validation; the capability door
    // independently rejects every never-minted receipt.
    bool preview_unminted = false;
};

// One tooling-visible JSON core for both host-entry and live-checkpoint
// maintenance receipts. Callers append only class-specific ranking/counter
// fields; state/reason/effects/victims/recovery/projected bytes cannot drift.
nlohmann::ordered_json server_cache_destruction_receipt_json(
    const common_cache_plan_destruction_receipt & receipt,
    uint64_t projected_bytes,
    const char * action_class = nullptr);

// Bounded pre-minimization shadow pass. `artifacts` is one normalized
// retention inventory: identities and leases were each inspected exactly once.
// Quotes are memoized by canonical victim-manifest digest; no mutation, lease
// advancement, or accounting claim occurs.
bool server_cache_destruction_quote_all(
    common_cache_plan_record & rec,
    int32_t legacy_plan_candidate,
    const std::vector<server_cache_destruction_artifact> & artifacts,
    uint64_t accounting_serial,
    const server_cache_destruction_preview_callback & preview,
    const server_cache_destruction_projection_callback & project,
    const server_cache_destruction_quote_options & options,
    common_cache_plan_destruction_counters & counters) noexcept;

// Exact-redundancy quote for one host victim. This is the same
// artifact classifier, canonical manifest digest, batch preview, and domain
// projection used by the pre-minimization B-candidate quote path. The caller
// supplies the separately-proved and pinned survivor at prepare time.
common_cache_plan_destruction_quote
server_cache_destruction_quote_redundant_host(
    const server_cache_destruction_artifact & victim,
    uint64_t accounting_serial,
    uint64_t admission_sequence,
    const server_cache_destruction_preview_callback & preview,
    const server_cache_destruction_projection_callback & project) noexcept;

// Independently owned artifacts reuse the exact single-victim
// quote without inheriting the host-consumption effect spelling.
common_cache_plan_destruction_quote
server_cache_destruction_quote_single_artifact(
    const server_cache_destruction_artifact & victim,
    common_cache_plan_destruction_effect_set effects,
    uint64_t accounting_serial,
    uint64_t admission_sequence,
    const server_cache_destruction_preview_callback & preview,
    const server_cache_destruction_projection_callback & project) noexcept;

void server_cache_destruction_select_quote(
    common_cache_plan_record & rec,
    common_cache_plan_destruction_counters & counters,
    common_cache_plan_destruction_effect_set permitted_effects = 0) noexcept;

// The lifecycle-off projection lives beside the production selector so its
// no-quote/refusal semantics cannot drift into a second server-context policy.
void server_cache_destruction_select_preview(
    common_cache_plan_record & rec,
    common_cache_plan_destruction_counters & counters,
    int32_t legacy_plan_candidate,
    bool lifecycle_available,
    common_cache_plan_destruction_effect_set permitted_effects = 0) noexcept;

void server_cache_destruction_finalize_projection(
    common_cache_plan_record & rec,
    const server_cache_yield_result & yield) noexcept;

bool server_cache_destruction_effect_matches(
    const common_cache_plan_destruction_receipt & quote,
    const common_cache_plan_destruction_effect_digest & current_effect,
    const std::vector<common_cache_plan_yield_domain> & quoted_domains,
    const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept;

common_cache_plan_destruction_effect_digest
server_cache_destruction_union_effect_digest(
    const std::vector<llama_cache_acct_op_id> & ops,
    const llama_cache_acct_release_set_preview & release);

common_cache_plan_destruction_recovery_digest
server_cache_destruction_recovery_source_digest(
    llama_cache_acct_artifact_id artifact,
    const std::vector<llama_cache_acct_op_id> & ops);

// Forward contract for the mutation-boundary certify-time recheck. The
// quote serial is evidence only; exact union/digest/domain equality decides.
common_cache_plan_destruction_reason server_cache_destruction_effect_recheck(
    const common_cache_plan_destruction_receipt & quote,
    const common_cache_plan_destruction_effect_digest & current_effect,
    const std::vector<common_cache_plan_yield_domain> & quoted_domains,
    const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept;

bool server_cache_destruction_has_effect(
    const common_cache_plan_record & rec,
    int32_t legacy_candidate,
    common_cache_plan_destruction_effect_set permitted_effects = 0) noexcept;

void server_cache_destruction_certify_receipt(
    common_cache_plan_destruction_receipt & receipt,
    common_cache_plan_displaced_fate fate,
    llama_cache_acct_artifact_id recovery_artifact,
    const std::vector<llama_cache_acct_op_id> & recovery_ops) noexcept;

// A non-policy recovery guard, separate from WS-D leases. The owner callback
// releases the underlying immutable host/catalog/live guard. Destruction
// authority requires its protected source to be disjoint from the victim union.
class server_cache_recovery_pin {
public:
    using release_fn = void (*)(void *) noexcept;

    server_cache_recovery_pin() = default;
    ~server_cache_recovery_pin();
    server_cache_recovery_pin(const server_cache_recovery_pin &) = delete;
    server_cache_recovery_pin & operator=(const server_cache_recovery_pin &) = delete;
    server_cache_recovery_pin(server_cache_recovery_pin &&) noexcept;
    server_cache_recovery_pin & operator=(server_cache_recovery_pin &&) noexcept;

    static server_cache_recovery_pin acquire(
        void * context,
        release_fn release,
        std::vector<llama_cache_acct_artifact_id> artifacts,
        std::vector<llama_cache_acct_op_id> ops) noexcept;

    bool valid() const noexcept { return context_ != nullptr && release_ != nullptr; }
    bool disjoint(
        const std::vector<llama_cache_acct_artifact_id> & artifacts,
        const std::vector<llama_cache_acct_op_id> & ops) const noexcept;
    bool binds_exact(
        llama_cache_acct_artifact_id artifact,
        const std::vector<llama_cache_acct_op_id> & ops) const noexcept;

private:
    void reset() noexcept;
    void * context_ = nullptr;
    release_fn release_ = nullptr;
    std::vector<llama_cache_acct_artifact_id> artifacts_;
    std::vector<llama_cache_acct_op_id> ops_;
};

enum class server_cache_prepare_release_status : uint8_t {
    prepared,
    invalid_quote,
    recovery_unavailable,
    serial_conflict,
    effect_drift,
    accounting_unavailable,
    internal_fault,
    _count,
};

struct server_cache_prepare_release_result;

class server_cache_prepared_release_capability {
public:
    server_cache_prepared_release_capability() = default;
    ~server_cache_prepared_release_capability() = default;
    server_cache_prepared_release_capability(
        const server_cache_prepared_release_capability &) = delete;
    server_cache_prepared_release_capability & operator=(
        const server_cache_prepared_release_capability &) = delete;
    server_cache_prepared_release_capability(
        server_cache_prepared_release_capability &&) noexcept = default;
    server_cache_prepared_release_capability & operator=(
        server_cache_prepared_release_capability &&) noexcept = default;

    bool ready() const noexcept { return release_.ready() && pin_.valid(); }
    uint64_t accounting_serial() const noexcept {
        return release_.accounting_serial();
    }

    // Accounting terminal. The API deliberately accepts no callback:
    // the prepare→physical-mutation→commit interval cannot re-enter the ledger
    // through this substrate. Same-thread ownership and unchanged serial are
    // asserted/checked by this terminal. On success, the caller owns the pin
    // for its longer B-execution dependency lifetime.
    common_cache_plan_destruction_reason commit(
        server_cache_recovery_pin & retained_pin) noexcept;

private:
    llama_cache_prepared_release_set release_;
    server_cache_recovery_pin pin_;
    std::thread::id scheduler_owner_;

    friend server_cache_prepare_release_result server_cache_prepare_release_set(
        const common_cache_plan_destruction_quote &,
        const std::vector<server_cache_destruction_artifact> &,
        llama_cache_acct_ledger &,
        uint64_t,
        const server_cache_destruction_projection_callback &,
        server_cache_recovery_pin &&) noexcept;
};

struct server_cache_prepare_release_result {
    server_cache_prepare_release_status status =
        server_cache_prepare_release_status::invalid_quote;
    common_cache_plan_destruction_reason reason =
        common_cache_plan_destruction_reason::manifest_incomplete;
    server_cache_prepared_release_capability capability;
};

// Fresh-serial certification. The caller first advances the lease
// lifecycle and builds `current_artifacts` with one fresh inspection per
// artifact. This compares identity/anchor/lease state, the canonical op-set
// bound inside the exact union digest, and projected release rows against the
// launch-time quote. The quote serial is deliberately ignored as an execution pin.
server_cache_prepare_release_result server_cache_prepare_release_set(
    const common_cache_plan_destruction_quote & quote,
    const std::vector<server_cache_destruction_artifact> & current_artifacts,
    llama_cache_acct_ledger & ledger,
    uint64_t fresh_accounting_serial,
    const server_cache_destruction_projection_callback & project,
    server_cache_recovery_pin && recovery_pin) noexcept;
