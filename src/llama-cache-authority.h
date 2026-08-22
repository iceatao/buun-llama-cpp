#pragma once

// llama-cache-authority.h — P2 F0a admission composer (library side).
//
// The policy-free ledger (llama-cache-accounting) records; the budget coordinator
// (llama-cache-budget) prices. This unit is the single place that COMPOSES them into the
// authoritative admission sequence F flips on:
//
//     snapshot() → fits(reserve-only plan) → reserve_if_serial(snapshot.serial)
//
// It is reserve-only and single-shot by contract:
//   - reserve-only: the plan credits NO planned release. Crediting a release before F0b owns an
//     atomic release/mutate transaction would let a reservation be admitted against bytes that are
//     not yet actually free (false admission).
//   - single-shot: on serial drift it returns serial_conflict immediately rather than retrying with
//     a possibly-stale capacity config; the shared multi-leaf primitive below owns the bounded
//     resnapshot→fits→reserve retry over one caller-sampled capacity config.
//
// `llama_cache_admit_reservation` (the single-leaf composer) is called in production only by the
// shared multi-leaf transaction primitive below; direct callers are unit tests. That primitive owns
// the mutation-adjacent stage/commit/abort sequence and is wired into two production mutation paths:
// F0b's host-cache publish (server-cache-authority) and F2's artifact catalog publish.

#include "llama-cache-accounting.h"
#include "llama-cache-budget.h"

#include <cstddef>
#include <memory>
#include <vector>

// One typed reservation request. Exactly one target domain; no release credits.
struct llama_cache_authority_request {
    llama_cache_acct_category        category = llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution     attribution;
    uint64_t                         expected_logical  = 0;
    uint64_t                         expected_resident = 0;
};

// Closed admission taxonomy. incomplete_evidence and budget_unavailable are kept distinct because
// fits() collapses every non-fit into `unavailable`; incomplete_evidence is claimed ONLY when the
// snapshot manifest is explicitly non-known (the fail-closed rule), never fabricated from an opaque
// coordinator refusal.
enum class llama_cache_admission_status : uint8_t {
    admitted,
    incomplete_evidence,
    budget_unavailable,
    exceeds_budget,
    serial_conflict,
    ledger_fault,
    internal_fault,     // the composer caught its own allocation/exception — fail-closed, never throws
    _count,
};

const char * llama_cache_admission_status_name(llama_cache_admission_status status) noexcept;

struct llama_cache_admission_result;
struct llama_cache_transaction_leaf;
class llama_cache_prepared_claim_group;

// Move-only handle to an admitted-but-not-yet-committed reservation. If it is destroyed while still
// holding a live op (an exception or early return before F0b's stage/commit), it aborts that op so a
// reserved operation can never leak (snapshot.live_ops would otherwise never return to zero). Only
// the composer mints an armed claim (constructor is private + friended), so no two claims can ever
// own the same op. The only disarm-without-abort path is F0b's commit-through-claim terminal, which
// disarms ONLY on a verified commit and hands the committed id to its durable artifact. Exposing a
// bare "discharge" would let a still-reserved (or committed-then-failed) op leak.
class llama_cache_reservation_claim {
public:
    llama_cache_reservation_claim() = default;
    ~llama_cache_reservation_claim();

    llama_cache_reservation_claim(const llama_cache_reservation_claim &)             = delete;
    llama_cache_reservation_claim & operator=(const llama_cache_reservation_claim &) = delete;
    llama_cache_reservation_claim(llama_cache_reservation_claim && other) noexcept;
    llama_cache_reservation_claim & operator=(llama_cache_reservation_claim && other) noexcept;

    bool                   has_op() const noexcept { return ledger_ != nullptr && op_.v != 0; }
    llama_cache_acct_op_id op()     const noexcept { return op_; }

    // Publish the staged accounting transaction through its sole owning claim. Success returns
    // the committed operation id to the durable artifact and disarms this claim; the artifact later
    // uses ledger.release() both for ordinary retirement and for rollback if publication fails.
    // Failure leaves the claim armed, so its destructor still aborts the reserved/staged operation.
    bool commit(uint64_t logical_bytes, llama_cache_acct_op_id & committed_op) noexcept;

private:
    llama_cache_reservation_claim(llama_cache_acct_ledger * ledger, llama_cache_acct_op_id op) noexcept;

    // Drop ownership without aborting; the single definition of "no longer holds an op".
    void release() noexcept { ledger_ = nullptr; op_ = {}; }
    void abort_if_live() noexcept;

    llama_cache_acct_ledger * ledger_ = nullptr;
    llama_cache_acct_op_id    op_     = {};

    friend llama_cache_admission_result llama_cache_admit_reservation(
            llama_cache_acct_ledger &, const llama_cache_budget_config &,
            const llama_cache_authority_request &) noexcept;
    friend class llama_cache_prepared_claim_group;
    friend llama_cache_prepared_claim_group
    llama_cache_prepare_reservation_transaction(
            llama_cache_acct_ledger &,
            const llama_cache_budget_config &,
            const std::vector<llama_cache_transaction_leaf> &) noexcept;
};

// The composer's result: a status plus, when admitted, the move-only claim that owns the reserved
// op until commit-through-claim hands the committed id to the durable artifact. Move-only.
struct llama_cache_admission_result {
    llama_cache_admission_status  status = llama_cache_admission_status::ledger_fault;
    llama_cache_reservation_claim claim;
};

// Run the authoritative admission sequence once. `budget_config` is the caller's point-in-time
// capacity/config (the server samples physical capacity immediately before calling). A local
// coordinator is used because reset() mutates coordinator state — a shared mutable coordinator is
// not safe across concurrent admissions. noexcept: every failure — including its own allocation —
// becomes a typed status, so no exception ever crosses the authority boundary into F0b.
llama_cache_admission_result llama_cache_admit_reservation(
        llama_cache_acct_ledger          & ledger,
        const llama_cache_budget_config  & budget_config,
        const llama_cache_authority_request & request) noexcept;

enum class llama_cache_prepare_release_status : uint8_t {
    prepared,
    invalid_argument,
    serial_conflict,
    ledger_fault,
    internal_fault,
    _count,
};

const char * llama_cache_prepare_release_status_name(
        llama_cache_prepare_release_status status) noexcept;

// Move-only, non-claiming preparation for a complete release union. Preparing
// only snapshots exact last-reference effects; destruction authority owns the
// later physical mutation and calls commit() exactly once at its terminal.
class llama_cache_prepared_release_set {
public:
    llama_cache_prepared_release_set() = default;
    ~llama_cache_prepared_release_set() = default;

    llama_cache_prepared_release_set(const llama_cache_prepared_release_set &) = delete;
    llama_cache_prepared_release_set & operator=(const llama_cache_prepared_release_set &) = delete;
    llama_cache_prepared_release_set(llama_cache_prepared_release_set &&) noexcept;
    llama_cache_prepared_release_set & operator=(llama_cache_prepared_release_set &&) noexcept;

    bool ready() const noexcept {
        return ledger_ != nullptr &&
               status_ == llama_cache_prepare_release_status::prepared;
    }
    llama_cache_prepare_release_status status() const noexcept { return status_; }
    uint64_t accounting_serial() const noexcept { return preview_.accounting_serial; }
    const std::vector<llama_cache_acct_op_id> & ops() const noexcept { return ops_; }
    const llama_cache_acct_release_set_preview & preview() const noexcept { return preview_; }

    // Allocation-free atomic terminal. serial_conflict means a forbidden
    // ledger write occurred after prepare; no operation was released.
    llama_cache_conditional_release_status commit() noexcept;

private:
    llama_cache_acct_ledger * ledger_ = nullptr;
    std::vector<llama_cache_acct_op_id> ops_;
    llama_cache_acct_release_set_preview preview_;
    llama_cache_prepare_release_status status_ =
        llama_cache_prepare_release_status::invalid_argument;

    friend llama_cache_prepared_release_set llama_cache_prepare_release_set(
        llama_cache_acct_ledger &,
        const std::vector<llama_cache_acct_op_id> &,
        uint64_t) noexcept;
};

// Fresh-serial prepare at the mutation boundary. The quote's earlier serial is
// evidence only; callers compare this canonical op set and exact union effect
// to the quote before permitting a physical mutation. An empty op set is the
// canonical known-zero release for logical ownership in fixed pooled storage;
// commit still checks the serial and performs no ledger mutation.
llama_cache_prepared_release_set llama_cache_prepare_release_set(
    llama_cache_acct_ledger & ledger,
    const std::vector<llama_cache_acct_op_id> & ops,
    uint64_t expected_serial) noexcept;

// One leaf in the shared all-or-nothing authority transaction. A zero existing_allocation asks
// the primitive to mint a fresh allocation; a nonzero one joins that immutable allocation and
// normally reserves zero new physical bytes while staging its full resident tuple. Output pointers
// are written only after EVERY leaf commits and the post-commit fault seam passes.
struct llama_cache_transaction_leaf {
    llama_cache_acct_category        category = llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution     attribution;
    uint64_t expected_logical = 0;
    uint64_t reserve_resident = 0;
    uint64_t stage_resident   = 0;
    llama_cache_acct_artifact_id    artifact;
    llama_cache_acct_content_digest content;
    llama_cache_acct_lineage_id     lineage;
    llama_cache_acct_alloc_id       existing_allocation;
    llama_cache_acct_op_id *        committed_op  = nullptr;
    llama_cache_acct_alloc_id *     allocation_out = nullptr;
};

// Library-owned fault vocabulary shared by F0b's translated server fault and F2's fake-shard
// tests. UINT32_MAX disables an indexed seam.
struct llama_cache_transaction_fault {
    uint32_t fail_stage_at  = UINT32_MAX;
    uint32_t fail_commit_at = UINT32_MAX;
    bool fail_after_commit  = false;
};

// Optional call-site preparation that must remain between "all capacity claims admitted" and
// "first C stage". F2 uses it to allocate catalog-owned shard storage without crossing the
// reserve-before-mutate boundary; F0b has no preparation hook. Exceptions are caught by the
// primitive and become internal_fault.
struct llama_cache_transaction_after_admit {
    void * context = nullptr;
    bool (*run)(void * context) = nullptr;
};

enum class llama_cache_transaction_status : uint8_t {
    committed,
    invalid_argument,
    admission_refused,
    after_admit_failed,
    stage_failed,
    commit_failed,
    post_commit_fault,
    internal_fault,
    _count,
};

const char * llama_cache_transaction_status_name(
        llama_cache_transaction_status status) noexcept;

struct llama_cache_transaction_result {
    llama_cache_transaction_status status =
        llama_cache_transaction_status::internal_fault;
    llama_cache_admission_status admission_status =
        llama_cache_admission_status::internal_fault;
    size_t failed_leaf = SIZE_MAX;
    uint32_t attempts = 0;
    uint64_t serial_retries = 0;
    uint64_t rolled_back = 0;
};

enum class llama_cache_prepare_status : uint8_t {
    prepared,
    invalid_argument,
    admission_refused,
    internal_fault,
    _count,
};

const char * llama_cache_prepare_status_name(
        llama_cache_prepare_status status) noexcept;

struct llama_cache_prepare_result {
    llama_cache_prepare_status status =
        llama_cache_prepare_status::internal_fault;
    llama_cache_admission_status admission_status =
        llama_cache_admission_status::internal_fault;
    size_t failed_leaf = SIZE_MAX;
    uint32_t attempts = 0;
    uint64_t serial_retries = 0;
};

// Split-phase form of the shared authority transaction. A prepared group owns
// every admitted reservation claim until its one materialize-and-commit
// terminal runs. Dropping or replacing the group aborts all still-live claims.
// This is the F3 streaming seam: capacity is proven before a bounded D2H
// materialization without introducing a second admission or rollback path.
class llama_cache_prepared_claim_group {
public:
    llama_cache_prepared_claim_group();
    ~llama_cache_prepared_claim_group();

    llama_cache_prepared_claim_group(
        const llama_cache_prepared_claim_group &) = delete;
    llama_cache_prepared_claim_group & operator=(
        const llama_cache_prepared_claim_group &) = delete;
    llama_cache_prepared_claim_group(
        llama_cache_prepared_claim_group &&) noexcept;
    llama_cache_prepared_claim_group & operator=(
        llama_cache_prepared_claim_group &&) noexcept;

    bool ready() const noexcept;
    const llama_cache_prepare_result & preparation() const noexcept;

    // Downward-only staging reprice. Every prepared leaf must have equal
    // logical/reserve/stage bytes. The admitted claims remain the same, so a
    // dependency-local shrink cannot fail due to a second capacity sample or
    // competing reservation.
    bool shrink_equal_reservations(
        const std::vector<uint64_t> & resident_bytes) noexcept;

    // Repartition a conservative prepared fence into exact final leaves while
    // preserving one uninterrupted capacity claim. The replacement may split
    // or merge leaves and may bind existing immutable allocations, but its
    // reserved aggregate can only decrease within each category/domain.
    bool repartition_downward(
        const std::vector<llama_cache_transaction_leaf> & leaves) noexcept;

    // Single terminal. The optional hook remains after all claims are admitted
    // and before the first stage. Re-entry or a failed preparation returns a
    // typed invalid/internal result and never exposes success outputs.
    llama_cache_transaction_result materialize_and_commit(
        const llama_cache_transaction_fault & fault = {},
        const llama_cache_transaction_after_admit & after_admit = {}) noexcept;

    // F3 prepares capacity before the content-addressed byte stream exists.
    // The terminal may bind the final artifact/content/lineage citations and
    // success-output locations, while every priced field remains identical
    // to the prepared leaf. Any shape change fails closed before staging.
    llama_cache_transaction_result materialize_and_commit(
        const std::vector<llama_cache_transaction_leaf> & finalized_leaves,
        const llama_cache_transaction_fault & fault = {},
        const llama_cache_transaction_after_admit & after_admit = {}) noexcept;

private:
    struct impl;
    explicit llama_cache_prepared_claim_group(
        std::unique_ptr<impl> state) noexcept;

    std::unique_ptr<impl> impl_;

    void abort_if_live() noexcept;

    friend llama_cache_prepared_claim_group
    llama_cache_prepare_reservation_transaction(
        llama_cache_acct_ledger &,
        const llama_cache_budget_config &,
        const std::vector<llama_cache_transaction_leaf> &) noexcept;
};

// Admit every leaf with the same bounded resnapshot/fits/reserve retry used by
// the historical one-shot helper, but retain the move-only claims for a later
// materialization terminal.
llama_cache_prepared_claim_group
llama_cache_prepare_reservation_transaction(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget_config,
        const std::vector<llama_cache_transaction_leaf> & leaves) noexcept;

// Shared F0/F2 transaction:
//   admit every leaf (bounded serial-conflict retry) → optional preparation → stage all →
//   commit all → post-commit fault seam → publish success-only outputs.
// Any failure releases already-committed ops and lets the move-only claims
// abort the rest. This one-shot API is a thin prepare + terminal wrapper.
llama_cache_transaction_result llama_cache_execute_reservation_transaction(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget_config,
        const std::vector<llama_cache_transaction_leaf> & leaves,
        const llama_cache_transaction_fault & fault = {},
        const llama_cache_transaction_after_admit & after_admit = {}) noexcept;
