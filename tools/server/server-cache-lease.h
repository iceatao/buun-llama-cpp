#pragma once

#include "server-cache-lifecycle.h"
#include "common-retention-sidecar.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class server_cache_lease_scope_kind : uint8_t {
    process = 0,
    session,
    context,
    lease,
    _count,
};

enum class server_cache_lease_class : uint8_t {
    none = 0,
    soft,
    hard,
    _count,
};

enum class server_cache_lease_eval_state : uint8_t {
    known = 0,
    unavailable,
    _count,
};

enum class server_cache_lease_eligibility : uint8_t {
    eligible = 0,
    hard_blocked,
    _count,
};

enum class server_cache_lease_fallback_state : uint8_t {
    available = 0,
    unavailable,
    invalid,
    _count,
};

enum class server_cache_lease_event_kind : uint8_t {
    grant_soft = 0,
    grant_hard,
    refuse_hard_unavailable,
    refuse_hard_invalid,
    renew,
    expire,
    release,
    invalidate_identity,
    mark_identity_unavailable,
    clear_identity_unavailable,
    clone,
    orphan_hard,
    _count,
};

struct server_cache_process_scope_id {
    uint64_t v = 0;
};
struct server_cache_session_scope_id {
    uint64_t v = 0;
};
struct server_cache_context_scope_id {
    uint64_t v = 0;
};
struct server_cache_explicit_lease_scope_id {
    uint64_t v = 0;
};
struct server_cache_lease_id {
    uint64_t v = 0;
    explicit operator bool() const noexcept { return v != 0; }
};

struct server_cache_lease_owner_id {
    uint64_t v = 0;
    explicit operator bool() const noexcept { return v != 0; }
};
struct server_cache_lease_identity_id {
    uint64_t v = 0;
    explicit operator bool() const noexcept { return v != 0; }
};

#define SERVER_CACHE_LEASE_ID_EQUALITY(type) \
    inline bool operator==(type a, type b) noexcept { return a.v == b.v; } \
    inline bool operator!=(type a, type b) noexcept { return !(a == b); }
SERVER_CACHE_LEASE_ID_EQUALITY(server_cache_process_scope_id)
SERVER_CACHE_LEASE_ID_EQUALITY(server_cache_session_scope_id)
SERVER_CACHE_LEASE_ID_EQUALITY(server_cache_context_scope_id)
SERVER_CACHE_LEASE_ID_EQUALITY(server_cache_explicit_lease_scope_id)
SERVER_CACHE_LEASE_ID_EQUALITY(server_cache_lease_id)
SERVER_CACHE_LEASE_ID_EQUALITY(server_cache_lease_identity_id)
SERVER_CACHE_LEASE_ID_EQUALITY(server_cache_lease_owner_id)
#undef SERVER_CACHE_LEASE_ID_EQUALITY

struct server_cache_lease_scope {
    server_cache_lease_scope_kind kind = server_cache_lease_scope_kind::context;
    uint64_t id = 0;

    static server_cache_lease_scope from(server_cache_process_scope_id value) noexcept;
    static server_cache_lease_scope from(server_cache_session_scope_id value) noexcept;
    static server_cache_lease_scope from(server_cache_context_scope_id value) noexcept;
    static server_cache_lease_scope from(server_cache_explicit_lease_scope_id value) noexcept;

    bool valid() const noexcept;
};

struct server_cache_lease_identity {
    // Canonical mirror of common_computation_frontier's three opaque
    // comparison keys. The contract scan ties the member names/types across
    // the two structs without making this observer-only library depend on the
    // server/common checkpoint holder.
    std::string execution_identity;
    std::string adapter_config_identity;
    std::string media_content_identity;

    bool valid() const noexcept;
};

bool operator==(
    const server_cache_lease_identity & a,
    const server_cache_lease_identity & b) noexcept;
inline bool operator!=(
        const server_cache_lease_identity & a,
        const server_cache_lease_identity & b) noexcept {
    return !(a == b);
}

struct server_cache_lease_subject {
    llama_cache_acct_artifact_id artifact;
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    int32_t owner_slot = -1;

    bool valid() const noexcept;
};

struct server_cache_lease_evaluation {
    server_cache_lease_eval_state state =
        server_cache_lease_eval_state::unavailable;
    server_cache_lease_class cls = server_cache_lease_class::none;
    server_cache_lease_eligibility eligibility =
        server_cache_lease_eligibility::eligible;
};

struct server_cache_lease_inspection_request {
    llama_cache_acct_artifact_id artifact;
    const server_cache_lease_identity * expected_identity = nullptr;
};

struct server_cache_lease_frontier {
    uint64_t sequence_epoch = 0;
    uint64_t token_count = 0;
    int64_t next_position = 0;

    bool valid() const noexcept {
        return sequence_epoch != 0 && next_position >= 0;
    }
};

inline bool operator==(
        const server_cache_lease_frontier & a,
        const server_cache_lease_frontier & b) noexcept {
    return a.sequence_epoch == b.sequence_epoch &&
           a.token_count == b.token_count &&
           a.next_position == b.next_position;
}

inline bool server_cache_lease_is_hard(
        const server_cache_lease_evaluation & lease) noexcept {
    return lease.cls == server_cache_lease_class::hard ||
           lease.eligibility == server_cache_lease_eligibility::hard_blocked;
}

struct server_cache_lease_event {
    server_cache_lease_event_kind kind = server_cache_lease_event_kind::grant_soft;
    server_cache_lease_id lease;
    server_cache_lease_id source_lease;
    server_cache_lease_scope scope;
    llama_cache_acct_artifact_id artifact;
    common_retention_artifact_kind artifact_kind =
        common_retention_artifact_kind::live_slot;
    int32_t owner_slot = -1;
    server_cache_lease_identity_id identity;
    server_cache_lease_class cls = server_cache_lease_class::none;
    server_cache_lease_fallback_state fallback =
        server_cache_lease_fallback_state::unavailable;
    uint64_t ttl_ns = 0;
    uint64_t ordinal = 0;
};

constexpr size_t SERVER_CACHE_LEASE_EVENT_RING = 64;

struct server_cache_lease_identity_record {
    server_cache_lease_identity_id id;
    server_cache_lease_identity value;
};

struct server_cache_lease_event_snapshot {
    std::array<server_cache_lease_event, SERVER_CACHE_LEASE_EVENT_RING> events = {};
    std::array<uint64_t, size_t(server_cache_lease_event_kind::_count)> totals = {};
    size_t size = 0;
    uint64_t first_ordinal = 0;
    uint64_t last_ordinal = 0;
    uint64_t overflows = 0;
    bool unavailable = false;
    std::vector<server_cache_lease_identity_record> identities;

    bool replay_available() const noexcept {
        return !unavailable && overflows == 0 &&
               (size == 0 || (first_ordinal == 1 && last_ordinal == size));
    }
};

struct server_cache_lease_replay_result {
    server_cache_lease_eval_state state =
        server_cache_lease_eval_state::unavailable;
    uint64_t last_ordinal = 0;
    std::vector<server_cache_lease_event> active;
    std::vector<server_cache_lease_identity_record> identities;
    std::vector<server_cache_lease_subject> identity_unavailable;
};

class server_cache_lease_clock {
public:
    virtual ~server_cache_lease_clock() = default;
    virtual uint64_t now_ns() noexcept = 0;
};

// A hard lease owns this move-only lifetime proof. The payload is deliberately
// type-erased here: retention and F catalogs remain the concrete content
// owners, while the one lease table owns only a shared lifetime pin.
class server_cache_durable_fallback_proof {
public:
    server_cache_durable_fallback_proof() = default;
    ~server_cache_durable_fallback_proof() = default;
    server_cache_durable_fallback_proof(
        server_cache_durable_fallback_proof &&) noexcept = default;
    server_cache_durable_fallback_proof & operator=(
        server_cache_durable_fallback_proof &&) noexcept = default;

    server_cache_durable_fallback_proof(
        const server_cache_durable_fallback_proof &) = delete;
    server_cache_durable_fallback_proof & operator=(
        const server_cache_durable_fallback_proof &) = delete;

    server_cache_lease_fallback_state state() const noexcept { return state_; }
    bool available() const noexcept {
        return state_ == server_cache_lease_fallback_state::available &&
               owner_ != nullptr;
    }

private:
    friend class server_cache_lease_table;
    friend server_cache_durable_fallback_proof
        server_cache_durable_fallback_proof_for_test(
            server_cache_lease_fallback_state,
            std::shared_ptr<void>) noexcept;
    server_cache_durable_fallback_proof(
            server_cache_lease_fallback_state state,
            std::shared_ptr<void> owner) noexcept :
        state_(state), owner_(std::move(owner)) {
    }
    server_cache_durable_fallback_proof retain() const noexcept {
        return { state_, owner_ };
    }

    server_cache_lease_fallback_state state_ =
        server_cache_lease_fallback_state::unavailable;
    std::shared_ptr<void> owner_;
};

// Private test door used to verify proof lifetime without granting production
// code a proof minting surface. Contract scans forbid production call sites.
server_cache_durable_fallback_proof
server_cache_durable_fallback_proof_for_test(
    server_cache_lease_fallback_state state,
    std::shared_ptr<void> owner) noexcept;

class server_cache_lease_fallback_provider {
public:
    virtual ~server_cache_lease_fallback_provider() = default;
    virtual server_cache_durable_fallback_proof acquire(
        const server_cache_lease_subject & subject,
        const server_cache_lease_identity & identity) noexcept = 0;
};

class server_cache_lease_table {
public:
    // 5 minutes. Shadow-only implicit soft-lease TTL; documented for later calibration.
    static constexpr uint64_t IMPLICIT_SOFT_TTL_NS =
        5ull * 60ull * 1000ull * 1000ull * 1000ull;

    explicit server_cache_lease_table(
        server_cache_lease_clock * clock = nullptr,
        server_cache_lease_fallback_provider * fallback = nullptr) noexcept;

    server_cache_context_scope_id new_context_scope() noexcept;

    server_cache_lease_id grant_soft(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        uint64_t ttl_ns) noexcept;
    server_cache_lease_id grant_hard(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        uint64_t ttl_ns) noexcept;
    // E1's holder-owned hard lease. `ttl_ns` is the holder inspection/
    // reattach deadline; it is deliberately not a destruction deadline.
    // The entry remains enforced until explicit release/owner close/restart.
    server_cache_lease_id grant_hard_owned(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        server_cache_lease_owner_id owner,
        const server_cache_lease_frontier & proven_frontier,
        uint64_t ttl_ns) noexcept;
    bool renew_owned(
        server_cache_lease_id lease,
        server_cache_lease_owner_id owner,
        const server_cache_lease_frontier & proven_frontier,
        uint64_t ttl_ns) noexcept;
    bool orphan_owner(server_cache_lease_owner_id owner) noexcept;
    bool orphan_owned_scope(
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) noexcept;
    bool release_owned_scope(
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) noexcept;
    bool lease_active(server_cache_lease_id lease) const noexcept;
    bool lease_subject_lost(server_cache_lease_id lease) const noexcept;
    bool owned_scope_active(
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) const noexcept;
    void bind_fallback_provider(
        server_cache_lease_fallback_provider * provider) noexcept;
    bool renew(server_cache_lease_id lease, uint64_t ttl_ns) noexcept;
    bool release(server_cache_lease_id lease) noexcept;

    void lifecycle_point() noexcept;
    void artifact_identity_unavailable(
        const server_cache_lease_subject & subject) noexcept;
    void artifact_retired(llama_cache_acct_artifact_id artifact) noexcept;
    bool artifact_cloned(
        const server_cache_lease_subject & source,
        const server_cache_lease_subject & destination,
        const server_cache_lease_identity & destination_identity) noexcept;
    // Move lease ownership across a new immutable observer artifact for the
    // same physical live slot. Identity and proven-frontier containment are
    // checked before any entry changes; a false result leaves the source
    // untouched so the caller can retire it through the normal fail-closed
    // subject_lost terminal.
    bool artifact_replaced(
        const server_cache_lease_subject & source,
        const server_cache_lease_subject & destination,
        const server_cache_lease_identity & destination_identity,
        const server_cache_lease_frontier & current_frontier) noexcept;
    bool artifact_rebound(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity) noexcept;

    server_cache_lease_evaluation evaluate(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity) noexcept;
    // Read-only planner seam: samples the current clock to ignore logically
    // expired leases, but never interns, expires, invalidates, records, or
    // changes table counters.
    server_cache_lease_evaluation inspect(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity) const noexcept;
    // Batch planner seam. It samples the clock once and indexes the immutable
    // lease/identity vectors once, avoiding a full lease scan per candidate.
    bool inspect_batch(
        const std::vector<server_cache_lease_inspection_request> & requests,
        std::vector<server_cache_lease_evaluation> & out) const noexcept;
    server_cache_lease_evaluation inspect_range(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity,
        uint64_t sequence_epoch,
        uint64_t first_token,
        uint64_t token_count) const noexcept;
    bool has_hard_lease() const noexcept;
    server_cache_destruction_verdict admit(
        const server_cache_destruction_request & request) noexcept;

    server_cache_lease_event_snapshot event_snapshot() const noexcept;
    uint64_t clock_samples() const noexcept { return n_clock_samples; }
    uint64_t unavailable_events() const noexcept { return n_unavailable; }

    static bool replay(
        const server_cache_lease_event_snapshot & snapshot,
        server_cache_lease_replay_result & out) noexcept;

private:
    struct entry {
        server_cache_lease_id lease;
        server_cache_lease_subject subject;
        server_cache_lease_scope scope;
        server_cache_lease_identity_id identity_id;
        server_cache_lease_class cls = server_cache_lease_class::none;
        uint64_t granted_at_ns = 0;
        uint64_t expires_at_ns = 0;
        uint64_t ttl_ns = 0;
        uint64_t last_event_ordinal = 0;
        server_cache_durable_fallback_proof fallback_proof;
        server_cache_lease_owner_id owner;
        server_cache_lease_frontier proven_frontier;
        bool explicit_hard = false;
        bool orphaned = false;
        bool subject_lost = false;
    };

    static entry clone_core(const entry & source) noexcept;

    uint64_t sample_now() noexcept;
    void mark_table_unavailable() noexcept;
    bool checked_deadline(uint64_t now, uint64_t ttl, uint64_t & out) noexcept;
    server_cache_lease_id issue_lease_id() noexcept;
    server_cache_lease_identity_id intern_identity(
        const server_cache_lease_identity & value) noexcept;
    void expire_due(uint64_t now) noexcept;
    void record(
        server_cache_lease_event_kind kind,
        const entry * lease,
        server_cache_lease_id source,
        server_cache_lease_fallback_state fallback) noexcept;
    entry * emit_grant(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        server_cache_lease_identity_id identity_id,
        server_cache_lease_class cls,
        uint64_t now,
        uint64_t deadline,
        uint64_t ttl_ns,
        server_cache_lease_event_kind kind,
        server_cache_durable_fallback_proof proof = {}) noexcept;
    entry * grant_hard_entry(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        uint64_t ttl_ns) noexcept;
    bool add_entry(entry && value, server_cache_lease_event_kind kind,
                   server_cache_lease_id source = {}) noexcept;
    static bool owned_scope_match(
        const entry & value,
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) noexcept;
    bool orphan_entry(entry & value) noexcept;
    void mark_subject_lost(size_t index) noexcept;
    void invalidate_entry(size_t index, server_cache_lease_event_kind kind) noexcept;
    void mark_identity_unavailable(
        const server_cache_lease_subject & subject) noexcept;
    void clear_identity_unavailable(llama_cache_acct_artifact_id artifact) noexcept;
    bool admit_checkpoint_ring(
        const server_cache_destruction_target & target,
        bool & saw_soft,
        bool & saw_hard) const noexcept;
    bool admit_scalar_artifact(
        const server_cache_destruction_target & target,
        bool & saw_soft,
        bool & saw_hard) const noexcept;

    server_cache_lease_clock * clock = nullptr;
    server_cache_lease_fallback_provider * fallback = nullptr;
    std::vector<entry> leases;
    std::vector<server_cache_lease_identity_record> identities;
    std::vector<server_cache_lease_subject> identity_unavailable;
    std::array<server_cache_lease_event, SERVER_CACHE_LEASE_EVENT_RING> events = {};
    std::array<uint64_t, size_t(server_cache_lease_event_kind::_count)> event_totals = {};
    uint64_t next_lease_id = 1;
    uint64_t next_identity_id = 1;
    uint64_t next_context_scope_id = 1;
    uint64_t next_event_ordinal = 1;
    uint64_t n_events = 0;
    uint64_t n_event_overflows = 0;
    uint64_t n_clock_samples = 0;
    uint64_t n_unavailable = 0;
    bool available = true;
};

server_cache_destruction_verdict server_cache_lease_evaluate_request(
    void * context,
    const server_cache_destruction_request & request) noexcept;

// Thin range-qualified door used by VBR enforcement and legacy skip guards.
// It delegates to the single lease table; it is not a second evaluator.
bool server_cache_hard_lease_blocks_range(
    const server_cache_lease_table * leases,
    llama_cache_acct_artifact_id artifact,
    const server_cache_lease_identity & identity,
    uint64_t sequence_epoch,
    uint64_t first_token,
    uint64_t token_count) noexcept;
bool server_cache_has_hard_lease(
    const server_cache_lease_table * leases) noexcept;
