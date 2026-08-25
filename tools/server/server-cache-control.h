#pragma once

#include "../../common/common-cache-family.h"
#include "server-cache-lease.h"
#include "server-retention-sidecar.h"

#include <array>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

class server_vbr_artifact_store;

enum class server_cache_control_status : uint8_t {
    ok = 0,
    invalid_request,
    not_supported,
    not_found,
    identity_unavailable,
    subject_busy,
    fallback_unavailable,
    fallback_invalid,
    hard_lease_blocked,
    lease_conflict,
    lease_expired,
    partially_stale,
    subject_lost,
    orphaned,
    already_released,
    profile_unfitted,
    capacity_refused,
    stale_capability,
    internal_fault,
    _count,
};

const char * server_cache_control_status_name(
    server_cache_control_status status) noexcept;

// Scheduler task-door precheck. Cache control's two-copy guarantee relies on lifecycle
// publication/floor enforcement; debug-only authority is observability and
// must refuse rather than construct a lease whose pin its erasers ignore.
server_cache_control_status server_cache_control_task_precheck(
    bool request_present,
    bool lifecycle_available,
    bool substrate_available) noexcept;

struct server_cache_control_token {
    uint64_t high = 0;
    uint64_t low = 0;
    explicit operator bool() const noexcept { return high != 0 && low != 0; }
};

inline bool operator==(
        server_cache_control_token a,
        server_cache_control_token b) noexcept {
    return a.high == b.high && a.low == b.low;
}

enum class server_cache_control_subject_kind : uint8_t {
    live_prefix = 0,
    host_snapshot,
    vbr_reference,
    live_checkpoint, // closed v1 rejection for both subjects and fallbacks.
    _count,
};

enum class server_cache_control_operation : uint8_t {
    holder_create = 0,
    holder_close,
    holder_reattach,
    family_register,
    family_bind,
    lease_acquire,
    lease_inspect,
    lease_renew,
    lease_release,
    events,
    _count,
};

struct server_cache_control_route_spec {
    std::string_view path;
    server_cache_control_operation operation;
};

inline constexpr std::array<server_cache_control_route_spec, 10>
        SERVER_CACHE_CONTROL_ROUTES {{
    { "/cache/holders/create", server_cache_control_operation::holder_create },
    { "/cache/holders/reattach", server_cache_control_operation::holder_reattach },
    { "/cache/holders/close", server_cache_control_operation::holder_close },
    { "/cache/families/register", server_cache_control_operation::family_register },
    { "/cache/families/bind", server_cache_control_operation::family_bind },
    { "/cache/leases/acquire", server_cache_control_operation::lease_acquire },
    { "/cache/leases/inspect", server_cache_control_operation::lease_inspect },
    { "/cache/leases/renew", server_cache_control_operation::lease_renew },
    { "/cache/leases/release", server_cache_control_operation::lease_release },
    { "/cache/events/query", server_cache_control_operation::events },
}};

inline bool server_cache_control_operation_for_path(
        std::string_view path,
        server_cache_control_operation & out) noexcept {
    for (const auto & route : SERVER_CACHE_CONTROL_ROUTES) {
        if (path.size() >= route.path.size() &&
                path.compare(path.size() - route.path.size(),
                             route.path.size(), route.path) == 0) {
            out = route.operation;
            return true;
        }
    }
    out = server_cache_control_operation::_count;
    return false;
}

inline bool server_cache_control_is_route(std::string_view path) noexcept {
    server_cache_control_operation ignored;
    return server_cache_control_operation_for_path(path, ignored);
}

enum class server_cache_control_protection_state : uint8_t {
    current = 0,
    partially_stale,
    subject_lost,
    orphaned,
    released,
    _count,
};

struct server_cache_control_selector {
    server_cache_control_subject_kind kind =
        server_cache_control_subject_kind::live_prefix;
    // This authority is scheduler-internal. The wire layer converts semantic selectors into
    // this exact association; raw keys never cross the HTTP boundary.
    server_retention_instance_key retention_key;
    std::string reference;
    std::string tenant_key;
    server_cache_lease_identity identity;
    server_cache_lease_frontier frontier;
};

struct server_cache_control_request {
    server_cache_control_token holder;
    server_cache_control_token recovery;
    server_cache_control_token lease;
    server_cache_control_token family;
    server_cache_control_token family_binding;
    common_cache_family_role family_role = common_cache_family_role::_count;
    std::string family_label;
    // The wire layer supplies a bounded client idempotency digest. Zero is allowed only
    // for scheduler-internal tests and receives no response-loss replay.
    uint64_t idempotency_key = 0;
    server_cache_lease_class requested_class = server_cache_lease_class::soft;
    uint64_t ttl_ns = 0;
    uint64_t after_ordinal = 0;
    uint32_t event_limit = 32;
    bool allow_soft_fallback = false;
    server_cache_control_selector subject;
    server_cache_control_selector fallback;
};

enum class server_cache_control_event_kind : uint8_t {
    grant = 0,
    refuse,
    renew,
    expire,
    release,
    _count,
};

struct server_cache_control_event_view {
    uint64_t ordinal = 0;
    uint64_t timestamp_ms = 0;
    server_cache_control_event_kind kind =
        server_cache_control_event_kind::refuse;
    server_cache_control_status status = server_cache_control_status::ok;
    server_cache_lease_class cls = server_cache_lease_class::none;
    server_cache_control_subject_kind subject_kind =
        server_cache_control_subject_kind::_count;
    common_cache_family_role family_role = common_cache_family_role::_count;
    server_cache_control_token lease;
};

struct server_cache_control_lease_summary {
    server_cache_control_token lease;
    server_cache_control_subject_kind subject_kind =
        server_cache_control_subject_kind::_count;
    server_cache_lease_frontier proven_frontier;
};

struct server_cache_control_family_summary {
    server_cache_control_token family;
    std::string label;
};

struct server_cache_control_result {
    server_cache_control_status status =
        server_cache_control_status::internal_fault;
    server_cache_control_token holder;
    server_cache_control_token holder_recovery;
    server_cache_control_token lease;
    server_cache_control_token family;
    server_cache_control_token family_binding;
    // Scheduler-internal resolved value. The wire layer serializes only opaque handles.
    common_cache_family_binding cache_family;
    std::string family_label;
    server_cache_lease_class granted_class = server_cache_lease_class::none;
    server_cache_control_subject_kind subject_kind =
        server_cache_control_subject_kind::_count;
    server_cache_control_protection_state protection =
        server_cache_control_protection_state::released;
    server_cache_lease_frontier lease_frontier;
    server_cache_lease_frontier proven_frontier;
    uint64_t expires_at_ns = 0;
    uint32_t max_leases = 0;
    bool events_overflowed = false;
    bool protected_bytes_known = false;
    bool fallback_pinned_bytes_known = false;
    uint64_t protected_bytes = 0;
    uint64_t fallback_pinned_bytes = 0;
    bool shared_fallback = false;
    server_cache_control_subject_kind fallback_kind =
        server_cache_control_subject_kind::_count;
    std::vector<server_cache_control_event_view> events;
    std::vector<server_cache_control_lease_summary> orphaned_leases;
    std::vector<server_cache_control_family_summary> families;
};

enum class server_cache_control_handle_kind : uint8_t {
    holder = 0,
    holder_recovery,
    lease,
    family,
    family_binding,
};

std::string server_cache_control_encode_handle(
    server_cache_control_handle_kind kind,
    server_cache_control_token token);
bool server_cache_control_decode_handle(
    server_cache_control_handle_kind kind,
    const std::string & text,
    server_cache_control_token & out) noexcept;
uint64_t server_cache_control_idempotency_digest(
    const std::string & text) noexcept;
bool server_cache_control_request_field_allowed(
    server_cache_control_operation operation,
    std::string_view field) noexcept;
bool server_cache_control_selector_field_allowed(
    server_cache_control_subject_kind kind,
    std::string_view field) noexcept;
const char * server_cache_control_subject_kind_name(
    server_cache_control_subject_kind kind) noexcept;
bool server_cache_control_parse_subject_kind(
    std::string_view name,
    server_cache_control_subject_kind & out) noexcept;
const char * server_cache_control_lease_class_name(
    server_cache_lease_class value) noexcept;
bool server_cache_control_parse_lease_class(
    std::string_view name,
    server_cache_lease_class & out) noexcept;
const char * server_cache_control_family_role_name(
    common_cache_family_role value) noexcept;
bool server_cache_control_parse_family_role(
    std::string_view name,
    common_cache_family_role & out) noexcept;
// Pure common-field parser. It validates the operation allowlist before any
// semantic selector tokenization; the scheduler-thread route then resolves
// only the already-authorized selector fields.
server_cache_control_status server_cache_control_prepare_request(
    server_cache_control_operation operation,
    const nlohmann::ordered_json & body,
    server_cache_control_request & out) noexcept;
nlohmann::ordered_json server_cache_control_json(
    server_cache_control_operation operation,
    const server_cache_control_result & result);

class server_cache_control_token_source {
public:
    virtual ~server_cache_control_token_source() = default;
    virtual bool next(server_cache_control_token & out) noexcept = 0;
};

struct server_cache_control_config {
    using refresh_subject_fn = bool (*)(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier) noexcept;
    using resolve_vbr_fn = server_cache_control_status (*)(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_subject & subject,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier,
        server_cache_durable_fallback_proof & pin) noexcept;
    using acquire_host_proof_fn = server_cache_durable_fallback_proof (*)(
        void * context,
        const server_cache_control_selector & selector) noexcept;
    using selector_evidence_fn = bool (*)(
        void * context,
        const server_cache_control_selector & selector,
        uint64_t & bytes,
        bool & shared) noexcept;
    server_cache_lease_table * leases = nullptr;
    server_retention_sidecar_store * retention = nullptr;
    server_vbr_artifact_store * artifacts = nullptr;
    server_cache_lease_clock * clock = nullptr;
    server_cache_control_token_source * tokens = nullptr;
    void * refresh_context = nullptr;
    refresh_subject_fn refresh_subject = nullptr;
    void * resolve_vbr_context = nullptr;
    resolve_vbr_fn resolve_vbr = nullptr;
    void * host_proof_context = nullptr;
    acquire_host_proof_fn acquire_host_proof = nullptr;
    void * selector_evidence_context = nullptr;
    selector_evidence_fn selector_evidence = nullptr;
    size_t max_holders = 64;
    size_t max_leases = 1024;
    size_t max_families = 1024;
    size_t max_family_bindings = 4096;
    // Model-free allocation-fault seams. Production must leave both defaults;
    // production code must never assign them outside tests.
    size_t test_fail_note_after = std::numeric_limits<size_t>::max();
    bool test_fail_remember = false;
};

// Scheduler-owned cache-control authority. It is also the lease table's one fallback
// provider: a proof is staged only while one scheduler transaction calls the
// existing grant/renew door, then consumed exactly once by acquire().
class server_cache_control_authority final :
        private server_cache_lease_fallback_provider {
public:
    explicit server_cache_control_authority(
        const server_cache_control_config & config) noexcept;
    ~server_cache_control_authority();
    server_cache_control_authority(const server_cache_control_authority &) = delete;
    server_cache_control_authority & operator=(
        const server_cache_control_authority &) = delete;

    server_cache_control_result execute(
        server_cache_control_operation operation,
        const server_cache_control_request & request) noexcept;
    // Completion launch resolves the opaque binding on the scheduler thread.
    // Closed/expired holders and unknown handles are indistinguishable misses.
    server_cache_control_status resolve_family_binding(
        server_cache_control_token token,
        common_cache_family_binding & out) noexcept;
    void lifecycle_point() noexcept;
    bool available() const noexcept;

private:
    struct impl;
    server_cache_durable_fallback_proof acquire(
        const server_cache_lease_subject & subject,
        const server_cache_lease_identity & identity) noexcept override;
    std::unique_ptr<impl> state_;
};
