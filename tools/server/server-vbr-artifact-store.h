#pragma once

#include "../../src/llama-cache-authority.h"
#include "../../src/llama-vbr-artifact-adopt.h"
#include "../../src/llama-vbr-artifact-catalog.h"
#include "../../src/llama-vbr-explicit-capture.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class server_prompt_cache_vbr_payload;
struct server_vbr_artifact_store_test_door;

enum class server_vbr_artifact_capture_status : uint8_t {
    ok = 0,
    unsupported,
    unavailable,
    invalid_slot,
    slot_processing,
    stale_frontier,
    identity_unavailable,
    unauthorized,
    required_companion_unavailable,
    admission_refused,
    cancelled,
    source_changed,
    internal_error,
    _count,
};

const char * server_vbr_artifact_capture_status_name(
    server_vbr_artifact_capture_status status) noexcept;

enum class server_vbr_artifact_import_status : uint8_t {
    ok = 0,
    unsupported,
    not_found,
    invalid_slot,
    slot_processing,
    slot_not_empty,
    validation_failed,
    report_only,
    stage_failed,
    adopt_failed,
    unavailable,
    internal_error,
    _count,
};

const char * server_vbr_artifact_import_status_name(
    server_vbr_artifact_import_status status) noexcept;

// Pure route/validator classifiers shared with the scheduler and CPU tests.
// `ok` means the caller may continue; no state is mutated here.
server_vbr_artifact_import_status server_vbr_artifact_import_route_precheck(
    bool store_available,
    bool slot_exists,
    bool slot_processing,
    bool target_available,
    bool slot_empty) noexcept;
server_vbr_artifact_import_status
server_vbr_artifact_import_validation_disposition(
    vbr_manifest_validation_status status,
    vbr_import_decision decision) noexcept;

enum class server_vbr_artifact_store_create_failure : uint8_t {
    none = 0,
    ledger_missing,
    budget_sampler_missing,
    topology_missing,
    pool_binding_missing,
    lane_missing,
    attention_child_missing,
    ring_size_invalid,
    chunk_size_invalid,
    budget_sample_failed,
    ring_create_failed,
    internal_error,
    _count,
};

const char * server_vbr_artifact_store_create_failure_name(
    server_vbr_artifact_store_create_failure failure) noexcept;

struct server_vbr_artifact_store_create_diagnostics {
    server_vbr_artifact_store_create_failure failure =
        server_vbr_artifact_store_create_failure::none;
    vbr_capture_stream_status ring_status =
        vbr_capture_stream_status::_count;
    vbr_capture_ring_create_failure ring_failure =
        vbr_capture_ring_create_failure::none;
    uint64_t requested_ring_bytes = 0;
    uint64_t attempted_ring_bytes = 0;
    uint64_t constructed_ring_bytes = 0;
    size_t chunk_bytes = 0;
    size_t lane_count = 0;
    uint32_t attention_children = 0;
};

struct server_vbr_artifact_store_config {
    using sample_budget_fn = bool (*)(
        void * context,
        llama_cache_budget_config & output) noexcept;

    llama_cache_acct_ledger * ledger = nullptr;
    llama_cache_acct_resource_domain pinned_domain;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    std::vector<vbr_capture_lane> lanes;
    uint32_t attention_children = 0;
    uint64_t ring_bytes = 0;
    size_t chunk_bytes = 0;
    // Baked-mean registry ID of the served model, for the representation identity digest.
    int turbo_meansub_id = 0;
    void * budget_context = nullptr;
    sample_budget_fn sample_budget = nullptr;
};

// Observe the artifact machinery's empty capacity rows after the one-shot
// manifest and before that domain's producer is certified. Device-scoped live
// rows are deliberately left to the live-memory observer.
bool server_vbr_artifact_store_observe_empty_accounting(
    llama_cache_acct_ledger & ledger,
    const llama_cache_acct_resource_domain & domain) noexcept;

// Configuration-owner hook for the store's dedicated pinned-host domain.
// No other producer has pinned state, so this composes empty observation,
// retention-sidecar certification, and the exact-domain proof.
bool server_vbr_artifact_store_configure_pinned_accounting(
    llama_cache_acct_ledger & ledger,
    const llama_cache_acct_resource_domain & domain) noexcept;

// Prove that every budget-participating cell the capture can price is known
// and certified in each exact topology-qualified domain. This is read-only and
// runs after the ordinary host/live producers have certified their rows.
bool server_vbr_artifact_store_verify_accounting(
    llama_cache_acct_ledger & ledger,
    const std::vector<llama_cache_acct_resource_domain> & domains) noexcept;

struct server_vbr_artifact_capture_output {
    server_vbr_artifact_capture_status status =
        server_vbr_artifact_capture_status::internal_error;
    vbr_explicit_capture_status library_status =
        vbr_explicit_capture_status::internal_error;
    vbr_explicit_capture_phase phase =
        vbr_explicit_capture_phase::validation;
    vbr_capture_stream_status inner_stream_status =
        vbr_capture_stream_status::_count;
    vbr_explicit_generation_failure generation_failure =
        vbr_explicit_generation_failure::none;
    vbr_explicit_size_failure size_failure =
        vbr_explicit_size_failure::none;
    vbr_capture_begin_diagnostics begin_diagnostics;
    std::string reference;
    vbr_artifact_consistency_kind consistency =
        vbr_artifact_consistency_kind::capture_exact;
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
};

// Scheduler-facing terminal for one row of an immutable projected capture
// assembly. A published/adopted row is usable only when payload is non-null;
// that owner holds the catalog reference and retires it when the last host
// alias is released. Other statuses are dependency-scoped soft failures.
struct server_vbr_projected_host_publish_result {
    uint64_t manifest_id = 0;
    vbr_projected_manifest_publish_status status =
        vbr_projected_manifest_publish_status::internal_error;
    std::shared_ptr<const server_prompt_cache_vbr_payload> payload;
};

struct server_vbr_projected_host_publish_diagnostics {
    vbr_projected_batch_publish_diagnostics catalog;
    uint64_t host_payloads_retained = 0;
    uint64_t postpublish_retirements = 0;
};

// Scheduler-visible accounting for one automatic projected capture. The
// immutable assembly and move-only publication capabilities never escape the
// store: this record carries only scalar evidence plus the catalog handoff
// outcome.
struct server_vbr_projected_host_capture_diagnostics {
    vbr_explicit_capture_status capture_status =
        vbr_explicit_capture_status::internal_error;
    vbr_explicit_capture_phase capture_phase =
        vbr_explicit_capture_phase::validation;
    vbr_capture_stream_status inner_stream_status =
        vbr_capture_stream_status::_count;
    uint64_t source_namespace = 0;
    uint64_t union_cells = 0;
    uint64_t planned_packed_bytes = 0;
    uint32_t size_pass_calls = 0;
    uint32_t projection_calls = 0;
    uint32_t unit_transfer_calls = 0;
    uint32_t transferred_units = 0;
    uint64_t companion_d2h_bytes = 0;
    uint64_t companion_d2h_reads = 0;
    uint32_t ring_operation_attempts = 0;
    uint32_t ring_operation_acquires = 0;
    uint32_t ring_operation_refusals = 0;
    enum class staging_status : uint8_t {
        not_called,
        zero_work_admitted,
        scheduler_refused,
        budget_failed,
        preparation_refused,
        prepared,
        invalid_quote,
        _count,
    } staging = staging_status::not_called;
    llama_cache_prepare_status staging_prepare_status =
        llama_cache_prepare_status::invalid_argument;
    llama_cache_admission_status staging_admission_status =
        llama_cache_admission_status::internal_fault;
    size_t staging_failed_leaf = SIZE_MAX;
    bool staging_reserved = false;
    vbr_capture_stream_stats transfer;
    server_vbr_projected_host_publish_diagnostics publication;
};

// Scheduler-owned last-mile admission at the exact pre-D2H boundary. The
// store holds the exact transfer-staging claim while this callback runs; the
// callback can only accept or refuse the immutable scalar quote. It must not
// retain the quote or mutate source memory.
struct server_vbr_projected_capture_admission {
    using quote = vbr_projected_capture_batch_request::pretransfer_quote;
    void * context = nullptr;
    bool (*admit)(void * context, const quote & quote) noexcept = nullptr;
    // Optional continuation probe after admission. False cancels the batch
    // between bounded recurrent writes or attention ring chunks.
    bool (*continue_transfer)(void * context) noexcept = nullptr;
};

struct server_vbr_artifact_store_counters {
    uint64_t requested = 0;
    uint64_t exact_published = 0;
    uint64_t refused = 0;
    uint64_t unavailable = 0;
    uint64_t internal_error = 0;
    uint64_t payload_bytes = 0;
    uint64_t stash_bytes = 0;
    uint64_t companion_bytes = 0;
    uint64_t pinned_bytes = 0;
    uint64_t chunks = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t dedup_hits = 0;
    uint64_t dedup_misses = 0;
    uint64_t staging_overlap_refusals = 0;
    std::array<uint64_t,
        size_t(vbr_explicit_capture_status::_count)> capture_outcomes = {};
    uint64_t imports_requested = 0;
    // Trusted cache owners that passed exact catalog/storage authentication
    // and entered the common import kernel (even when target validation later
    // refuses). This keeps the credential-free authority boundary observable.
    uint64_t host_imports_authenticated = 0;
    uint64_t host_imports_succeeded = 0;
    uint64_t imports_succeeded = 0;
    uint64_t imports_report_only = 0;
    uint64_t imports_not_found = 0;
    uint64_t imports_refused = 0;
    uint64_t imports_unavailable = 0;
    std::array<uint64_t, size_t(vbr_import_decision::_count)>
        import_decisions = {};
    std::array<uint64_t, size_t(vbr_manifest_validation_status::_count)>
        validation_outcomes = {};
};

struct server_vbr_artifact_import_target {
    using prepare_publish_fn = bool (*)(
        void * context,
        const std::vector<llama_token> & tokens,
        uint64_t sequence_epoch) noexcept;
    using publish_fn = void (*)(void * context) noexcept;

    llama_memory_i * memory = nullptr;
    llama_seq_id destination = -1;
    std::string execution_identity;
    std::string adapter_config_identity;
    bool previously_observed = false;
    void * publish_context = nullptr;
    prepare_publish_fn prepare_publish = nullptr;
    publish_fn publish = nullptr;
};

struct server_vbr_artifact_import_request : server_vbr_artifact_import_target {
    std::string reference;
    std::string tenant_key;
};

struct server_vbr_artifact_import_output {
    server_vbr_artifact_import_status status =
        server_vbr_artifact_import_status::internal_error;
    vbr_manifest_validation_status validation_status =
        vbr_manifest_validation_status::internal_error;
    vbr_adopt_stage_status stage_status =
        vbr_adopt_stage_status::internal_error;
    vbr_downward_reserve_status downward_reserve_status =
        vbr_downward_reserve_status::not_attempted;
    vbr_adopt_status adopt_status = vbr_adopt_status::internal_error;
    bool adopt_attempted = false;
    vbr_adopt_phase phase = vbr_adopt_phase::consume_capabilities;
    vbr_downward_adopt_subphase downward_subphase =
        vbr_downward_adopt_subphase::none;
    uint32_t downward_edge = UINT32_MAX;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_chunks = 0;
    uint64_t rollback_count = 0;
    vbr_import_decision decision = vbr_import_decision::reject;
    vbr_artifact_consistency_kind consistency =
        vbr_artifact_consistency_kind::live_rebased;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t companion_bytes = 0;
};

// A lower-precision variant may be useful only while negotiation is still
// representation-dependent and before adoption has begun moving payload.
bool server_vbr_artifact_import_variant_fallback_safe(
    const server_vbr_artifact_import_output & output) noexcept;

// Server-internal opaque-reference authorization index. It exposes only one
// indistinguishable miss result; there is no enumeration or tenant-agnostic
// lookup door.
class server_vbr_artifact_reference_index {
public:
    bool publish(
        std::string reference,
        std::string tenant_key,
        llama_cache_acct_artifact_id artifact) noexcept;
    bool authorize(
        const std::string & reference,
        const std::string & tenant_key,
        llama_cache_acct_artifact_id & artifact) const noexcept;

private:
    struct binding {
        std::string tenant_key;
        llama_cache_acct_artifact_id artifact;
    };
    std::map<std::string, binding> entries_;
};

// F3.3 server owner for the internal catalog/ring. Explicit control APIs
// return and resolve tenant-bound opaque handles only through the exact
// capture-time tenant key. The trusted scheduler path below instead receives
// typed host owners directly; neither path exposes catalog enumeration or a
// tenant-agnostic control lookup.
class server_vbr_artifact_store {
public:
    static std::unique_ptr<server_vbr_artifact_store> create(
        const server_vbr_artifact_store_config & config,
        server_vbr_artifact_capture_status & status,
        server_vbr_artifact_store_create_diagnostics * diagnostics =
            nullptr) noexcept;

    ~server_vbr_artifact_store();
    server_vbr_artifact_store(const server_vbr_artifact_store &) = delete;
    server_vbr_artifact_store & operator=(
        const server_vbr_artifact_store &) = delete;

    server_vbr_artifact_capture_output capture(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        const std::string & tenant_key) noexcept;

    // Publish an already sealed projected assembly through this store's
    // canonical catalog and current sampled budget. Structural failure clears
    // the complete output. Successfully published rows are returned as
    // cache-owned typed payloads directly; no tenant reference is minted.
    bool publish_projected_host_batch(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        std::vector<server_vbr_projected_host_publish_result> & output,
        server_vbr_projected_host_publish_diagnostics * diagnostics = nullptr)
        noexcept;

    // Trusted scheduler composition of the H2 capture coordinator and the
    // catalog handoff above. Runtime transport/topology/representation
    // authority remains store-owned; callers provide only bounded semantic
    // manifests and the admitted aggregate payload runway.
    bool capture_projected_host_batch(
        llama_memory_i & memory,
        std::vector<vbr_projected_capture_manifest_request> manifests,
        uint64_t max_packed_bytes,
        std::vector<server_vbr_projected_host_publish_result> & output,
        const server_vbr_projected_capture_admission * admission = nullptr,
        server_vbr_projected_host_capture_diagnostics * diagnostics = nullptr)
        noexcept;

    server_vbr_artifact_import_output import(
        server_vbr_artifact_import_request request) noexcept;

    // Trusted scheduler import of a cache-owned immutable package. The shared
    // owner pins catalog storage for the complete validate/stage/adopt
    // transaction; no tenant handle is minted or resolved.
    server_vbr_artifact_import_output import_host_payload(
        server_vbr_artifact_import_target request,
        std::shared_ptr<const server_prompt_cache_vbr_payload> payload)
        noexcept;

    // Scheduler-only E1 resolver. Authorization is identical to import and
    // the returned move-only package is the durable catalog pin. No raw
    // artifact lookup or tenant-agnostic enumeration is exposed.
    bool resolve_control_reference(
        const std::string & reference,
        const std::string & tenant_key,
        vbr_artifact_package_view & package) noexcept;

    // H1 host-cache adapter. Authorization is identical to explicit import;
    // the catalog's sealed publication is the validation proof, so retaining
    // the immutable capability performs no payload read or rehash. The
    // returned shared owner holds one catalog borrow for all host aliases.
    bool retain_host_payload(
        const std::string & reference,
        const std::string & tenant_key,
        std::shared_ptr<const server_prompt_cache_vbr_payload> & payload)
        noexcept;

    const server_vbr_artifact_store_counters & counters() const noexcept;
    uint32_t attention_children() const noexcept;

private:
    server_vbr_artifact_import_output import_package(
        server_vbr_artifact_import_target request,
        const vbr_artifact_package_view & package) noexcept;
    struct impl;
    explicit server_vbr_artifact_store(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
    friend struct server_vbr_artifact_store_test_door;
};

// Model-free production-wiring probe. It exposes the exact transport portion
// of the same stage policy used by import(), without publishing a second raw
// core or transport construction API.
struct server_vbr_artifact_store_test_door {
    struct projected_staging_lifecycle_result {
        bool initial_admitted = false;
        bool shrink_admitted = false;
        bool growth_refused = false;
        bool live_at_publication = false;
        uint32_t scheduler_calls = 0;
        uint32_t budget_samples = 0;
        server_vbr_projected_host_capture_diagnostics::staging_status
            staging = server_vbr_projected_host_capture_diagnostics::
                staging_status::not_called;
        llama_cache_prepare_result preparation;
        llama_cache_acct_snapshot initial;
        llama_cache_acct_snapshot scheduler;
        llama_cache_acct_snapshot shrunk;
        llama_cache_acct_snapshot publication;
        llama_cache_acct_snapshot after;
    };

    static bool import_transport_policy(
        const server_vbr_artifact_store & store,
        vbr_adopt_stage_policy & policy) noexcept;
    static void fail_projected_host_adoption_once(
        server_vbr_artifact_store & store) noexcept;
    static bool projected_staging_lifecycle(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        const vbr_projected_capture_batch_request::pretransfer_quote & initial,
        const vbr_projected_capture_batch_request::pretransfer_quote & shrink,
        const vbr_projected_capture_batch_request::pretransfer_quote & growth,
        projected_staging_lifecycle_result & result) noexcept;
    static bool projected_staging_initial(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        const vbr_projected_capture_batch_request::pretransfer_quote & quote,
        bool scheduler_accept,
        projected_staging_lifecycle_result & result) noexcept;
};
