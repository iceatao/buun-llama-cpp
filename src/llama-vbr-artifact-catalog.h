#pragma once

#include "llama-vbr-artifact-capture.h"

#include <cstdint>
#include <memory>
#include <vector>

// Internal F2.2 immutable artifact catalog. It is deliberately absent from
// public llama.h and has no server/backend dependency.
enum class llama_vbr_artifact_publish_status : uint8_t {
    published = 0,
    adopted,
    invalid_argument,
    shard_failed,
    duplicate_completion,
    missing_completion,
    format_rejected,
    accounting_unavailable,
    admission_refused,
    stage_failed,
    commit_failed,
    publication_failed,
    internal_error,
    _count,
};

struct llama_vbr_artifact_domain_binding {
    uint32_t topology_index = UINT32_MAX;
    uint16_t device_ordinal = UINT16_MAX;
    llama_cache_acct_resource_domain domain;
};

struct llama_vbr_artifact_publish_result {
    llama_vbr_artifact_publish_status status =
        llama_vbr_artifact_publish_status::internal_error;
    llama_cache_acct_artifact_id reference_artifact;
    llama_cache_acct_content_digest unit_content;
    llama_cache_acct_lineage_id reference_lineage;
};

struct llama_vbr_artifact_catalog_snapshot {
    uint64_t blobs = 0;
    uint64_t stashes = 0;
    uint64_t references = 0;
    uint64_t published = 0;
    uint64_t adopted = 0;
    uint64_t refusals = 0;
    uint64_t staging_overlap_refusals = 0;
};

struct llama_vbr_artifact_reference_tokens {
    llama_cache_acct_artifact_id artifact;
    llama_cache_acct_content_digest unit_content;
    llama_cache_acct_lineage_id lineage;
};

struct llama_vbr_projected_publication_request {
    uint64_t manifest_id = 0;
    std::vector<vbr_artifact_portable_accounting_row> accounting;
    // Subset of unit_payload bytes this manifest must be able to materialize
    // freshly. Other unit rows are zero-reserve placeholders that bind to a
    // preceding manifest's immutable allocation at publication.
    std::vector<vbr_artifact_portable_accounting_row> reserve_accounting;
    bool reserve_accounting_explicit = false;
};

// Catalog-owned split-phase durable-publication fence. Preparation resolves
// portable accounting through this catalog and admits the complete bounded
// row set before projected D2H. The opaque move-only owner is later
// repartitioned onto content-addressed publication leaves; dropping it aborts
// every still-live reservation.
class llama_vbr_projected_publication_claim {
public:
    llama_vbr_projected_publication_claim() noexcept;
    ~llama_vbr_projected_publication_claim();
    llama_vbr_projected_publication_claim(
        const llama_vbr_projected_publication_claim &) = delete;
    llama_vbr_projected_publication_claim & operator=(
        const llama_vbr_projected_publication_claim &) = delete;
    llama_vbr_projected_publication_claim(
        llama_vbr_projected_publication_claim &&) noexcept;
    llama_vbr_projected_publication_claim & operator=(
        llama_vbr_projected_publication_claim &&) noexcept;

    bool ready() const noexcept;
    uint64_t manifest_id() const noexcept;
    const llama_cache_prepare_result & preparation() const noexcept;

private:
    struct impl;
    explicit llama_vbr_projected_publication_claim(
        std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;

    friend class llama_vbr_artifact_catalog;
};

// One pre-D2H fence for a complete projected batch. Physical unit rows are
// reserved once for the batch while every manifest retains its own metadata
// and zero-reserve reference leaves. Dependency shrink repartitions this
// owner in place; only the final sealed assembly partitions it into the
// manifest-local claims consumed by publication.
class llama_vbr_projected_publication_batch_claim {
public:
    llama_vbr_projected_publication_batch_claim() noexcept;
    ~llama_vbr_projected_publication_batch_claim();
    llama_vbr_projected_publication_batch_claim(
        const llama_vbr_projected_publication_batch_claim &) = delete;
    llama_vbr_projected_publication_batch_claim & operator=(
        const llama_vbr_projected_publication_batch_claim &) = delete;
    llama_vbr_projected_publication_batch_claim(
        llama_vbr_projected_publication_batch_claim &&) noexcept;
    llama_vbr_projected_publication_batch_claim & operator=(
        llama_vbr_projected_publication_batch_claim &&) noexcept;

    bool ready() const noexcept;
    uint32_t manifests() const noexcept;
    const llama_cache_prepare_result & preparation() const noexcept;

private:
    struct impl;
    explicit llama_vbr_projected_publication_batch_claim(
        std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;

    friend class llama_vbr_artifact_catalog;
};

enum class vbr_artifact_resolve_status : uint8_t {
    ok = 0,
    not_found,
    busy,
    unavailable,
    internal_error,
    _count,
};

enum class vbr_artifact_retire_status : uint8_t {
    retired = 0,
    busy,
    not_found,
    internal_error,
    _count,
};

struct vbr_artifact_allocation_view {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    uint64_t logical = 0;
    uint64_t resident = 0;
    llama_cache_acct_alloc_id allocation;
    llama_cache_acct_artifact_id artifact;
    llama_cache_acct_content_digest content;
    llama_cache_acct_lineage_id lineage;
};

struct vbr_artifact_unit_view {
    vbr_unit_version_id unit_version_id;
    vbr_payload_digest payload_digest;
    vbr_artifact_unit_descriptor descriptor;
    std::vector<std::shared_ptr<const artifact_segment_chain>> payload_shards;
    std::vector<std::shared_ptr<const artifact_segment_chain>> stash_shards;
    std::vector<vbr_artifact_allocation_view> payload_allocations;
    std::vector<vbr_artifact_allocation_view> stash_allocations;
};

struct vbr_artifact_companion_view {
    vbr_artifact_companion_payload descriptor;
    std::shared_ptr<const artifact_segment_chain> payload;
};

// Reference-local authenticated projection retained with a catalog package.
// The projected unit owns the complete packed shard while this proof names
// the exact byte ranges authorized by one logical manifest.
struct vbr_artifact_projected_range_view {
    uint32_t unit_index = UINT32_MAX;
    uint32_t shard_index = UINT32_MAX;
    vbr_capture_range_proof proof;
};

enum class vbr_projected_manifest_publish_status : uint8_t {
    published = 0,
    adopted,
    dependency_unavailable,
    companion_unavailable,
    metadata_invalid,
    accounting_unavailable,
    admission_refused,
    publication_failed,
    internal_error,
    _count,
};

// Admission and companion evidence for one logical row of an immutable
// projected assembly. Semantic identity, placement, generation, controller
// policy, unit geometry, payload sources, and content IDs come exclusively
// from the sealed assembly. The caller supplies only the portable accounting
// budget and the topology namespace in which those rows are interpreted.
// Keeping this envelope narrow prevents an H2 scheduler from manufacturing a
// mostly-placeholder artifact package that the catalog then overwrites.
struct vbr_projected_manifest_publication {
    uint64_t manifest_id = 0;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_artifact_portable_accounting_row> accounting;
    std::vector<vbr_capture_sealed_companion> companions;
};

struct vbr_projected_manifest_publish_result {
    uint64_t manifest_id = 0;
    vbr_projected_manifest_publish_status status =
        vbr_projected_manifest_publish_status::internal_error;
    llama_vbr_artifact_publish_result publication;
};

struct vbr_projected_batch_publish_diagnostics {
    uint64_t ready_manifests = 0;
    uint64_t published_manifests = 0;
    uint64_t dependency_unavailable = 0;
    uint64_t main_payload_bytes_rehashed = 0;
    uint64_t companion_payload_hash_bytes = 0;
};

class llama_vbr_artifact_catalog;

enum class vbr_artifact_prepared_retire_status : uint8_t {
    retired = 0,
    retired_projection_stale,
    unavailable,
    _count,
};

// Move-only scheduler capability for retiring one or more host-owned catalog
// references through one exact accounting union. Preparation performs every
// allocation and last-reference preview; commit is allocation-free.
class vbr_artifact_prepared_retire {
public:
    vbr_artifact_prepared_retire() noexcept;
    vbr_artifact_prepared_retire(vbr_artifact_prepared_retire &&) noexcept;
    vbr_artifact_prepared_retire & operator=(
        vbr_artifact_prepared_retire &&) noexcept;
    ~vbr_artifact_prepared_retire();

    vbr_artifact_prepared_retire(
        const vbr_artifact_prepared_retire &) = delete;
    vbr_artifact_prepared_retire & operator=(
        const vbr_artifact_prepared_retire &) = delete;

    bool ready() const noexcept;
    const llama_cache_acct_release_set_preview & preview() const noexcept;
    vbr_artifact_prepared_retire_status commit() noexcept;
    void reset() noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class llama_vbr_artifact_catalog;
};

class vbr_artifact_package_view;

struct vbr_artifact_package_set_view {
    const vbr_artifact_package_view * const * data = nullptr;
    size_t size = 0;
};

struct vbr_artifact_retire_resident_preview {
    uint64_t resident = 0;
    bool known = false;
};

// A catalog lease exposes only immutable restore inputs that passed the
// catalog's sealed publication transaction. Resolving or retaining a view is
// therefore an O(metadata) capability operation, not a request to re-read and
// rehash its payload. Explicit import boundaries may still validate bytes.
// The catalog is the single owner and must outlive every view.
class vbr_artifact_package_view {
public:
    vbr_artifact_package_view() = default;
    vbr_artifact_package_view(vbr_artifact_package_view && other) noexcept;
    vbr_artifact_package_view & operator=(vbr_artifact_package_view && other) noexcept;
    ~vbr_artifact_package_view();

    vbr_artifact_package_view(const vbr_artifact_package_view &) = delete;
    vbr_artifact_package_view & operator=(const vbr_artifact_package_view &) = delete;

    explicit operator bool() const noexcept { return owner_ != nullptr; }
    bool same_catalog(const vbr_artifact_package_view & other) const noexcept {
        return owner_ != nullptr && owner_ == other.owner_;
    }
    // Process-local ownership check used by the automatic prompt-cache
    // adapter. A sealed catalog package is already charged in exactly one
    // ledger; logical aliases must never duplicate those physical bytes in a
    // second host-cache accounting transaction.
    bool accounted_by(const llama_cache_acct_ledger * ledger) const noexcept;
    // Transfer the catalog reference's retirement authority to this view.
    // The view already owns one borrow; later aliases retain that immutable
    // view rather than minting another physical reference.
    bool claim_host_ownership() noexcept;
    bool host_owned() const noexcept { return host_owned_; }
    bool prepare_owned_retire(
        const std::vector<const vbr_artifact_package_view *> & packages,
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept;
    bool preview_owned_retire(
        const std::vector<const vbr_artifact_package_view *> & packages,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept;
    bool preview_owned_retire_resident_batch(
        const std::vector<vbr_artifact_package_set_view> & package_sets,
        uint64_t expected_serial,
        std::vector<vbr_artifact_retire_resident_preview> & out) const noexcept;
    bool preview_owned_retire_resident_conditioned_batch(
        vbr_artifact_package_set_view baseline,
        const std::vector<vbr_artifact_package_set_view> & package_sets,
        uint64_t expected_serial,
        std::vector<vbr_artifact_retire_resident_preview> & out) const noexcept;
    llama_cache_acct_artifact_id reference_artifact() const noexcept;
    const std::vector<vbr_artifact_portable_topology> & topologies() const noexcept;
    const vbr_artifact_reference_manifest & manifest() const noexcept;
    const std::vector<vbr_artifact_unit_view> & units() const noexcept;
    const std::vector<vbr_artifact_companion_view> & companions() const noexcept;
    const std::vector<vbr_artifact_projected_range_view> &
        projected_ranges() const noexcept;
    const std::vector<vbr_artifact_allocation_view> &
        reference_allocations() const noexcept;
    vbr_artifact_status validate() const noexcept;
    vbr_artifact_resolve_status retain(
        vbr_artifact_package_view & output) const noexcept;
    void reset() noexcept;

private:
    struct storage;
    friend class llama_vbr_artifact_catalog;
    llama_vbr_artifact_catalog * owner_ = nullptr;
    std::shared_ptr<const storage> storage_;
    bool host_owned_ = false;
};

class llama_vbr_artifact_catalog : public vbr_unit_version_sink {
public:
    explicit llama_vbr_artifact_catalog(llama_cache_acct_ledger & ledger);
    ~llama_vbr_artifact_catalog();

    llama_vbr_artifact_catalog(const llama_vbr_artifact_catalog &) = delete;
    llama_vbr_artifact_catalog & operator=(const llama_vbr_artifact_catalog &) = delete;

    // Bind every portable topology/device ordinal through C's canonical
    // topology interner. Callers use the returned domains in their one-shot
    // completeness manifest and point-in-time budget configuration.
    bool bind_topologies(
        const std::vector<vbr_artifact_portable_topology> & topologies,
        std::vector<llama_vbr_artifact_domain_binding> & bindings) noexcept;

    // Create measured-zero cells for every package accounting row plus the F3
    // temporary leaves in those same capacity domains. This never certifies a
    // producer and never resets a cell already configured by this catalog.
    bool configure_accounting(const vbr_artifact_package & package) noexcept;

    // F3 explicit-capture preparation door: atomically requires/binds the
    // package topology set and creates every measured-zero accounting cell.
    bool prepare_capture_package(
        const vbr_artifact_package & package) noexcept;

    // Prepare the conservative durable rows that a projected batch may
    // publish. Rows must be nonzero, canonical (unique role/domain), and use
    // equal logical/resident bytes. This performs the only budget admission;
    // content dedup later may only repartition the fence downward.
    llama_vbr_projected_publication_claim
    prepare_projected_publication_claim(
        uint64_t manifest_id,
        const std::vector<vbr_artifact_portable_accounting_row> & rows,
        const llama_cache_budget_config & budget) noexcept;

    // Batch form used by automatic capture. The physical unit union is
    // reserved exactly once; later manifest groups carry zero-reserve
    // reference placeholders until the first publication materializes the
    // shared allocations. No second ledger snapshot or capacity decision is
    // permitted after this call succeeds.
    llama_vbr_projected_publication_batch_claim
    prepare_projected_publication_claims(
        const std::vector<llama_vbr_projected_publication_request> & requests,
        const llama_cache_budget_config & budget) noexcept;

    // Replace the admitted batch inventory with an ordered dependency-local
    // subset. Aggregate reserved bytes may only decrease; the claim remains
    // one owner and the scheduler is not re-entered.
    bool shrink_projected_publication_claims(
        llama_vbr_projected_publication_batch_claim & claim,
        const std::vector<llama_vbr_projected_publication_request> & requests)
        noexcept;

    // Final ownership handoff after assembly. Counts and manifest identities
    // are taken from the batch claim, so partial or swapped output is
    // unrepresentable.
    bool partition_projected_publication_claims(
        llama_vbr_projected_publication_batch_claim && claim,
        std::vector<llama_vbr_projected_publication_claim> & output) noexcept;

    // F3.1 streaming path and the abstract F3.2 capture sink entry point.
    std::unique_ptr<vbr_capture_build> begin_capture(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        vbr_capture_stream_status & status,
        vbr_capture_begin_diagnostics * diagnostics =
            nullptr) noexcept override;

    // H1 dependency-scoped publication. Structural assembly corruption or a
    // malformed publication inventory clears all output and returns false.
    // Missing/stale unit or companion evidence is reported per manifest;
    // unaffected rows publish independently. Main payload bytes are never
    // reread: authority comes exclusively from the opaque sealed assembly.
    bool publish_projected_batch(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        const llama_cache_budget_config & budget,
        std::vector<vbr_projected_manifest_publish_result> & output,
        vbr_projected_batch_publish_diagnostics * diagnostics = nullptr,
        const llama_cache_transaction_fault & fault = {})
        noexcept;

    // Prepared-capacity counterpart used only by the automatic store path.
    // Claims are consumed per ready manifest; no capacity sample or second
    // durable admission occurs after D2H.
    bool publish_projected_batch_claimed(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        std::vector<llama_vbr_projected_publication_claim> && claims,
        std::vector<vbr_projected_manifest_publish_result> & output,
        vbr_projected_batch_publish_diagnostics * diagnostics = nullptr,
        const llama_cache_transaction_fault & fault = {})
        noexcept;

    // Automatic-capture form. The intact batch fence reaches catalog
    // normalization so a dependency-local invalid row cannot strand the
    // shared physical reserve in an unusable manifest claim.
    bool publish_projected_batch_claimed(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        llama_vbr_projected_publication_batch_claim && claim,
        std::vector<vbr_projected_manifest_publish_result> & output,
        vbr_projected_batch_publish_diagnostics * diagnostics = nullptr,
        const llama_cache_transaction_fault & fault = {}) noexcept;

    // Release every ledger reference owned by this checkpoint reference.
    // Physical payload/stash bytes discharge only when C observes the last op.
    vbr_artifact_resolve_status resolve_reference(
        llama_cache_acct_artifact_id reference,
        vbr_artifact_package_view & out) noexcept;

    // O(1) authenticity check for an already host-owned immutable view. A
    // host owner deliberately cannot be resolved into a second borrow; this
    // predicate lets the trusted scheduler return that same capability to
    // its originating catalog without reopening ownership.
    bool owns_host_package(
        const vbr_artifact_package_view & package) const noexcept;
    // Scheduler-only handoff for references returned by one just-completed
    // projected publication. The input must be nonempty and unique, and the
    // output must be empty. Allocation or structural failure leaves every
    // reference unborrowed and unowned; success materializes the complete
    // immutable views and transfers host retirement authority atomically.
    bool claim_fresh_host_batch(
        const std::vector<llama_cache_acct_artifact_id> & references,
        std::vector<vbr_artifact_package_view> & output) noexcept;
    vbr_artifact_retire_status retire(
        llama_cache_acct_artifact_id reference) noexcept;
    // Allocation-free compensating terminal for a reference that was just
    // published but could not be handed to its intended owner. The caller
    // must hold no view and the reference must not be host-owned.
    vbr_artifact_retire_status discard_unowned_reference(
        llama_cache_acct_artifact_id reference) noexcept;

    bool reference_tokens(
        llama_cache_acct_artifact_id reference,
        llama_vbr_artifact_reference_tokens & out) const noexcept;
    llama_vbr_artifact_catalog_snapshot snapshot() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;

    llama_vbr_artifact_publish_result publish_stream(
        const vbr_artifact_package & package,
        const std::vector<vbr_verified_segment> & segments,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        void * prepared_stream_state) noexcept;

    llama_vbr_artifact_publish_result publish_stream_complete(
        vbr_artifact_package package,
        const std::vector<vbr_verified_segment> & segments,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        void * prepared_stream_state,
        bool sealed_projected = false,
        const std::vector<vbr_artifact_projected_range_view> *
            projected_ranges = nullptr,
        uint64_t * payload_bytes_rehashed = nullptr,
        llama_vbr_projected_publication_claim * projected_claim = nullptr)
        noexcept;

    bool publish_projected_batch_impl(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        const llama_cache_budget_config & budget,
        std::vector<vbr_projected_manifest_publish_result> & output,
        vbr_projected_batch_publish_diagnostics * diagnostics,
        const llama_cache_transaction_fault & fault,
        std::vector<llama_vbr_projected_publication_claim> * claims,
        llama_vbr_projected_publication_batch_claim * batch_claim) noexcept;

    std::unique_ptr<vbr_capture_build> begin_capture_impl(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        bool charge_transfer_staging,
        vbr_capture_stream_status & status,
        vbr_capture_begin_diagnostics * diagnostics) noexcept;

    friend class llama_vbr_artifact_catalog_stream_build;
    friend class vbr_artifact_package_view;
    friend class vbr_artifact_prepared_retire;
    vbr_artifact_resolve_status materialize_reference_locked(
        llama_cache_acct_artifact_id reference,
        std::shared_ptr<const vbr_artifact_package_view::storage> & output);
    bool accounted_by(const llama_cache_acct_ledger * ledger) const noexcept;
    bool claim_host_ownership(
        llama_cache_acct_artifact_id reference) noexcept;
    bool prepare_owned_retire(
        const std::vector<llama_cache_acct_artifact_id> & references,
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) noexcept;
    bool preview_owned_retire(
        const std::vector<llama_cache_acct_artifact_id> & references,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept;
    bool preview_owned_retire_resident_batch(
        const std::vector<vbr_artifact_package_set_view> & package_sets,
        uint64_t expected_serial,
        std::vector<vbr_artifact_retire_resident_preview> & out) const noexcept;
    bool preview_owned_retire_resident_conditioned_batch(
        vbr_artifact_package_set_view baseline,
        const std::vector<vbr_artifact_package_set_view> & package_sets,
        uint64_t expected_serial,
        std::vector<vbr_artifact_retire_resident_preview> & out) const noexcept;
    vbr_artifact_prepared_retire_status commit_owned_retire(
        uint64_t token,
        const std::vector<llama_cache_acct_artifact_id> & references,
        const std::vector<vbr_unit_version_id> & unit_ids,
        const std::vector<vbr_stash_payload_id> & stash_ids,
        llama_cache_prepared_release_set & release) noexcept;
    void cancel_owned_retire(
        uint64_t token,
        const std::vector<llama_cache_acct_artifact_id> & references,
        const std::vector<vbr_unit_version_id> & unit_ids,
        const std::vector<vbr_stash_payload_id> & stash_ids,
        llama_cache_prepared_release_set & release) noexcept;
    void release_reference_lease(
        llama_cache_acct_artifact_id reference,
        bool host_owned) noexcept;
};

const char * llama_vbr_artifact_publish_status_name(
    llama_vbr_artifact_publish_status status) noexcept;
