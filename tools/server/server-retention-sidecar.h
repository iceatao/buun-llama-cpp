#pragma once

#include "common-retention-sidecar.h"
#include "server-cache-lease.h"
#include "../../src/llama-cache-accounting.h"

#include <memory>
#include <unordered_map>
#include <vector>

constexpr size_t SERVER_RETENTION_MAX_CANDIDATES = 8192;

struct common_prompt_checkpoint;
struct server_prompt_cache_state;
class server_cache_recovery_pin;

struct server_retention_instance_key {
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    int32_t owner_slot = -1;
    uintptr_t instance = 0;

    static server_retention_instance_key for_slot(int32_t slot_id) noexcept {
        return {
            common_retention_artifact_kind::live_slot,
            slot_id,
            uintptr_t(slot_id) + 1,
        };
    }

    static server_retention_instance_key for_checkpoint(
            int32_t owner_slot,
            const common_prompt_checkpoint * checkpoint) noexcept {
        return {
            common_retention_artifact_kind::checkpoint,
            owner_slot,
            reinterpret_cast<uintptr_t>(checkpoint),
        };
    }

    static server_retention_instance_key for_host_entry(
            const server_prompt_cache_state * entry) noexcept {
        return {
            common_retention_artifact_kind::host_entry,
            -1,
            reinterpret_cast<uintptr_t>(entry),
        };
    }
};

inline bool operator==(
        const server_retention_instance_key & a,
        const server_retention_instance_key & b) {
    return a.kind == b.kind &&
           a.owner_slot == b.owner_slot &&
           a.instance == b.instance;
}

struct server_retention_instance_key_hash {
    size_t operator()(const server_retention_instance_key & key) const noexcept;
};

enum class server_retention_candidate_availability : uint8_t {
    available = 0,
    in_flight_mutation,
    backing_missing_or_stale,
    _count,
};

struct server_retention_candidate {
    llama_cache_acct_artifact_id artifact_id;
    server_retention_instance_key instance_key;
    common_retention_artifact_record record;
    common_retention_lineage_record lineage;
    // The descriptor charge is provenance only. It is excluded from the payload
    // budget and must never be credited as eviction yield.
    llama_cache_acct_op_id provenance_op;
    // Exact physical release ownership. Host rows are joined by the
    // backing resolver; live checkpoints carry it in the catalog itself.
    std::vector<llama_cache_acct_op_id> release_ops;
    server_retention_candidate_availability avail =
        server_retention_candidate_availability::backing_missing_or_stale;
};

// Allocation-free retention-value view. Pressure observation needs only immutable
// score/lineage scalars; it must not copy accounting-operation vectors or the
// shared turn table while a request is making room.
struct server_retention_value_snapshot {
    llama_cache_acct_artifact_id artifact_id;
    server_retention_instance_key instance_key;
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    common_retention_stamp stamp;
    common_retention_lineage_record lineage;
    uint64_t external_shared_coverage_tokens = 0;
};

enum class server_retention_value_snapshot_status : uint8_t {
    complete = 0,
    overflow,
    unavailable,
};

struct server_retention_value_snapshot_result {
    server_retention_value_snapshot_status status =
        server_retention_value_snapshot_status::unavailable;
    size_t size = 0;
};

using server_retention_value_snapshot_visitor = bool (*)(
    void *, const server_retention_value_snapshot &) noexcept;

// Bounded exact-token radix index used to lower cross-lineage shared-prefix
// coverage into the pure retention projector. Same-lineage aliases never count as
// external coverage. Any allocation, arithmetic, or cardinality failure
// poisons the index so callers cannot consume a partially indexed view.
class server_retention_prefix_index {
public:
    using prefix_visitor = bool (*)(
        void *, llama_cache_acct_artifact_id, uint64_t) noexcept;

    server_retention_prefix_index() noexcept;
    ~server_retention_prefix_index();

    server_retention_prefix_index(const server_retention_prefix_index &) = delete;
    server_retention_prefix_index & operator=(const server_retention_prefix_index &) = delete;

    bool publish(
        llama_cache_acct_artifact_id artifact,
        uint64_t lineage_id,
        const std::vector<llama_token> & tokens) noexcept;
    bool clone(
        llama_cache_acct_artifact_id source,
        llama_cache_acct_artifact_id destination,
        uint64_t lineage_id) noexcept;
    bool exact_matches(
        llama_cache_acct_artifact_id artifact,
        const std::vector<llama_token> & tokens) const noexcept;
    void retire(llama_cache_acct_artifact_id artifact) noexcept;
    bool external_shared_coverage(
        llama_cache_acct_artifact_id artifact,
        uint64_t & coverage_tokens,
        llama_cache_acct_artifact_id excluded = {},
        uint64_t * compared_tokens = nullptr) const noexcept;
    bool visit_prefixes(
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_visitor visitor) const noexcept;
    // Visit every stored terminal tied for the longest nonzero common prefix
    // with tokens. Unlike visit_prefixes(), the terminal may extend beyond or
    // diverge after that prefix. Results are bounded by the index cardinality
    // and ordered by artifact id.
    bool visit_longest_common_prefix(
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_visitor visitor) const noexcept;
    // Visit every stored terminal with a nonzero common prefix. Unlike the
    // longest-only view, this lets a semantic owner reject a longer terminal
    // without hiding a shorter eligible one. Results are bounded by the index
    // cardinality and ordered by artifact id.
    bool visit_common_prefixes(
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_visitor visitor,
        uint64_t * compared_tokens = nullptr,
        uint64_t * visited_nodes = nullptr) const noexcept;

    bool available() const noexcept;
    size_t size() const noexcept;
    size_t node_count() const noexcept;
    uint64_t token_bytes() const noexcept;
    uint64_t source_token_bytes() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct server_retention_lineage_ticket {
    common_retention_pool pool = common_retention_pool::attention;
    uint64_t lineage_id = 0;
    std::shared_ptr<const common_retention_turn_table> turns;

    bool valid() const noexcept { return lineage_id != 0 && bool(turns); }
};

// Allocation-free checkpoint creation-path view. The three comparison strings are
// stored in the catalog once when the immutable checkpoint member is
// published (and copied once when that member is cloned). Repeated thinning
// scans consume only scalar fields plus the lease-table result; they never
// rebuild media/adapter identities or copy the serialized retention record.
struct server_retention_checkpoint_inventory {
    llama_cache_acct_artifact_id artifact_id;
    common_retention_pool pool = common_retention_pool::attention;
    uint64_t stable_id = 0;
    bool mandatory_anchor = false;
    bool release_owned = false;
    bool recovery_pinned = false;
    bool identity_known = false;
    server_cache_lease_evaluation lease;
};

void server_cache_acct_mark_shadow_unavailable(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer) noexcept;

llama_cache_acct_op_id server_cache_acct_charge_shadow(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer,
        const llama_cache_acct_attribution & attribution,
        uint64_t logical_bytes,
        uint64_t resident_bytes) noexcept;

// Observer-owned retention catalog. This server layer owns process-local accounting handles;
// the common codec remains a pure, serializable value format.
class server_retention_sidecar_store {
public:
    using prefix_instance_visitor = bool (*)(
        void *, const server_retention_instance_key &, uint64_t) noexcept;

    ~server_retention_sidecar_store();

    server_retention_sidecar_store();
    server_retention_sidecar_store(const server_retention_sidecar_store &) = delete;
    server_retention_sidecar_store & operator=(const server_retention_sidecar_store &) = delete;

    void configure(
        llama_cache_acct_ledger * ledger,
        const llama_cache_acct_resource_domain & domain,
        server_cache_lease_table * leases = nullptr) noexcept;
    // The optional seam deterministically exercises allocation refusal; no
    // production caller passes it.
    bool enable_prefix_tracking(bool force_failure_for_test = false) noexcept;
    bool publish_prefix(
        const server_retention_instance_key & key,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens) noexcept;
    // Clone the exact immutable token block already indexed for source into
    // destination. The new terminal remains independently retireable, while
    // large prefix storage is charged once across live/host aliases.
    bool clone_prefix(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept;
    // Clone an exact host frontier while preserving the cheap shared-prefix
    // path when source coverage already matches. If the semantic prompt has
    // advanced beyond the live sidecar coverage, mint a same-lineage record
    // at the authenticated token frontier and index that exact block.
    bool clone_exact_prefix(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens) noexcept;
    bool visit_prefix_instances(
        common_retention_pool pool,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_instance_visitor visitor) const noexcept;
    bool visit_longest_common_prefix_instances(
        common_retention_pool pool,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_instance_visitor visitor) const noexcept;
    bool visit_common_prefix_instances(
        common_retention_pool pool,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_instance_visitor visitor) const noexcept;
    bool prefix_tracking_enabled() const noexcept;
    bool prefix_tracking_available() const noexcept;
    bool publish(
        const server_retention_instance_key & key,
        common_retention_pool pool,
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t turn_token_count,
        uint64_t coverage_tokens,
        bool coverage_valid,
        const server_cache_lease_identity * checkpoint_identity = nullptr,
        const server_cache_lease_frontier * replacement_frontier = nullptr,
        const server_retention_instance_key * lineage_source = nullptr,
        const server_retention_lineage_ticket * lineage_ticket = nullptr,
        bool defer_lineage_admission = false) noexcept;
    bool clone(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept;
    // Read-only half of clone() used to keep compound payload publication
    // outside a partial sidecar transition. This performs bounded catalog
    // lookups only; it neither allocates nor changes availability.
    bool clone_source_available(
        const server_retention_instance_key & source) const noexcept;
    // Divergent reuse credits the immutable source separately, then admits
    // the destination on probation without copied frequency or leases.
    bool branch(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination,
        const server_retention_instance_key * destination_lineage_source =
            nullptr,
        bool defer_lineage_admission = false) noexcept;
    // Prefix projection is a divergent branch whose artifact geometry is the
    // selected parent prefix, not the complete source artifact.
    bool branch_prefix(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination,
        uint64_t destination_coverage_tokens,
        bool defer_lineage_admission = false) noexcept;
    bool rebind(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept;
    bool prepare_for_launch(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept;
    bool prepared_for_launch(
        const server_retention_instance_key & destination) const noexcept;
    // Occupied restore prepares a second live-slot association under a private
    // instance key.  The read half proves that replacing the canonical slot is
    // an allocation-free terminal; the write half only changes an existing
    // association value, erases the private alias, and retires the displaced
    // descriptor. A refusal leaves both associations untouched.
    bool prepared_launch_destination_swappable(
        const server_retention_instance_key & prepared,
        const server_retention_instance_key & occupied) const noexcept;
    bool swap_prepared_launch_destination(
        const server_retention_instance_key & prepared,
        const server_retention_instance_key & occupied) noexcept;
    bool consume_prepared_launch(
        const server_retention_instance_key & destination,
        server_retention_lineage_ticket & source) noexcept;
    void abandon_prepared_launch(
        const server_retention_instance_key & destination) noexcept;
    common_retention_credit_result credit_reuse(
        const server_retention_instance_key & source) noexcept;
    bool acquire_lineage_ticket(
        const server_retention_instance_key & source,
        server_retention_lineage_ticket & out) noexcept;
    void release_lineage_ticket(
        server_retention_lineage_ticket & ticket) noexcept;
    common_retention_credit_result credit_reuse(
        const server_retention_lineage_ticket & source) noexcept;
    bool activate_lineage_ticket(
        const server_retention_lineage_ticket & ticket) noexcept;
    bool set_lineage_prior(
        const server_retention_instance_key & source,
        uint32_t prior_milli) noexcept;
    // One pressure/reclaim wave advances competition once, regardless of how
    // many victims it later removes. The same terminal serves observation and
    // the authoritative reclaim call site.
    bool begin_competition_wave() noexcept;
    bool lineage_for_instance(
        const server_retention_instance_key & key,
        common_retention_lineage_record & out) const noexcept;
    server_retention_value_snapshot_result value_snapshots(
        void * context,
        server_retention_value_snapshot_visitor visitor,
        llama_cache_acct_artifact_id excluded = {}) const noexcept;
    // Lifecycle choke points retire associations directly.
    // This can consolidate onto retire-by-artifact-id once admission
    // owns the catalog mutation rather than merely carrying the strong id.
    void retire(const server_retention_instance_key & key) noexcept;
    void retire_slot(int32_t owner_slot) noexcept;
    llama_cache_acct_artifact_id artifact_id(
        const server_retention_instance_key & key) const noexcept;
    bool candidate_for_instance(
        const server_retention_instance_key & key,
        server_retention_candidate & out) const noexcept;
    // Restore-time no-copy lookup: return an independently unowned checkpoint
    // artifact without copying its potentially large turn-boundary record.
    bool checkpoint_admission_artifact(
        const server_retention_instance_key & key,
        llama_cache_acct_artifact_id & artifact) const noexcept;
    bool checkpoint_inventory(
        const server_retention_instance_key & key,
        server_retention_checkpoint_inventory & out) const noexcept;
    // Checkpoint payload ownership. Host saves and non-consuming restores
    // join the immutable plane allocations instead of charging copied bytes;
    // each independently retireable logical checkpoint still owns one exact
    // operation reference per unique plane allocation.
    // Takes ownership on entry; failure releases every supplied operation.
    bool attach_release_ops(
        const server_retention_instance_key & key,
        std::vector<llama_cache_acct_op_id> ops) noexcept;
    server_cache_recovery_pin acquire_recovery_pin(
        const server_retention_instance_key & key) noexcept;
    bool recovery_pinned(
        const server_retention_instance_key & key) const noexcept;
    // The prepared capability already released the payload operations. Drop
    // only the sidecar descriptor/association after the physical mutation.
    void retire_after_committed_release(
        const server_retention_instance_key & key) noexcept;
    // Complete-slot retirement terminal. The selected artifact set is the
    // capability's committed manifest. Every slot association must belong to
    // it before payload and descriptor operation ownership is cleared.
    bool retire_slot_after_committed_release(
        int32_t owner_slot,
        const std::vector<llama_cache_acct_artifact_id> & selected_attention,
        const std::vector<llama_cache_acct_artifact_id> & selected_recurrent) noexcept;
    std::vector<server_retention_candidate> candidate_snapshot() const noexcept;

    common_retention_sidecar_snapshot snapshot() const noexcept;
    bool import_snapshot(const common_retention_sidecar_snapshot & snapshot) noexcept;
    bool export_bytes(std::vector<uint8_t> & out) const noexcept;

    uint64_t live_bytes() const noexcept { return bytes_live; }
    uint64_t publish_ok() const noexcept { return n_publish_ok; }
    uint64_t unavailable() const noexcept { return n_unavailable; }
    uint64_t competition_epoch_value() const noexcept {
        return competition_epoch;
    }

private:
    struct prefix_tracking;
    struct catalog_entry {
        common_retention_artifact_record record;
        server_cache_lease_identity checkpoint_identity;
        uint64_t encoded_size = 0;
        llama_cache_acct_op_id accounting_op;
        std::vector<llama_cache_acct_op_id> release_ops;
        uint32_t recovery_pins = 0;
        server_retention_sidecar_store * owner = nullptr;
        llama_cache_acct_artifact_id artifact;
        bool checkpoint_identity_known = false;
        bool prefix_indexed = false;
        bool retire_pending = false;
        server_retention_lineage_ticket prepared_source;
    };
    struct lineage_entry {
        common_retention_lineage_record record;
        uint32_t refs = 0;
        bool admitted = false;
    };
    struct turn_table_entry {
        std::shared_ptr<const common_retention_turn_table> table;
        llama_cache_acct_op_id accounting_op;
        uint64_t bytes = 0;
        uint32_t refs = 0;
    };
    using catalog_map = std::unordered_map<uint64_t, catalog_entry>;
    using lineage_map = std::unordered_map<uint64_t, lineage_entry>;
    using turn_table_map = std::unordered_map<
        const common_retention_turn_table *, turn_table_entry>;
    using association_map = std::unordered_map<
        server_retention_instance_key,
        llama_cache_acct_artifact_id,
        server_retention_instance_key_hash>;

    bool install(
        const server_retention_instance_key & key,
        common_retention_artifact_record && record,
        const server_cache_lease_identity * checkpoint_identity,
        const server_cache_lease_frontier * replacement_frontier) noexcept;
    const catalog_entry * find_clone_source(
        const server_retention_instance_key & source) const noexcept;
    bool branch_impl(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination,
        const server_retention_instance_key * destination_lineage_source,
        bool defer_lineage_admission,
        const uint64_t * destination_coverage_tokens) noexcept;
    void retire_catalog_entry(catalog_map::iterator entry) noexcept;
    bool retain_lineage(
        common_retention_pool pool,
        uint64_t lineage_id) noexcept;
    void release_lineage(
        common_retention_pool pool,
        uint64_t lineage_id) noexcept;
    bool retain_turn_table(
        const std::shared_ptr<const common_retention_turn_table> & table,
        llama_cache_acct_artifact_id attribution) noexcept;
    void release_turn_table(
        const std::shared_ptr<const common_retention_turn_table> & table) noexcept;
    bool create_lineage(
        common_retention_pool pool,
        uint64_t & lineage_id) noexcept;
    bool advance_competition_epoch() noexcept;
    void retire_association(association_map::iterator it) noexcept;
    static void release_recovery_pin(void * context) noexcept;
    void mark_unavailable() noexcept;
    static llama_cache_acct_artifact_id qualified_artifact_id(
        common_retention_pool pool, uint64_t stable_id) noexcept;
    static uint64_t qualified_lineage_id(
        common_retention_pool pool, uint64_t lineage_id) noexcept;

    llama_cache_acct_ledger * ledger = nullptr;
    server_cache_lease_table * leases = nullptr;
    llama_cache_acct_resource_domain domain;
    common_retention_allocator allocator;
    association_map associations;
    catalog_map catalog;
    lineage_map lineages;
    turn_table_map turn_tables;
    common_retention_frequency_config frequency_config;
    std::unique_ptr<prefix_tracking> prefixes;
    bool prefix_tracking_requested = false;
    uint64_t competition_epoch = 1;
    uint64_t bytes_live = 0;
    uint64_t turn_table_bytes_live = 0;
    uint64_t n_publish_ok = 0;
    uint64_t n_unavailable = 0;
};
