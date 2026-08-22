#include "server-vbr-artifact-store.h"

#include "server-prompt-cache-payload.h"

#include "build-info.h"

#include "../../src/llama-sha256.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <utility>

namespace {

// One prefix keeps the reference builder and the authorizer in lock-step.
constexpr char VBR_REFERENCE_PREFIX[] = "vbrref_";

bool capture_capacity_category_applies(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        bool include_live_scope) {
    const auto row = llama_cache_budget_classify(category);
    if (row.participation !=
            llama_cache_budget_capacity_participation::
                participating) {
        return false;
    }
    if (row.scope ==
            llama_cache_budget_residency_scope::by_domain) {
        return domain.residency ==
                   llama_cache_acct_residency::device ||
               domain.residency ==
                   llama_cache_acct_residency::pinned_host ||
               domain.residency ==
                   llama_cache_acct_residency::pageable_host;
    }
    if (row.scope ==
            llama_cache_budget_residency_scope::host) {
        return
            (domain.residency ==
                 llama_cache_acct_residency::pinned_host ||
             domain.residency ==
                 llama_cache_acct_residency::pageable_host);
    }
    return include_live_scope &&
           row.scope ==
               llama_cache_budget_residency_scope::device &&
           domain.residency ==
               llama_cache_acct_residency::device;
}

std::string opaque_reference(
        uint64_t nonce,
        uint64_t ordinal,
        llama_cache_acct_artifact_id artifact,
        const std::string & tenant) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] = "buun.vbr.server-reference/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u64(nonce);
    writer.u64(ordinal);
    writer.u64(artifact.v);
    writer.string(tenant.data(), tenant.size());
    const auto digest = writer.finish();
    static constexpr char HEX[] = "0123456789abcdef";
    std::string out = VBR_REFERENCE_PREFIX;
    out.reserve(out.size() + 32);
    for (size_t i = 0; i < 16; ++i) {
        out.push_back(HEX[digest[i] >> 4]);
        out.push_back(HEX[digest[i] & 0x0f]);
    }
    return out;
}

server_vbr_artifact_capture_status map_status(
        vbr_explicit_capture_status status) {
    switch (status) {
        case vbr_explicit_capture_status::ok:
            return server_vbr_artifact_capture_status::ok;
        case vbr_explicit_capture_status::not_armed:
        case vbr_explicit_capture_status::unsupported_layout:
            return server_vbr_artifact_capture_status::unsupported;
        case vbr_explicit_capture_status::slot_not_idle:
            return server_vbr_artifact_capture_status::slot_processing;
        case vbr_explicit_capture_status::identity_unavailable:
            return server_vbr_artifact_capture_status::identity_unavailable;
        case vbr_explicit_capture_status::required_companion_unavailable:
            return server_vbr_artifact_capture_status::
                required_companion_unavailable;
        case vbr_explicit_capture_status::admission_refused:
            return server_vbr_artifact_capture_status::admission_refused;
        case vbr_explicit_capture_status::source_changed:
            return server_vbr_artifact_capture_status::source_changed;
        case vbr_explicit_capture_status::generation_unavailable:
        case vbr_explicit_capture_status::registry_busy:
        case vbr_explicit_capture_status::recovery_pending:
        case vbr_explicit_capture_status::geometry_mismatch:
        case vbr_explicit_capture_status::stash_inconsistent:
        case vbr_explicit_capture_status::size_overflow:
        case vbr_explicit_capture_status::ring_unavailable:
        case vbr_explicit_capture_status::transfer_failed:
        case vbr_explicit_capture_status::short_read:
        case vbr_explicit_capture_status::event_failed:
        case vbr_explicit_capture_status::hash_mismatch:
        case vbr_explicit_capture_status::dedup_validation_failed:
        case vbr_explicit_capture_status::accounting_failed:
        case vbr_explicit_capture_status::publication_failed:
            return server_vbr_artifact_capture_status::unavailable;
        case vbr_explicit_capture_status::internal_error:
        case vbr_explicit_capture_status::_count:
            return server_vbr_artifact_capture_status::internal_error;
    }
    return server_vbr_artifact_capture_status::internal_error;
}

struct live_import_context {
    llama_memory_i * memory = nullptr;
    llama_cache_acct_ledger * ledger = nullptr;
    const vbr_artifact_package_view * package = nullptr;
    const std::vector<llama_vbr_artifact_domain_binding> * bindings = nullptr;
    llama_seq_id destination = -1;
    vbr_target_validation_snapshot snapshot;
};

bool import_inspect_target(
        const void * opaque,
        llama_memory_i & memory,
        const std::vector<llama_memory_tree_child> &,
        vbr_target_validation_snapshot & output) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    if (!context || context->memory != &memory) {
        return false;
    }
    try {
        output = context->snapshot;
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

uint64_t import_accounting_serial(const void * opaque) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    return context && context->ledger
        ? context->ledger->snapshot().serial : 0;
}

uint64_t import_policy_epoch(const void * opaque) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    // This is deliberately a fresh read even though target recheck also reads
    // the policy: the validator compares two independent live observations to
    // close the adopt-time TOCTOU window.
    return context && context->memory
        ? vbr_explicit_import_policy_epoch(*context->memory) : 0;
}

bool import_target_recheck(
        const void * opaque,
        const vbr_target_empty_fingerprint & expected) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    return context && context->memory &&
        vbr_explicit_import_target_recheck(
            *context->memory, context->destination, expected);
}

bool import_downward_digest(
        const void * opaque,
        std::array<uint8_t, 32> & output) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    if (!context || !context->memory || !context->package ||
        !context->bindings) {
        return false;
    }
    vbr_target_validation_snapshot snapshot;
    vbr_downward_policy_projection projection;
    if (!vbr_explicit_import_target_snapshot(
            *context->memory, context->destination,
            *context->package, *context->bindings, false,
            1, snapshot, &projection)) {
        return false;
    }
    output = projection.tree_digest;
    return std::any_of(output.begin(), output.end(),
        [](uint8_t byte) { return byte != 0; });
}

bool import_parse_companion(
        const void * opaque,
        const vbr_artifact_companion_payload & descriptor,
        const artifact_segment_chain & source,
        const vbr_target_companion_snapshot & target,
        std::unique_ptr<vbr_parsed_companion_image> & output) noexcept {
    if (descriptor.kind != vbr_artifact_companion_kind::recurrent) {
        output.reset();
        return false;
    }
    return vbr_parse_recurrent_companion(
        opaque, descriptor, source, target, output);
}

bool import_reserve_downward(
        void * opaque,
        const std::vector<vbr_validated_child_plan> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept {
    auto * context = static_cast<live_import_context *>(opaque);
    return context && context->memory &&
        vbr_explicit_import_reserve_downward(
            *context->memory, plans, ledger, budget, output);
}

bool add_bytes(uint64_t & total, uint64_t value) noexcept {
    if (value > UINT64_MAX - total) {
        return false;
    }
    total += value;
    return true;
}

bool package_bytes(
        const vbr_artifact_package_view & package,
        uint64_t & payload,
        uint64_t & companions) noexcept {
    payload = 0;
    companions = 0;
    for (const auto & unit : package.units()) {
        for (const auto & shard : unit.descriptor.shards) {
            if (!add_bytes(payload, shard.payload_bytes)) {
                return false;
            }
        }
        for (const auto & stash : unit.descriptor.clean_stash.shards) {
            if (!add_bytes(payload, stash.payload_bytes)) {
                return false;
            }
        }
    }
    for (const auto & companion : package.companions()) {
        if (!add_bytes(companions, companion.descriptor.payload_bytes)) {
            return false;
        }
    }
    return true;
}

} // namespace

server_vbr_artifact_import_status server_vbr_artifact_import_route_precheck(
        bool store_available,
        bool slot_exists,
        bool slot_processing,
        bool target_available,
        bool slot_empty) noexcept {
    if (!store_available) {
        return server_vbr_artifact_import_status::unsupported;
    }
    if (!slot_exists) {
        return server_vbr_artifact_import_status::invalid_slot;
    }
    if (slot_processing) {
        return server_vbr_artifact_import_status::slot_processing;
    }
    if (!target_available) {
        return server_vbr_artifact_import_status::unavailable;
    }
    return slot_empty
        ? server_vbr_artifact_import_status::ok
        : server_vbr_artifact_import_status::slot_not_empty;
}

server_vbr_artifact_import_status
server_vbr_artifact_import_validation_disposition(
        vbr_manifest_validation_status status,
        vbr_import_decision decision) noexcept {
    if (status != vbr_manifest_validation_status::validated) {
        return server_vbr_artifact_import_status::validation_failed;
    }
    switch (decision) {
        case vbr_import_decision::native_import:
        case vbr_import_decision::live_rebased:
        case vbr_import_decision::downward_rebase:
            return server_vbr_artifact_import_status::ok;
        case vbr_import_decision::rebuild:
        case vbr_import_decision::cold:
            return server_vbr_artifact_import_status::report_only;
        case vbr_import_decision::reject:
        case vbr_import_decision::_count:
            return server_vbr_artifact_import_status::validation_failed;
    }
    return server_vbr_artifact_import_status::validation_failed;
}

bool server_vbr_artifact_import_variant_fallback_safe(
        const server_vbr_artifact_import_output & output) noexcept {
    if (output.adopt_attempted || output.h2d_bytes != 0 ||
        output.h2d_chunks != 0) {
        return false;
    }
    switch (output.status) {
        case server_vbr_artifact_import_status::validation_failed:
        case server_vbr_artifact_import_status::report_only:
        case server_vbr_artifact_import_status::stage_failed:
            return true;
        case server_vbr_artifact_import_status::ok:
        case server_vbr_artifact_import_status::unsupported:
        case server_vbr_artifact_import_status::not_found:
        case server_vbr_artifact_import_status::invalid_slot:
        case server_vbr_artifact_import_status::slot_processing:
        case server_vbr_artifact_import_status::slot_not_empty:
        case server_vbr_artifact_import_status::adopt_failed:
        case server_vbr_artifact_import_status::unavailable:
        case server_vbr_artifact_import_status::internal_error:
        case server_vbr_artifact_import_status::_count:
            return false;
    }
    return false;
}

bool server_vbr_artifact_reference_index::publish(
        std::string reference,
        std::string tenant_key,
        llama_cache_acct_artifact_id artifact) noexcept {
    try {
        return reference.rfind(VBR_REFERENCE_PREFIX, 0) == 0 &&
               !tenant_key.empty() && artifact.v != 0 &&
               entries_.emplace(
                   std::move(reference),
                   binding { std::move(tenant_key), artifact }).second;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_reference_index::authorize(
        const std::string & reference,
        const std::string & tenant_key,
        llama_cache_acct_artifact_id & artifact) const noexcept {
    artifact = {};
    try {
        if (reference.rfind(VBR_REFERENCE_PREFIX, 0) != 0 || tenant_key.empty()) {
            return false;
        }
        const auto found = entries_.find(reference);
        if (found == entries_.end() ||
            found->second.tenant_key != tenant_key) {
            return false;
        }
        artifact = found->second.artifact;
        return artifact.v != 0;
    } catch (...) {
        artifact = {};
        return false;
    }
}

struct server_vbr_artifact_store::impl {
    llama_cache_acct_ledger * ledger = nullptr;
    llama_vbr_artifact_catalog catalog;
    std::unique_ptr<vbr_pinned_chunk_ring> ring;
    std::shared_ptr<vbr_h2d_chunk_ring> import_ring;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    std::vector<llama_vbr_artifact_domain_binding> domain_bindings;
    std::vector<llama_vbr_artifact_domain_binding> policy_bindings;
    std::vector<vbr_h2d_lane_binding> h2d_lanes;
    llama_cache_acct_resource_domain pinned_domain;
    uint64_t import_ring_bytes = 0;
    size_t import_chunk_bytes = 0;
    void * budget_context = nullptr;
    server_vbr_artifact_store_config::sample_budget_fn sample_budget = nullptr;
    int turbo_meansub_id = 0;
    server_vbr_artifact_store_counters counters;
    uint64_t nonce = 0;
    uint64_t next_reference = 1;
    uint32_t n_attention_children = 0;
    server_vbr_artifact_reference_index references;
    bool fail_projected_host_adoption_once = false;

    explicit impl(llama_cache_acct_ledger & ledger)
        : ledger(&ledger), catalog(ledger) {
    }

    bool bind_import_transport(vbr_adopt_stage_policy & policy) const noexcept {
        if (!ledger || !import_ring || h2d_lanes.empty() ||
            import_ring_bytes == 0 || import_chunk_bytes == 0) {
            return false;
        }
        policy.ledger = ledger;
        policy.lanes = h2d_lanes;
        policy.pinned_domain = pinned_domain;
        policy.pinned_ring_bytes = import_ring_bytes;
        policy.chunk_bytes = import_chunk_bytes;
        policy.persistent_ring = import_ring;
        return true;
    }
};

server_vbr_artifact_store::server_vbr_artifact_store(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {
}

server_vbr_artifact_store::~server_vbr_artifact_store() = default;

bool server_vbr_artifact_store_test_door::import_transport_policy(
        const server_vbr_artifact_store & store,
        vbr_adopt_stage_policy & policy) noexcept {
    policy = {};
    return store.impl_ && store.impl_->bind_import_transport(policy);
}

void server_vbr_artifact_store_test_door::
fail_projected_host_adoption_once(
        server_vbr_artifact_store & store) noexcept {
    if (store.impl_) {
        store.impl_->fail_projected_host_adoption_once = true;
    }
}

bool server_vbr_artifact_store_observe_empty_accounting(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) noexcept {
    try {
        if (!llama_cache_acct_resource_domain_valid(domain) ||
            (domain.residency !=
                 llama_cache_acct_residency::device &&
             domain.residency !=
                 llama_cache_acct_residency::pinned_host &&
             domain.residency !=
                 llama_cache_acct_residency::pageable_host)) {
            return false;
        }

        const auto before = ledger.snapshot();
        if (before.completeness_manifest !=
                llama_cache_acct_known::known) {
            return false;
        }
        if (std::none_of(
                before.completeness.begin(),
                before.completeness.end(),
                [&](const llama_cache_acct_completeness_row & row) {
                    return row.domain == domain &&
                           row.state !=
                               llama_cache_acct_known::
                                   unavailable;
                })) {
            return false;
        }

        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            if (!capture_capacity_category_applies(
                    category, domain, false)) {
                continue;
            }
            const auto cell = std::find_if(
                before.cells.begin(), before.cells.end(),
                [&](const llama_cache_acct_cell_row & row) {
                    return row.category == category &&
                           row.domain == domain;
                });
            if (cell == before.cells.end()) {
                return false;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                const auto value =
                    cell->cell.measures[size_t(measure)];
                if (value.state ==
                        llama_cache_acct_known::unavailable ||
                    (value.state == llama_cache_acct_known::known &&
                     value.value != 0)) {
                    return false;
                }
            }
        }

        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            if (!capture_capacity_category_applies(
                    category, domain, false)) {
                continue;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                ledger.gauge_set(category, domain, measure, 0);
            }
        }
        const auto after = ledger.snapshot();
        if (after.faults_invalid_transition !=
                before.faults_invalid_transition ||
            after.faults_overflow != before.faults_overflow ||
            after.faults_allocation !=
                before.faults_allocation) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_store_verify_accounting(
        llama_cache_acct_ledger & ledger,
        const std::vector<llama_cache_acct_resource_domain> &
            domains) noexcept {
    try {
        const auto snapshot = ledger.snapshot();
        if (snapshot.completeness_manifest !=
                llama_cache_acct_known::known ||
            domains.empty()) {
            return false;
        }
        for (size_t i = 0; i < domains.size(); ++i) {
            const auto & domain = domains[i];
            if (!llama_cache_acct_resource_domain_valid(domain) ||
                std::find(
                    domains.begin(), domains.begin() + i,
                    domain) != domains.begin() + i) {
                return false;
            }
            bool has_requirement = false;
            for (const auto & row : snapshot.completeness) {
                if (row.domain == domain) {
                    has_requirement = true;
                    if (row.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
            }
            if (!has_requirement) {
                return false;
            }
        }
        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            const auto classification =
                llama_cache_budget_classify(category);
            for (const auto & domain : domains) {
                if (!capture_capacity_category_applies(
                        category, domain, true)) {
                    continue;
                }
                const auto cell = std::find_if(
                    snapshot.cells.begin(), snapshot.cells.end(),
                    [&](const llama_cache_acct_cell_row & row) {
                        return row.category == category &&
                               row.domain == domain;
                    });
                if (cell == snapshot.cells.end() ||
                    cell->certification !=
                        llama_cache_acct_known::known) {
                    return false;
                }
                const auto resident =
                    cell->cell.measures[size_t(
                        llama_cache_acct_measure::
                            resident_allocated)];
                if (resident.state !=
                        llama_cache_acct_known::known) {
                    return false;
                }
                if (classification.mode ==
                        llama_cache_budget_accounting_mode::
                            transactional) {
                    const auto reserved =
                        cell->cell.measures[size_t(
                            llama_cache_acct_measure::reserved)];
                    if (reserved.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
                if (classification.scope !=
                        llama_cache_budget_residency_scope::
                            device) {
                    const auto logical =
                        cell->cell.measures[size_t(
                            llama_cache_acct_measure::
                                logical_payload)];
                    if (logical.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_store_configure_pinned_accounting(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) noexcept {
    const auto canonical =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    if (domain != canonical ||
        !server_vbr_artifact_store_observe_empty_accounting(
            ledger, domain) ||
        !ledger.certify_complete(
            domain,
            llama_cache_acct_producer::retention_sidecar)) {
        return false;
    }
    return server_vbr_artifact_store_verify_accounting(
        ledger, { domain });
}

std::unique_ptr<server_vbr_artifact_store>
server_vbr_artifact_store::create(
        const server_vbr_artifact_store_config & config,
        server_vbr_artifact_capture_status & status,
        server_vbr_artifact_store_create_diagnostics * diagnostics) noexcept {
    status = server_vbr_artifact_capture_status::unavailable;
    server_vbr_artifact_store_create_diagnostics observed;
    observed.requested_ring_bytes = config.ring_bytes;
    observed.chunk_bytes = config.chunk_bytes;
    observed.lane_count = config.lanes.size();
    observed.attention_children = config.attention_children;
    const auto fail =
        [&](server_vbr_artifact_store_create_failure failure) {
            observed.failure = failure;
            if (diagnostics) {
                *diagnostics = observed;
            }
        };
    try {
        if (config.ledger == nullptr) {
            fail(server_vbr_artifact_store_create_failure::ledger_missing);
            return nullptr;
        }
        if (config.sample_budget == nullptr) {
            fail(server_vbr_artifact_store_create_failure::
                budget_sampler_missing);
            return nullptr;
        }
        if (config.topologies.empty()) {
            fail(server_vbr_artifact_store_create_failure::
                topology_missing);
            return nullptr;
        }
        if (config.pool_bindings.empty()) {
            fail(server_vbr_artifact_store_create_failure::
                pool_binding_missing);
            return nullptr;
        }
        if (config.lanes.empty()) {
            fail(server_vbr_artifact_store_create_failure::lane_missing);
            return nullptr;
        }
        if (config.attention_children == 0) {
            fail(server_vbr_artifact_store_create_failure::
                attention_child_missing);
            return nullptr;
        }
        if (config.ring_bytes == 0 ||
            config.ring_bytes >
                VBR_CAPTURE_PINNED_RING_MAX_BYTES) {
            fail(server_vbr_artifact_store_create_failure::
                ring_size_invalid);
            return nullptr;
        }
        if (config.chunk_bytes == 0 ||
            config.lanes.size() >
                std::numeric_limits<uint64_t>::max()/2 ||
            uint64_t(config.lanes.size()*2) >
                std::numeric_limits<uint64_t>::max() /
                    uint64_t(config.chunk_bytes)) {
            fail(server_vbr_artifact_store_create_failure::
                chunk_size_invalid);
            return nullptr;
        }
        auto state = std::make_unique<impl>(*config.ledger);
        state->topologies = config.topologies;
        state->pool_bindings = config.pool_bindings;
        state->pinned_domain = config.pinned_domain;
        state->import_ring_bytes = config.ring_bytes;
        state->import_chunk_bytes = config.chunk_bytes;
        state->budget_context = config.budget_context;
        state->sample_budget = config.sample_budget;
        state->turbo_meansub_id = config.turbo_meansub_id;
        state->n_attention_children = config.attention_children;
        if (!state->catalog.bind_topologies(
                config.topologies, state->domain_bindings)) {
            fail(server_vbr_artifact_store_create_failure::topology_missing);
            return nullptr;
        }
        state->policy_bindings = state->domain_bindings;
        state->policy_bindings.push_back({
            UINT32_MAX, UINT16_MAX,
            llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host),
        });
        state->policy_bindings.push_back({
            UINT32_MAX, UINT16_MAX, state->pinned_domain,
        });
        state->h2d_lanes.resize(config.lanes.size());
        std::vector<bool> lane_bound(config.lanes.size(), false);
        for (const auto & pool : config.pool_bindings) {
            if (pool.lane >= config.lanes.size()) {
                fail(server_vbr_artifact_store_create_failure::lane_missing);
                return nullptr;
            }
            const auto domain = std::find_if(
                state->domain_bindings.begin(),
                state->domain_bindings.end(),
                [&](const llama_vbr_artifact_domain_binding & binding) {
                    return binding.topology_index == pool.topology_index &&
                           binding.device_ordinal == pool.device_ordinal;
                });
            if (domain == state->domain_bindings.end()) {
                fail(server_vbr_artifact_store_create_failure::pool_binding_missing);
                return nullptr;
            }
            const auto & capture_lane = config.lanes[pool.lane];
            auto & lane = state->h2d_lanes[pool.lane];
            if (lane_bound[pool.lane] &&
                (lane.domain != domain->domain ||
                 lane.device != capture_lane.device ||
                 lane.backend != capture_lane.backend)) {
                fail(server_vbr_artifact_store_create_failure::lane_missing);
                return nullptr;
            }
            lane = {
                domain->domain, capture_lane.device,
                capture_lane.backend, capture_lane.force_synchronous,
            };
            lane_bound[pool.lane] = true;
        }
        if (std::find(lane_bound.begin(), lane_bound.end(), false) !=
                lane_bound.end()) {
            fail(server_vbr_artifact_store_create_failure::lane_missing);
            return nullptr;
        }
        std::random_device random;
        state->nonce = (uint64_t(random()) << 32) ^ random();
        if (state->nonce == 0) {
            state->nonce = 1;
        }

        llama_cache_budget_config budget;
        if (!state->sample_budget(state->budget_context, budget)) {
            fail(server_vbr_artifact_store_create_failure::
                budget_sample_failed);
            return nullptr;
        }
        vbr_capture_ring_accounting accounting {
            config.ledger, config.pinned_domain, &budget,
        };
        const uint64_t minimum_ring_bytes =
            uint64_t(config.lanes.size()*2) *
            uint64_t(config.chunk_bytes);
        uint64_t attempt = config.ring_bytes;
        for (;;) {
            observed.attempted_ring_bytes = attempt;
            auto core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
                vbr_bounded_pinned_ring_core::create(
                    config.lanes, attempt, config.chunk_bytes,
                    &accounting, observed.ring_failure));
            if (core) {
                state->ring = vbr_pinned_chunk_ring::attach(core);
                state->import_ring = vbr_h2d_chunk_ring::attach(
                    std::move(core), state->h2d_lanes);
                observed.ring_status = state->ring && state->import_ring
                    ? vbr_capture_stream_status::ok
                    : vbr_capture_stream_status::internal_error;
                if (!state->ring || !state->import_ring) {
                    observed.ring_failure =
                        vbr_capture_ring_create_failure::internal_error;
                }
            } else {
                observed.ring_status =
                    vbr_capture_ring_failure_status(
                        observed.ring_failure);
            }
            if (state->ring && state->import_ring) {
                break;
            }
            state->ring.reset();
            state->import_ring.reset();
            // Pinned allocation pressure is recoverable without weakening
            // the ring protocol: two chunks per physical lane are sufficient
            // for bounded producer/consumer overlap. Other failures are
            // evidence/configuration failures and remain fail closed.
            if (observed.ring_failure !=
                    vbr_capture_ring_create_failure::
                        host_buffer_allocation_failed ||
                attempt <= minimum_ring_bytes) {
                break;
            }
            uint64_t next =
                (attempt/2/uint64_t(config.chunk_bytes)) *
                uint64_t(config.chunk_bytes);
            next = std::max(next, minimum_ring_bytes);
            if (next >= attempt) {
                break;
            }
            attempt = next;
        }
        if (!state->ring) {
            status = observed.ring_status ==
                        vbr_capture_stream_status::accounting_refused
                ? server_vbr_artifact_capture_status::admission_refused
                : observed.ring_status ==
                        vbr_capture_stream_status::internal_error
                    ? server_vbr_artifact_capture_status::internal_error
                    : server_vbr_artifact_capture_status::unavailable;
            fail(server_vbr_artifact_store_create_failure::
                ring_create_failed);
            return nullptr;
        }
        observed.constructed_ring_bytes =
            state->ring->capacity_bytes();
        state->import_ring_bytes = observed.constructed_ring_bytes;
        state->counters.pinned_bytes =
            observed.constructed_ring_bytes;
        status = server_vbr_artifact_capture_status::ok;
        if (diagnostics) {
            *diagnostics = observed;
        }
        return std::unique_ptr<server_vbr_artifact_store>(
            new server_vbr_artifact_store(std::move(state)));
    } catch (...) {
        status = server_vbr_artifact_capture_status::internal_error;
        fail(server_vbr_artifact_store_create_failure::internal_error);
        return nullptr;
    }
}

server_vbr_artifact_capture_output server_vbr_artifact_store::capture(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        const std::string & tenant_key) noexcept {
    server_vbr_artifact_capture_output output;
    impl_->counters.requested++;
    try {
        if (tenant_key.empty()) {
            output.status =
                server_vbr_artifact_capture_status::unauthorized;
            impl_->counters.refused++;
            return output;
        }
        llama_cache_budget_config budget;
        if (!impl_->sample_budget(impl_->budget_context, budget)) {
            output.status =
                server_vbr_artifact_capture_status::unavailable;
            impl_->counters.unavailable++;
            return output;
        }
        request.ring = impl_->ring.get();
        request.topologies = impl_->topologies;
        request.pool_bindings = impl_->pool_bindings;
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
            impl_->turbo_meansub_id,
        };
        request.representation_context = &representation_policy;
        request.representation_identity =
            vbr_explicit_capture_representation_identity;

        vbr_explicit_capture_accounting accounting;
        accounting.budget = &budget;
        accounting.context = &impl_->catalog;
        accounting.prepare = [](
                void * context,
                const vbr_artifact_package & package) noexcept {
            return static_cast<llama_vbr_artifact_catalog *>(context)
                ->prepare_capture_package(package);
        };
        const auto result = vbr_capture_explicit_manifest(
            memory, request, impl_->catalog, accounting);
        output.library_status = result.status;
        output.phase = result.phase;
        output.inner_stream_status =
            result.inner_stream_status;
        output.generation_failure =
            result.generation_failure;
        output.size_failure =
            result.size_failure;
        output.begin_diagnostics =
            result.begin_diagnostics;
        if (result.status != vbr_explicit_capture_status::_count) {
            impl_->counters.capture_outcomes[size_t(result.status)]++;
        }
        output.status = map_status(result.status);
        output.controllers = result.controllers;
        output.units = result.units;
        output.companions = result.companions;
        output.payload_bytes = result.payload_bytes;
        output.stash_bytes = result.stash_bytes;
        output.companion_bytes = result.companion_bytes;
        output.chunks = result.chunks;
        output.backpressure_waits = result.backpressure_waits;
        output.event_completions = result.event_completions;
        output.synchronous_fallbacks = result.synchronous_fallbacks;
        if (result.status != vbr_explicit_capture_status::ok) {
            if (output.status ==
                    server_vbr_artifact_capture_status::admission_refused) {
                impl_->counters.refused++;
            } else if (output.status ==
                    server_vbr_artifact_capture_status::internal_error) {
                impl_->counters.internal_error++;
            } else {
                impl_->counters.unavailable++;
            }
            return output;
        }
        if (result.sink.reference_artifact.v == 0) {
            output.status =
                server_vbr_artifact_capture_status::internal_error;
            impl_->counters.internal_error++;
            return output;
        }
        const auto after = impl_->catalog.snapshot();
        output.dedup = result.sink.adopted;
        output.reference = opaque_reference(
            impl_->nonce, impl_->next_reference++,
            result.sink.reference_artifact, tenant_key);
        if (!impl_->references.publish(
                output.reference, tenant_key,
                result.sink.reference_artifact)) {
            (void) impl_->catalog.retire(
                result.sink.reference_artifact);
            output.reference.clear();
            output.status =
                server_vbr_artifact_capture_status::internal_error;
            impl_->counters.internal_error++;
            return output;
        }
        output.consistency = vbr_artifact_consistency_kind::capture_exact;
        impl_->counters.exact_published++;
        impl_->counters.payload_bytes += result.payload_bytes;
        impl_->counters.stash_bytes += result.stash_bytes;
        impl_->counters.companion_bytes += result.companion_bytes;
        impl_->counters.chunks += result.chunks;
        impl_->counters.event_completions +=
            result.event_completions;
        impl_->counters.synchronous_fallbacks +=
            result.synchronous_fallbacks;
        impl_->counters.backpressure_waits += result.backpressure_waits;
        if (output.dedup) {
            impl_->counters.dedup_hits++;
        } else {
            impl_->counters.dedup_misses++;
        }
        impl_->counters.staging_overlap_refusals =
            after.staging_overlap_refusals;
        return output;
    } catch (...) {
        output.status =
            server_vbr_artifact_capture_status::internal_error;
        impl_->counters.internal_error++;
        return output;
    }
}

bool server_vbr_artifact_store::publish_projected_host_batch(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        std::vector<server_vbr_projected_host_publish_result> & output,
        server_vbr_projected_host_publish_diagnostics * diagnostics) noexcept {
    output.clear();
    if (diagnostics) {
        *diagnostics = {};
    }
    if (!impl_ || !impl_->sample_budget) {
        return false;
    }

    llama_cache_budget_config budget;
    if (!impl_->sample_budget(impl_->budget_context, budget)) {
        return false;
    }

    const size_t manifest_count = assembly.manifests().size();
    std::vector<vbr_projected_manifest_publish_result> published;
    std::vector<llama_cache_acct_artifact_id> handoff_references;
    std::vector<size_t> handoff_indices;
    std::vector<vbr_artifact_package_view> handoff_packages;
    // Allocate every cardinality-sized adapter arena before the catalog can
    // commit a row. The catalog handoff may still reject allocation or
    // structure atomically; compensating discard is allocation-free.
    try {
        output.resize(manifest_count);
        published.reserve(manifest_count);
        handoff_references.resize(manifest_count);
        handoff_indices.resize(manifest_count);
        handoff_packages.reserve(manifest_count);
    } catch (...) {
        output.clear();
        return false;
    }

    server_vbr_projected_host_publish_diagnostics measured;
    if (!impl_->catalog.publish_projected_batch(
            assembly, std::move(publications), budget,
            published, &measured.catalog)) {
        output.clear();
        return false;
    }

    const auto is_published = [](vbr_projected_manifest_publish_status status) {
        return status == vbr_projected_manifest_publish_status::published ||
               status == vbr_projected_manifest_publish_status::adopted;
    };
    if (published.size() != output.size()) {
        for (const auto & row : published) {
            if (is_published(row.status) &&
                row.publication.reference_artifact.v != 0) {
                (void) impl_->catalog.discard_unowned_reference(
                    row.publication.reference_artifact);
            }
        }
        output.clear();
        return false;
    }

    size_t handoff_count = 0;
    size_t injected_failure = SIZE_MAX;
    for (size_t i = 0; i < published.size(); ++i) {
        const auto & row = published[i];
        auto & terminal = output[i];
        terminal.manifest_id = row.manifest_id;
        terminal.status = row.status;
        if (!is_published(row.status)) {
            continue;
        }

        const auto reference = row.publication.reference_artifact;
        if (impl_->fail_projected_host_adoption_once &&
            injected_failure == SIZE_MAX) {
            injected_failure = i;
            impl_->fail_projected_host_adoption_once = false;
            continue;
        }
        if (reference.v == 0) {
            terminal.status =
                vbr_projected_manifest_publish_status::internal_error;
            measured.postpublish_retirements++;
            continue;
        }
        handoff_references[handoff_count] = reference;
        handoff_indices[handoff_count] = i;
        ++handoff_count;
    }
    handoff_references.resize(handoff_count);
    handoff_indices.resize(handoff_count);

    const bool handed_off = handoff_count == 0 ||
        impl_->catalog.claim_fresh_host_batch(
            handoff_references, handoff_packages);
    for (size_t i = 0; i < handoff_count; ++i) {
        auto & terminal = output[handoff_indices[i]];
        if (handed_off) {
            terminal.payload = server_prompt_cache_vbr_payload::adopt(
                std::move(handoff_packages[i]));
        }
        if (terminal.payload) {
            measured.host_payloads_retained++;
            continue;
        }

        // A successfully claimed view retires itself on reset; a failed atomic
        // handoff leaves the reference unowned for allocation-free discard.
        if (handed_off) {
            handoff_packages[i].reset();
        } else {
            (void) impl_->catalog.discard_unowned_reference(
                handoff_references[i]);
        }
        terminal.status =
            vbr_projected_manifest_publish_status::internal_error;
        measured.postpublish_retirements++;
    }
    if (injected_failure != SIZE_MAX) {
        auto & terminal = output[injected_failure];
        (void) impl_->catalog.discard_unowned_reference(
            published[injected_failure].publication.reference_artifact);
        terminal.status =
            vbr_projected_manifest_publish_status::internal_error;
        measured.postpublish_retirements++;
    }
    if (diagnostics) {
        *diagnostics = measured;
    }
    return true;
}

bool server_vbr_artifact_store::capture_projected_host_batch(
        llama_memory_i & memory,
        std::vector<vbr_projected_capture_manifest_request> manifests,
        uint64_t max_packed_bytes,
        std::vector<server_vbr_projected_host_publish_result> & output,
        server_vbr_projected_host_capture_diagnostics * diagnostics) noexcept {
    output.clear();
    if (diagnostics) {
        *diagnostics = {};
    }
    if (!impl_ || !impl_->ring || manifests.empty() ||
        max_packed_bytes == 0) {
        return false;
    }

    try {
        vbr_projected_capture_batch_request request;
        request.idle_decode_thread = true;
        request.max_packed_bytes = max_packed_bytes;
        request.manifests = std::move(manifests);
        request.ring = impl_->ring.get();
        request.topologies = impl_->topologies;
        request.pool_bindings = impl_->pool_bindings;
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
            impl_->turbo_meansub_id,
        };
        request.representation_context = &representation_policy;
        request.representation_identity =
            vbr_explicit_capture_representation_identity;

        auto captured = vbr_capture_projected_batch(memory, request);
        server_vbr_projected_host_capture_diagnostics measured;
        measured.capture_status = captured.status;
        measured.capture_phase = captured.phase;
        measured.inner_stream_status = captured.inner_stream_status;
        measured.source_namespace = captured.source_namespace;
        measured.union_cells = captured.union_cells;
        measured.planned_packed_bytes = captured.planned_packed_bytes;
        measured.size_pass_calls = captured.size_pass_calls;
        measured.projection_calls = captured.projection_calls;
        measured.unit_transfer_calls = captured.unit_transfer_calls;
        measured.transferred_units = captured.transferred_units;
        measured.transfer = captured.transfer;
        if (captured.status != vbr_explicit_capture_status::ok ||
            !captured.assembly ||
            captured.publications.size() !=
                captured.assembly.manifests().size()) {
            if (diagnostics) {
                *diagnostics = measured;
            }
            return false;
        }

        const bool published = publish_projected_host_batch(
            captured.assembly, std::move(captured.publications), output,
            &measured.publication);
        if (diagnostics) {
            *diagnostics = measured;
        }
        return published;
    } catch (...) {
        output.clear();
        return false;
    }
}

server_vbr_artifact_import_output server_vbr_artifact_store::import(
        server_vbr_artifact_import_request request) noexcept {
    server_vbr_artifact_import_output output;
    impl_->counters.imports_requested++;
    const auto fail = [&](server_vbr_artifact_import_status status,
                          uint64_t & counter) {
        output.status = status;
        ++counter;
        return output;
    };
    try {
        llama_cache_acct_artifact_id artifact;
        if (!impl_->references.authorize(
                request.reference, request.tenant_key, artifact)) {
            return fail(server_vbr_artifact_import_status::not_found,
                        impl_->counters.imports_not_found);
        }
        if (!request.memory || request.destination < 0 ||
            request.execution_identity.empty() ||
            request.adapter_config_identity.empty() ||
            !request.prepare_publish || !request.publish) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }

        vbr_artifact_package_view package;
        const auto resolved = impl_->catalog.resolve_reference(
            artifact, package);
        if (resolved != vbr_artifact_resolve_status::ok || !package) {
            const auto status = resolved == vbr_artifact_resolve_status::not_found
                ? server_vbr_artifact_import_status::not_found
                : server_vbr_artifact_import_status::unavailable;
            return status == server_vbr_artifact_import_status::not_found
                ? fail(status, impl_->counters.imports_not_found)
                : fail(status, impl_->counters.imports_unavailable);
        }
        return import_package(std::move(request), package);
    } catch (...) {
        return fail(server_vbr_artifact_import_status::internal_error,
                    impl_->counters.imports_unavailable);
    }
}

server_vbr_artifact_import_output
server_vbr_artifact_store::import_host_payload(
        server_vbr_artifact_import_target request,
        std::shared_ptr<const server_prompt_cache_vbr_payload> payload)
        noexcept {
    server_vbr_artifact_import_output output;
    impl_->counters.imports_requested++;
    if (!payload || !payload->retirement_owned() ||
        !payload->accounted_by(impl_->ledger) || !payload->package()) {
        output.status = server_vbr_artifact_import_status::unavailable;
        impl_->counters.imports_unavailable++;
        return output;
    }
    if (!impl_->catalog.owns_host_package(payload->package())) {
        output.status = server_vbr_artifact_import_status::not_found;
        impl_->counters.imports_not_found++;
        return output;
    }
    impl_->counters.host_imports_authenticated++;
    auto imported = import_package(
        std::move(request), payload->package());
    if (imported.status == server_vbr_artifact_import_status::ok) {
        impl_->counters.host_imports_succeeded++;
    }
    return imported;
}

server_vbr_artifact_import_output server_vbr_artifact_store::import_package(
        server_vbr_artifact_import_target request,
        const vbr_artifact_package_view & package) noexcept {
    server_vbr_artifact_import_output output;
    const auto fail = [&](server_vbr_artifact_import_status status,
                          uint64_t & counter) {
        output.status = status;
        ++counter;
        return output;
    };
    try {
        if (!request.memory || request.destination < 0 ||
            request.execution_identity.empty() ||
            request.adapter_config_identity.empty() ||
            !request.prepare_publish || !request.publish || !package) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        if (!package_bytes(
                package, output.payload_bytes, output.companion_bytes)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }

        llama_cache_budget_config budget;
        if (!impl_->sample_budget(impl_->budget_context, budget)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        const auto accounting_snapshot = impl_->ledger->snapshot();
        live_import_context context;
        context.memory = request.memory;
        context.ledger = impl_->ledger;
        context.package = &package;
        context.bindings = &impl_->domain_bindings;
        context.destination = request.destination;
        vbr_downward_policy_projection downward_projection;
        bool downward = false;
        if (!vbr_explicit_import_target_snapshot(
                *request.memory, request.destination, package,
                impl_->domain_bindings, request.previously_observed,
                accounting_snapshot.serial, context.snapshot,
                &downward_projection, &downward)) {
            output.validation_status =
                vbr_manifest_validation_status::unavailable;
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        // Idleness is the SCHEDULER's fact to assert, not the library's: the
        // route handler admits imports only on an idle, deferred-safe slot, so
        // the store vouches for it here on the snapshot the validator consumes.
        context.snapshot.scheduler_idle = true;

        llama_cache_budget_plan downward_budget;
        downward_budget.accounting_serial = accounting_snapshot.serial;
        vbr_adopt_policy policy;
        policy.authorized = true;
        policy.identity = {
            request.execution_identity,
            request.adapter_config_identity,
            package.manifest().identity.media_content_identity,
            package.manifest().identity.sequence_epoch,
            package.manifest().identity.next_position,
            &package.manifest().token_block.tokens,
        };
        policy.destination_sequence = request.destination;
        policy.adoption_nonce = impl_->next_reference++;
        if (policy.adoption_nonce == 0) {
            policy.adoption_nonce = impl_->next_reference++;
        }
        policy.domain_bindings = impl_->policy_bindings;
        policy.accounting_snapshot = &accounting_snapshot;
        policy.budget_config = &budget;
        policy.context = &context;
        policy.inspect_target = import_inspect_target;
        policy.parse_companion = import_parse_companion;
        policy.recheck_target_empty = import_target_recheck;
        policy.read_accounting_serial = import_accounting_serial;
        policy.read_policy_epoch = import_policy_epoch;
        if (downward) {
            policy.downward_budget_plan = &downward_budget;
            policy.downward_projection = &downward_projection;
            policy.read_downward_tree_digest = import_downward_digest;
        }
        auto validated = vbr_validate_unit_manifest(
            *request.memory, package, policy);
        output.validation_status = validated.status;
        output.decision = validated.decision;
        if (validated.status != vbr_manifest_validation_status::_count) {
            impl_->counters.validation_outcomes[size_t(validated.status)]++;
        }
        if (validated.decision != vbr_import_decision::_count) {
            impl_->counters.import_decisions[size_t(validated.decision)]++;
        }
        const auto disposition =
            server_vbr_artifact_import_validation_disposition(
                validated.status, validated.decision);
        if (disposition ==
                server_vbr_artifact_import_status::validation_failed) {
            return fail(disposition, impl_->counters.imports_refused);
        }
        if (disposition == server_vbr_artifact_import_status::report_only) {
            return fail(disposition, impl_->counters.imports_report_only);
        }
        if (!validated.proof ||
            !request.prepare_publish(
                request.publish_context,
                package.manifest().token_block.tokens,
                package.manifest().identity.sequence_epoch)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }

        vbr_adopt_stage_policy stage_policy;
        if (!impl_->bind_import_transport(stage_policy)) {
            return fail(server_vbr_artifact_import_status::stage_failed,
                        impl_->counters.imports_refused);
        }
        stage_policy.budget = &budget;
        stage_policy.downward_context = &context;
        stage_policy.reserve_downward = import_reserve_downward;
        auto staged = vbr_stage_validated_manifest(
            std::move(validated.proof), stage_policy);
        output.stage_status = staged.status;
        output.downward_reserve_status = staged.downward_status;
        if (staged.status != vbr_adopt_stage_status::staged ||
            !staged.manifest || !staged.staged) {
            return fail(server_vbr_artifact_import_status::stage_failed,
                        impl_->counters.imports_refused);
        }

        vbr_composite_publish_hooks hooks;
        hooks.publish = [](void * opaque) noexcept {
            auto * pair = static_cast<std::pair<
                server_vbr_artifact_import_target::publish_fn,
                void *> *>(opaque);
            pair->first(pair->second);
        };
        // The publish hook needs both the immutable owner-token context and
        // the server metadata callback. Use a no-throw local adapter while
        // retaining the memory pointer as the validated capability token.
        std::pair<server_vbr_artifact_import_target::publish_fn, void *>
            publish { request.publish, request.publish_context };
        hooks.context = &publish;
        hooks.owner_token = request.memory;
        hooks.validate_owner_token = [](
                const void * opaque, const void * token,
                const llama_memory_i * target) noexcept {
            return opaque != nullptr && token == target;
        };
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(request.memory, tree)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        for (const auto & child : tree) {
            if (child.recurrent) {
                hooks.companions.push_back(
                    vbr_recurrent_companion_adoption_provider(
                        *child.recurrent));
            }
        }
        const auto adopted = vbr_adopt_empty_manifest(
            *request.memory, request.destination,
            std::move(*staged.manifest), std::move(*staged.staged),
            *impl_->ledger, hooks);
        output.adopt_attempted = true;
        output.adopt_status = adopted.status;
        output.phase = adopted.phase;
        output.downward_subphase = adopted.downward_subphase;
        output.downward_edge = adopted.downward_edge;
        output.h2d_bytes = adopted.h2d_bytes;
        output.h2d_chunks = adopted.h2d_chunks;
        output.rollback_count = adopted.rollback_count;
        output.decision = adopted.decision;
        output.consistency = adopted.consistency;
        output.units = adopted.units;
        output.companions = adopted.companions;
        if (adopted.status != vbr_adopt_status::adopted) {
            return fail(server_vbr_artifact_import_status::adopt_failed,
                        impl_->counters.imports_refused);
        }
        output.status = server_vbr_artifact_import_status::ok;
        impl_->counters.imports_succeeded++;
        return output;
    } catch (...) {
        return fail(server_vbr_artifact_import_status::internal_error,
                    impl_->counters.imports_unavailable);
    }
}

bool server_vbr_artifact_store::resolve_control_reference(
        const std::string & reference,
        const std::string & tenant_key,
        vbr_artifact_package_view & package) noexcept {
    package.reset();
    try {
        llama_cache_acct_artifact_id artifact;
        if (!impl_->references.authorize(reference, tenant_key, artifact)) {
            return false;
        }
        return impl_->catalog.resolve_reference(artifact, package) ==
                   vbr_artifact_resolve_status::ok &&
               package && package.validate() == vbr_artifact_status::ok;
    } catch (...) {
        package.reset();
        return false;
    }
}

bool server_vbr_artifact_store::retain_host_payload(
        const std::string & reference,
        const std::string & tenant_key,
        std::shared_ptr<const server_prompt_cache_vbr_payload> & payload)
        noexcept {
    payload.reset();
    vbr_artifact_package_view package;
    llama_cache_acct_artifact_id artifact;
    // A catalog view can only name a package that passed the sealed
    // publication transaction. Retaining that immutable capability must not
    // re-read and rehash multi-GiB payloads; explicit import/control remains
    // the boundary that performs read-time validation.
    if (!impl_->references.authorize(reference, tenant_key, artifact) ||
        impl_->catalog.resolve_reference(artifact, package) !=
            vbr_artifact_resolve_status::ok ||
        !package) {
        return false;
    }
    payload = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(package));
    return bool(payload);
}

const server_vbr_artifact_store_counters &
server_vbr_artifact_store::counters() const noexcept {
    return impl_->counters;
}

uint32_t server_vbr_artifact_store::attention_children() const noexcept {
    return impl_->n_attention_children;
}

const char * server_vbr_artifact_store_create_failure_name(
        server_vbr_artifact_store_create_failure failure) noexcept {
    switch (failure) {
        case server_vbr_artifact_store_create_failure::none:
            return "none";
        case server_vbr_artifact_store_create_failure::ledger_missing:
            return "ledger_missing";
        case server_vbr_artifact_store_create_failure::
                budget_sampler_missing:
            return "budget_sampler_missing";
        case server_vbr_artifact_store_create_failure::topology_missing:
            return "topology_missing";
        case server_vbr_artifact_store_create_failure::
                pool_binding_missing:
            return "pool_binding_missing";
        case server_vbr_artifact_store_create_failure::lane_missing:
            return "lane_missing";
        case server_vbr_artifact_store_create_failure::
                attention_child_missing:
            return "attention_child_missing";
        case server_vbr_artifact_store_create_failure::
                ring_size_invalid:
            return "ring_size_invalid";
        case server_vbr_artifact_store_create_failure::
                chunk_size_invalid:
            return "chunk_size_invalid";
        case server_vbr_artifact_store_create_failure::
                budget_sample_failed:
            return "budget_sample_failed";
        case server_vbr_artifact_store_create_failure::
                ring_create_failed:
            return "ring_create_failed";
        case server_vbr_artifact_store_create_failure::internal_error:
            return "internal_error";
        case server_vbr_artifact_store_create_failure::_count:
            break;
    }
    return "invalid";
}

const char * server_vbr_artifact_capture_status_name(
        server_vbr_artifact_capture_status status) noexcept {
    switch (status) {
        case server_vbr_artifact_capture_status::ok: return "ok";
        case server_vbr_artifact_capture_status::unsupported: return "unsupported";
        case server_vbr_artifact_capture_status::unavailable: return "unavailable";
        case server_vbr_artifact_capture_status::invalid_slot: return "invalid_slot";
        case server_vbr_artifact_capture_status::slot_processing: return "slot_processing";
        case server_vbr_artifact_capture_status::stale_frontier: return "stale_frontier";
        case server_vbr_artifact_capture_status::identity_unavailable: return "identity_unavailable";
        case server_vbr_artifact_capture_status::unauthorized: return "unauthorized";
        case server_vbr_artifact_capture_status::required_companion_unavailable: return "required_companion_unavailable";
        case server_vbr_artifact_capture_status::admission_refused: return "admission_refused";
        case server_vbr_artifact_capture_status::source_changed: return "source_changed";
        case server_vbr_artifact_capture_status::internal_error: return "internal_error";
        case server_vbr_artifact_capture_status::_count: return "_count";
    }
    return "_count";
}

const char * server_vbr_artifact_import_status_name(
        server_vbr_artifact_import_status status) noexcept {
    switch (status) {
        case server_vbr_artifact_import_status::ok: return "ok";
        case server_vbr_artifact_import_status::unsupported: return "unsupported";
        case server_vbr_artifact_import_status::not_found: return "not_found";
        case server_vbr_artifact_import_status::invalid_slot: return "invalid_slot";
        case server_vbr_artifact_import_status::slot_processing: return "slot_processing";
        case server_vbr_artifact_import_status::slot_not_empty: return "slot_not_empty";
        case server_vbr_artifact_import_status::validation_failed: return "validation_failed";
        case server_vbr_artifact_import_status::report_only: return "report_only";
        case server_vbr_artifact_import_status::stage_failed: return "stage_failed";
        case server_vbr_artifact_import_status::adopt_failed: return "adopt_failed";
        case server_vbr_artifact_import_status::unavailable: return "unavailable";
        case server_vbr_artifact_import_status::internal_error: return "internal_error";
        case server_vbr_artifact_import_status::_count: break;
    }
    return "_count";
}
