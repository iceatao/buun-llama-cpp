#include "llama-vbr-artifact-validate.h"

#include "llama-cache-budget.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace {

vbr_manifest_validation_result terminal_result(
        vbr_manifest_validation_status status,
        vbr_import_decision decision = vbr_import_decision::reject) {
    vbr_manifest_validation_result result;
    result.status = status;
    result.decision = decision;
    return result;
}

vbr_import_decision fallback_decision(const vbr_adopt_policy & policy) {
    if (policy.allow_rebuild) {
        return vbr_import_decision::rebuild;
    }
    if (policy.allow_cold) {
        return vbr_import_decision::cold;
    }
    return vbr_import_decision::reject;
}

vbr_manifest_validation_status codec_status(vbr_artifact_status status) {
    switch (status) {
        case vbr_artifact_status::ok:
            return vbr_manifest_validation_status::validated;
        case vbr_artifact_status::unsupported_version:
            return vbr_manifest_validation_status::unsupported_artifact_version;
        case vbr_artifact_status::checksum_mismatch:
        case vbr_artifact_status::content_id_mismatch:
            return vbr_manifest_validation_status::checksum_or_digest_mismatch;
        case vbr_artifact_status::topology_mismatch:
            return vbr_manifest_validation_status::topology_mismatch;
        case vbr_artifact_status::generation_mismatch:
            return vbr_manifest_validation_status::generation_mismatch;
        case vbr_artifact_status::accounting_unavailable:
            return vbr_manifest_validation_status::accounting_unavailable;
        case vbr_artifact_status::invalid_argument:
        case vbr_artifact_status::malformed:
        case vbr_artifact_status::out_of_bounds:
            return vbr_manifest_validation_status::malformed;
        case vbr_artifact_status::internal_error:
        case vbr_artifact_status::_count:
            return vbr_manifest_validation_status::internal_error;
    }
    return vbr_manifest_validation_status::internal_error;
}

bool identity_matches(
        const vbr_artifact_reference_manifest & manifest,
        const vbr_adopt_policy & policy) {
    return manifest.identity.execution_identity ==
               policy.identity.execution_identity &&
           manifest.identity.adapter_config_identity ==
               policy.identity.adapter_config_identity &&
           manifest.identity.media_content_identity ==
               policy.identity.media_content_identity &&
           manifest.identity.sequence_epoch ==
               policy.identity.sequence_epoch &&
           manifest.identity.next_position ==
               policy.identity.requested_frontier;
}

bool target_domain_for(
        const vbr_artifact_portable_domain & portable,
        const vbr_adopt_policy & policy,
        llama_cache_acct_resource_domain & output) {
    const auto found = std::find_if(
        policy.domain_bindings.begin(), policy.domain_bindings.end(),
        [&](const llama_vbr_artifact_domain_binding & binding) {
            if (portable.kind ==
                    llama_cache_acct_domain_kind::device_topology) {
                return binding.topology_index == portable.topology_index &&
                       binding.device_ordinal == portable.device_ordinal &&
                       binding.domain.residency == portable.residency;
            }
            return binding.topology_index == UINT32_MAX &&
                   binding.device_ordinal == UINT16_MAX &&
                   binding.domain.residency == portable.residency &&
                   binding.domain.kind == portable.kind;
        });
    if (found == policy.domain_bindings.end()) {
        return false;
    }
    output = found->domain;
    return true;
}

const vbr_target_unit_snapshot * find_target_unit(
        const vbr_target_child_snapshot & child,
        uint32_t logical_unit_id) {
    const auto found = std::find_if(
        child.units.begin(), child.units.end(),
        [&](const vbr_target_unit_snapshot & unit) {
            return unit.logical_unit_id == logical_unit_id;
        });
    return found == child.units.end() ? nullptr : &*found;
}

const vbr_artifact_unit_reference * find_reference(
        const vbr_artifact_reference_manifest & manifest,
        const vbr_artifact_unit_descriptor & descriptor) {
    const auto found = std::find_if(
        manifest.unit_references.begin(),
        manifest.unit_references.end(),
        [&](const vbr_artifact_unit_reference & reference) {
            return reference.lineage_uuid == descriptor.lineage_uuid &&
                   reference.logical_unit_id ==
                       descriptor.logical_unit_id &&
                   reference.repr_gen == descriptor.repr_gen &&
                   reference.unit_version_id.valid();
        });
    return found == manifest.unit_references.end() ? nullptr : &*found;
}

bool authorized_placement_plan(
        const vbr_artifact_reference_manifest & manifest,
        uint32_t child_id,
        const vbr_artifact_unit_reference & reference,
        std::vector<vbr_artifact_stream_placement> & placements,
        std::vector<vbr_authorized_cell_run> & runs) {
    std::vector<uint32_t> cells;
    placements.clear();
    runs.clear();
    for (uint32_t stream_ref : reference.authorized_stream_refs) {
        const auto found = std::find_if(
            manifest.stream_placements.begin(),
            manifest.stream_placements.end(),
            [&](const vbr_artifact_stream_placement & placement) {
                return placement.child_id == child_id &&
                       placement.stream_index == stream_ref;
            });
        if (found == manifest.stream_placements.end()) {
            return false;
        }
        placements.push_back(*found);
        for (const auto & cell : found->cells) {
            cells.push_back(cell.physical_cell);
        }
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    for (uint32_t cell : cells) {
        if (runs.empty() ||
            uint64_t(runs.back().first_physical_cell) +
                    runs.back().cell_count != cell) {
            runs.push_back({ cell, 1 });
        } else {
            ++runs.back().cell_count;
        }
    }
    return !runs.empty();
}

bool stash_full_prefix(const vbr_artifact_stash_reference & stash) {
    if (stash.valid_rows == 0 ||
        stash.valid_rows > UINT32_MAX ||
        stash.captured_sink_count < stash.valid_rows) {
        return false;
    }
    std::vector<uint32_t> cells;
    for (const auto & page : stash.covered_sink_pages) {
        const uint64_t base =
            uint64_t(page.page_index) * VBR_GENERATION_PAGE_CELLS;
        for (uint32_t bit = 0; bit < VBR_GENERATION_PAGE_CELLS; ++bit) {
            if ((page.covered_mask[bit / 64] &
                 (uint64_t(1) << (bit % 64))) == 0) {
                continue;
            }
            const uint64_t cell = base + bit;
            if (cell > UINT32_MAX) {
                return false;
            }
            cells.push_back(uint32_t(cell));
        }
    }
    std::sort(cells.begin(), cells.end());
    if (cells.size() < stash.valid_rows) {
        return false;
    }
    for (uint32_t i = 0; i < stash.valid_rows; ++i) {
        if (cells[i] != i) {
            return false;
        }
    }
    return true;
}

bool same_geometry(
        const vbr_artifact_unit_descriptor & source,
        const vbr_target_unit_snapshot & target) {
    return source.n_stream == 1 &&
           source.unified == target.unified &&
           source.wm_cells <= target.wm_cells &&
           source.rank == target.rank &&
           source.dimensions == target.dimensions &&
           source.row_alignment == target.row_alignment &&
           source.recoverability == target.recoverability &&
           source.side == target.side &&
           source.layout == target.layout &&
           source.row_codec_version == target.row_codec_version &&
           target.shards.size() == source.shards.size();
}

bool shard_domain_matches(
        const vbr_artifact_shard_descriptor & source,
        const vbr_target_shard_snapshot & target,
        uint64_t source_wm_cells,
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy) {
    llama_cache_acct_resource_domain resolved;
    const vbr_artifact_portable_domain portable {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        source.topology_index,
        source.device_ordinal,
    };
    return target_domain_for(portable, policy, resolved) &&
           target.shard_index == source.shard_index &&
           target.topology_index == source.topology_index &&
           target.device_ordinal == source.device_ordinal &&
           source.topology_index < package.topologies().size() &&
           target.topology_digest ==
               package.topologies()[source.topology_index].digest &&
           target.logical_offset == source.logical_offset &&
           target.row_count == source.row_count &&
           target.domain == resolved &&
           target.pool_cookie != nullptr &&
           target.row_bytes != 0 &&
           source_wm_cells <= UINT64_MAX / target.row_bytes &&
           target.mapped_bytes >= source_wm_cells * target.row_bytes;
}

bool digest_nonzero(const std::array<uint8_t, 32> & digest) {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t value) {
        return value != 0;
    });
}

bool downward_recipe_complete(
        const vbr_artifact_unit_descriptor & source,
        vbr_repr_domain source_domain,
        const vbr_target_unit_snapshot & target) {
    vbr_downward_recipe resolved;
    const auto status = vbr_downward_resolve_recipe(
        static_cast<ggml_type>(source.current_type),
        static_cast<ggml_type>(target.current_type),
        static_cast<ggml_type>(target.controller_floor_type),
        target.downward_movable, resolved);
    return target.downward_supported &&
           target.downward_type == target.current_type &&
           target.downward_recipe_id == VBR_DOWNWARD_RECIPE_ID &&
           target.downward_recipe_version == VBR_DOWNWARD_RECIPE_VERSION &&
           status == vbr_downward_recipe_status::resolved &&
           resolved == target.downward_recipe &&
           digest_nonzero(target.downward_build_identity_digest) &&
           target.downward_row_bytes != 0 &&
           target.downward_mapped_bytes != 0 &&
           target.downward_transfer_bytes != 0 &&
           target.downward_codec_workspace_bytes != 0 &&
           target.downward_meansub_model_id >= 0 &&
           !(source_domain == vbr_repr_domain::tapped &&
             target.downward_domain == vbr_repr_domain::full);
}

bool same_representation(
        const vbr_artifact_unit_descriptor & source,
        const vbr_target_unit_snapshot & target) {
    return source.current_type == target.current_type &&
           source.last_source_type == target.last_source_type &&
           source.promote_hops == target.promote_hops &&
           source.last_transition == target.last_transition &&
           source.representation.kind == target.representation_kind &&
           source.representation.codec_id == target.codec_id &&
           source.representation.codec_version == target.codec_version &&
           source.representation.reference_digest ==
               target.representation_reference_digest &&
           source.representation.source_loss_history ==
               target.source_loss_history &&
           source.representation.checkpoint_codec_hops ==
               target.checkpoint_codec_hops &&
           source.codebook_digest == target.codebook_digest &&
           source.rotation_digest == target.rotation_digest &&
           source.meansub_digest == target.meansub_digest;
}

bool add_checked(uint64_t a, uint64_t b, uint64_t & output) {
    if (b > UINT64_MAX - a) {
        return false;
    }
    output = a + b;
    return true;
}

bool price_plan(
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_budget_config & config,
        const llama_cache_budget_plan & plan,
        llama_cache_budget_fit_state & state) {
    llama_cache_budget_coordinator coordinator;
    if (!coordinator.reset(snapshot, config)) {
        return false;
    }
    const auto result = coordinator.fits(plan);
    state = result.state;
    return result.accounting_serial == snapshot.serial;
}

bool accounting_plan(
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy,
        std::vector<llama_cache_transaction_leaf> & leaves,
        llama_cache_budget_plan & plan) {
    const auto & manifest = package.manifest();
    plan.accounting_serial = policy.accounting_snapshot->serial;
    using domain_key = std::tuple<uint8_t, uint8_t, uint16_t, uint64_t>;
    using accounting_key =
        std::tuple<uint8_t, uint8_t, uint8_t, uint16_t, uint64_t>;
    std::map<domain_key, uint64_t> totals;
    std::map<accounting_key, std::pair<uint64_t, uint64_t>> expected_existing;
    std::map<accounting_key, std::pair<uint64_t, uint64_t>> actual_existing;
    std::set<uint64_t> existing_allocations;
    const auto append_existing = [&](const vbr_artifact_allocation_view & value) {
        if (!value.allocation || value.artifact.v == 0 ||
            value.content.v == 0 || value.lineage.v == 0 ||
            value.logical == 0 || value.resident == 0) {
            return false;
        }
        if (!existing_allocations.insert(value.allocation.v).second) {
            return true;
        }
        const accounting_key key {
            uint8_t(value.category), uint8_t(value.domain.residency),
            uint8_t(value.domain.kind), value.domain.device_ordinal.v,
            value.domain.topology.v,
        };
        auto & actual = actual_existing[key];
        if (!add_checked(actual.first, value.logical, actual.first) ||
            !add_checked(actual.second, value.resident, actual.second)) {
            return false;
        }
        llama_cache_transaction_leaf leaf;
        leaf.category = value.category;
        leaf.domain = value.domain;
        leaf.attribution = {
            llama_cache_acct_attr_kind::artifact,
            -1,
            value.artifact,
        };
        leaf.expected_logical = value.logical;
        leaf.reserve_resident = 0;
        leaf.stage_resident = value.resident;
        leaf.artifact = value.artifact;
        leaf.content = value.content;
        leaf.lineage = value.lineage;
        leaf.existing_allocation = value.allocation;
        leaves.push_back(leaf);
        return true;
    };
    for (const auto & unit : package.units()) {
        for (const auto & allocation : unit.payload_allocations) {
            if (!append_existing(allocation)) {
                return false;
            }
        }
        for (const auto & allocation : unit.stash_allocations) {
            if (!append_existing(allocation)) {
                return false;
            }
        }
    }
    for (const auto & allocation : package.reference_allocations()) {
        if (allocation.category ==
                llama_cache_acct_category::artifact_descriptor_metadata ||
            allocation.category ==
                llama_cache_acct_category::artifact_reference_metadata) {
            continue;
        }
        if (!append_existing(allocation)) {
            return false;
        }
    }

    for (const auto & row : manifest.accounting) {
        if (row.role ==
                vbr_artifact_accounting_role::descriptor_metadata ||
            row.role ==
                vbr_artifact_accounting_role::reference_metadata) {
            continue;
        }
        llama_cache_acct_resource_domain domain;
        if (!target_domain_for(row.domain, policy, domain)) {
            return false;
        }
        const accounting_key key {
            uint8_t(vbr_artifact_accounting_category(row.role)),
            uint8_t(domain.residency), uint8_t(domain.kind),
            domain.device_ordinal.v, domain.topology.v,
        };
        auto & expected = expected_existing[key];
        if (!add_checked(
                expected.first, row.logical_bytes, expected.first) ||
            !add_checked(
                expected.second, row.resident_bytes, expected.second)) {
            return false;
        }
    }
    if (actual_existing != expected_existing) {
        return false;
    }

    // Descriptor/reference receipts are destination-local metadata. Unlike
    // immutable payload/stash/companion storage they deliberately mint fresh
    // allocations during F4 adoption.
    for (const auto & row : manifest.accounting) {
        if (row.role !=
                vbr_artifact_accounting_role::descriptor_metadata &&
            row.role !=
                vbr_artifact_accounting_role::reference_metadata) {
            continue;
        }
        llama_cache_acct_resource_domain domain;
        if (!target_domain_for(row.domain, policy, domain)) {
            return false;
        }
        const auto category = vbr_artifact_accounting_category(row.role);
        const auto source_receipt = std::find_if(
            package.reference_allocations().begin(),
            package.reference_allocations().end(),
            [&](const vbr_artifact_allocation_view & value) {
                return value.category == category &&
                       value.domain == domain &&
                       value.logical == row.logical_bytes &&
                       value.resident == row.resident_bytes &&
                       value.content.v != 0 && value.lineage.v != 0;
            });
        if (source_receipt == package.reference_allocations().end()) {
            return false;
        }
        llama_cache_transaction_leaf leaf;
        leaf.category = category;
        leaf.domain = domain;
        leaf.attribution = {
            row.attribution, -1, package.reference_artifact(),
        };
        leaf.expected_logical = row.logical_bytes;
        leaf.reserve_resident = row.resident_bytes;
        leaf.stage_resident = row.resident_bytes;
        leaf.artifact = package.reference_artifact();
        leaf.content = source_receipt->content;
        leaf.lineage = source_receipt->lineage;
        leaves.push_back(leaf);
        const auto key = std::make_tuple(
            uint8_t(domain.residency), uint8_t(domain.kind),
            domain.device_ordinal.v,
            domain.topology.v);
        uint64_t next;
        if (!add_checked(totals[key], leaf.reserve_resident, next)) {
            return false;
        }
        totals[key] = next;
    }
    for (const auto & total : totals) {
        llama_cache_budget_plan_entry entry;
        const auto found = std::find_if(
            leaves.begin(), leaves.end(),
            [&](const llama_cache_transaction_leaf & leaf) {
                return std::make_tuple(
                           uint8_t(leaf.domain.residency),
                           uint8_t(leaf.domain.kind),
                           leaf.domain.device_ordinal.v,
                           leaf.domain.topology.v) == total.first;
            });
        if (found == leaves.end()) {
            return false;
        }
        entry.domain = found->domain;
        entry.reserve_bytes = total.second;
        plan.entries.push_back(entry);
    }
    return true;
}

} // namespace

const char * vbr_import_schedule_status_name(
        vbr_import_schedule_status status) noexcept {
    switch (status) {
        case vbr_import_schedule_status::exact: return "exact";
        case vbr_import_schedule_status::downward: return "downward";
        case vbr_import_schedule_status::upward_same_domain:
            return "upward_same_domain";
        case vbr_import_schedule_status::upward_cross_domain:
            return "upward_cross_domain";
        case vbr_import_schedule_status::mixed_direction_unsupported:
            return "mixed_direction_unsupported";
        case vbr_import_schedule_status::unavailable: return "unavailable";
        case vbr_import_schedule_status::_count: break;
    }
    return "invalid";
}

vbr_import_schedule_status vbr_classify_import_schedule_units(
        const std::vector<vbr_import_schedule_unit> & units) noexcept {
    if (units.empty()) {
        return vbr_import_schedule_status::unavailable;
    }
    bool has_downward = false;
    bool has_upward = false;
    bool has_cross_domain_upward = false;
    for (const auto & unit : units) {
        if (unit.source_type < 0 || unit.target_type < 0) {
            return vbr_import_schedule_status::unavailable;
        }
        const auto source = static_cast<ggml_type>(unit.source_type);
        const auto target = static_cast<ggml_type>(unit.target_type);
        if (unit.source_domain != vbr_downward_tier_domain(source) ||
            unit.target_domain != vbr_downward_tier_domain(target)) {
            return vbr_import_schedule_status::unavailable;
        }
        vbr_downward_recipe recipe;
        const auto relation = vbr_downward_resolve_recipe(
            source, target, GGML_TYPE_TURBO1_TCQ, true, recipe);
        if (relation == vbr_downward_recipe_status::resolved) {
            has_downward = true;
        } else if (relation ==
                   vbr_downward_recipe_status::upward_forbidden) {
            has_upward = true;
            has_cross_domain_upward |=
                unit.source_domain != unit.target_domain;
        } else if (relation != vbr_downward_recipe_status::equal_tier) {
            return vbr_import_schedule_status::unavailable;
        }
    }
    if (has_downward && has_upward) {
        return vbr_import_schedule_status::mixed_direction_unsupported;
    }
    if (has_upward) {
        return has_cross_domain_upward
            ? vbr_import_schedule_status::upward_cross_domain
            : vbr_import_schedule_status::upward_same_domain;
    }
    return has_downward
        ? vbr_import_schedule_status::downward
        : vbr_import_schedule_status::exact;
}

bool vbr_quote_import_schedule(
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & package,
        vbr_import_schedule_quote & output) noexcept {
    output = {};
    try {
        if (!package || package.validate() != vbr_artifact_status::ok ||
            !package.manifest().manifest_digest.valid() ||
            target.memory_instance_cookie == 0 ||
            target.target_state_serial == 0 ||
            target.tree_shape_digest == 0 || target.policy_epoch == 0 ||
            target.children.empty() || package.units().empty()) {
            return false;
        }
        output.manifest_digest_ = package.manifest().manifest_digest;
        output.memory_instance_cookie_ = target.memory_instance_cookie;
        output.target_state_serial_ = target.target_state_serial;
        output.accounting_serial_ = target.accounting_serial;
        output.tree_shape_digest_ = target.tree_shape_digest;
        output.policy_epoch_ = target.policy_epoch;

        for (size_t child_index = 0;
             child_index < target.children.size(); ++child_index) {
            const auto & child = target.children[child_index];
            if (child.child_id != child_index) {
                output = {};
                return false;
            }
            for (size_t unit_index = 0;
                 unit_index < child.units.size(); ++unit_index) {
                if (child.units[unit_index].logical_unit_id != unit_index) {
                    output = {};
                    return false;
                }
            }
        }
        output.units_.reserve(package.units().size());
        for (const auto & source : package.units()) {
            if (source.descriptor.child_id >= target.children.size()) {
                output = {};
                return false;
            }
            const auto & child =
                target.children[source.descriptor.child_id];
            if (source.descriptor.logical_unit_id >= child.units.size() ||
                source.descriptor.current_type < 0 ||
                child.units[source.descriptor.logical_unit_id].current_type < 0) {
                output = {};
                return false;
            }
            const auto & unit =
                child.units[source.descriptor.logical_unit_id];
            const auto source_type = static_cast<ggml_type>(
                source.descriptor.current_type);
            const auto target_type = static_cast<ggml_type>(
                unit.current_type);
            const auto source_domain = vbr_downward_tier_domain(source_type);
            const auto target_domain = vbr_downward_tier_domain(target_type);
            if (unit.current_domain != target_domain) {
                output = {};
                return false;
            }
            for (const auto & shard : source.descriptor.shards) {
                uint64_t next = 0;
                if (!add_checked(
                        output.source_payload_bytes_,
                        shard.payload_bytes, next)) {
                    output = {};
                    return false;
                }
                output.source_payload_bytes_ = next;
            }
            for (const auto & shard : unit.shards) {
                uint64_t next = 0;
                if (!add_checked(
                        output.target_mapped_bytes_,
                        shard.mapped_bytes, next)) {
                    output = {};
                    return false;
                }
                output.target_mapped_bytes_ = next;
            }
            output.units_.push_back({
                source.descriptor.child_id,
                source.descriptor.logical_unit_id,
                source.descriptor.current_type,
                unit.current_type,
                source_domain,
                target_domain,
            });
        }
        output.status_ = vbr_classify_import_schedule_units(output.units_);
        return output.status_ != vbr_import_schedule_status::unavailable;
    } catch (...) {
        output = {};
        return false;
    }
}

bool vbr_import_schedule_quote_matches(
        const vbr_import_schedule_quote & quote,
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & package) noexcept {
    if (!package ||
        quote.status_ == vbr_import_schedule_status::unavailable ||
        quote.status_ == vbr_import_schedule_status::_count ||
        quote.manifest_digest_ != package.manifest().manifest_digest ||
        quote.memory_instance_cookie_ != target.memory_instance_cookie ||
        quote.target_state_serial_ != target.target_state_serial ||
        quote.accounting_serial_ != target.accounting_serial ||
        quote.tree_shape_digest_ != target.tree_shape_digest ||
        quote.policy_epoch_ != target.policy_epoch) {
        return false;
    }
    if (quote.units_.size() != package.units().size()) {
        return false;
    }
    uint64_t source_payload_bytes = 0;
    uint64_t target_mapped_bytes = 0;
    for (size_t i = 0; i < quote.units_.size(); ++i) {
        const auto & expected = quote.units_[i];
        const auto & source = package.units()[i].descriptor;
        if (expected.child_id != source.child_id ||
            expected.logical_unit_id != source.logical_unit_id ||
            expected.source_type != source.current_type ||
            source.child_id >= target.children.size() ||
            target.children[source.child_id].child_id != source.child_id ||
            source.logical_unit_id >=
                target.children[source.child_id].units.size()) {
            return false;
        }
        const auto & unit = target.children[source.child_id].units[
            source.logical_unit_id];
        if (unit.logical_unit_id != source.logical_unit_id ||
            expected.target_type != unit.current_type ||
            expected.source_domain != vbr_downward_tier_domain(
                static_cast<ggml_type>(source.current_type)) ||
            expected.target_domain != unit.current_domain ||
            expected.target_domain != vbr_downward_tier_domain(
                static_cast<ggml_type>(unit.current_type))) {
            return false;
        }
        for (const auto & shard : source.shards) {
            uint64_t next = 0;
            if (!add_checked(
                    source_payload_bytes, shard.payload_bytes, next)) {
                return false;
            }
            source_payload_bytes = next;
        }
        for (const auto & shard : unit.shards) {
            uint64_t next = 0;
            if (!add_checked(
                    target_mapped_bytes, shard.mapped_bytes, next)) {
                return false;
            }
            target_mapped_bytes = next;
        }
    }
    return source_payload_bytes == quote.source_payload_bytes_ &&
           target_mapped_bytes == quote.target_mapped_bytes_;
}

vbr_validated_manifest::vbr_validated_manifest(
        vbr_validated_manifest &&) noexcept = default;
vbr_validated_manifest & vbr_validated_manifest::operator=(
        vbr_validated_manifest &&) noexcept = default;
vbr_validated_manifest::~vbr_validated_manifest() = default;
vbr_parsed_companion_image::~vbr_parsed_companion_image() = default;

vbr_manifest_validation_result vbr_validate_unit_manifest_snapshot(
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy) noexcept {
    try {
        if (!package) {
            return terminal_result(
                vbr_manifest_validation_status::unsupported_artifact_version);
        }
        if (policy.schedule_quote != nullptr) {
            if (!vbr_import_schedule_quote_matches(
                    *policy.schedule_quote, target, package)) {
                return terminal_result(
                    vbr_manifest_validation_status::unavailable);
            }
            switch (policy.schedule_quote->status()) {
                case vbr_import_schedule_status::exact:
                case vbr_import_schedule_status::downward:
                    break;
                case vbr_import_schedule_status::upward_same_domain:
                case vbr_import_schedule_status::upward_cross_domain:
                case vbr_import_schedule_status::mixed_direction_unsupported:
                    return terminal_result(
                        vbr_manifest_validation_status::unavailable);
                case vbr_import_schedule_status::unavailable:
                case vbr_import_schedule_status::_count:
                    return terminal_result(
                        vbr_manifest_validation_status::unavailable);
            }
        } else {
            const auto codec = package.validate();
            if (codec != vbr_artifact_status::ok) {
                return terminal_result(codec_status(codec));
            }
        }
        const auto & manifest = package.manifest();
        if (manifest.version <
                VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT) {
            return terminal_result(
                vbr_manifest_validation_status::restore_metadata_missing,
                fallback_decision(policy));
        }
        if (manifest.version > VBR_UNIT_ARTIFACT_FORMAT_VERSION) {
            return terminal_result(
                vbr_manifest_validation_status::unsupported_artifact_version);
        }
        if (!policy.authorized) {
            return terminal_result(vbr_manifest_validation_status::unauthorized);
        }
        if (!identity_matches(manifest, policy)) {
            return terminal_result(vbr_manifest_validation_status::identity_mismatch);
        }
        if (policy.identity.tokens == nullptr ||
            manifest.token_block.tokens != *policy.identity.tokens ||
            manifest.token_block.tokens.size() !=
                size_t(manifest.identity.token_count)) {
            return terminal_result(
                vbr_manifest_validation_status::token_block_mismatch);
        }
        if (target.memory_instance_cookie == 0 ||
            target.tree_shape_digest == 0 || target.children.empty() ||
            target.children.size() !=
                manifest.generation.controllers.size()) {
            return terminal_result(
                vbr_manifest_validation_status::memory_tree_mismatch);
        }
        for (size_t i = 0; i < target.children.size(); ++i) {
            if (target.children[i].child_id !=
                    manifest.generation.controllers[i].child_id ||
                target.children[i].dependency_mode !=
                    manifest.generation.controllers[i].dependency_mode) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
        }
        if (!target.scheduler_idle) {
            return terminal_result(vbr_manifest_validation_status::target_not_idle);
        }
        if (!target.destination_sequence_absent) {
            return terminal_result(vbr_manifest_validation_status::target_not_empty);
        }
        std::set<uint32_t> target_child_ids;
        std::set<const void *> target_memory_cookies;
        std::vector<vbr_controller_instance_id> target_instances;
        for (const auto & child : target.children) {
            const bool duplicate_instance = std::any_of(
                target_instances.begin(), target_instances.end(),
                [&](vbr_controller_instance_id instance) {
                    return instance == child.instance_id;
                });
            if (child.memory_cookie == nullptr ||
                !target_child_ids.insert(child.child_id).second ||
                !target_memory_cookies.insert(child.memory_cookie).second ||
                duplicate_instance) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            target_instances.push_back(child.instance_id);
            if (!child.empty) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_empty);
            }
            if (!child.dedicated) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_dedicated);
            }
            if (!child.armed ||
                !vbr_controller_instance_id_is_set(child.instance_id)) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_armed);
            }
            if (child.policy_epoch != target.policy_epoch ||
                std::any_of(
                    child.units.begin(), child.units.end(),
                    [](const vbr_target_unit_snapshot & unit) {
                        return unit.n_stream != 1 || unit.v_trans;
                    })) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_armed);
            }
        }

        std::vector<vbr_validated_child_plan> child_plans;
        vbr_tracker_install_plan tracker;
        bool needs_live_rebase =
            manifest.consistency.kind ==
                vbr_artifact_consistency_kind::live_rebased;
        bool needs_downward = false;
        for (size_t controller_index = 0;
             controller_index < manifest.generation.controllers.size();
             ++controller_index) {
            const auto & controller =
                manifest.generation.controllers[controller_index];
            const auto * target_child = &target.children[controller_index];
            if (target_child->dependency_mode !=
                    controller.dependency_mode) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            const size_t source_units = size_t(std::count_if(
                package.units().begin(), package.units().end(),
                [&](const vbr_artifact_unit_view & unit) {
                    return unit.descriptor.child_id == controller.child_id;
                }));
            if (target_child->units.size() != source_units ||
                controller.units.size() != source_units) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            if (target_child->previously_observed) {
                needs_live_rebase = true;
            }
            if (controller.child_id >= manifest.controller_policy.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::policy_mismatch);
            }
            const auto & controller_policy =
                manifest.controller_policy[controller.child_id];
            const bool projected_policy = policy.allow_downward &&
                policy.downward_projection != nullptr &&
                policy.downward_projection->status ==
                    vbr_downward_policy_status::coherent &&
                controller_index <
                    policy.downward_projection->child_type_digests.size() &&
                controller_index <
                    policy.downward_projection->final_types.size() &&
                target_child->controller_policy.current_type_vector_digest ==
                    policy.downward_projection->child_type_digests[controller_index];
            if (controller_policy.child_id != controller.child_id ||
                controller_policy.dependency_mode !=
                    controller.dependency_mode ||
                target_child->controller_policy.degrade_order_digest !=
                    controller_policy.degrade_order_digest ||
                target_child->controller_policy.policy_digest !=
                    controller_policy.policy_digest ||
                target_child->controller_policy.floor_type !=
                    controller_policy.floor_type ||
                target_child->controller_policy.pressure_independent_settings !=
                    controller_policy.pressure_independent_settings ||
                target_child->controller_policy.n_stream !=
                    controller_policy.n_stream ||
                target_child->controller_policy.unified !=
                    controller_policy.unified ||
                target_child->controller_policy.wm_cells !=
                    controller_policy.wm_cells ||
                (!projected_policy &&
                 (target_child->controller_policy.cursor !=
                      controller_policy.cursor ||
                  target_child->controller_policy.current_type_vector_digest !=
                      controller_policy.current_type_vector_digest)) ||
                target_child->controller_policy.completed_wave !=
                    controller_policy.completed_wave) {
                return terminal_result(
                    vbr_manifest_validation_status::policy_mismatch);
            }
            if (!target_child->generation_compatible) {
                return terminal_result(
                    vbr_manifest_validation_status::generation_mismatch);
            }
            if (!target_child->ownership_compatible) {
                return terminal_result(
                    vbr_manifest_validation_status::ownership_mismatch);
            }
            if (!target_child->stash_compatible) {
                return terminal_result(
                    vbr_manifest_validation_status::stash_inconsistent);
            }
            vbr_tracker_install_child tracker_child;
            tracker_child.child_id = controller.child_id;
            tracker_child.transition =
                vbr_tracker_install_transition::native_clone;
            tracker_child.lineage_uuid = controller.lineage_uuid;
            tracker_child.target_instance = target_child->instance_id;
            tracker_child.global_generation =
                controller.global_generation;
            tracker_child.units = controller.units;
            tracker.children.push_back(std::move(tracker_child));
        }

        for (const auto & unit : package.units()) {
            const auto & descriptor = unit.descriptor;
            if (descriptor.child_id >= target.children.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            const auto * target_child =
                &target.children[descriptor.child_id];
            const auto * target_unit = find_target_unit(
                *target_child, descriptor.logical_unit_id);
            if (target_unit == nullptr) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            if (!same_geometry(descriptor, *target_unit)) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            if (descriptor.shards.empty()) {
                return terminal_result(
                    vbr_manifest_validation_status::topology_mismatch);
            }
            for (size_t shard_index = 0;
                 shard_index < descriptor.shards.size(); ++shard_index) {
                const auto & shard = descriptor.shards[shard_index];
                const auto & target_shard =
                    target_unit->shards[shard_index];
                if (!shard_domain_matches(
                        shard, target_shard, descriptor.wm_cells,
                        package, policy)) {
                    return terminal_result(
                        vbr_manifest_validation_status::topology_mismatch);
                }
            }
            bool downward = false;
            const auto & controller =
                manifest.generation.controllers[descriptor.child_id];
            if (descriptor.logical_unit_id >= controller.units.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::generation_mismatch);
            }
            const vbr_repr_domain source_domain =
                controller.units[descriptor.logical_unit_id].domain;
            const bool representation_matches =
                same_representation(descriptor, *target_unit);
            if (representation_matches &&
                target_unit->current_domain != source_domain) {
                return terminal_result(
                    vbr_manifest_validation_status::representation_mismatch);
            }
            if (!representation_matches) {
                if (descriptor.current_type ==
                        target_unit->current_type) {
                    const bool codec_identity_changed =
                        descriptor.codebook_digest !=
                            target_unit->codebook_digest ||
                        descriptor.rotation_digest !=
                            target_unit->rotation_digest ||
                        descriptor.meansub_digest !=
                            target_unit->meansub_digest;
                    return terminal_result(codec_identity_changed ?
                        vbr_manifest_validation_status::codebook_mismatch :
                        vbr_manifest_validation_status::representation_mismatch);
                }
                if (!policy.allow_downward ||
                    descriptor.codebook_digest != target_unit->codebook_digest ||
                    descriptor.rotation_digest != target_unit->rotation_digest ||
                    descriptor.meansub_digest != target_unit->meansub_digest ||
                    !downward_recipe_complete(
                        descriptor, source_domain, *target_unit)) {
                    return terminal_result(
                        vbr_manifest_validation_status::representation_mismatch);
                }
                downward = true;
                needs_downward = true;
            }
            if (!downward) {
                for (size_t i = 0; i < descriptor.shards.size(); ++i) {
                    if (target_unit->shards[i].row_bytes !=
                            descriptor.shards[i].row_bytes ||
                        target_unit->shards[i].mapped_bytes <
                            descriptor.shards[i].payload_bytes) {
                        return terminal_result(
                            vbr_manifest_validation_status::geometry_mismatch);
                    }
                }
            }
            const auto * reference = find_reference(manifest, descriptor);
            if (reference == nullptr) {
                return terminal_result(
                    vbr_manifest_validation_status::generation_mismatch);
            }
            std::vector<vbr_artifact_stream_placement> placements;
            std::vector<vbr_authorized_cell_run> runs;
            if (!authorized_placement_plan(
                    manifest, descriptor.child_id, *reference,
                    placements, runs)) {
                return terminal_result(
                    vbr_manifest_validation_status::ownership_mismatch);
            }
            for (const auto & placement : placements) {
                for (const auto & cell : placement.cells) {
                    if (cell.physical_cell >= descriptor.wm_cells) {
                        return terminal_result(
                            vbr_manifest_validation_status::ownership_mismatch);
                    }
                }
            }

            vbr_validated_stash_action stash_action =
                vbr_validated_stash_action::_count;
            switch (descriptor.clean_stash_state) {
                case vbr_artifact_clean_stash_state::absent_at_source:
                    stash_action =
                        vbr_validated_stash_action::none_at_source;
                    // A full-domain F16/raw unit has no clean-stash by
                    // construction; that honest absence is capture-exact.
                    // A tapped unit without its source-present clean prefix
                    // cannot preserve native generations and must rebase.
                    if (source_domain != vbr_repr_domain::full) {
                        needs_live_rebase = true;
                    }
                    break;
                case vbr_artifact_clean_stash_state::omitted_source_present:
                    stash_action =
                        vbr_validated_stash_action::omit_live_rebased;
                    needs_live_rebase = true;
                    break;
                case vbr_artifact_clean_stash_state::present:
                    if (!reference->has_stash_reference) {
                        return terminal_result(
                            vbr_manifest_validation_status::stash_inconsistent);
                    }
                    if (stash_full_prefix(reference->stash_reference)) {
                        stash_action =
                            vbr_validated_stash_action::restore_exact;
                    } else {
                        stash_action =
                            vbr_validated_stash_action::omit_live_rebased;
                        needs_live_rebase = true;
                    }
                    break;
                case vbr_artifact_clean_stash_state::_count:
                    return terminal_result(
                        vbr_manifest_validation_status::stash_inconsistent);
            }

            vbr_validated_child_plan plan;
            plan.child_id = descriptor.child_id;
            plan.dependency_mode = target_child->dependency_mode;
            plan.logical_unit_id = descriptor.logical_unit_id;
            plan.target_pool_cookie = target_unit->shards[0].pool_cookie;
            plan.descriptor = descriptor;
            plan.authorized_runs = runs;
            plan.placements = std::move(placements);
            plan.selected_target_type = target_unit->current_type;
            plan.source_domain = source_domain;
            plan.selected_target_domain = downward ?
                target_unit->downward_domain : target_unit->current_domain;
            plan.transcode_recipe_id = downward ?
                target_unit->downward_recipe_id : 0;
            plan.transcode_recipe_version = downward ?
                target_unit->downward_recipe_version : 0;
            plan.transcode_build_identity_digest = downward ?
                target_unit->downward_build_identity_digest :
                std::array<uint8_t, 32> {};
            // The degrade cursor is controller-wide. A mixed projection may
            // transcode only some units, but every unit publishes under the
            // same projected cursor.
            plan.target_controller_cursor =
                target_child->controller_policy.cursor;
            if (downward) {
                if (policy.downward_projection == nullptr ||
                    descriptor.child_id >=
                        policy.downward_projection->final_types.size() ||
                    descriptor.child_id >=
                        policy.downward_projection->child_type_digests.size() ||
                    descriptor.logical_unit_id >=
                        policy.downward_projection->final_types[descriptor.child_id].size() ||
                    policy.downward_projection->final_types[descriptor.child_id]
                        [descriptor.logical_unit_id] !=
                            static_cast<ggml_type>(target_unit->current_type) ||
                    !digest_nonzero(policy.downward_projection->tree_digest)) {
                    return terminal_result(
                        vbr_manifest_validation_status::policy_mismatch);
                }
                plan.transcode_recipe = target_unit->downward_recipe;
                plan.transcode_policy_digest =
                    policy.downward_projection->child_type_digests[
                        descriptor.child_id];
                plan.transcode_tree_digest =
                    policy.downward_projection->tree_digest;
                plan.transcode_meansub_model_id =
                    target_unit->downward_meansub_model_id;
                const auto build_identity = vbr_downward_build_identity(
                    target_unit->downward_recipe,
                    target_unit->downward_meansub_model_id,
                    target_unit->meansub_digest,
                    plan.transcode_policy_digest,
                    plan.transcode_tree_digest);
                if (!digest_nonzero(build_identity) ||
                    build_identity !=
                        target_unit->downward_build_identity_digest) {
                    return terminal_result(
                        vbr_manifest_validation_status::codebook_mismatch);
                }
            }
            plan.downward = downward;
            // Downward import regenerates the target-tier sink stash before
            // the canonical outgoing tapped edge. A source-tier stash is not a
            // target-tier byte image and must never be copied as if exact.
            plan.stash_action = downward
                ? vbr_validated_stash_action::omit_live_rebased
                : stash_action;
            plan.target_row_bytes = downward ?
                target_unit->downward_row_bytes :
                target_unit->shards[0].row_bytes;
            plan.target_mapped_bytes = downward ?
                target_unit->downward_mapped_bytes :
                target_unit->shards[0].mapped_bytes;
            plan.transfer_bytes = downward ?
                target_unit->downward_transfer_bytes : 0;
            plan.codec_workspace_bytes = downward ?
                target_unit->downward_codec_workspace_bytes : 0;
            plan.unit_reference = *reference;
            plan.controller_policy =
                manifest.controller_policy[descriptor.child_id];
            plan.operation_target.instance_id = target_child->instance_id;
            plan.operation_target.operation_class =
                vbr_operation_class::state_api;
            plan.operation_target.registrant_mask =
                vbr_registrant_bit(
                    vbr_mutation_registrant::whole_import);
            plan.operation_target.child_phase =
                vbr_operation_phase::mutate;
            plan.operation_target.stream = VBR_STREAM_ANY;
            plan.operation_target.seq_id = policy.destination_sequence;
            plan.operation_target.range = {
                0, std::numeric_limits<llama_pos>::max(),
            };
            if (unit.payload_shards.size() != descriptor.shards.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::malformed);
            }
            for (size_t i = 0; i < descriptor.shards.size(); ++i) {
                vbr_validated_shard_plan shard;
                shard.shard_index = uint32_t(i);
                shard.target_pool_cookie =
                    target_unit->shards[i].pool_cookie;
                if (!target_domain_for(
                        { llama_cache_acct_residency::device,
                          llama_cache_acct_domain_kind::device_topology,
                          descriptor.shards[i].topology_index,
                          descriptor.shards[i].device_ordinal },
                        policy, shard.domain)) {
                    return terminal_result(
                        vbr_manifest_validation_status::topology_mismatch);
                }
                shard.logical_offset =
                    descriptor.shards[i].logical_offset;
                shard.row_count = descriptor.shards[i].row_count;
                shard.row_bytes = descriptor.shards[i].row_bytes;
                shard.target_row_bytes = target_unit->shards[i].row_bytes;
                shard.target_mapped_bytes = target_unit->shards[i].mapped_bytes;
                shard.payload_bytes =
                    descriptor.shards[i].payload_bytes;
                shard.source = unit.payload_shards[i];
                if (!shard.source ||
                    shard.source->size() != shard.payload_bytes) {
                    return terminal_result(
                        vbr_manifest_validation_status::malformed);
                }
                plan.shards.push_back(std::move(shard));
            }
            child_plans.push_back(std::move(plan));
        }

        std::vector<vbr_validated_companion_plan> companion_plans;
        if (package.companions().size() != manifest.companions.size()) {
            return terminal_result(
                vbr_manifest_validation_status::required_companion_unavailable);
        }
        for (const auto & companion : package.companions()) {
            const auto target_companion = std::find_if(
                target.companions.begin(), target.companions.end(),
                [&](const vbr_target_companion_snapshot & value) {
                    return value.kind == companion.descriptor.kind &&
                           value.format_version ==
                               companion.descriptor.format_version &&
                           value.build_identity_digest ==
                               companion.descriptor.build_identity_digest;
                });
            if (target_companion == target.companions.end() ||
                !target_companion->available ||
                target_companion->target_cookie == nullptr ||
                !companion.payload ||
                companion.payload->size() !=
                    companion.descriptor.payload_bytes) {
                return terminal_result(
                    vbr_manifest_validation_status::required_companion_unavailable);
            }
            std::unique_ptr<vbr_parsed_companion_image> parsed;
            if (policy.parse_companion == nullptr ||
                !policy.parse_companion(
                    policy.context, companion.descriptor,
                    *companion.payload, *target_companion, parsed) ||
                !parsed || parsed->kind() != companion.descriptor.kind ||
                parsed->format_version() !=
                    companion.descriptor.format_version) {
                return terminal_result(
                    vbr_manifest_validation_status::required_companion_unavailable);
            }
            vbr_validated_companion_plan plan;
            plan.descriptor = companion.descriptor;
            plan.target_cookie = target_companion->target_cookie;
            plan.source = companion.payload;
            plan.parsed = std::move(parsed);
            companion_plans.push_back(std::move(plan));
        }

        if (policy.accounting_snapshot == nullptr ||
            policy.budget_config == nullptr ||
            policy.accounting_snapshot->schema_version !=
                LLAMA_CACHE_ACCT_SCHEMA_VERSION ||
            policy.accounting_snapshot->serial !=
                target.accounting_serial ||
            policy.accounting_snapshot->completeness_manifest !=
                llama_cache_acct_known::known) {
            return terminal_result(
                vbr_manifest_validation_status::accounting_unavailable);
        }
        std::vector<llama_cache_transaction_leaf> leaves;
        llama_cache_budget_plan native_plan;
        if (!accounting_plan(
                package, policy, leaves, native_plan)) {
            return terminal_result(
                vbr_manifest_validation_status::accounting_unavailable);
        }
        llama_cache_budget_fit_state native_fit;
        if (!price_plan(
                *policy.accounting_snapshot, *policy.budget_config,
                native_plan, native_fit) ||
            native_fit == llama_cache_budget_fit_state::unavailable) {
            return terminal_result(
                vbr_manifest_validation_status::budget_unavailable);
        }
        if (needs_downward) {
            if (policy.downward_budget_plan == nullptr ||
                policy.downward_projection == nullptr ||
                policy.read_downward_tree_digest == nullptr ||
                !digest_nonzero(policy.downward_projection->tree_digest)) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            // An empty downward projection is only acceptable through the
            // already-priced native plan. It is conservative (source bytes
            // are never smaller than the declared downward target) and keeps
            // an empty caller projection from making admission vacuous.
            llama_cache_budget_fit_state downward_fit = native_fit;
            if (policy.downward_budget_plan->accounting_serial !=
                    target.accounting_serial ||
                (!policy.downward_budget_plan->entries.empty() &&
                 !price_plan(
                     *policy.accounting_snapshot, *policy.budget_config,
                     *policy.downward_budget_plan, downward_fit)) ||
                downward_fit ==
                    llama_cache_budget_fit_state::unavailable) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            if (downward_fit == llama_cache_budget_fit_state::exceeds) {
                return terminal_result(
                    vbr_manifest_validation_status::validated,
                    fallback_decision(policy));
            }
        } else if (native_fit == llama_cache_budget_fit_state::exceeds) {
            return terminal_result(
                vbr_manifest_validation_status::validated,
                fallback_decision(policy));
        }

        vbr_import_decision decision;
        if (policy.schedule_quote != nullptr &&
            ((needs_downward && policy.schedule_quote->status() !=
                                  vbr_import_schedule_status::downward) ||
             (!needs_downward && policy.schedule_quote->status() !=
                                   vbr_import_schedule_status::exact))) {
            return terminal_result(
                vbr_manifest_validation_status::unavailable);
        }
        if (needs_downward) {
            decision = vbr_import_decision::downward_rebase;
        } else if (needs_live_rebase || !policy.allow_native) {
            if (!policy.allow_live_rebased) {
                decision = fallback_decision(policy);
            } else {
                decision = vbr_import_decision::live_rebased;
            }
        } else {
            if (!policy.native_instance_available) {
                return terminal_result(
                    vbr_manifest_validation_status::native_lineage_unavailable);
            }
            decision = vbr_import_decision::native_import;
        }
        if (decision == vbr_import_decision::rebuild ||
            decision == vbr_import_decision::cold ||
            decision == vbr_import_decision::reject) {
            return terminal_result(
                vbr_manifest_validation_status::validated, decision);
        }

        vbr_target_empty_fingerprint fingerprint;
        fingerprint.memory_instance_cookie = target.memory_instance_cookie;
        fingerprint.target_state_serial = target.target_state_serial;
        fingerprint.accounting_serial = target.accounting_serial;
        fingerprint.tree_shape_digest = target.tree_shape_digest;
        fingerprint.policy_epoch = target.policy_epoch;
        for (const auto & child : target.children) {
            fingerprint.previously_observed |= child.previously_observed;
            fingerprint.children.push_back({
                child.child_id, child.memory_cookie,
                child.state_serial, child.instance_id,
            });
        }
        if (policy.recheck_target_empty == nullptr ||
            !policy.recheck_target_empty(policy.context, fingerprint) ||
            policy.read_accounting_serial == nullptr ||
            policy.read_policy_epoch == nullptr ||
            policy.read_accounting_serial(policy.context) !=
                target.accounting_serial ||
            policy.read_policy_epoch(policy.context) !=
                target.policy_epoch) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        if (policy.adoption_nonce == 0) {
            return terminal_result(vbr_manifest_validation_status::internal_error);
        }
        if (decision != vbr_import_decision::native_import) {
            for (auto & child : tracker.children) {
                if (child.child_id >= target.children.size()) {
                    return terminal_result(
                        vbr_manifest_validation_status::internal_error);
                }
                child.transition =
                    vbr_tracker_install_transition::whole_import;
                child.lineage_uuid =
                    target.children[child.child_id].lineage_uuid;
                child.global_generation = 1;
                child.units.assign(
                    target.children[child.child_id].units.size(), {});
                std::vector<bool> initialized(child.units.size(), false);
                for (const auto & plan : child_plans) {
                    if (plan.child_id != child.child_id) {
                        continue;
                    }
                    if (plan.logical_unit_id >= child.units.size() ||
                        initialized[plan.logical_unit_id]) {
                        return terminal_result(
                            vbr_manifest_validation_status::internal_error);
                    }
                    auto & fresh = child.units[plan.logical_unit_id];
                    fresh.repr_gen = 1;
                    fresh.current_type = plan.selected_target_type;
                    fresh.last_source_type = plan.selected_target_type;
                    fresh.domain = plan.selected_target_domain;
                    fresh.promote_hops = 0;
                    fresh.last_transition =
                        vbr_repr_transition::whole_import;
                    initialized[plan.logical_unit_id] = true;
                }
                if (std::find(initialized.begin(), initialized.end(), false) !=
                        initialized.end()) {
                    return terminal_result(
                        vbr_manifest_validation_status::internal_error);
                }
            }
        }
        vbr_artifact_package_view retained;
        if (package.retain(retained) != vbr_artifact_resolve_status::ok) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        auto proof = std::unique_ptr<vbr_validated_manifest>(
            new vbr_validated_manifest());
        proof->source_lease_ = std::move(retained);
        proof->decision_ = decision;
        proof->target_ = std::move(fingerprint);
        proof->manifest_digest_ = manifest.manifest_digest;
        proof->capture_generation_id_ =
            manifest.capture_generation_id;
        proof->authenticated_identity_ = policy.identity;
        proof->token_block_ = manifest.token_block;
        proof->children_ = std::move(child_plans);
        proof->companions_ = std::move(companion_plans);
        proof->accounting_leaves_ = std::move(leaves);
        proof->tracker_install_ = std::move(tracker);
        proof->adoption_nonce_ = policy.adoption_nonce;
        proof->recheck_context_ = policy.context;
        proof->recheck_target_empty_ = policy.recheck_target_empty;
        proof->read_accounting_serial_ = policy.read_accounting_serial;
        proof->read_policy_epoch_ = policy.read_policy_epoch;
        proof->read_downward_tree_digest_ =
            policy.read_downward_tree_digest;

        vbr_manifest_validation_result result;
        result.status = vbr_manifest_validation_status::validated;
        result.decision = decision;
        result.proof = std::move(proof);
        return result;
    } catch (...) {
        return terminal_result(vbr_manifest_validation_status::internal_error);
    }
}

vbr_manifest_validation_result vbr_validate_unit_manifest(
        llama_memory_i & target,
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy) noexcept {
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&target, tree)) {
            return terminal_result(
                vbr_manifest_validation_status::memory_tree_mismatch);
        }
        if (policy.inspect_target == nullptr) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        vbr_target_validation_snapshot snapshot;
        if (!policy.inspect_target(
                policy.context, target, tree, snapshot)) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        // The snapshot carries one child per ATTENTION tree child; recurrent
        // children travel as companions (checked by the snapshot core against
        // the manifest's companion set). Pairing against tree.size() would make
        // every hybrid (attention+recurrent) import unsatisfiable: the snapshot
        // core requires children == controllers == attention count.
        size_t snapshot_index = 0;
        for (const auto & child : tree) {
            if (child.attention == nullptr) {
                continue;
            }
            if (snapshot_index >= snapshot.children.size() ||
                snapshot.children[snapshot_index].child_id != child.child_id ||
                snapshot.children[snapshot_index].dependency_mode !=
                    child.dependency_mode) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            ++snapshot_index;
        }
        if (snapshot_index != snapshot.children.size()) {
            return terminal_result(
                vbr_manifest_validation_status::memory_tree_mismatch);
        }
        return vbr_validate_unit_manifest_snapshot(
            snapshot, package, policy);
    } catch (...) {
        return terminal_result(vbr_manifest_validation_status::internal_error);
    }
}

const char * vbr_import_decision_name(
        vbr_import_decision decision) noexcept {
    switch (decision) {
        case vbr_import_decision::native_import: return "native_import";
        case vbr_import_decision::live_rebased: return "live_rebased";
        case vbr_import_decision::downward_rebase: return "downward_rebase";
        case vbr_import_decision::rebuild: return "rebuild";
        case vbr_import_decision::cold: return "cold";
        case vbr_import_decision::reject: return "reject";
        case vbr_import_decision::_count: break;
    }
    return "invalid";
}

const char * vbr_manifest_validation_status_name(
        vbr_manifest_validation_status status) noexcept {
    switch (status) {
        case vbr_manifest_validation_status::validated: return "validated";
        case vbr_manifest_validation_status::unauthorized: return "unauthorized";
        case vbr_manifest_validation_status::unsupported_artifact_version: return "unsupported_artifact_version";
        case vbr_manifest_validation_status::restore_metadata_missing: return "restore_metadata_missing";
        case vbr_manifest_validation_status::malformed: return "malformed";
        case vbr_manifest_validation_status::checksum_or_digest_mismatch: return "checksum_or_digest_mismatch";
        case vbr_manifest_validation_status::identity_mismatch: return "identity_mismatch";
        case vbr_manifest_validation_status::token_block_mismatch: return "token_block_mismatch";
        case vbr_manifest_validation_status::memory_tree_mismatch: return "memory_tree_mismatch";
        case vbr_manifest_validation_status::target_not_idle: return "target_not_idle";
        case vbr_manifest_validation_status::target_not_empty: return "target_not_empty";
        case vbr_manifest_validation_status::target_not_dedicated: return "target_not_dedicated";
        case vbr_manifest_validation_status::target_not_armed: return "target_not_armed";
        case vbr_manifest_validation_status::geometry_mismatch: return "geometry_mismatch";
        case vbr_manifest_validation_status::topology_mismatch: return "topology_mismatch";
        case vbr_manifest_validation_status::representation_mismatch: return "representation_mismatch";
        case vbr_manifest_validation_status::codebook_mismatch: return "codebook_mismatch";
        case vbr_manifest_validation_status::policy_mismatch: return "policy_mismatch";
        case vbr_manifest_validation_status::generation_mismatch: return "generation_mismatch";
        case vbr_manifest_validation_status::ownership_mismatch: return "ownership_mismatch";
        case vbr_manifest_validation_status::stash_inconsistent: return "stash_inconsistent";
        case vbr_manifest_validation_status::required_companion_unavailable: return "required_companion_unavailable";
        case vbr_manifest_validation_status::accounting_unavailable: return "accounting_unavailable";
        case vbr_manifest_validation_status::budget_unavailable: return "budget_unavailable";
        case vbr_manifest_validation_status::native_lineage_unavailable: return "native_lineage_unavailable";
        case vbr_manifest_validation_status::unavailable: return "unavailable";
        case vbr_manifest_validation_status::internal_error: return "internal_error";
        case vbr_manifest_validation_status::_count: break;
    }
    return "invalid";
}
