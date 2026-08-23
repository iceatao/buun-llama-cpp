#include "llama-vbr-occupied-replacement.h"

#include "llama-vbr-artifact-validate.h"
#include "llama-vbr-identity-digest.h"
#include "llama-sha256.h"

#include <algorithm>
#include <limits>
#include <utility>

struct vbr_occupied_replacement_guard::map {
    std::vector<vbr_occupied_replacement_cell_mapping> mappings;
    std::vector<vbr_occupied_replacement_relocation_run> relocation_runs;
    std::vector<vbr_occupied_replacement_relocation_run> recovery_runs;
    vbr_occupied_replacement_strategy strategy =
        vbr_occupied_replacement_strategy::_count;
    uint64_t packed_rows_expanded = 0;
};

namespace {

bool occupied_projected_packed_rows(
        const vbr_artifact_package_view & incoming,
        const vbr_artifact_stream_placement & placement,
        std::vector<uint64_t> & packed_rows) {
    packed_rows.clear();
    packed_rows.resize(placement.cells.size(), UINT64_MAX);
    const auto & proofs = incoming.projected_ranges();
    if (proofs.empty()) {
        for (size_t i = 0; i < placement.cells.size(); ++i) {
            packed_rows[i] = placement.cells[i].physical_cell;
        }
        return true;
    }
    if (proofs.size() > VBR_OCCUPIED_REPLACEMENT_MAX_RUNS) {
        return false;
    }

    std::vector<const vbr_artifact_cell_placement *> physical;
    physical.reserve(placement.cells.size());
    for (const auto & cell : placement.cells) {
        physical.push_back(&cell);
    }
    std::sort(physical.begin(), physical.end(), [](const auto * lhs,
                                                   const auto * rhs) {
        return lhs->physical_cell < rhs->physical_cell;
    });
    for (size_t i = 1; i < physical.size(); ++i) {
        if (physical[i-1]->physical_cell == physical[i]->physical_cell) {
            return false;
        }
    }

    size_t expected_proofs = 0;
    for (const auto & unit : incoming.units()) {
        if (unit.descriptor.shards.size() > SIZE_MAX-expected_proofs) {
            return false;
        }
        expected_proofs += unit.descriptor.shards.size();
    }
    if (proofs.size() != expected_proofs) {
        return false;
    }
    std::vector<std::pair<uint32_t, uint32_t>> seen;
    seen.reserve(proofs.size());
    std::vector<std::pair<uint64_t, uint64_t>> canonical_ranges;
    std::vector<std::pair<uint64_t, uint64_t>> normalized_ranges;
    bool canonical = false;
    for (const auto & selected : proofs) {
        if (selected.unit_index >= incoming.units().size() || !selected.proof) {
            return false;
        }
        const auto & unit = incoming.units()[selected.unit_index];
        const auto shard = std::find_if(
            unit.descriptor.shards.begin(), unit.descriptor.shards.end(),
            [&](const auto & value) {
                return value.shard_index == selected.shard_index;
            });
        if (shard == unit.descriptor.shards.end() || shard->row_bytes == 0 ||
            selected.proof.root() != shard->section_checksum ||
            selected.proof.total_bytes() != shard->payload_bytes) {
            return false;
        }
        seen.push_back({ selected.unit_index, selected.shard_index });
        normalized_ranges.clear();
        size_t selected_rows = 0;
        for (const auto & range : selected.proof.ranges()) {
            if (range.size == 0 || range.offset%shard->row_bytes != 0 ||
                range.size%shard->row_bytes != 0 ||
                range.offset > shard->payload_bytes ||
                range.size > shard->payload_bytes-range.offset) {
                return false;
            }
            const uint64_t first = range.offset/shard->row_bytes;
            const uint64_t count = range.size/shard->row_bytes;
            if (count > physical.size()-selected_rows) {
                return false;
            }
            if (!normalized_ranges.empty() &&
                normalized_ranges.back().first <=
                    UINT64_MAX-normalized_ranges.back().second &&
                normalized_ranges.back().first+
                    normalized_ranges.back().second == first &&
                count <= UINT64_MAX-normalized_ranges.back().second) {
                normalized_ranges.back().second += count;
            } else {
                normalized_ranges.push_back({ first, count });
            }
            selected_rows += size_t(count);
        }
        if (selected_rows != physical.size()) {
            return false;
        }
        if (!canonical) {
            canonical_ranges = normalized_ranges;
            size_t cell_index = 0;
            for (const auto & range : canonical_ranges) {
                for (uint64_t row = 0; row < range.second; ++row) {
                    const auto logical = physical[cell_index++]->logical_position;
                    if (logical < 0 || uint64_t(logical) >= packed_rows.size() ||
                        row > UINT64_MAX-range.first) {
                        return false;
                    }
                    packed_rows[size_t(logical)] = range.first+row;
                }
            }
        } else if (normalized_ranges != canonical_ranges) {
            return false;
        }
        canonical = true;
    }
    std::sort(seen.begin(), seen.end());
    return canonical &&
        std::adjacent_find(seen.begin(), seen.end()) == seen.end() &&
        std::none_of(packed_rows.begin(), packed_rows.end(),
                     [](uint64_t value) { return value == UINT64_MAX; });
}

bool occupied_unit_schedule_equal(
        const vbr_artifact_unit_descriptor & incoming,
        const vbr_artifact_unit_descriptor & recovery) noexcept {
    if (incoming.child_id != 0 || recovery.child_id != 0 ||
        incoming.logical_unit_id != recovery.logical_unit_id ||
        incoming.current_type != recovery.current_type ||
        incoming.last_source_type != recovery.last_source_type ||
        incoming.promote_hops != recovery.promote_hops ||
        incoming.last_transition != recovery.last_transition ||
        incoming.representation.kind != recovery.representation.kind ||
        incoming.representation.codec_id != recovery.representation.codec_id ||
        incoming.representation.codec_version !=
            recovery.representation.codec_version ||
        incoming.representation.reference_digest !=
            recovery.representation.reference_digest ||
        incoming.representation.source_loss_history !=
            recovery.representation.source_loss_history ||
        incoming.representation.checkpoint_codec_hops !=
            recovery.representation.checkpoint_codec_hops ||
        incoming.recoverability != recovery.recoverability ||
        incoming.side != recovery.side ||
        incoming.layout != vbr_artifact_layout::row_major ||
        recovery.layout != vbr_artifact_layout::row_major ||
        incoming.n_stream != 1 || recovery.n_stream != 1 ||
        !incoming.unified || !recovery.unified ||
        incoming.rank != recovery.rank ||
        incoming.dimensions[1] != recovery.dimensions[1] ||
        incoming.dimensions[2] != recovery.dimensions[2] ||
        incoming.dimensions[3] != recovery.dimensions[3] ||
        incoming.row_alignment != recovery.row_alignment ||
        incoming.row_codec_version != recovery.row_codec_version ||
        incoming.codebook_digest != recovery.codebook_digest ||
        incoming.rotation_digest != recovery.rotation_digest ||
        incoming.meansub_digest != recovery.meansub_digest ||
        incoming.meansub_model_id != recovery.meansub_model_id ||
        incoming.meansub_layer != recovery.meansub_layer ||
        incoming.meansub_baked != recovery.meansub_baked ||
        incoming.clean_stash_state !=
            vbr_artifact_clean_stash_state::absent_at_source ||
        recovery.clean_stash_state !=
            vbr_artifact_clean_stash_state::absent_at_source ||
        incoming.clean_stash.valid_rows != 0 ||
        recovery.clean_stash.valid_rows != 0 ||
        !incoming.clean_stash.shards.empty() ||
        !recovery.clean_stash.shards.empty() ||
        incoming.shards.size() != recovery.shards.size() ||
        incoming.shards.empty()) {
        return false;
    }
    for (size_t i = 0; i < incoming.shards.size(); ++i) {
        const auto & a = incoming.shards[i];
        const auto & b = recovery.shards[i];
        if (a.shard_index != i || b.shard_index != i ||
            a.topology_index != b.topology_index ||
            a.device_ordinal != b.device_ordinal ||
            a.column_count != b.column_count ||
            a.row_bytes != b.row_bytes) {
            return false;
        }
    }
    return true;
}

bool occupied_target_unit_matches(
        const vbr_target_unit_snapshot & target,
        const vbr_artifact_unit_descriptor & descriptor,
        const vbr_checkpoint_unit_generation & captured,
        const vbr_occupied_replacement_unit_currency & live) noexcept {
    if (target.child_id != 0 || live.child_id != 0 ||
        target.logical_unit_id != descriptor.logical_unit_id ||
        live.logical_unit_id != descriptor.logical_unit_id ||
        descriptor.repr_gen != captured.repr_gen ||
        live.generation.repr_gen != captured.repr_gen ||
        target.current_type != descriptor.current_type ||
        captured.current_type != descriptor.current_type ||
        live.generation.current_type != descriptor.current_type ||
        target.last_source_type != descriptor.last_source_type ||
        captured.last_source_type != descriptor.last_source_type ||
        live.generation.last_source_type != descriptor.last_source_type ||
        target.promote_hops != descriptor.promote_hops ||
        captured.promote_hops != descriptor.promote_hops ||
        live.generation.promote_hops != descriptor.promote_hops ||
        target.last_transition != descriptor.last_transition ||
        captured.last_transition != descriptor.last_transition ||
        live.generation.last_transition != descriptor.last_transition ||
        target.current_domain != captured.domain ||
        live.generation.domain != captured.domain ||
        target.representation_kind != descriptor.representation.kind ||
        target.codec_id != descriptor.representation.codec_id ||
        target.codec_version != descriptor.representation.codec_version ||
        target.representation_reference_digest !=
            descriptor.representation.reference_digest ||
        target.source_loss_history !=
            descriptor.representation.source_loss_history ||
        target.checkpoint_codec_hops !=
            descriptor.representation.checkpoint_codec_hops ||
        target.recoverability != descriptor.recoverability ||
        target.side != descriptor.side || target.layout != descriptor.layout ||
        target.row_codec_version != descriptor.row_codec_version ||
        target.codebook_digest != descriptor.codebook_digest ||
        target.rotation_digest != descriptor.rotation_digest ||
        target.meansub_digest != descriptor.meansub_digest ||
        target.meansub_model_id != descriptor.meansub_model_id ||
        target.meansub_layer != descriptor.meansub_layer ||
        target.meansub_baked != descriptor.meansub_baked ||
        target.n_stream != 1 || !target.unified || target.v_trans ||
        target.wm_cells != descriptor.wm_cells ||
        target.rank != descriptor.rank ||
        target.dimensions != descriptor.dimensions ||
        target.row_alignment != descriptor.row_alignment ||
        target.shards.size() != descriptor.shards.size()) {
        return false;
    }
    for (size_t i = 0; i < target.shards.size(); ++i) {
        const auto & a = target.shards[i];
        const auto & b = descriptor.shards[i];
        if (a.shard_index != i || a.pool_cookie == nullptr ||
            a.topology_index != b.topology_index ||
            a.device_ordinal != b.device_ordinal ||
            a.logical_offset != b.logical_offset ||
            a.row_count != b.row_count || a.row_bytes != b.row_bytes ||
            a.mapped_bytes < b.payload_bytes) {
            return false;
        }
    }
    return true;
}

std::array<uint8_t, 32> occupied_currency_digest(
        const vbr_target_validation_snapshot & target,
        const vbr_occupied_replacement_observation & observation) {
    llama_sha256_writer writer;
    static constexpr char domain[] =
        "buun.vbr.occupied-replacement/currency/v1";
    writer.string(domain, sizeof(domain)-1);
    writer.u64(target.memory_instance_cookie);
    writer.u64(target.target_state_serial);
    writer.u64(target.accounting_serial);
    writer.u64(target.tree_shape_digest);
    writer.u64(target.policy_epoch);
    writer.u64(target.scheduler_idle ? 1 : 0);
    writer.u64(target.destination_sequence_absent ? 1 : 0);
    writer.u64(target.children.size());
    for (const auto & child : target.children) {
        writer.u64(child.child_id);
        writer.u64(uint64_t(child.dependency_mode));
        writer.u64(uint64_t(reinterpret_cast<uintptr_t>(child.memory_cookie)));
        writer.u64(child.empty ? 1 : 0);
        writer.u64(child.dedicated ? 1 : 0);
        writer.u64(child.armed ? 1 : 0);
        writer.u64(child.lineage_uuid.hi);
        writer.u64(child.lineage_uuid.lo);
        writer.u64(child.instance_id.hi);
        writer.u64(child.instance_id.lo);
        writer.u64(child.state_serial);
        writer.u64(child.policy_epoch);
        writer.u64(child.units.size());
        for (const auto & unit : child.units) {
            writer.u64(unit.logical_unit_id);
            writer.u64(uint64_t(unit.current_type));
            writer.u64(unit.codec_id);
            writer.u64(unit.codec_version);
            writer.bytes(unit.representation_reference_digest.data(),
                         unit.representation_reference_digest.size());
            writer.bytes(unit.codebook_digest.data(),
                         unit.codebook_digest.size());
            writer.bytes(unit.rotation_digest.data(),
                         unit.rotation_digest.size());
            writer.bytes(unit.meansub_digest.data(),
                         unit.meansub_digest.size());
            writer.u64(unit.wm_cells);
        }
    }
    writer.u64(target.companions.size());
    writer.u64(uint64_t(observation.destination));
    writer.u64(observation.sequence_epoch);
    writer.u64(observation.controller_generation);
    writer.u64(observation.representation_epoch);
    writer.u64(observation.cell_capacity);
    writer.u64(observation.cell_count);
    for (size_t i = 0; i < observation.cell_count; ++i) {
        const auto & cell = observation.cells[i];
        writer.u64(cell.stream_index);
        writer.u64(cell.physical_cell);
        writer.u64(uint64_t(cell.logical_position));
        writer.u64(uint64_t(cell.ext_x));
        writer.u64(uint64_t(cell.ext_y));
        writer.u64(uint64_t(cell.owner_sequence));
        writer.u64(cell.reference_count);
    }
    writer.u64(observation.unit_count);
    for (size_t i = 0; i < observation.unit_count; ++i) {
        const auto & unit = observation.units[i];
        writer.u64(unit.child_id);
        writer.u64(unit.logical_unit_id);
        writer.u64(unit.generation.repr_gen);
        writer.u64(unit.generation.publish_seq);
        writer.u64(uint64_t(unit.generation.current_type));
        writer.u64(uint64_t(unit.generation.last_source_type));
        writer.u64(uint64_t(unit.generation.domain));
        writer.u64(unit.generation.promote_hops);
        writer.u64(uint64_t(unit.generation.last_transition));
    }
    return writer.finish();
}

vbr_occupied_replacement_guard_status occupied_guard_validate(
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & incoming,
        const vbr_artifact_package_view & recovery,
        const vbr_occupied_replacement_observation & observation,
        vbr_occupied_replacement_guard::map * build,
        const vbr_occupied_replacement_guard::map * expected) {
    if (!incoming || !recovery || observation.destination < 0 ||
        observation.cell_capacity == 0 ||
        (observation.cell_count != 0 && observation.cells == nullptr) ||
        (observation.unit_count != 0 && observation.units == nullptr)) {
        return vbr_occupied_replacement_guard_status::invalid_argument;
    }
    if (target.memory_instance_cookie == 0 || target.tree_shape_digest == 0 ||
        target.accounting_serial == 0 || target.policy_epoch == 0 ||
        target.destination_sequence_absent || target.children.size() != 1 ||
        target.companions.size() != 0 ||
        incoming.companions().size() != 0 || recovery.companions().size() != 0 ||
        incoming.manifest().generation.controllers.size() != 1 ||
        recovery.manifest().generation.controllers.size() != 1 ||
        incoming.manifest().controller_policy.size() != 1 ||
        recovery.manifest().controller_policy.size() != 1 ||
        incoming.manifest().stream_placements.size() != 1 ||
        recovery.manifest().stream_placements.size() != 1) {
        return target.companions.empty() && incoming.companions().empty() &&
                       recovery.companions().empty()
            ? vbr_occupied_replacement_guard_status::unsupported_tree
            : vbr_occupied_replacement_guard_status::companion_unavailable;
    }
    if (observation.cell_capacity > VBR_OCCUPIED_REPLACEMENT_MAX_CELLS) {
        return vbr_occupied_replacement_guard_status::cell_limit_exceeded;
    }
    const auto & child = target.children.front();
    const auto & recovery_controller =
        recovery.manifest().generation.controllers.front();
    const auto & incoming_controller =
        incoming.manifest().generation.controllers.front();
    const auto & recovery_policy = recovery.manifest().controller_policy.front();
    const auto & incoming_policy = incoming.manifest().controller_policy.front();
    if (child.child_id != 0 || child.empty || !child.dedicated || !child.armed ||
        !child.generation_compatible || !child.ownership_compatible ||
        !child.stash_compatible || child.policy_epoch != target.policy_epoch ||
        recovery_controller.child_id != 0 || incoming_controller.child_id != 0 ||
        recovery_controller.dependency_mode !=
            checkpoint_child_dependency_mode::live_guarded ||
        incoming_controller.dependency_mode !=
            checkpoint_child_dependency_mode::live_guarded ||
        child.dependency_mode != recovery_controller.dependency_mode ||
        child.lineage_uuid != recovery_controller.lineage_uuid ||
        observation.controller_generation !=
            recovery_controller.global_generation ||
        observation.representation_epoch != child.state_serial ||
        !vbr_artifact_controller_policy_equal(child.controller_policy,
                                               recovery_policy)) {
        return vbr_occupied_replacement_guard_status::generation_mismatch;
    }
    if (incoming_policy.degrade_order_digest != recovery_policy.degrade_order_digest ||
        incoming_policy.floor_type != recovery_policy.floor_type ||
        incoming_policy.pressure_independent_settings !=
            recovery_policy.pressure_independent_settings ||
        incoming_policy.n_stream != 1 || recovery_policy.n_stream != 1 ||
        !incoming_policy.unified || !recovery_policy.unified ||
        incoming_policy.current_type_vector_digest !=
            recovery_policy.current_type_vector_digest ||
        incoming.units().size() != recovery.units().size() ||
        incoming.units().empty() || child.units.size() != recovery.units().size() ||
        observation.unit_count != recovery.units().size() ||
        recovery_controller.units.size() != recovery.units().size() ||
        incoming_controller.units.size() != incoming.units().size()) {
        return vbr_occupied_replacement_guard_status::representation_mismatch;
    }
    for (size_t i = 0; i < recovery.units().size(); ++i) {
        const auto & rd = recovery.units()[i].descriptor;
        const auto & id = incoming.units()[i].descriptor;
        if (rd.logical_unit_id != i || id.logical_unit_id != i ||
            !occupied_unit_schedule_equal(id, rd) ||
            !occupied_target_unit_matches(
                child.units[i], rd, recovery_controller.units[i],
                observation.units[i])) {
            return vbr_occupied_replacement_guard_status::representation_mismatch;
        }
    }
    const auto & recovery_placement =
        recovery.manifest().stream_placements.front();
    const auto & incoming_placement =
        incoming.manifest().stream_placements.front();
    const auto recovery_tokens = recovery.manifest().token_block.tokens.size();
    const auto incoming_tokens = incoming.manifest().token_block.tokens.size();
    if (recovery.manifest().identity.token_count <= 0 ||
        incoming.manifest().identity.token_count <= 0 ||
        uint64_t(recovery.manifest().identity.token_count) != recovery_tokens ||
        uint64_t(incoming.manifest().identity.token_count) != incoming_tokens ||
        recovery.manifest().identity.next_position != llama_pos(recovery_tokens) ||
        incoming.manifest().identity.next_position != llama_pos(incoming_tokens) ||
        observation.sequence_epoch != recovery.manifest().identity.sequence_epoch ||
        recovery_placement.child_id != 0 || incoming_placement.child_id != 0 ||
        recovery_placement.stream_index != 0 ||
        incoming_placement.stream_index != 0 ||
        recovery_placement.source_sequence != observation.destination ||
        recovery_placement.computation_frontier != llama_pos(recovery_tokens) ||
        incoming_placement.computation_frontier != llama_pos(incoming_tokens) ||
        recovery_placement.cells.size() != recovery_tokens ||
        incoming_placement.cells.size() != incoming_tokens ||
        observation.cell_count != recovery_tokens) {
        return vbr_occupied_replacement_guard_status::frontier_mismatch;
    }
    for (size_t logical = 0; logical < recovery_tokens; ++logical) {
        if (recovery_placement.cells[logical].logical_position !=
                llama_pos(logical)) {
            return vbr_occupied_replacement_guard_status::unsupported_layout;
        }
    }
    for (size_t logical = 0; logical < incoming_tokens; ++logical) {
        if (incoming_placement.cells[logical].logical_position !=
                llama_pos(logical)) {
            return vbr_occupied_replacement_guard_status::unsupported_layout;
        }
    }
    const size_t free_cells = observation.cell_capacity-observation.cell_count;
    const auto strategy = incoming_tokens <= free_cells
        ? vbr_occupied_replacement_strategy::provisional_free_cells
        : incoming_tokens == recovery_tokens
            ? vbr_occupied_replacement_strategy::recycle_incumbent_cells
            : vbr_occupied_replacement_strategy::_count;
    if (strategy == vbr_occupied_replacement_strategy::_count) {
        return vbr_occupied_replacement_guard_status::capacity_unavailable;
    }
    if (build) {
        build->strategy = strategy;
    } else if (!expected || expected->strategy != strategy) {
        return vbr_occupied_replacement_guard_status::currency_changed;
    }

    std::vector<uint64_t> packed_rows;
    std::vector<uint64_t> recovery_packed_rows;
    if (build && (!occupied_projected_packed_rows(
                      incoming, incoming_placement, packed_rows) ||
                  (strategy ==
                       vbr_occupied_replacement_strategy::recycle_incumbent_cells &&
                   !occupied_projected_packed_rows(
                       recovery, recovery_placement,
                       recovery_packed_rows)))) {
        return vbr_occupied_replacement_guard_status::unsupported_layout;
    }
    uint32_t previous_physical = 0;
    for (size_t i = 0; i < observation.cell_count; ++i) {
        const auto & live = observation.cells[i];
        if (live.stream_index != 0 ||
            live.physical_cell >= observation.cell_capacity ||
            (i != 0 && live.physical_cell <= previous_physical) ||
            live.reference_count != 1 ||
            live.owner_sequence != observation.destination ||
            live.logical_position < 0 ||
            uint64_t(live.logical_position) >= recovery_tokens) {
            return vbr_occupied_replacement_guard_status::ownership_mismatch;
        }
        previous_physical = live.physical_cell;
        const auto & sealed = recovery_placement.cells[
            size_t(live.logical_position)];
        if (sealed.physical_cell != live.physical_cell ||
            sealed.ext_x != live.ext_x || sealed.ext_y != live.ext_y) {
            return vbr_occupied_replacement_guard_status::ownership_mismatch;
        }
    }
    const auto append_run = [](
            std::vector<vbr_occupied_replacement_relocation_run> & runs,
            uint64_t source_packed_row,
            uint32_t destination_physical_cell) {
        if (!runs.empty()) {
            auto & run = runs.back();
            if (run.cell_count != UINT32_MAX &&
                run.first_source_packed_row <= UINT64_MAX-run.cell_count &&
                run.first_source_packed_row+run.cell_count == source_packed_row &&
                uint64_t(run.first_destination_physical_cell)+run.cell_count ==
                    destination_physical_cell) {
                ++run.cell_count;
                return true;
            }
        }
        if (runs.size() >= VBR_OCCUPIED_REPLACEMENT_MAX_RUNS) {
            return false;
        }
        runs.push_back({ source_packed_row, destination_physical_cell, 1 });
        return true;
    };
    const auto mapping_matches = [](
            const vbr_occupied_replacement_cell_mapping & lhs,
            const vbr_occupied_replacement_cell_mapping & rhs) {
        return lhs.source_stream == rhs.source_stream &&
               lhs.source_physical_cell == rhs.source_physical_cell &&
               lhs.source_packed_row == rhs.source_packed_row &&
               lhs.destination_physical_cell == rhs.destination_physical_cell &&
               lhs.logical_position == rhs.logical_position &&
               lhs.ext_x == rhs.ext_x && lhs.ext_y == rhs.ext_y;
    };

    if (strategy ==
            vbr_occupied_replacement_strategy::recycle_incumbent_cells) {
        for (size_t logical = 0; logical < incoming_tokens; ++logical) {
            const auto & source = incoming_placement.cells[logical];
            const auto & incumbent = recovery_placement.cells[logical];
            const uint64_t source_packed_row = build
                ? packed_rows[logical]
                : expected && logical < expected->mappings.size()
                    ? expected->mappings[logical].source_packed_row
                    : UINT64_MAX;
            const vbr_occupied_replacement_cell_mapping value {
                0, source.physical_cell, source_packed_row,
                incumbent.physical_cell, source.logical_position,
                source.ext_x, source.ext_y,
            };
            if (build) {
                build->mappings.push_back(value);
                if (!append_run(
                        build->relocation_runs, source_packed_row,
                        incumbent.physical_cell) ||
                    !append_run(
                        build->recovery_runs, recovery_packed_rows[logical],
                        incumbent.physical_cell)) {
                    return vbr_occupied_replacement_guard_status::run_limit_exceeded;
                }
            } else if (!expected || logical >= expected->mappings.size() ||
                       !mapping_matches(expected->mappings[logical], value)) {
                return vbr_occupied_replacement_guard_status::currency_changed;
            }
        }
        if (!build && (!expected ||
                expected->mappings.size() != incoming_tokens ||
                expected->relocation_runs.empty() ||
                expected->recovery_runs.empty())) {
            return vbr_occupied_replacement_guard_status::currency_changed;
        }
        return vbr_occupied_replacement_guard_status::ready;
    }

    size_t occupied_index = 0;
    size_t mapped = 0;
    size_t run_index = 0;
    uint32_t run_offset = 0;
    for (uint32_t physical = 0; physical < observation.cell_capacity; ++physical) {
        const bool occupied = occupied_index < observation.cell_count &&
            observation.cells[occupied_index].physical_cell == physical;
        if (occupied) {
            ++occupied_index;
            continue;
        }
        if (mapped < incoming_tokens) {
            const auto & source = incoming_placement.cells[mapped];
            const uint64_t source_packed_row = build
                ? packed_rows[mapped]
                : expected && mapped < expected->mappings.size()
                    ? expected->mappings[mapped].source_packed_row
                    : UINT64_MAX;
            const vbr_occupied_replacement_cell_mapping value {
                0, source.physical_cell, source_packed_row, physical,
                source.logical_position, source.ext_x, source.ext_y,
            };
            if (build) {
                build->mappings.push_back(value);
            } else if (!expected || mapped >= expected->mappings.size()) {
                return vbr_occupied_replacement_guard_status::currency_changed;
            } else {
                const auto & old = expected->mappings[mapped];
                if (!mapping_matches(old, value)) {
                    return vbr_occupied_replacement_guard_status::currency_changed;
                }
            }

            if (build) {
                if (!append_run(
                        build->relocation_runs, source_packed_row, physical)) {
                    return vbr_occupied_replacement_guard_status::
                        run_limit_exceeded;
                }
            } else {
                if (!expected || run_index >= expected->relocation_runs.size()) {
                    return vbr_occupied_replacement_guard_status::currency_changed;
                }
                const auto & run = expected->relocation_runs[run_index];
                if (run_offset >= run.cell_count ||
                    run.first_source_packed_row+run_offset != source_packed_row ||
                    uint64_t(run.first_destination_physical_cell)+run_offset !=
                        physical) {
                    return vbr_occupied_replacement_guard_status::currency_changed;
                }
                if (++run_offset == run.cell_count) {
                    ++run_index;
                    run_offset = 0;
                }
            }
            ++mapped;
        }
    }
    if (mapped != incoming_tokens || occupied_index != observation.cell_count ||
        (!build && (!expected || expected->mappings.size() != mapped ||
                    expected->relocation_runs.size() != run_index ||
                    run_offset != 0 || !expected->recovery_runs.empty()))) {
        return vbr_occupied_replacement_guard_status::currency_changed;
    }
    return vbr_occupied_replacement_guard_status::ready;
}

} // namespace
vbr_occupied_replacement_guard::vbr_occupied_replacement_guard() noexcept =
    default;
vbr_occupied_replacement_guard::~vbr_occupied_replacement_guard() = default;
vbr_occupied_replacement_guard::vbr_occupied_replacement_guard(
        vbr_occupied_replacement_guard &&) noexcept = default;
vbr_occupied_replacement_guard &
vbr_occupied_replacement_guard::operator=(
        vbr_occupied_replacement_guard &&) noexcept = default;

bool vbr_occupied_replacement_guard::ready() const noexcept {
    return map_ && incoming_ && recovery_ && destination_ >= 0 &&
        map_->strategy != vbr_occupied_replacement_strategy::_count &&
        !map_->mappings.empty() && !map_->relocation_runs.empty() &&
        (map_->strategy !=
             vbr_occupied_replacement_strategy::recycle_incumbent_cells ||
        !map_->recovery_runs.empty()) &&
        accounting_serial_ != 0 && vbr_digest_nonzero(currency_digest_);
}

llama_seq_id vbr_occupied_replacement_guard::destination() const noexcept {
    return destination_;
}

uint64_t vbr_occupied_replacement_guard::accounting_serial() const noexcept {
    return accounting_serial_;
}

llama_cache_acct_artifact_id
vbr_occupied_replacement_guard::incoming_artifact() const noexcept {
    return incoming_ ? incoming_.reference_artifact()
                     : llama_cache_acct_artifact_id {};
}

llama_cache_acct_artifact_id
vbr_occupied_replacement_guard::recovery_artifact() const noexcept {
    return recovery_ ? recovery_.reference_artifact()
                     : llama_cache_acct_artifact_id {};
}

vbr_occupied_replacement_strategy
vbr_occupied_replacement_guard::strategy() const noexcept {
    return map_ ? map_->strategy : vbr_occupied_replacement_strategy::_count;
}

const std::vector<vbr_occupied_replacement_cell_mapping> &
vbr_occupied_replacement_guard::cell_mapping() const noexcept {
    static const std::vector<vbr_occupied_replacement_cell_mapping> empty;
    return map_ ? map_->mappings : empty;
}

const std::vector<vbr_occupied_replacement_relocation_run> &
vbr_occupied_replacement_guard::relocation_runs() const noexcept {
    static const std::vector<vbr_occupied_replacement_relocation_run> empty;
    return map_ ? map_->relocation_runs : empty;
}

const std::vector<vbr_occupied_replacement_relocation_run> &
vbr_occupied_replacement_guard::recovery_runs() const noexcept {
    static const std::vector<vbr_occupied_replacement_relocation_run> empty;
    return map_ ? map_->recovery_runs : empty;
}

const vbr_artifact_package_view &
vbr_occupied_replacement_guard::recovery_package() const noexcept {
    return recovery_;
}

uint64_t vbr_occupied_replacement_guard::packed_rows_expanded() const noexcept {
    return map_ ? map_->packed_rows_expanded : 0;
}

void vbr_occupied_replacement_guard::reset() noexcept {
    map_.reset();
    incoming_.reset();
    recovery_.reset();
    currency_digest_ = {};
    direct_currency_digest_ = {};
    memory_ = nullptr;
    cache_ = nullptr;
    destination_ = -1;
    accounting_serial_ = 0;
}

vbr_occupied_replacement_guard_status
vbr_prepare_occupied_replacement_guard(
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & incoming,
        const vbr_artifact_package_view & recovery,
        const vbr_occupied_replacement_observation & observation,
        vbr_occupied_replacement_guard & output,
        const vbr_import_schedule_quote * authenticated_incoming,
        const vbr_import_schedule_quote * authenticated_recovery) noexcept {
    output.reset();
    try {
        vbr_import_schedule_quote recovery_quote;
        vbr_import_schedule_quote incoming_quote;
        const auto * incoming_authority = authenticated_incoming;
        const auto * recovery_authority = authenticated_recovery;
        if ((!recovery_authority &&
             !vbr_quote_import_schedule(target, recovery, recovery_quote)) ||
            (!incoming_authority &&
             !vbr_quote_import_schedule(target, incoming, incoming_quote))) {
            return vbr_occupied_replacement_guard_status::representation_mismatch;
        }
        if (!recovery_authority) {
            recovery_authority = &recovery_quote;
        }
        if (!incoming_authority) {
            incoming_authority = &incoming_quote;
        }
        if (
            recovery_authority->status() != vbr_import_schedule_status::exact ||
            incoming_authority->status() != vbr_import_schedule_status::exact ||
            !vbr_import_schedule_quote_matches(
                *recovery_authority, target, recovery) ||
            !vbr_import_schedule_quote_matches(
                *incoming_authority, target, incoming)) {
            return vbr_occupied_replacement_guard_status::representation_mismatch;
        }
        auto shared = std::make_shared<vbr_occupied_replacement_guard::map>();
        shared->mappings.reserve(
            incoming.manifest().stream_placements.front().cells.size());
        shared->relocation_runs.reserve(std::min<size_t>(
            VBR_OCCUPIED_REPLACEMENT_MAX_RUNS,
            observation.cell_count + 1));
        shared->recovery_runs.reserve(std::min<size_t>(
            VBR_OCCUPIED_REPLACEMENT_MAX_RUNS,
            observation.cell_count + 1));
        const auto status = occupied_guard_validate(
            target, incoming, recovery, observation,
            shared.get(), nullptr);
        if (status != vbr_occupied_replacement_guard_status::ready) {
            return status;
        }
        shared->packed_rows_expanded = incoming.projected_ranges().empty()
            ? 0 : shared->mappings.size();
        vbr_artifact_package_view incoming_lease;
        vbr_artifact_package_view recovery_lease;
        if (incoming.retain(incoming_lease) != vbr_artifact_resolve_status::ok ||
            recovery.retain(recovery_lease) != vbr_artifact_resolve_status::ok) {
            return vbr_occupied_replacement_guard_status::currency_changed;
        }
        const auto digest = occupied_currency_digest(target, observation);
        if (!vbr_digest_nonzero(digest)) {
            return vbr_occupied_replacement_guard_status::internal_error;
        }
        output.map_ = std::move(shared);
        output.incoming_ = std::move(incoming_lease);
        output.recovery_ = std::move(recovery_lease);
        output.currency_digest_ = digest;
        output.destination_ = observation.destination;
        output.accounting_serial_ = target.accounting_serial;
        return vbr_occupied_replacement_guard_status::ready;
    } catch (...) {
        output.reset();
        return vbr_occupied_replacement_guard_status::internal_error;
    }
}

vbr_occupied_replacement_guard_status
vbr_recheck_occupied_replacement_guard(
        vbr_occupied_replacement_guard & guard,
        const vbr_target_validation_snapshot & target,
        const vbr_occupied_replacement_observation & observation) noexcept {
    try {
        if (!guard.ready() || target.accounting_serial !=
                guard.accounting_serial_ || observation.destination !=
                guard.destination_) {
            guard.reset();
            return vbr_occupied_replacement_guard_status::currency_changed;
        }
        const auto status = occupied_guard_validate(
            target, guard.incoming_, guard.recovery_, observation,
            nullptr, guard.map_.get());
        if (status != vbr_occupied_replacement_guard_status::ready ||
            occupied_currency_digest(target, observation) !=
                guard.currency_digest_) {
            guard.reset();
            return status == vbr_occupied_replacement_guard_status::ready
                ? vbr_occupied_replacement_guard_status::currency_changed
                : status;
        }
        return status;
    } catch (...) {
        guard.reset();
        return vbr_occupied_replacement_guard_status::internal_error;
    }
}
const char * vbr_occupied_replacement_guard_status_name(
        vbr_occupied_replacement_guard_status status) noexcept {
    switch (status) {
        case vbr_occupied_replacement_guard_status::ready: return "ready";
        case vbr_occupied_replacement_guard_status::invalid_argument: return "invalid_argument";
        case vbr_occupied_replacement_guard_status::unsupported_tree: return "unsupported_tree";
        case vbr_occupied_replacement_guard_status::unsupported_layout: return "unsupported_layout";
        case vbr_occupied_replacement_guard_status::companion_unavailable: return "companion_unavailable";
        case vbr_occupied_replacement_guard_status::frontier_mismatch: return "frontier_mismatch";
        case vbr_occupied_replacement_guard_status::representation_mismatch: return "representation_mismatch";
        case vbr_occupied_replacement_guard_status::generation_mismatch: return "generation_mismatch";
        case vbr_occupied_replacement_guard_status::ownership_mismatch: return "ownership_mismatch";
        case vbr_occupied_replacement_guard_status::cell_limit_exceeded: return "cell_limit_exceeded";
        case vbr_occupied_replacement_guard_status::run_limit_exceeded: return "run_limit_exceeded";
        case vbr_occupied_replacement_guard_status::capacity_unavailable: return "capacity_unavailable";
        case vbr_occupied_replacement_guard_status::currency_changed: return "currency_changed";
        case vbr_occupied_replacement_guard_status::internal_error: return "internal_error";
        case vbr_occupied_replacement_guard_status::_count: return "_count";
    }
    return "_count";
}
