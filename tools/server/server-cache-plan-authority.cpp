#include "server-cache-plan-authority.h"

#include "common-cache-plan-estimate.h"

int32_t server_cache_plan_host_source(
        const common_cache_plan_record & rec,
        int32_t candidate) noexcept {
    if (candidate < 0 || uint32_t(candidate) >= rec.n_inventory) {
        return -1;
    }
    const auto & row = rec.inventory[size_t(candidate)];
    if (row.is_chain()) {
        const int32_t base = row.component_ids[0];
        return base >= 0 && uint32_t(base) < rec.n_inventory
            ? rec.inventory[size_t(base)].source_id : -1;
    }
    return row.provider == common_cache_plan_provider::host_cache_entry
        ? row.source_id : -1;
}

common_cache_plan_destruction_effect_set server_cache_destruction_effects_for(
        const common_cache_plan_record & rec,
        int32_t candidate,
        int32_t legacy_candidate,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    if (candidate < 0 || legacy_candidate < 0 ||
        uint32_t(candidate) >= rec.n_inventory ||
        uint32_t(legacy_candidate) >= rec.n_inventory) {
        return 0;
    }
    const auto & planned = rec.inventory[size_t(candidate)];
    const auto & legacy = rec.inventory[size_t(legacy_candidate)];
    common_cache_plan_destruction_effect_set effects = 0;
    if (planned.target_slot_id != legacy.target_slot_id) {
        if (rec.selection == common_cache_plan_selection::similarity &&
            planned.provider == common_cache_plan_provider::live_slot &&
            planned.f_keep_known && planned.f_keep >= 1.0) {
            // The sole zero-destruction cross-target case.
        } else {
            effects |= common_cache_plan_destruction_effect_bit(
                rec.selection == common_cache_plan_selection::similarity
                    ? common_cache_plan_destruction_effect::
                          destructive_similarity_retarget
                    : common_cache_plan_destruction_effect::
                          cross_target_displacement);
        }
    }
    const bool legacy_uses_live_target =
        common_cache_plan_provider_is_live(legacy.provider);
    const bool destruction_certification_available =
        (permitted_effects &
         server_cache_plan_nonconsuming_host_effects(true)) != 0;
    if (planned.target_slot_id == legacy.target_slot_id &&
        ((planned.provider == common_cache_plan_provider::cold_replay &&
          legacy.provider != common_cache_plan_provider::cold_replay) ||
         (destruction_certification_available &&
          (planned.provider == common_cache_plan_provider::host_cache_entry ||
           planned.is_chain()) && legacy_uses_live_target))) {
        // Cold replacement is the established planner effect. Occupied restoration adds host restore
        // to the same physical class only when lifecycle certification exists;
        // lifecycle-off therefore preserves the previously authorized
        // same-target host-restore behavior byte-for-byte.
        // The schema-v6 name predates non-consuming host restore. Its physical
        // class is the stable contract: any certified same-target whole-state
        // replacement destroys the live slot, whether replacement bytes come
        // from cold replay or an immutable host snapshot.
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::same_target_cold_replacement);
    }
    const int32_t planned_host = server_cache_plan_host_source(rec, candidate);
    const int32_t legacy_host = server_cache_plan_host_source(rec, legacy_candidate);
    if (planned_host >= 0 && planned_host != legacy_host) {
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::different_host_source_consumption);
    }
    // Physical non-effects (lifecycle's non-consuming host restore) and
    // mutation-boundary destruction certificates share this single row-opening mask.
    return effects & ~permitted_effects;
}

uint64_t server_cache_plan_capability_fold(
        uint64_t hash,
        uint64_t value) noexcept {
    // FNV-1a with an explicit value delimiter. This is a process-local drift
    // detector, not a durable/content identity.
    for (unsigned i = 0; i < 8; ++i) {
        hash = (hash ^ uint8_t(value >> (8*i))) * 1099511628211ull;
    }
    return (hash ^ 0xffu) * 1099511628211ull;
}

bool server_cache_plan_assign_source_id(
        int32_t & instance_source_id,
        int32_t & next_source_id,
        int32_t & source_id) noexcept {
    if (instance_source_id >= 0) {
        source_id = instance_source_id;
        return true;
    }
    if (next_source_id < 0 ||
        next_source_id > SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID ||
        next_source_id >= int32_t(COMMON_CACHE_PLAN_MAX_CANDIDATES)) {
        source_id = -1;
        return false;
    }
    instance_source_id = next_source_id++;
    source_id = instance_source_id;
    return true;
}

void server_cache_plan_authority::plan_before_mutation(
        common_cache_plan_record & rec,
        uint64_t capability_before,
        uint64_t capability_after) noexcept {
    rec.authority = {};
    common_cache_plan_authority_fallback fallback =
        common_cache_plan_authority_fallback::none;
    if (capability_before != capability_after) {
        rec.clear_planner_outputs();
        rec.planner_status = common_cache_plan_planner_status::incomplete_evidence;
        fallback = common_cache_plan_authority_fallback::stale_capability;
    } else {
        common_cache_plan_run_planner(rec);
    }
    common_cache_plan_derive_shadow_authority(rec, configured_level, fallback);
    rec.authority_prequalified =
        rec.planner_status == common_cache_plan_planner_status::ok &&
        capability_before == capability_after;
    rec.planner_precomputed = true;
}

void server_cache_plan_authority::fail_closed(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason) noexcept {
    rec.clear_planner_outputs();
    rec.planner_status = common_cache_plan_planner_status::internal_fault;
    common_cache_plan_derive_shadow_authority(rec, configured_level, reason);
    const auto decision_level = server_cache_plan_level_of(rec.selection);
    if (decision_level != common_cache_plan_authority_level::off &&
        decision_level != common_cache_plan_authority_level::_count &&
        server_cache_plan_level_enabled(configured_level, decision_level)) {
        rec.authority.state =
            common_cache_plan_authority_state::fallback_legacy;
        rec.authority.fallback_reason = reason;
    }
    rec.authority_prequalified = false;
    rec.planner_precomputed = true;
}

int32_t server_cache_plan_legacy_candidate(
        const common_cache_plan_record & rec,
        int32_t target_slot_id,
        bool host_lookup_enabled) noexcept {
    int32_t live = -1;
    int32_t host = -1;
    double f_keep = -1.0;
    double sim = 0.0;

    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain()) {
            continue;
        }
        if (candidate.provider == common_cache_plan_provider::live_slot) {
            live = int32_t(i);
            if (candidate.f_keep_known) {
                f_keep = candidate.f_keep;
            }
            if (candidate.sim_known) {
                sim = candidate.sim;
            }
            break;
        }
    }

    // Reproduce the legacy host selector's strict two-axis improvement and
    // insertion-order tie behavior. Invalid rows never enter that selector.
    for (uint32_t i = 0; host_lookup_enabled && i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain() ||
            candidate.provider != common_cache_plan_provider::host_cache_entry ||
            !candidate.viable() || !candidate.f_keep_known ||
            !candidate.sim_known) {
            continue;
        }
        if (f_keep < candidate.f_keep && sim < candidate.sim) {
            f_keep = candidate.f_keep;
            sim = candidate.sim;
            host = int32_t(i);
        }
    }

    const int32_t host_source = host >= 0
        ? rec.inventory[size_t(host)].source_id : -1;
    int32_t checkpoint = -1;
    int32_t checkpoint_ordinal = -1;
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain() ||
            candidate.provider != common_cache_plan_provider::live_context_checkpoint ||
            !candidate.viable()) {
            continue;
        }
        const int32_t ordinal =
            server_cache_plan_checkpoint_ordinal_from_source_id(
                candidate.source_id, host >= 0 ? host_source : -1);
        if (host >= 0) {
            if (!candidate.component_only ||
                candidate.dependent_host_source_id != host_source) {
                continue;
            }
        } else if (candidate.component_only) {
            continue;
        }
        // The shipped selector scans newest-to-oldest, so the greatest forward
        // ordinal is its first viable checkpoint.
        if (ordinal >= 0 && ordinal > checkpoint_ordinal) {
            checkpoint_ordinal = ordinal;
            checkpoint = int32_t(i);
        }
    }

    if (checkpoint >= 0) {
        if (host < 0) {
            return checkpoint;
        }
        const auto * chain = rec.find_chain(
            common_cache_plan_provider::host_cache_entry, host, checkpoint);
        const int32_t chain_id = chain
            ? int32_t(chain - rec.inventory.data()) : -1;
        return chain_id >= 0 ? chain_id : host;
    }
    if (host >= 0) {
        return host;
    }
    if (live >= 0 && rec.inventory[size_t(live)].viable()) {
        return live;
    }
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id == target_slot_id &&
            candidate.provider == common_cache_plan_provider::cold_replay &&
            !candidate.is_chain() && candidate.viable()) {
            return int32_t(i);
        }
    }
    return -1;
}

bool server_cache_plan_execution_from_candidate(
        const common_cache_plan_record & rec,
        int32_t candidate,
        int32_t target_slot_id,
        server_cache_plan_execution & out) noexcept {
    out = {};
    if (candidate < 0 || uint32_t(candidate) >= rec.n_inventory) {
        return false;
    }
    const auto & selected = rec.inventory[size_t(candidate)];
    if (selected.target_slot_id != target_slot_id || !selected.viable()) {
        return false;
    }
    out.target = target_slot_id;
    if (selected.is_chain()) {
        const int32_t host = selected.component_ids[0];
        const int32_t checkpoint = selected.component_ids[1];
        if (host < 0 || checkpoint < 0 || uint32_t(host) >= rec.n_inventory ||
            uint32_t(checkpoint) >= rec.n_inventory) {
            return false;
        }
        const auto & h = rec.inventory[size_t(host)];
        const auto & c = rec.inventory[size_t(checkpoint)];
        if (h.target_slot_id != target_slot_id || c.target_slot_id != target_slot_id ||
            h.provider != common_cache_plan_provider::host_cache_entry ||
            c.provider != common_cache_plan_provider::live_context_checkpoint ||
            !c.component_only || c.dependent_host_source_id != h.source_id ||
            !h.viable() || !c.viable()) {
            return false;
        }
        out.kind = server_cache_plan_execution_kind::host_checkpoint_restore;
        out.host_source_id = h.source_id;
        out.checkpoint_source_id = c.source_id;
        return true;
    }
    switch (selected.provider) {
        case common_cache_plan_provider::live_slot:
            out.kind = server_cache_plan_execution_kind::live_replay;
            return true;
        case common_cache_plan_provider::host_cache_entry:
            out.kind = server_cache_plan_execution_kind::host_restore;
            out.host_source_id = selected.source_id;
            return true;
        case common_cache_plan_provider::live_context_checkpoint:
            if (selected.component_only) {
                return false;
            }
            out.kind = server_cache_plan_execution_kind::checkpoint_restore;
            out.checkpoint_source_id = selected.source_id;
            return true;
        case common_cache_plan_provider::cold_replay:
            out.kind = server_cache_plan_execution_kind::cold_replay;
            return true;
        case common_cache_plan_provider::_count:
            break;
    }
    return false;
}

static bool inside_pre_da_safety_envelope(
        const common_cache_plan_record & rec,
        int32_t planned_candidate,
        int32_t legacy_candidate,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    return server_cache_destruction_effects_for(
        rec, planned_candidate, legacy_candidate, permitted_effects) == 0;
}

static common_cache_plan_authority_fallback pre_da_envelope_refusal_reason(
        const common_cache_plan_record & rec,
        const server_cache_plan_execution & planned,
        const server_cache_plan_execution & legacy) noexcept {
    // Schema 5 has no eviction_evidence_unavailable spelling. At LRU, use its
    // existing budget/lease availability reason only for the destruction fence: a
    // target change, or a cold replacement of retained same-target state.
    // Consuming a different host source remains destruction authority, just as
    // it does at every earlier ratchet.
    if (rec.selection == common_cache_plan_selection::lru &&
        (planned.target != legacy.target ||
         (planned.kind == server_cache_plan_execution_kind::cold_replay &&
          legacy.kind != server_cache_plan_execution_kind::cold_replay))) {
        return common_cache_plan_authority_fallback::budget_or_lease_unavailable;
    }
    return common_cache_plan_authority_fallback::
        destruction_authority_required;
}

server_cache_plan_execution server_cache_plan_authority::authorize(
        common_cache_plan_record & rec,
        int32_t legacy_target_slot_id,
        bool host_lookup_enabled,
        bool target_identity_matches,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    server_cache_plan_execution execution;
    const auto decision_level = server_cache_plan_level_of(rec.selection);
    if (decision_level == common_cache_plan_authority_level::off ||
        decision_level == common_cache_plan_authority_level::_count) {
        return execution;
    }
    if (!server_cache_plan_level_enabled(configured_level, decision_level)) {
        // Preserve a planner refusal (no profile, incomplete evidence, ...).
        // tier_not_enabled describes only an otherwise-qualified plan whose
        // decision ratchet has not landed yet.
        if (rec.authority.fallback_reason ==
                common_cache_plan_authority_fallback::none &&
            rec.authority_prequalified &&
            rec.planner_status == common_cache_plan_planner_status::ok) {
            rec.authority.fallback_reason =
                common_cache_plan_authority_fallback::tier_not_enabled;
        }
        return execution;
    }
    const int32_t legacy_plan_candidate = server_cache_plan_legacy_candidate(
        rec, legacy_target_slot_id, host_lookup_enabled);
    rec.authority.legacy_plan_candidate = legacy_plan_candidate;
    if (!server_cache_plan_candidate_prequalified(rec)) {
        fallback_legacy(rec,
            rec.authority.fallback_reason !=
                    common_cache_plan_authority_fallback::none
                ? rec.authority.fallback_reason
                : common_cache_plan_authority_fallback::internal_fault);
        return execution;
    }
    const int32_t planned_target_slot_id = server_cache_plan_planned_target(
        rec, configured_level, legacy_target_slot_id);
    if (planned_target_slot_id < 0) {
        fallback_legacy(rec,
            common_cache_plan_authority_fallback::incomplete_evidence);
        return {};
    }
    if (!server_cache_plan_execution_from_candidate(
            rec, rec.shadow_choice, planned_target_slot_id, execution)) {
        fallback_legacy(rec,
            common_cache_plan_authority_fallback::internal_fault);
        return {};
    }
    if (!target_identity_matches &&
        execution.kind != server_cache_plan_execution_kind::cold_replay) {
        // Identity feasibility is known before mutation; this is incomplete
        // planner evidence, not capability drift discovered at execution.
        fallback_legacy(rec,
            common_cache_plan_authority_fallback::incomplete_evidence);
        return {};
    }
    server_cache_plan_execution legacy_execution;
    if (!server_cache_plan_execution_from_candidate(
            rec, legacy_plan_candidate, legacy_target_slot_id,
            legacy_execution)) {
        fallback_legacy(rec,
            common_cache_plan_authority_fallback::internal_fault);
        return {};
    }
    if (!inside_pre_da_safety_envelope(
            rec, rec.shadow_choice, legacy_plan_candidate,
            permitted_effects)) {
        fallback_legacy(rec,
            pre_da_envelope_refusal_reason(
                rec, execution, legacy_execution));
        return {};
    }
    rec.authority.state = common_cache_plan_authority_state::authoritative;
    rec.authority.fallback_reason = common_cache_plan_authority_fallback::none;
    return execution;
}

void server_cache_plan_authority::fallback_legacy(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason) noexcept {
    rec.authority.state = common_cache_plan_authority_state::fallback_legacy;
    rec.authority.fallback_reason = reason;
}

bool server_cache_plan_demote_for_coverage_recovery(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        server_cache_plan_execution & execution,
        int64_t pos_min,
        int64_t pos_min_threshold) noexcept {
    if (!server_cache_plan_requires_coverage_recovery(
            execution, pos_min, pos_min_threshold)) {
        return false;
    }
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability);
    execution.clear();
    return true;
}

bool server_cache_plan_demote_for_vbr_low_lcp_reset(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        server_cache_plan_execution & execution,
        bool reset_applied) noexcept {
    if (!reset_applied || !execution.authoritative()) {
        return false;
    }
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability);
    execution.clear();
    return true;
}

bool server_cache_plan_revalidate_checkpoint_execution(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        server_cache_plan_execution & execution,
        size_t checkpoint_count,
        bool eligible,
        int32_t & ordinal) noexcept {
    if (server_cache_plan_checkpoint_override_ordinal(
            execution, checkpoint_count, eligible, ordinal)) {
        return true;
    }
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability);
    execution.clear();
    return false;
}

void server_cache_plan_authority::finalize_execution(
        common_cache_plan_record & rec) noexcept {
    common_cache_plan_finalize_shadow_authority(rec);
    if (rec.authority.state == common_cache_plan_authority_state::authoritative &&
        rec.authority.executed_plan_candidate !=
            rec.authority.planner_plan_candidate) {
        fallback_legacy(rec,
            common_cache_plan_authority_fallback::internal_fault);
    }
    counters.observe(rec.authority, rec.authority_prequalified);
}

server_cache_plan_live_evaluation server_cache_plan_evaluate_live(
        bool busy,
        bool has_payload,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens,
        uint64_t source_tokens) noexcept {
    server_cache_plan_live_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.f_keep = source_tokens ? float(lcp_tokens) / float(source_tokens) : -1.0f;
    out.reason = busy ? COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY :
                 !has_payload ? COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE :
                 lcp_tokens == 0 ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_live(
        common_cache_plan_candidate * row,
        const server_cache_plan_live_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->f_keep = evaluation.f_keep;
    row->f_keep_known = evaluation.f_keep >= 0.0f;
    row->note_reject(evaluation.reason);
}

server_cache_plan_host_evaluation server_cache_plan_evaluate_host(
        bool payload_present,
        bool identity_matches,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens,
        uint64_t source_tokens,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_host_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.payload_bytes = payload_bytes;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.f_keep = source_tokens ? float(lcp_tokens) / float(source_tokens) : 0.0f;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !identity_matches ? COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH :
                 out.f_keep < 0.25f ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_host(
        common_cache_plan_candidate * row,
        const server_cache_plan_host_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->f_keep = evaluation.f_keep;
    row->f_keep_known = true;
    row->note_reject(evaluation.reason);
}

server_cache_plan_checkpoint_evaluation server_cache_plan_evaluate_checkpoint(
        bool payload_present,
        bool frontier_current,
        bool recurrent,
        bool checkpoint_lineage_matches,
        int64_t pos_min,
        int64_t pos_max,
        int64_t next_position,
        int64_t min_position_threshold,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_checkpoint_evaluation out;
    out.lcp_tokens = pos_max >= 0 ? uint64_t(pos_max) : 0;
    out.payload_bytes = payload_bytes;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !frontier_current ? COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID :
                 !checkpoint_lineage_matches ? COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED :
                 recurrent
                    ? (pos_max < next_position
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT)
                    : (pos_max <= next_position &&
                       (pos_min < min_position_threshold || pos_min == 0)
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT);
    return out;
}

void server_cache_plan_apply_checkpoint(
        common_cache_plan_candidate * row,
        const server_cache_plan_checkpoint_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->note_reject(evaluation.reason);
}
