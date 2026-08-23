#include "common-cache-plan.h"
#include "common-cache-plan-estimate.h" // tie-floor constants echoed into shadow JSON

#include <nlohmann/json.hpp>

#include <stdexcept>

using json = nlohmann::ordered_json;

bool common_cache_plan_projected_release_bytes(
        const std::vector<common_cache_plan_yield_domain> & domains,
        uint64_t & total) noexcept {
    total = 0;
    for (const auto & row : domains) {
        if (row.projected_release_bytes.state !=
                llama_cache_acct_known::known ||
            row.projected_release_bytes.value > UINT64_MAX - total) {
            total = 0;
            return false;
        }
        total += row.projected_release_bytes.value;
    }
    return true;
}

void common_cache_plan_fill_actual_yield(
        common_cache_plan_yield_record & yield,
        const std::vector<common_cache_plan_yield_domain> & projected,
        const std::vector<common_cache_plan_yield_domain> & observed_after) noexcept {
    yield.actual_domains.clear();
    try {
        yield.actual_domains.reserve(projected.size());
        for (const auto & row : projected) {
            const auto current = std::find_if(
                observed_after.begin(), observed_after.end(),
                [&](const auto & observed) {
                    return observed.domain == row.domain;
                });
            if (current == observed_after.end() ||
                row.current_resident_bytes.state !=
                    llama_cache_acct_known::known ||
                current->current_resident_bytes.state !=
                    llama_cache_acct_known::known ||
                current->current_resident_bytes.value >
                    row.current_resident_bytes.value) {
                yield.actual_domains.clear();
                yield.actual_state =
                    common_cache_plan_yield_actual_state::unavailable;
                return;
            }
            common_cache_plan_actual_yield_domain actual;
            actual.domain = row.domain;
            actual.before_bytes = row.current_resident_bytes;
            actual.after_bytes = current->current_resident_bytes;
            actual.released_bytes = llama_cache_acct_value::measured(
                row.current_resident_bytes.value -
                current->current_resident_bytes.value);
            yield.actual_domains.push_back(std::move(actual));
        }
        yield.actual_state = common_cache_plan_yield_actual_state::measured;
    } catch (...) {
        yield.actual_domains.clear();
        yield.actual_state = common_cache_plan_yield_actual_state::unavailable;
    }
}

// Exhaustive name tables for the B0 closed enums. Switch-based with no default case so a new
// member without a name is a compile-time -Wswitch error, and the single unreachable return
// keeps release builds defined. These are the ONLY spellings of these names — CI bans replicas.

const char * common_cache_plan_reason_name(common_cache_plan_reason r) {
    switch (r) {
#define COMMON_CACHE_PLAN_REASON_NAME_CASE(sym, name, val) case COMMON_CACHE_PLAN_REASON_##sym: return name;
        COMMON_CACHE_PLAN_REASON_LIST(COMMON_CACHE_PLAN_REASON_NAME_CASE)
#undef COMMON_CACHE_PLAN_REASON_NAME_CASE
        case COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL: break;
    }
    return "invalid";
}

const char * common_cache_plan_disposition_name(common_cache_plan_disposition d) {
    switch (d) {
        case common_cache_plan_disposition::accepted:              return "accepted";
        case common_cache_plan_disposition::rejected_invalid:      return "rejected_invalid";
        case common_cache_plan_disposition::valid_not_chosen_cost: return "valid_not_chosen_cost";
        case common_cache_plan_disposition::unavailable:           return "unavailable";
        case common_cache_plan_disposition::_count:                break;
    }
    return "invalid";
}

const char * common_cache_plan_provider_name(common_cache_plan_provider p) {
    switch (p) {
        case common_cache_plan_provider::live_slot:               return "live_slot";
        case common_cache_plan_provider::live_context_checkpoint: return "live_context_checkpoint";
        case common_cache_plan_provider::host_cache_entry:        return "host_cache_entry";
        case common_cache_plan_provider::cold_replay:             return "cold_replay";
        case common_cache_plan_provider::_count:                  break;
    }
    return "invalid";
}

const char * common_cache_plan_payload_kind_name(
        common_cache_plan_payload_kind kind) noexcept {
    switch (kind) {
        case common_cache_plan_payload_kind::unavailable:  return "unavailable";
        case common_cache_plan_payload_kind::fixed_state:  return "fixed_state";
        case common_cache_plan_payload_kind::vbr_artifact: return "vbr_artifact";
        case common_cache_plan_payload_kind::_count:       break;
    }
    return "invalid";
}

const char * common_cache_plan_outcome_name(common_cache_plan_outcome o) {
    switch (o) {
        case common_cache_plan_outcome::unknown:                       return "unknown";
        case common_cache_plan_outcome::restored:                      return "restored";
        case common_cache_plan_outcome::restore_failed_fell_back_cold: return "restore_failed_fell_back_cold";
        case common_cache_plan_outcome::cold:                          return "cold";
        case common_cache_plan_outcome::_count:                        break;
    }
    return "invalid";
}

const char * common_cache_plan_selection_name(common_cache_plan_selection s) {
    switch (s) {
        case common_cache_plan_selection::none:       return "none";
        case common_cache_plan_selection::by_id:      return "by_id";
        case common_cache_plan_selection::similarity: return "similarity";
        case common_cache_plan_selection::route_home: return "route_home";
        case common_cache_plan_selection::lru:        return "lru";
        case common_cache_plan_selection::_count:     break;
    }
    return "invalid";
}

const char * common_cache_plan_authority_level_name(common_cache_plan_authority_level level) {
    switch (level) {
        case common_cache_plan_authority_level::off:        return "off";
        case common_cache_plan_authority_level::by_id:      return "by_id";
        case common_cache_plan_authority_level::similarity: return "similarity";
        case common_cache_plan_authority_level::route_home: return "route_home";
        case common_cache_plan_authority_level::lru:        return "lru";
        case common_cache_plan_authority_level::_count:     break;
    }
    return "invalid";
}

common_cache_plan_authority_level common_cache_plan_authority_level_parse(
        const std::string & value) {
    if (value == "off") {
        return common_cache_plan_authority_level::off;
    }
    if (value == "by_id") {
        return common_cache_plan_authority_level::by_id;
    }
    if (value == "similarity") {
        return common_cache_plan_authority_level::similarity;
    }
    if (value == "route_home") {
        return common_cache_plan_authority_level::route_home;
    }
    if (value == "lru") {
        return common_cache_plan_authority_level::lru;
    }
    throw std::invalid_argument("invalid cache-plan authority level: " + value);
}

const char * common_cache_plan_authority_state_name(common_cache_plan_authority_state state) {
    switch (state) {
        case common_cache_plan_authority_state::shadow:          return "shadow";
        case common_cache_plan_authority_state::authoritative:   return "authoritative";
        case common_cache_plan_authority_state::fallback_legacy: return "fallback_legacy";
        case common_cache_plan_authority_state::_count:          break;
    }
    return "invalid";
}

const char * common_cache_plan_authority_fallback_name(
        common_cache_plan_authority_fallback reason) {
    switch (reason) {
        case common_cache_plan_authority_fallback::none: return "none";
        case common_cache_plan_authority_fallback::tier_not_enabled:
            return "tier_not_enabled";
        case common_cache_plan_authority_fallback::no_profile: return "no_profile";
        case common_cache_plan_authority_fallback::profile_unfitted:
            return "profile_unfitted";
        case common_cache_plan_authority_fallback::invalid_calibration:
            return "invalid_calibration";
        case common_cache_plan_authority_fallback::incomplete_evidence:
            return "incomplete_evidence";
        case common_cache_plan_authority_fallback::stale_capability:
            return "stale_capability";
        case common_cache_plan_authority_fallback::destruction_authority_required:
            return "destruction_authority_required";
        case common_cache_plan_authority_fallback::budget_or_lease_unavailable:
            return "budget_or_lease_unavailable";
        case common_cache_plan_authority_fallback::destruction_not_certified:
            return "destruction_not_certified";
        case common_cache_plan_authority_fallback::internal_fault:
            return "internal_fault";
        case common_cache_plan_authority_fallback::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_destruction_state_name(
        common_cache_plan_destruction_state state) {
    switch (state) {
        case common_cache_plan_destruction_state::not_required: return "not_required";
        case common_cache_plan_destruction_state::quoted:       return "quoted";
        case common_cache_plan_destruction_state::certified:    return "certified";
        case common_cache_plan_destruction_state::executed:     return "executed";
        case common_cache_plan_destruction_state::refused:      return "refused";
        case common_cache_plan_destruction_state::failed:       return "failed";
        case common_cache_plan_destruction_state::_count:       break;
    }
    return "invalid";
}

const char * common_cache_plan_destruction_reason_name(
        common_cache_plan_destruction_reason reason) {
    switch (reason) {
        case common_cache_plan_destruction_reason::none: return "none";
        case common_cache_plan_destruction_reason::lifecycle_disabled: return "lifecycle_disabled";
        case common_cache_plan_destruction_reason::manifest_incomplete: return "manifest_incomplete";
        case common_cache_plan_destruction_reason::identity_unavailable: return "identity_unavailable";
        case common_cache_plan_destruction_reason::mandatory_anchor: return "mandatory_anchor";
        case common_cache_plan_destruction_reason::lease_unavailable: return "lease_unavailable";
        case common_cache_plan_destruction_reason::hard_lease_blocked: return "hard_lease_blocked";
        case common_cache_plan_destruction_reason::accounting_unavailable: return "accounting_unavailable";
        case common_cache_plan_destruction_reason::effect_drift: return "effect_drift";
        case common_cache_plan_destruction_reason::release_evidence_unavailable: return "release_evidence_unavailable";
        case common_cache_plan_destruction_reason::recovery_unavailable: return "recovery_unavailable";
        case common_cache_plan_destruction_reason::profile_unfitted: return "profile_unfitted";
        case common_cache_plan_destruction_reason::capacity_refused: return "capacity_refused";
        case common_cache_plan_destruction_reason::mutation_failed: return "mutation_failed";
        case common_cache_plan_destruction_reason::internal_fault: return "internal_fault";
        case common_cache_plan_destruction_reason::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_destruction_effect_name(
        common_cache_plan_destruction_effect effect) {
    switch (effect) {
        case common_cache_plan_destruction_effect::none: return "none";
        case common_cache_plan_destruction_effect::cross_target_displacement: return "cross_target_displacement";
        case common_cache_plan_destruction_effect::destructive_similarity_retarget: return "destructive_similarity_retarget";
        case common_cache_plan_destruction_effect::same_target_cold_replacement: return "same_target_cold_replacement";
        case common_cache_plan_destruction_effect::different_host_source_consumption: return "different_host_source_consumption";
        case common_cache_plan_destruction_effect::checkpoint_member_drop: return "checkpoint_member_drop";
        case common_cache_plan_destruction_effect::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_destruction_class_name(
        common_cache_plan_destruction_class action) {
    switch (action) {
        case common_cache_plan_destruction_class::slot_drop: return "slot_drop";
        case common_cache_plan_destruction_class::live_range_drop: return "live_range_drop";
        case common_cache_plan_destruction_class::host_artifact_drop: return "host_artifact_drop";
        case common_cache_plan_destruction_class::checkpoint_drop: return "checkpoint_drop";
        case common_cache_plan_destruction_class::token_ledger_truncate: return "token_ledger_truncate";
        case common_cache_plan_destruction_class::mandatory_recovery_reset: return "mandatory_recovery_reset";
        case common_cache_plan_destruction_class::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_destruction_physical_reason_name(
        common_cache_plan_destruction_physical_reason reason) {
    switch (reason) {
        case common_cache_plan_destruction_physical_reason::slot_rebind: return "slot_rebind";
        case common_cache_plan_destruction_physical_reason::idle_reclaim: return "idle_reclaim";
        case common_cache_plan_destruction_physical_reason::prompt_trim: return "prompt_trim";
        case common_cache_plan_destruction_physical_reason::cache_capacity: return "cache_capacity";
        case common_cache_plan_destruction_physical_reason::cache_update: return "cache_update";
        case common_cache_plan_destruction_physical_reason::prompt_clear: return "prompt_clear";
        case common_cache_plan_destruction_physical_reason::checkpoint_replace: return "checkpoint_replace";
        case common_cache_plan_destruction_physical_reason::mandatory_recovery: return "mandatory_recovery";
        case common_cache_plan_destruction_physical_reason::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_destruction_lease_verdict_name(
        common_cache_plan_destruction_lease_verdict verdict) {
    switch (verdict) {
        case common_cache_plan_destruction_lease_verdict::unavailable: return "unavailable";
        case common_cache_plan_destruction_lease_verdict::unleased: return "unleased";
        case common_cache_plan_destruction_lease_verdict::soft_leased: return "soft_leased";
        case common_cache_plan_destruction_lease_verdict::hard_leased: return "hard_leased";
        case common_cache_plan_destruction_lease_verdict::mandatory_recovery: return "mandatory_recovery";
        case common_cache_plan_destruction_lease_verdict::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_displaced_fate_name(
        common_cache_plan_displaced_fate fate) {
    switch (fate) {
        case common_cache_plan_displaced_fate::unavailable: return "unavailable";
        case common_cache_plan_displaced_fate::retained_live: return "retained_live";
        case common_cache_plan_displaced_fate::retained_host: return "retained_host";
        case common_cache_plan_displaced_fate::retained_sealed_artifact: return "retained_sealed_artifact";
        case common_cache_plan_displaced_fate::exact_duplicate: return "exact_duplicate";
        case common_cache_plan_displaced_fate::exact_replay_recipe: return "exact_replay_recipe";
        case common_cache_plan_displaced_fate::destroyed_by_policy: return "destroyed_by_policy";
        case common_cache_plan_displaced_fate::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_recovery_citation_name(
        common_cache_plan_recovery_citation citation) {
    switch (citation) {
        case common_cache_plan_recovery_citation::unavailable: return "unavailable";
        case common_cache_plan_recovery_citation::resolved: return "resolved";
        case common_cache_plan_recovery_citation::prospective: return "prospective";
        case common_cache_plan_recovery_citation::_count: break;
    }
    return "invalid";
}

const char * common_cache_plan_destruction_comparison_name(
        common_cache_plan_destruction_comparison comparison) {
    switch (comparison) {
        case common_cache_plan_destruction_comparison::not_compared: return "not_compared";
        case common_cache_plan_destruction_comparison::matched: return "matched";
        case common_cache_plan_destruction_comparison::differed: return "differed";
        case common_cache_plan_destruction_comparison::ds6_insufficient_yield: return "ds6_insufficient_yield";
        case common_cache_plan_destruction_comparison::ds6_unsupported_required: return "ds6_unsupported_required";
        case common_cache_plan_destruction_comparison::ds6_unavailable: return "ds6_unavailable";
        case common_cache_plan_destruction_comparison::_count: break;
    }
    return "invalid";
}

void common_cache_plan_destruction_counters::observe(
        common_cache_plan_selection tier,
        const common_cache_plan_destruction_receipt & receipt,
        bool observe_classification) noexcept {
    // Unlike B-A's one-receipt observe(), D-A calls this once per candidate.
    // The server boundary publishes the selected/finalized receipt exactly once.
    const size_t t = size_t(tier);
    if (t >= n_tiers) {
        return;
    }
    for (uint8_t raw = uint8_t(common_cache_plan_destruction_effect::none) + 1;
         raw < uint8_t(common_cache_plan_destruction_effect::_count); ++raw) {
        const auto effect = common_cache_plan_destruction_effect(raw);
        if (!common_cache_plan_destruction_effect_has(receipt.effects, effect)) {
            continue;
        }
        const size_t c = size_t(common_cache_plan_destruction_class_for_effect(effect));
        if (c >= n_classes) {
            continue;
        }
        if (receipt.state == common_cache_plan_destruction_state::quoted) {
            quoted[t][c]++;
        } else if (receipt.state == common_cache_plan_destruction_state::certified) {
            certified[t][c]++;
        } else if (receipt.state == common_cache_plan_destruction_state::executed) {
            executed[t][c]++;
            if (receipt.actual_accounting_serial == 0) {
                actual_yield_unavailable[t][c]++;
            }
        }
    }
    if (receipt.state == common_cache_plan_destruction_state::refused ||
        receipt.state == common_cache_plan_destruction_state::failed) {
        const size_t r = size_t(receipt.reason);
        if (r < n_reasons) {
            refused[t][r]++;
        }
    }
    // State transitions may publish the same candidate at quoted, certified,
    // and executed. Lease and recovery classification are once-per-candidate,
    // selected by the caller at the terminal evidence-bearing observation.
    if (observe_classification) {
        const size_t v = size_t(receipt.lease_verdict);
        if (v < n_verdicts) {
            lease_verdict[t][v]++;
        }
        const size_t f = size_t(receipt.displaced_fate);
        if (f < n_fates) {
            recovery_outcome[t][f]++;
        }
    }
}

static common_cache_plan_authority_fallback planner_fallback(
        common_cache_plan_planner_status status) noexcept {
    switch (status) {
        case common_cache_plan_planner_status::ok:
        case common_cache_plan_planner_status::not_attempted:
            return common_cache_plan_authority_fallback::none;
        case common_cache_plan_planner_status::no_profile:
            return common_cache_plan_authority_fallback::no_profile;
        case common_cache_plan_planner_status::profile_unfitted:
            return common_cache_plan_authority_fallback::profile_unfitted;
        case common_cache_plan_planner_status::invalid_calibration:
            return common_cache_plan_authority_fallback::invalid_calibration;
        case common_cache_plan_planner_status::incomplete_evidence:
            return common_cache_plan_authority_fallback::incomplete_evidence;
        case common_cache_plan_planner_status::internal_fault:
            return common_cache_plan_authority_fallback::internal_fault;
        case common_cache_plan_planner_status::_count:
            break;
    }
    return common_cache_plan_authority_fallback::internal_fault;
}

void common_cache_plan_derive_shadow_authority(
        common_cache_plan_record & rec,
        common_cache_plan_authority_level configured_level,
        common_cache_plan_authority_fallback fallback_reason) noexcept {
    auto & receipt = rec.authority;
    receipt = {};
    receipt.configured_level = configured_level;
    receipt.state = common_cache_plan_authority_state::shadow;
    receipt.planner_plan_candidate =
        rec.planner_status == common_cache_plan_planner_status::ok
            ? rec.shadow_choice : -1;
    if (receipt.planner_plan_candidate >= 0 &&
        uint32_t(receipt.planner_plan_candidate) < rec.n_inventory) {
        receipt.decision_tier =
            rec.inventory[size_t(receipt.planner_plan_candidate)].origin_tier;
    }
    receipt.fallback_reason = fallback_reason ==
            common_cache_plan_authority_fallback::none
        ? planner_fallback(rec.planner_status) : fallback_reason;
}

void common_cache_plan_finalize_shadow_authority(common_cache_plan_record & rec) noexcept {
    if (!rec.planner_precomputed) {
        common_cache_plan_derive_shadow_authority(
            rec, common_cache_plan_authority_level::off,
            common_cache_plan_authority_fallback::none);
        // Observer-only records predate authority fallbacks: preserve their
        // schema-v5 meaning even when the shadow planner itself refused.
        rec.authority.fallback_reason =
            common_cache_plan_authority_fallback::none;
    }
    auto & receipt = rec.authority;
    receipt.legacy_tier = rec.selection;
    // Preserve a pre-mutation legacy counterfactual through every later
    // fallback. A similarity retarget can demote after switching slots; its
    // shipped plan then belongs to the planner target even though the state is
    // fallback_legacy. Only records that never computed a counterfactual may
    // derive one from the final shipped plan.
    if (receipt.legacy_plan_candidate < 0) {
        receipt.legacy_plan_candidate = rec.shipped_plan_candidate;
    }
    receipt.executed_plan_candidate = rec.shipped_plan_candidate;
    receipt.disagreed = receipt.legacy_plan_candidate >= 0 &&
                        receipt.planner_plan_candidate >= 0 &&
                        receipt.legacy_plan_candidate != receipt.planner_plan_candidate;
}

void common_cache_plan_authority_counters::observe(
        const common_cache_plan_authority_receipt & receipt,
        bool qualified) noexcept {
    last_receipt = receipt;
    has_receipt = true;

    const size_t tier = size_t(receipt.legacy_tier);
    if (tier >= size_t(common_cache_plan_selection::_count)) {
        return;
    }
    observed[tier]++;
    if (receipt.planner_plan_candidate >= 0) {
        (receipt.disagreed ? disagree : agree)[tier]++;
    }
    const size_t decision_tier = size_t(receipt.decision_tier);
    if (qualified && decision_tier < size_t(common_cache_plan_selection::_count)) {
        authority_eligible[decision_tier]++;
    }
    if (receipt.state == common_cache_plan_authority_state::authoritative) {
        if (decision_tier < size_t(common_cache_plan_selection::_count)) {
            if (!qualified) {
                authority_eligible[decision_tier]++;
            }
            authority_executed[decision_tier]++;
        }
    } else if (receipt.state == common_cache_plan_authority_state::fallback_legacy) {
        fallback_legacy[tier]++;
        const size_t reason = size_t(receipt.fallback_reason);
        if (reason < size_t(common_cache_plan_authority_fallback::_count)) {
            fallback_reason[reason]++;
        }
    }
}

const char * common_cache_plan_inventory_state_name(common_cache_plan_inventory_state s) {
    switch (s) {
        case common_cache_plan_inventory_state::unobserved: return "unobserved";
        case common_cache_plan_inventory_state::complete:   return "complete";
        case common_cache_plan_inventory_state::truncated_by_shipped_short_circuit:
            return "truncated_by_shipped_short_circuit";
        case common_cache_plan_inventory_state::overflowed: return "overflowed";
        case common_cache_plan_inventory_state::_count:     break;
    }
    return "invalid";
}

const char * common_cache_plan_planner_status_name(common_cache_plan_planner_status st) {
    switch (st) {
        case common_cache_plan_planner_status::not_attempted:       return "not_attempted";
        case common_cache_plan_planner_status::ok:                  return "ok";
        case common_cache_plan_planner_status::no_profile:          return "no_profile";
        case common_cache_plan_planner_status::profile_unfitted:    return "profile_unfitted";
        case common_cache_plan_planner_status::invalid_calibration: return "invalid_calibration";
        case common_cache_plan_planner_status::incomplete_evidence: return "incomplete_evidence";
        case common_cache_plan_planner_status::internal_fault:      return "internal_fault";
        case common_cache_plan_planner_status::_count:              break;
    }
    return "invalid";
}

const char * common_cache_plan_yield_status_name(common_cache_plan_yield_status st) {
    switch (st) {
        case common_cache_plan_yield_status::fits:                 return "fits";
        case common_cache_plan_yield_status::insufficient_yield:   return "insufficient_yield";
        case common_cache_plan_yield_status::unsupported_required: return "unsupported_required";
        case common_cache_plan_yield_status::unavailable:          return "unavailable";
        case common_cache_plan_yield_status::_count:                break;
    }
    return "invalid";
}

const char * common_cache_plan_yield_plan_state_name(common_cache_plan_yield_plan_state st) {
    switch (st) {
        case common_cache_plan_yield_plan_state::not_required: return "not_required";
        case common_cache_plan_yield_plan_state::planned:      return "planned";
        case common_cache_plan_yield_plan_state::unavailable:  return "unavailable";
        case common_cache_plan_yield_plan_state::_count:        break;
    }
    return "invalid";
}

const char * common_cache_plan_yield_actual_state_name(common_cache_plan_yield_actual_state st) {
    switch (st) {
        case common_cache_plan_yield_actual_state::not_observed: return "not_observed";
        case common_cache_plan_yield_actual_state::measured:     return "measured";
        case common_cache_plan_yield_actual_state::unavailable:  return "unavailable";
        case common_cache_plan_yield_actual_state::_count:        break;
    }
    return "invalid";
}

const char * common_cache_acct_category_name(llama_cache_acct_category c) {
    switch (c) {
        case llama_cache_acct_category::live_attention_state:                 return "live_attention_state";
        case llama_cache_acct_category::live_recurrent_state:                 return "live_recurrent_state";
        case llama_cache_acct_category::recurrent_rollback_planes:            return "recurrent_rollback_planes";
        case llama_cache_acct_category::full_snapshot_payload:                return "full_snapshot_payload";
        case llama_cache_acct_category::checkpoint_state_payload:             return "checkpoint_state_payload";
        case llama_cache_acct_category::typed_accelerator_payload:            return "typed_accelerator_payload";
        case llama_cache_acct_category::checkpoint_generation_page_metadata:  return "checkpoint_generation_page_metadata";
        case llama_cache_acct_category::checkpoint_generation_unit_metadata:  return "checkpoint_generation_unit_metadata";
        case llama_cache_acct_category::live_generation_metadata:             return "live_generation_metadata";
        case llama_cache_acct_category::ownership_index_metadata:             return "ownership_index_metadata";
        case llama_cache_acct_category::unit_version_payload:                 return "unit_version_payload";
        case llama_cache_acct_category::clean_stash_payload:                  return "clean_stash_payload";
        case llama_cache_acct_category::artifact_descriptor_metadata:         return "artifact_descriptor_metadata";
        case llama_cache_acct_category::artifact_reference_metadata:          return "artifact_reference_metadata";
        case llama_cache_acct_category::transfer_staging:                     return "transfer_staging";
        case llama_cache_acct_category::codec_workspace:                      return "codec_workspace";
        case llama_cache_acct_category::pinned_preimage_ring:                 return "pinned_preimage_ring";
        case llama_cache_acct_category::rolling_window_tape:                  return "rolling_window_tape";
        case llama_cache_acct_category::container_overhead:                   return "container_overhead";
        case llama_cache_acct_category::_count:                               break;
    }
    return "invalid";
}

const char * common_cache_acct_residency_name(llama_cache_acct_residency r) {
    switch (r) {
        case llama_cache_acct_residency::device:         return "device";
        case llama_cache_acct_residency::pinned_host:    return "pinned_host";
        case llama_cache_acct_residency::pageable_host:  return "pageable_host";
        case llama_cache_acct_residency::disk:           return "disk";
        case llama_cache_acct_residency::remote:         return "remote";
        case llama_cache_acct_residency::not_applicable: return "not_applicable";
        case llama_cache_acct_residency::_count:         break;
    }
    return "invalid";
}

const char * common_cache_acct_domain_kind_name(llama_cache_acct_domain_kind k) {
    switch (k) {
        case llama_cache_acct_domain_kind::not_applicable: return "not_applicable";
        case llama_cache_acct_domain_kind::device_topology: return "device_topology";
        case llama_cache_acct_domain_kind::_count:          break;
    }
    return "invalid";
}

const char * common_cache_acct_producer_name(llama_cache_acct_producer p) {
    switch (p) {
        case llama_cache_acct_producer::observer_init: return "observer_init";
        case llama_cache_acct_producer::host_cache:    return "host_cache";
        case llama_cache_acct_producer::live_memory:   return "live_memory";
        case llama_cache_acct_producer::retention_sidecar: return "retention_sidecar";
        case llama_cache_acct_producer::_count:        break;
    }
    return "invalid";
}

const char * common_cache_acct_measure_name(llama_cache_acct_measure m) {
    switch (m) {
        case llama_cache_acct_measure::logical_payload:    return "logical_payload";
        case llama_cache_acct_measure::resident_allocated: return "resident_allocated";
        case llama_cache_acct_measure::reserved:           return "reserved";
        case llama_cache_acct_measure::transient_peak:     return "transient_peak";
        case llama_cache_acct_measure::_count:             break;
    }
    return "invalid";
}

const char * common_cache_acct_known_name(llama_cache_acct_known k) {
    switch (k) {
        case llama_cache_acct_known::known:       return "known";
        case llama_cache_acct_known::unknown:     return "unknown";
        case llama_cache_acct_known::unavailable: return "unavailable";
        case llama_cache_acct_known::_count:      break;
    }
    return "invalid";
}

const char * common_cache_acct_unit_name(llama_cache_acct_unit u) {
    switch (u) {
        case llama_cache_acct_unit::bytes:      return "bytes";
        case llama_cache_acct_unit::tokens:     return "tokens";
        case llama_cache_acct_unit::operations: return "operations";
        case llama_cache_acct_unit::_count:     break;
    }
    return "invalid";
}

const char * common_cache_acct_cost_kind_name(llama_cache_acct_cost_kind k) {
    switch (k) {
        case llama_cache_acct_cost_kind::restore:   return "restore";
        case llama_cache_acct_cost_kind::replay:    return "replay";
        case llama_cache_acct_cost_kind::transfer:  return "transfer";
        case llama_cache_acct_cost_kind::eviction:  return "eviction";
        case llama_cache_acct_cost_kind::workspace: return "workspace";
        case llama_cache_acct_cost_kind::_count:    break;
    }
    return "invalid";
}

void common_cache_plan_compose_chains(common_cache_plan_record & rec) {
    rec.shipped_plan_candidate = rec.selected[size_t(rec.chosen)];

    auto * host = rec.selected_row(common_cache_plan_provider::host_cache_entry);
    if (!host || !host->delivered) {
        return;
    }
    // Only checkpoints exposed by THIS delivered host entry can compose with it.
    // Pre-mutation live checkpoints were destroyed by the restore, and checkpoints exposed
    // by another host are unreachable alternatives. Prefer the pre-mutation inventory chain
    // with the exact component ordinals: the authority receipt compares stable inventory
    // ordinals, so appending a duplicate here would fabricate a disagreement.
    const int32_t host_ord =
        rec.selected[size_t(common_cache_plan_provider::host_cache_entry)];
    const int32_t sel_ckpt =
        rec.selected[size_t(common_cache_plan_provider::live_context_checkpoint)];
    const uint32_t n_before = rec.n_inventory; // chains appended below are not siblings
    bool shipped_chain_recorded = false;
    for (uint32_t i = 0; i < n_before; i++) {
        auto & sib = rec.inventory[i];
        if (sib.provider != common_cache_plan_provider::live_context_checkpoint ||
            sib.target_slot_id != host->target_slot_id ||
            !sib.component_only ||
            sib.dependent_host_source_id != host->source_id) {
            continue;
        }
        if (!sib.viable()) {
            continue;
        }
        common_cache_plan_candidate * chain = rec.find_chain(
            common_cache_plan_provider::host_cache_entry, host_ord, int32_t(i));
        if (!chain) {
            chain = rec.add_chain(common_cache_plan_provider::host_cache_entry,
                                  host_ord, int32_t(i));
            if (!chain) {
                break; // derived_plans_incomplete latched; planner will refuse
            }
            chain->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
        }
        if ((int32_t) i == sel_ckpt && sib.delivered) {
            chain->accept();
            chain->delivered = true;
            rec.shipped_plan_candidate = int32_t(chain - rec.inventory.data());
            shipped_chain_recorded = true;
        }
    }
    // a composed delivery whose chain could not be recorded has NO honest shipped-plan
    // ordinal — the bare dependent checkpoint must not stand in for it (verify-r3
    // finding 2)
    if (sel_ckpt >= 0 && rec.inventory[size_t(sel_ckpt)].delivered &&
        !shipped_chain_recorded) {
        rec.shipped_plan_candidate = -1;
    }
}

json common_cache_plan_value_json(const llama_cache_acct_value & v) {
    if (v.state == llama_cache_acct_known::known) {
        return json(v.value);
    }
    return json(common_cache_acct_known_name(v.state));
}

static json cache_acct_domain_json(const llama_cache_acct_resource_domain & domain) {
    json out = {
        { "residency", common_cache_acct_residency_name(domain.residency) },
        { "kind",      common_cache_acct_domain_kind_name(domain.kind) },
    };
    if (domain.kind == llama_cache_acct_domain_kind::not_applicable) {
        out["device_ordinal"] = "not_applicable";
        out["topology_id"] = "not_applicable";
        return out;
    }
    out["device_ordinal"] = domain.device_ordinal.v;
    out["topology_id"] = domain.topology.v;
    return out;
}

static json cache_plan_yield_json(const common_cache_plan_yield_record & yield) {
    json selected_attention = json::array();
    for (const auto artifact : yield.selected_attention) {
        selected_attention.push_back(artifact.v);
    }
    json selected_recurrent = json::array();
    for (const auto artifact : yield.selected_recurrent) {
        selected_recurrent.push_back(artifact.v);
    }
    json unsupported = json::array();
    for (const auto artifact : yield.unsupported) {
        unsupported.push_back(artifact.v);
    }

    json projected_domains = json::array();
    for (const auto & row : yield.projected_domains) {
        projected_domains.push_back(json {
            { "domain",            cache_acct_domain_json(row.domain) },
            { "current_resident",  common_cache_plan_value_json(
                                           row.current_resident_bytes) },
            { "fit_before",        common_cache_plan_value_json(
                                           row.fit_before_bytes) },
            { "projected_release", common_cache_plan_value_json(
                                           row.projected_release_bytes) },
            { "projected_reserve", common_cache_plan_value_json(
                                           row.projected_reserve_bytes) },
            { "projected_after",   common_cache_plan_value_json(
                                           row.projected_after_bytes) },
        });
    }

    json actual_domains = json::array();
    for (const auto & row : yield.actual_domains) {
        actual_domains.push_back(json {
            { "domain",   cache_acct_domain_json(row.domain) },
            { "before",   common_cache_plan_value_json(row.before_bytes) },
            { "released", common_cache_plan_value_json(row.released_bytes) },
            { "after",    common_cache_plan_value_json(row.after_bytes) },
        });
    }

    return json {
        { "status",               common_cache_plan_yield_status_name(
                                        yield.status) },
        { "plan_state",           common_cache_plan_yield_plan_state_name(
                                        yield.plan_state) },
        { "actual_state",         common_cache_plan_yield_actual_state_name(
                                        yield.actual_state) },
        { "yield_policy_version", yield.yield_policy_version },
        { "accounting_serial",    yield.accounting_serial },
        { "selected", json {
            { "attention", std::move(selected_attention) },
            { "recurrent", std::move(selected_recurrent) },
        } },
        { "unsupported",       std::move(unsupported) },
        { "projected_domains", std::move(projected_domains) },
        { "actual_domains",    std::move(actual_domains) },
    };
}

static json cache_plan_destruction_json(
        const common_cache_plan_destruction_receipt & receipt) {
    json effects = json::array();
    for (uint8_t raw = uint8_t(common_cache_plan_destruction_effect::none) + 1;
         raw < uint8_t(common_cache_plan_destruction_effect::_count); ++raw) {
        const auto effect = common_cache_plan_destruction_effect(raw);
        if (!common_cache_plan_destruction_effect_has(receipt.effects, effect)) {
            continue;
        }
        effects.push_back({
            { "effect", common_cache_plan_destruction_effect_name(effect) },
            { "action_class", common_cache_plan_destruction_class_name(
                                  common_cache_plan_destruction_class_for_effect(effect)) },
            { "physical_reason", common_cache_plan_destruction_physical_reason_name(
                                      common_cache_plan_destruction_physical_reason_for_effect(effect)) },
        });
    }
    json attention = json::array();
    for (const auto artifact : receipt.selected_attention) {
        attention.push_back(artifact.v);
    }
    json recurrent = json::array();
    for (const auto artifact : receipt.selected_recurrent) {
        recurrent.push_back(artifact.v);
    }
    const auto digest_json = [](const auto & digest) -> json {
        return digest.valid()
            ? json(common_cache_plan_sha256_hex_digest(digest.bytes()))
            : json(common_cache_acct_known_name(
                  llama_cache_acct_known::unavailable));
    };
    json recovery_source = common_cache_acct_known_name(
        llama_cache_acct_known::unavailable);
    if (receipt.recovery_source_artifact_id.v != 0 &&
        receipt.recovery_source_manifest_digest.valid()) {
        recovery_source = json {
            { "artifact_id", receipt.recovery_source_artifact_id.v },
            { "manifest_digest", digest_json(
                  receipt.recovery_source_manifest_digest) },
        };
    }
    return json {
        { "policy_version", receipt.policy_version },
        { "state", common_cache_plan_destruction_state_name(receipt.state) },
        { "reason", common_cache_plan_destruction_reason_name(receipt.reason) },
        { "payload_kind", receipt.payload_kind ==
                common_cache_plan_payload_kind::unavailable
            ? json(nullptr)
            : json(common_cache_plan_payload_kind_name(receipt.payload_kind)) },
        { "effects", std::move(effects) },
        { "lease_verdict", common_cache_plan_destruction_lease_verdict_name(receipt.lease_verdict) },
        { "displaced_fate", common_cache_plan_displaced_fate_name(receipt.displaced_fate) },
        { "recovery_citation", common_cache_plan_recovery_citation_name(receipt.recovery_citation) },
        { "recovery_source", std::move(recovery_source) },
        { "post_finalize_comparison", common_cache_plan_destruction_comparison_name(receipt.post_finalize_comparison) },
        { "plan_candidate", receipt.plan_candidate >= 0
              ? json(receipt.plan_candidate)
              : json(common_cache_acct_known_name(
                    llama_cache_acct_known::unavailable)) },
        { "admission_sequence", receipt.admission_sequence },
        { "quote_duration_us", receipt.quote_duration_us },
        { "quote_accounting_serial", receipt.quote_accounting_serial },
        { "actual_accounting_serial", receipt.actual_accounting_serial },
        { "manifest_digest", digest_json(receipt.manifest_digest) },
        { "union_effect_digest", digest_json(receipt.union_effect_digest) },
        { "selected", json {
            { "attention", std::move(attention) },
            { "recurrent", std::move(recurrent) },
        } },
    };
}

// phase-bit spellings (single source; CI scans ban replicas like the other name tables)
static json cache_plan_phases_json(uint8_t phases_seen) {
    static constexpr struct { uint8_t bit; const char * name; } bits[] = {
        { COMMON_CACHE_PLAN_PHASE_BY_ID,      "by_id_route"  },
        { COMMON_CACHE_PLAN_PHASE_SIMILARITY, "similarity_scan" },
        { COMMON_CACHE_PLAN_PHASE_ROUTE_HOME, "route_home_scan" },
        { COMMON_CACHE_PLAN_PHASE_LRU,        "lru_scan"     },
        { COMMON_CACHE_PLAN_PHASE_HOST_SCAN,  "host_scan"    },
        { COMMON_CACHE_PLAN_PHASE_CKPT_SCAN,  "ckpt_scan"    },
        { COMMON_CACHE_PLAN_PHASE_CHAIN,      "chain"        },
    };
    json out = json::array();
    for (const auto & b : bits) {
        if (phases_seen & b.bit) {
            out.push_back(b.name);
        }
    }
    return out;
}

static json cache_plan_ordinal_json(int32_t ordinal) {
    return ordinal >= 0
        ? json(ordinal)
        : json(common_cache_acct_known_name(llama_cache_acct_known::unavailable));
}

static json cache_plan_payload_kind_json(
        common_cache_plan_payload_kind kind) {
    return kind == common_cache_plan_payload_kind::unavailable
        ? json(nullptr)
        : json(common_cache_plan_payload_kind_name(kind));
}

json common_cache_plan_record_json(const common_cache_plan_record & rec) {
    const bool finalized = rec.outcome != common_cache_plan_outcome::unknown;

    json cands = json::array();
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        const auto & c = rec.inventory[i];
        json jc = {
            { "id",            i },
            { "target_slot_id", cache_plan_ordinal_json(c.target_slot_id) },
            { "origin_tier",   common_cache_plan_selection_name(c.origin_tier) },
            { "provider",      common_cache_plan_provider_name(c.provider) },
            { "payload_kind",  cache_plan_payload_kind_json(c.payload_kind) },
            { "phases",        cache_plan_phases_json(c.phases_seen) },
            { "disposition",   common_cache_plan_disposition_name(c.disposition) },
            { "reason",        common_cache_plan_reason_name(c.reason) },
            { "delivered",     c.delivered },
            { "lcp_tokens",    common_cache_plan_value_json(c.lcp_tokens) },
            { "payload_bytes", common_cache_plan_value_json(c.payload_bytes) },
        };
        if (c.source_id >= 0)  { jc["source_id"] = c.source_id; }
        if (c.component_only)  { jc["component_only"] = true; }
        if (c.sim_known)       { jc["sim"]       = c.sim; }
        if (c.f_keep_known)    { jc["f_keep"]    = c.f_keep; }
        if (c.spec_capable_known) { jc["spec_capable"] = c.spec_capable; }
        if (c.t_last_used_us.state == llama_cache_acct_known::known) {
            jc["t_last_used_us"] =
                common_cache_plan_value_json(c.t_last_used_us);
        }
        if (c.siblings_scanned > 0) {
            jc["siblings_scanned"]        = c.siblings_scanned;
            jc["siblings_rejected_epoch"] = c.siblings_rejected_epoch;
        }
        if (c.is_chain()) {
            // explicit marker: a chain row's `provider` names its BASE provider, so
            // provider histograms must not count it as a plain entry
            jc["is_chain"] = true;
            json comps = json::array();
            for (const int32_t comp : c.component_ids) {
                if (comp >= 0) {
                    comps.push_back(comp);
                }
            }
            jc["components"] = std::move(comps);
        }
        // per-candidate economics only once an estimator produced them — absence on the wire
        // IS the typed-unavailable state; emitting five unavailable slots per row would bloat
        // every pre-planner record
        bool any_estimated = c.predicted_total_us.state == llama_cache_acct_known::known;
        for (const auto & term : c.cost_terms) {
            any_estimated = any_estimated || term.estimated_us.state == llama_cache_acct_known::known
                                          || term.raw.state          == llama_cache_acct_known::known;
        }
        if (any_estimated) {
            json terms = json::object();
            for (const auto & term : c.cost_terms) {
                // wire discipline: absence IS the typed-unavailable state, per term — the
                // D-owned kinds (transfer/eviction) would otherwise add two dead objects
                // to every estimated row (kind->unit is fixed schema, nothing is lost)
                if (term.raw.state          != llama_cache_acct_known::known &&
                    term.estimated_us.state != llama_cache_acct_known::known) {
                    continue;
                }
                json jt = {
                    { "raw",          common_cache_plan_value_json(term.raw) },
                    { "unit",         common_cache_acct_unit_name(term.raw_unit) },
                    { "estimated_us", common_cache_plan_value_json(term.estimated_us) },
                };
                // the estimator version is metadata OF an estimate: emitting it while the
                // estimate is unavailable would fabricate evidence
                if (term.estimated_us.state == llama_cache_acct_known::known) {
                    jt["estimator_version"] = term.estimator_version;
                }
                terms[common_cache_acct_cost_kind_name(term.kind)] = std::move(jt);
            }
            jc["cost_terms"]         = std::move(terms);
            jc["predicted_total_us"] =
                common_cache_plan_value_json(c.predicted_total_us);
        }
        cands.push_back(std::move(jc));
    }

    // per-provider observed-inventory completeness over the declared (shipped-visited) domain
    json inv_states = json::object();
    for (size_t p = 0; p < size_t(common_cache_plan_provider::_count); p++) {
        const auto st = rec.inventory_states[p];
        if (st != common_cache_plan_inventory_state::unobserved) {
            inv_states[common_cache_plan_provider_name(common_cache_plan_provider(p))] =
                common_cache_plan_inventory_state_name(st);
        }
    }

    // causal delivery chain: the selected candidates that actually applied state, in shipped
    // order (live slot prefix → host snapshot → context checkpoint). `chosen` is terminal.
    json chain = json::array();
    for (const auto prov : { common_cache_plan_provider::live_slot,
                             common_cache_plan_provider::host_cache_entry,
                             common_cache_plan_provider::live_context_checkpoint }) {
        const int32_t sel = rec.selected[size_t(prov)];
        if (sel >= 0 && uint32_t(sel) < rec.n_inventory && rec.inventory[size_t(sel)].delivered) {
            chain.push_back(common_cache_plan_provider_name(prov));
        }
    }

    json out = {
        { "schema_version",    rec.schema_version },
        { "id_task",           rec.id_task },
        { "id_slot",           rec.id_slot },
        { "selection",         common_cache_plan_selection_name(rec.selection) },
        { "identity", json {
            { "model",              common_cache_plan_value_json(rec.identity.model_digest) },
            { "execution",          common_cache_plan_value_json(rec.identity.execution_digest) },
            { "adapter_config",     common_cache_plan_value_json(rec.identity.adapter_config_digest) },
            { "media_content",      common_cache_plan_value_json(rec.identity.media_content_digest) },
            { "tokenizer_template", common_cache_plan_value_json(rec.identity.tokenizer_template_digest) },
            { "prefix_tokens",      common_cache_plan_value_json(rec.identity.prefix_token_digest) },
        } },
        { "candidates",        std::move(cands) },
        { "inventory_states",  std::move(inv_states) },
        { "delivered_chain",   std::move(chain) },
        { "seq_cp_capability", true }, // copy_state_to's primitive exists on every build
        { "chosen",            finalized ? common_cache_plan_provider_name(rec.chosen) : "unknown" },
        { "outcome",           common_cache_plan_outcome_name(rec.outcome) },
        { "n_prompt_tokens",   common_cache_plan_value_json(rec.n_prompt_tokens) },
        { "n_reused_tokens",   common_cache_plan_value_json(rec.n_reused_tokens) },
        { "n_replayed_tokens", common_cache_plan_value_json(rec.n_replayed_tokens) },
        { "ttft_us",           common_cache_plan_value_json(rec.ttft_us) },
    };
    if (!rec.calibration_profile.empty()) {
        out["calibration_profile"] = rec.calibration_profile;
    }
    if (finalized) {
        const int32_t sel = rec.selected[size_t(rec.chosen)];
        if (sel >= 0) {
            out["chosen_candidate"] = sel;
        }
        // the COMPLETE shipped plan (chain ordinal on composed deliveries) — offline
        // agreement runs against THIS, never the terminal provider's bare row
        if (rec.shipped_plan_candidate >= 0) {
            out["shipped_plan_candidate"] = rec.shipped_plan_candidate;
        }
        out["planner_status"] = common_cache_plan_planner_status_name(rec.planner_status);
        out["yield"] = cache_plan_yield_json(rec.yield);
        out["destruction"] = cache_plan_destruction_json(rec.destruction);
    }
    out["authority"] = json {
        { "policy_version",          rec.authority.policy_version },
        { "configured_level",        common_cache_plan_authority_level_name(
                                           rec.authority.configured_level) },
        { "legacy_tier",             common_cache_plan_selection_name(
                                           rec.authority.legacy_tier) },
        { "decision_tier",           common_cache_plan_selection_name(
                                           rec.authority.decision_tier) },
        { "state",                   common_cache_plan_authority_state_name(
                                           rec.authority.state) },
        { "legacy_plan_candidate",   cache_plan_ordinal_json(
                                           rec.authority.legacy_plan_candidate) },
        { "planner_plan_candidate",  cache_plan_ordinal_json(
                                           rec.authority.planner_plan_candidate) },
        { "executed_plan_candidate", cache_plan_ordinal_json(
                                           rec.authority.executed_plan_candidate) },
        { "fallback_reason",         common_cache_plan_authority_fallback_name(
                                           rec.authority.fallback_reason) },
        { "disagreed",               rec.authority.disagreed },
    };
    // shadow-planner result: an object when computed, the typed unavailable name otherwise
    // (string-sentinel follows the acct-value wire convention used record-wide). The tie
    // floors are chooser SEMANTICS: without them on the wire, logs spanning builds with
    // different floors would merge into one silently-mixed agreement rate.
    if (rec.shadow_choice >= 0) {
        json ties = json::array();
        for (uint32_t i = 0; i < rec.n_shadow_ties; i++) {
            ties.push_back(rec.shadow_tie_set[i]);
        }
        out["shadow"] = json {
            { "choice",  rec.shadow_choice },
            { "tie_set", std::move(ties) },
            { "tie_floor", json {
                { "rel",    COMMON_CACHE_PLAN_TIE_REL_FLOOR },
                { "abs_us", COMMON_CACHE_PLAN_TIE_ABS_FLOOR_US },
            } },
        };
    } else {
        out["shadow"] = common_cache_acct_known_name(llama_cache_acct_known::unavailable);
    }
    if (rec.sim_best_any_known) {
        out["sim_best_any"] = rec.sim_best_any;
    }

    if (finalized) {
        // touched cells only (state != unknown) — a known ZERO is an observation and is
        // emitted; an unknown cell is silence, never a zero
        json cells = json::array();
        for (const auto & row : rec.acct.cells) {
            for (size_t m = 0; m < size_t(llama_cache_acct_measure::_count); m++) {
                const auto & cell = row.cell.measures[m];
                if (cell.state == llama_cache_acct_known::unknown) {
                    continue;
                }
                cells.push_back(json {
                    { "category", common_cache_acct_category_name(row.category) },
                    { "domain",   cache_acct_domain_json(row.domain) },
                    { "certification", common_cache_acct_known_name(row.certification) },
                    { "measure",  common_cache_acct_measure_name(llama_cache_acct_measure(m)) },
                    { "value",    common_cache_plan_value_json(cell) },
                });
            }
        }

        json topologies = json::array();
        for (const auto & row : rec.acct.topologies) {
            json devices = json::array();
            for (const auto & identity : row.topology.device_identities) {
                devices.push_back(common_cache_plan_sha256_hex_digest(identity.bytes()));
            }
            topologies.push_back(json {
                { "id",                  row.id.v },
                { "version",             row.topology.version },
                { "digest",              common_cache_plan_sha256_hex_digest(
                                                row.topology.digest.bytes()) },
                { "split_mode",          row.topology.split_mode },
                { "main_device_ordinal", row.topology.main_device.v },
                { "device_identities",   std::move(devices) },
                { "shard_weights",       row.topology.shard_weights },
                { "weight_denominator",  LLAMA_CACHE_ACCT_SHARD_WEIGHT_DENOMINATOR },
            });
        }

        // Configuration-owned required manifest, reported row-for-row. There is
        // intentionally no server-wide completeness scalar in accounting schema v2.
        json completeness = json::array();
        for (const auto & row : rec.acct.completeness) {
            completeness.push_back(json {
                { "domain",   cache_acct_domain_json(row.domain) },
                { "producer", common_cache_acct_producer_name(row.producer) },
                { "state",    common_cache_acct_known_name(row.state) },
            });
        }
        out["accounting"] = json {
            { "schema_version", rec.acct.schema_version },
            { "serial",         rec.acct.serial },
            { "completeness_manifest",
                common_cache_acct_known_name(rec.acct.completeness_manifest) },
            { "completeness",   std::move(completeness) },
            { "topologies",     std::move(topologies) },
            { "live_ops",       rec.acct.live_ops },
            { "cells",          std::move(cells) },
            { "faults", json {
                { "invalid_transition", rec.acct.faults_invalid_transition },
                { "overflow",           rec.acct.faults_overflow },
                { "unknown_id",         rec.acct.faults_unknown_id },
                { "allocation",         rec.acct.faults_allocation },
            } },
        };
    }

    return out;
}
