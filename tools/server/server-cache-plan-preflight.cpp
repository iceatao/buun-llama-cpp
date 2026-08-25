#include "server-cache-plan-preflight-internal.h"

#include "server-cache-plan-authority.h"

#include <algorithm>
#include <array>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

json public_value(const llama_cache_acct_value & value) {
    return value.state == llama_cache_acct_known::known
        ? json(value.value) : json(nullptr);
}

const char * preflight_status_name(
        server_cache_plan_preflight_status status) noexcept {
    switch (status) {
        case server_cache_plan_preflight_status::ok:             return "ok";
        case server_cache_plan_preflight_status::no_target:      return "no_target";
        case server_cache_plan_preflight_status::internal_fault: return "internal_fault";
        case server_cache_plan_preflight_status::_count:         break;
    }
    return "invalid";
}

const char * expected_path_name(
        server_cache_plan_preflight_expected_path path) noexcept {
    switch (path) {
        case server_cache_plan_preflight_expected_path::legacy:
            return "legacy";
        case server_cache_plan_preflight_expected_path::
                 planner_if_still_current:
            return "planner_if_still_current";
        case server_cache_plan_preflight_expected_path::
                 conditional_on_destruction_certification:
            return "conditional_on_destruction_certification";
        case server_cache_plan_preflight_expected_path::_count:
            break;
    }
    return "invalid";
}

const char * target_relation_name(
        server_cache_plan_preflight_target_relation relation) noexcept {
    switch (relation) {
        case server_cache_plan_preflight_target_relation::unavailable:
            return "unavailable";
        case server_cache_plan_preflight_target_relation::forced_slot:
            return "forced_slot";
        case server_cache_plan_preflight_target_relation::same_as_legacy:
            return "same_as_legacy";
        case server_cache_plan_preflight_target_relation::retarget:
            return "retarget";
        case server_cache_plan_preflight_target_relation::_count:
            break;
    }
    return "invalid";
}

const char * cache_hit_name(
        server_cache_plan_preflight_cache_hit hit) noexcept {
    switch (hit) {
        case server_cache_plan_preflight_cache_hit::unavailable:
            return "unavailable";
        case server_cache_plan_preflight_cache_hit::miss:    return "miss";
        case server_cache_plan_preflight_cache_hit::partial: return "partial";
        case server_cache_plan_preflight_cache_hit::full:    return "full";
        case server_cache_plan_preflight_cache_hit::_count:  break;
    }
    return "invalid";
}

const char * assessment_name(
        common_cache_plan_destruction_state state) noexcept {
    switch (state) {
        case common_cache_plan_destruction_state::not_required:
            return "not_required";
        case common_cache_plan_destruction_state::quoted:
            return "eligible_at_snapshot";
        case common_cache_plan_destruction_state::refused:
            return "blocked";
        case common_cache_plan_destruction_state::failed:
            return "unavailable";
        case common_cache_plan_destruction_state::certified:
        case common_cache_plan_destruction_state::executed:
        case common_cache_plan_destruction_state::_count:
            break;
    }
    // A read-only preview cannot produce certified/executed.
    return "unavailable";
}

const char * protection_name(
        common_cache_plan_destruction_lease_verdict verdict) noexcept {
    switch (verdict) {
        case common_cache_plan_destruction_lease_verdict::unleased:
            return "none";
        case common_cache_plan_destruction_lease_verdict::soft_leased:
            return "weighted";
        case common_cache_plan_destruction_lease_verdict::hard_leased:
        case common_cache_plan_destruction_lease_verdict::mandatory_recovery:
            return "hard";
        case common_cache_plan_destruction_lease_verdict::unavailable:
            return "unavailable";
        case common_cache_plan_destruction_lease_verdict::_count:
            break;
    }
    return "unavailable";
}

json public_cost_term(const llama_cache_acct_cost_term & term) {
    const char * quantity = common_cache_acct_unit_name(term.raw_unit);
    return json {
        { quantity, public_value(term.raw) },
        { "estimated_us", public_value(term.estimated_us) },
    };
}

json public_effects(common_cache_plan_destruction_effect_set effects) {
    json out = json::array();
    for (uint8_t raw =
             uint8_t(common_cache_plan_destruction_effect::none) + 1;
         raw < uint8_t(common_cache_plan_destruction_effect::_count);
         ++raw) {
        const auto effect = common_cache_plan_destruction_effect(raw);
        if (!common_cache_plan_destruction_effect_has(effects, effect)) {
            continue;
        }
        out.push_back({
            { "effect", common_cache_plan_destruction_effect_name(effect) },
            { "action_class", common_cache_plan_destruction_class_name(
                  common_cache_plan_destruction_class_for_effect(effect)) },
            { "physical_reason",
              common_cache_plan_destruction_physical_reason_name(
                  common_cache_plan_destruction_physical_reason_for_effect(
                      effect)) },
        });
    }
    return out;
}

} // namespace

bool server_cache_plan_local_source_registry::get_or_assign(
        uintptr_t instance,
        int32_t & source_id) {
    auto [it, inserted] = source_ids_.emplace(instance, -1);
    (void) inserted;
    return server_cache_plan_assign_source_id(
        it->second, next_source_id_, source_id);
}

bool server_cache_plan_local_source_registry::find(
        uintptr_t instance,
        int32_t & source_id) const noexcept {
    const auto found = source_ids_.find(instance);
    if (found == source_ids_.end() || found->second < 0) {
        source_id = -1;
        return false;
    }
    source_id = found->second;
    return true;
}

server_cache_plan_preflight_semantics server_cache_plan_preflight_semantics_for(
        bool is_preflight,
        bool native_completion,
        bool update_cache,
        bool prompt_cache_available,
        bool adapter_matches) noexcept {
    server_cache_plan_preflight_semantics out;
    out.completion_semantics = is_preflight || native_completion;
    out.host_lookup_enabled = update_cache && prompt_cache_available &&
                              out.completion_semantics && adapter_matches;
    out.recovery_citation = prompt_cache_available && out.completion_semantics
        ? common_cache_plan_recovery_citation::prospective
        : common_cache_plan_recovery_citation::unavailable;
    return out;
}

static bool tier_enabled(const common_cache_plan_record & rec) noexcept {
    const auto decision = server_cache_plan_level_of(rec.selection);
    return decision != common_cache_plan_authority_level::off &&
           decision != common_cache_plan_authority_level::_count &&
           server_cache_plan_level_enabled(
               rec.authority.configured_level, decision);
}

server_cache_plan_preflight_expected_path
server_cache_plan_preflight_derive_expected_path(
        const common_cache_plan_record & rec,
        bool planner_inputs_current) noexcept {
    if (!planner_inputs_current ||
        rec.planner_status != common_cache_plan_planner_status::ok ||
        !server_cache_plan_shadow_choice_valid(rec)) {
        return server_cache_plan_preflight_expected_path::legacy;
    }
    if (!tier_enabled(rec)) {
        return server_cache_plan_preflight_expected_path::legacy;
    }
    if (rec.destruction.plan_candidate == rec.shadow_choice &&
        rec.destruction.effects != 0 &&
        rec.destruction.state ==
            common_cache_plan_destruction_state::quoted) {
        return server_cache_plan_preflight_expected_path::
            conditional_on_destruction_certification;
    }
    if (rec.destruction.plan_candidate == rec.shadow_choice &&
        rec.destruction.effects != 0) {
        return server_cache_plan_preflight_expected_path::legacy;
    }
    return server_cache_plan_preflight_expected_path::
        planner_if_still_current;
}

static llama_cache_acct_value term_raw(
        const common_cache_plan_candidate & candidate,
        llama_cache_acct_cost_kind kind) noexcept {
    return candidate.cost_terms[size_t(kind)].raw;
}

bool server_cache_plan_preflight_build_view(
        const common_cache_plan_record & rec,
        int32_t legacy_target_slot_id,
        bool planner_inputs_current,
        server_cache_plan_preflight_view & out) noexcept {
    try {
        out = {};
        out.status = server_cache_plan_preflight_status::ok;
        out.planner_status = rec.planner_status;
        out.configured_level = rec.authority.configured_level;
        out.selection_tier = rec.selection;
        out.fallback_reason = rec.authority.fallback_reason;
        if (rec.planner_status == common_cache_plan_planner_status::ok &&
            !planner_inputs_current) {
            out.fallback_reason =
                common_cache_plan_authority_fallback::stale_capability;
        } else if (rec.planner_status ==
                       common_cache_plan_planner_status::ok &&
                   !tier_enabled(rec)) {
            out.fallback_reason =
                common_cache_plan_authority_fallback::tier_not_enabled;
        }
        out.prompt_tokens = rec.n_prompt_tokens;
        out.expected_path = server_cache_plan_preflight_derive_expected_path(
            rec, planner_inputs_current);

        for (uint32_t i = 0; i < rec.n_inventory; ++i) {
            const auto & candidate = rec.inventory[i];
            if (candidate.reason == COMMON_CACHE_PLAN_REASON_NONE) {
                continue;
            }
            auto found = std::find_if(
                out.miss_reasons.begin(), out.miss_reasons.end(),
                [&](const auto & row) {
                    return row.provider == candidate.provider &&
                           row.reason == candidate.reason;
                });
            if (found == out.miss_reasons.end()) {
                out.miss_reasons.push_back({
                    candidate.provider, candidate.reason, 1,
                });
            } else {
                found->count++;
            }
        }

        if (rec.planner_status != common_cache_plan_planner_status::ok ||
            !server_cache_plan_shadow_choice_valid(rec)) {
            return true;
        }
        const auto & selected = rec.inventory[size_t(rec.shadow_choice)];
        out.provider = selected.provider;
        out.provider_available = true;
        out.target_relation = rec.selection == common_cache_plan_selection::by_id
            ? server_cache_plan_preflight_target_relation::forced_slot
            : (selected.target_slot_id == legacy_target_slot_id
                ? server_cache_plan_preflight_target_relation::same_as_legacy
                : server_cache_plan_preflight_target_relation::retarget);
        out.cost_terms = selected.cost_terms;
        for (const auto & term : selected.cost_terms) {
            if (term.estimated_us.state == llama_cache_acct_known::known) {
                out.estimator_version = term.estimator_version;
                break;
            }
        }
        out.predicted_replay_tokens = term_raw(
            selected, llama_cache_acct_cost_kind::replay);
        out.predicted_restore_bytes = term_raw(
            selected, llama_cache_acct_cost_kind::restore);
        out.predicted_ttft_us = selected.predicted_total_us;
        if (out.prompt_tokens.state == llama_cache_acct_known::known &&
            out.predicted_replay_tokens.state ==
                llama_cache_acct_known::known &&
            out.predicted_replay_tokens.value <= out.prompt_tokens.value) {
            out.predicted_reuse_tokens = llama_cache_acct_value::measured(
                out.prompt_tokens.value - out.predicted_replay_tokens.value);
        }
        if (selected.provider == common_cache_plan_provider::cold_replay) {
            out.cache_hit = server_cache_plan_preflight_cache_hit::miss;
        } else if (out.predicted_replay_tokens.state ==
                       llama_cache_acct_known::known) {
            out.cache_hit = out.predicted_replay_tokens.value == 0
                ? server_cache_plan_preflight_cache_hit::full
                : server_cache_plan_preflight_cache_hit::partial;
        }

        if (rec.destruction.plan_candidate == rec.shadow_choice ||
            rec.destruction.state ==
                common_cache_plan_destruction_state::not_required) {
            out.destruction.state = rec.destruction.state;
            out.destruction.reason = rec.destruction.reason;
            out.destruction.effects = rec.destruction.effects;
            out.destruction.protection = rec.destruction.lease_verdict;
            out.destruction.displaced_fate = rec.destruction.displaced_fate;
            out.destruction.recovery = rec.destruction.recovery_citation;
            uint64_t projected = 0;
            const auto quote = std::find_if(
                rec.destruction_quotes.begin(), rec.destruction_quotes.end(),
                [&](const auto & candidate) {
                    return candidate.receipt.plan_candidate == rec.shadow_choice;
                });
            if (quote != rec.destruction_quotes.end() &&
                common_cache_plan_projected_release_bytes(
                    quote->projected_domains, projected)) {
                out.destruction.projected_release_bytes =
                    llama_cache_acct_value::measured(projected);
            }
            out.destruction.estimated_destruction_us =
                selected.cost_terms[size_t(
                    llama_cache_acct_cost_kind::eviction)].estimated_us;
        } else {
            // A receipt for another candidate says nothing about the selected
            // union. Report that evidence gap instead of implying the selected
            // candidate carried a malformed manifest.
            out.destruction.state =
                common_cache_plan_destruction_state::failed;
            out.destruction.reason =
                common_cache_plan_destruction_reason::
                    release_evidence_unavailable;
        }
        return true;
    } catch (...) {
        out = {};
        out.status = server_cache_plan_preflight_status::internal_fault;
        return false;
    }
}

json server_cache_plan_preflight_json(
        const server_cache_plan_preflight_view & view) {
    json miss_reasons = json::array();
    for (const auto & row : view.miss_reasons) {
        miss_reasons.push_back({
            { "provider", common_cache_plan_provider_name(row.provider) },
            { "reason", common_cache_plan_reason_name(row.reason) },
            { "count", row.count },
        });
    }

    const auto & terms = view.cost_terms;
    json planner = {
        { "status", common_cache_plan_planner_status_name(
              view.planner_status) },
        { "configured_level", common_cache_plan_authority_level_name(
              view.configured_level) },
        { "selection_tier", common_cache_plan_selection_name(
              view.selection_tier) },
        { "expected_path", expected_path_name(view.expected_path) },
        { "fallback_reason", common_cache_plan_authority_fallback_name(
              view.fallback_reason) },
        { "provider", view.provider_available
              ? json(common_cache_plan_provider_name(view.provider))
              : json(nullptr) },
        { "target_relation", view.target_relation ==
                  server_cache_plan_preflight_target_relation::unavailable
              ? json(nullptr)
              : json(target_relation_name(view.target_relation)) },
        { "cache_hit", view.cache_hit ==
                  server_cache_plan_preflight_cache_hit::unavailable
              ? json(nullptr) : json(cache_hit_name(view.cache_hit)) },
        { "prompt_tokens", public_value(view.prompt_tokens) },
        { "predicted_reuse_tokens",
          public_value(view.predicted_reuse_tokens) },
        { "predicted_replay_tokens",
          public_value(view.predicted_replay_tokens) },
        { "predicted_restore_bytes",
          public_value(view.predicted_restore_bytes) },
        { "predicted_ttft_us", public_value(view.predicted_ttft_us) },
        { "estimate_scope", "cache_path_only" },
        { "estimator_version", view.estimator_version == 0
              ? json(nullptr) : json(view.estimator_version) },
        { "cost_terms", {
            { "replay", public_cost_term(terms[size_t(
                  llama_cache_acct_cost_kind::replay)]) },
            { "restore", public_cost_term(terms[size_t(
                  llama_cache_acct_cost_kind::restore)]) },
            { "workspace", public_cost_term(terms[size_t(
                  llama_cache_acct_cost_kind::workspace)]) },
            { "transfer", public_cost_term(terms[size_t(
                  llama_cache_acct_cost_kind::transfer)]) },
            { "eviction", public_cost_term(terms[size_t(
                  llama_cache_acct_cost_kind::eviction)]) },
        } },
    };

    const auto & destruction = view.destruction;
    json destruction_json = {
        { "required", destruction.effects != 0 },
        { "assessment", assessment_name(destruction.state) },
        { "reason", common_cache_plan_destruction_reason_name(
              destruction.reason) },
        { "effects", public_effects(destruction.effects) },
        { "protection", protection_name(destruction.protection) },
        { "displaced_fate", common_cache_plan_displaced_fate_name(
              destruction.displaced_fate) },
        { "recovery", common_cache_plan_recovery_citation_name(
              destruction.recovery) },
        { "projected_release_bytes",
          public_value(destruction.projected_release_bytes) },
        { "estimated_destruction_us",
          public_value(destruction.estimated_destruction_us) },
    };

    return json {
        { "object", "cache_plan_preflight" },
        { "schema_version", 1 },
        { "cache_plan_schema_version", COMMON_CACHE_PLAN_SCHEMA_VERSION },
        { "authoritative", false },
        { "reservation", "none" },
        { "valid_until", nullptr },
        { "status", preflight_status_name(view.status) },
        { "planner", std::move(planner) },
        { "destruction", std::move(destruction_json) },
        { "miss_reasons", std::move(miss_reasons) },
        { "limitations", json::array({
            "point_in_time",
            "no_reservation",
            "queue_and_contention_not_modeled",
            "post_generation_maintenance_not_modeled",
        }) },
    };
}

bool server_cache_plan_preflight_exposure_allowed(
        const std::string & hostname,
        size_t api_key_count) noexcept {
    const bool local = hostname == "127.0.0.1" || hostname == "::1" ||
                       hostname == "localhost" ||
                       (hostname.size() >= 5 &&
                        hostname.compare(hostname.size() - 5, 5, ".sock") == 0);
    return local && api_key_count <= 1;
}

bool server_cache_plan_preflight_request_field_allowed(
        std::string_view field) noexcept {
    static constexpr std::array<std::string_view, 5> accepted = {
        "prompt", "id_slot", "cache_prompt", "lora",
        "message_delimiters",
    };
    return std::find(accepted.begin(), accepted.end(), field) !=
           accepted.end();
}
