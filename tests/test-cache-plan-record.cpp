// Decision-record contract tests (schema v5): band monotonicity (compile-time),
// multi-failure first-reason precedence including out-of-order arrival, valid-loser
// disposition, per-entry inventory merge/overflow/completeness semantics, selection
// mapping, planner-output clearing, unknown-vs-zero on measured fields, and exhaustive
// name tables (every member of every closed enum must produce a non-"invalid" name).

#include "common-cache-plan.h"
#include "../tools/server/server-cache-lifecycle.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

// multi-failure: the earliest precedence band is THE reason regardless of arrival order
static void test_precedence() {
    common_cache_plan_candidate c;
    c.note_reject(COMMON_CACHE_PLAN_REASON_PERSISTENT_BUDGET_EXCEEDED);   // 500 first
    c.note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);                // 206 arrives later
    c.note_reject(COMMON_CACHE_PLAN_REASON_KV_TYPE_MISMATCH);             // 402 later still
    CHECK(c.reason == COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
    CHECK(c.disposition == common_cache_plan_disposition::rejected_invalid);

    // identity always dominates
    c.note_reject(COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);      // 102
    CHECK(c.reason == COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);
}

// a valid loser is not an invalid candidate — and an invalidity ever observed dominates cost
static void test_valid_loser() {
    common_cache_plan_candidate c;
    c.note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(c.disposition == common_cache_plan_disposition::valid_not_chosen_cost);

    c.note_reject(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
    CHECK(c.reason == COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
    CHECK(c.disposition == common_cache_plan_disposition::rejected_invalid);

    // and the reverse order: cost after invalidity never resurrects the candidate
    common_cache_plan_candidate d;
    d.note_reject(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
    d.note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(d.disposition == common_cache_plan_disposition::rejected_invalid);
}

// per-entry inventory: cross-phase merge on (target, provider, source), never duplicate rows for
// one physical candidate; phases accumulate; selection maps to a row
static void test_inventory_merge() {
    common_cache_plan_record rec;
    CHECK(rec.n_inventory == 0);
    for (size_t p = 0; p < size_t(common_cache_plan_provider::_count); p++) {
        CHECK(rec.inventory_states[p] == common_cache_plan_inventory_state::unobserved);
        CHECK(rec.selected[p] == -1);
        CHECK(rec.selected_row(common_cache_plan_provider(p)) == nullptr);
    }

    // slot 3 visited by similarity, then again by LRU: ONE row, both phase bits
    auto * a = rec.find_or_add(common_cache_plan_provider::live_slot, 3,
                               COMMON_CACHE_PLAN_PHASE_SIMILARITY);
    CHECK(a != nullptr);
    a->sim = 0.4; a->sim_known = true;
    auto * b = rec.find_or_add(common_cache_plan_provider::live_slot, 3,
                               COMMON_CACHE_PLAN_PHASE_LRU);
    CHECK(b == a);
    CHECK(rec.n_inventory == 1);
    CHECK(a->phases_seen == (COMMON_CACHE_PLAN_PHASE_SIMILARITY | COMMON_CACHE_PLAN_PHASE_LRU));
    CHECK(a->sim_known); // the later observation did not erase earlier similarity evidence

    // the same provider/source offered to another target is another executable plan
    auto * other_target = rec.find_or_add(
        common_cache_plan_provider::live_slot, 3,
        COMMON_CACHE_PLAN_PHASE_SIMILARITY, 9,
        common_cache_plan_selection::similarity);
    CHECK(other_target != nullptr && other_target != a);
    CHECK(other_target->target_slot_id == 9);
    CHECK(other_target->origin_tier == common_cache_plan_selection::similarity);

    // same source id under a DIFFERENT provider is a different physical candidate
    auto * h = rec.find_or_add(common_cache_plan_provider::host_cache_entry, 3,
                               COMMON_CACHE_PLAN_PHASE_HOST_SCAN);
    CHECK(h != nullptr && h != a);
    CHECK(rec.n_inventory == 3);

    // first observation flips unobserved -> complete
    CHECK(rec.inventory_states[size_t(common_cache_plan_provider::live_slot)] ==
          common_cache_plan_inventory_state::complete);

    // selection round-trips through the ordinal mapping
    rec.select(common_cache_plan_provider::live_slot, a);
    CHECK(rec.selected_row(common_cache_plan_provider::live_slot) == a);
    rec.select(common_cache_plan_provider::live_slot, nullptr);
    CHECK(rec.selected_row(common_cache_plan_provider::live_slot) == nullptr);

    // rows carry typed-unknown measured fields until a shipped loop fills them
    CHECK(h->lcp_tokens.state == llama_cache_acct_known::unknown);
    CHECK(h->t_last_used_us.state == llama_cache_acct_known::unknown);
    CHECK(!h->delivered);
}

// capacity exhaustion: overflow latches, append stops, shipped-side calls keep succeeding
// as no-ops (nullptr), and the latched state never downgrades
static void test_inventory_overflow() {
    common_cache_plan_record rec;
    for (size_t i = 0; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; i++) {
        CHECK(rec.find_or_add(common_cache_plan_provider::host_cache_entry, (int32_t) i,
                              COMMON_CACHE_PLAN_PHASE_HOST_SCAN) != nullptr);
    }
    CHECK(rec.n_inventory == COMMON_CACHE_PLAN_MAX_CANDIDATES);
    CHECK(rec.find_or_add(common_cache_plan_provider::live_slot, 0,
                          COMMON_CACHE_PLAN_PHASE_LRU) == nullptr);
    CHECK(rec.inventory_states[size_t(common_cache_plan_provider::live_slot)] ==
          common_cache_plan_inventory_state::overflowed);
    // overflow never downgrades
    rec.note_inventory_complete(common_cache_plan_provider::live_slot);
    rec.note_inventory_truncated(common_cache_plan_provider::live_slot);
    CHECK(rec.inventory_states[size_t(common_cache_plan_provider::live_slot)] ==
          common_cache_plan_inventory_state::overflowed);
    // an EXISTING row is still found after capacity is exhausted (merge, not append)
    CHECK(rec.find_or_add(common_cache_plan_provider::host_cache_entry, 0,
                          COMMON_CACHE_PLAN_PHASE_HOST_SCAN) != nullptr);
    // a dropped derived plan latches the record-level flag WITHOUT touching any
    // provider's inventory state
    const auto host_state_before =
        rec.inventory_states[size_t(common_cache_plan_provider::host_cache_entry)];
    CHECK(rec.add_chain(common_cache_plan_provider::host_cache_entry, 0, 1) == nullptr);
    CHECK(rec.derived_plans_incomplete);
    CHECK(rec.inventory_states[size_t(common_cache_plan_provider::host_cache_entry)] ==
          host_state_before);
}

// Truncation marks a shipped short-circuit; complete never overwrites it.
static void test_inventory_truncation() {
    common_cache_plan_record rec;
    rec.find_or_add(common_cache_plan_provider::live_context_checkpoint, 0,
                    COMMON_CACHE_PLAN_PHASE_CKPT_SCAN);
    rec.note_inventory_truncated(common_cache_plan_provider::live_context_checkpoint);
    rec.note_inventory_complete(common_cache_plan_provider::live_context_checkpoint);
    CHECK(rec.inventory_states[size_t(common_cache_plan_provider::live_context_checkpoint)] ==
          common_cache_plan_inventory_state::truncated_by_shipped_short_circuit);
}

// revocation clears every non-cold delivery; planner-fault clearing wipes planner outputs
// only, leaving rejection evidence intact
static void test_revoke_and_planner_clear() {
    common_cache_plan_record rec;
    auto * s = rec.find_or_add(common_cache_plan_provider::live_slot, 0, COMMON_CACHE_PLAN_PHASE_LRU);
    auto * k = rec.find_or_add(common_cache_plan_provider::live_context_checkpoint, 0,
                               COMMON_CACHE_PLAN_PHASE_CKPT_SCAN);
    auto * cold = rec.find_or_add(common_cache_plan_provider::cold_replay, -1, uint8_t(0));
    s->delivered = k->delivered = cold->delivered = true;
    rec.revoke_deliveries();
    CHECK(!s->delivered && !k->delivered);
    CHECK(cold->delivered); // cold is a final-state fact, never revoked

    // simulated planner outputs
    k->predicted_total_us = llama_cache_acct_value::measured(42);
    k->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].estimated_us =
        llama_cache_acct_value::measured(41);
    rec.shadow_choice  = 1;
    rec.shadow_tie_set[0] = 1; rec.n_shadow_ties = 1;
    rec.yield.status = common_cache_plan_yield_status::fits;
    rec.yield.plan_state = common_cache_plan_yield_plan_state::not_required;
    rec.yield.accounting_serial = 17;
    k->note_reject(COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);

    rec.clear_planner_outputs();
    CHECK(rec.shadow_choice == -1 && rec.n_shadow_ties == 0);
    CHECK(k->predicted_total_us.state == llama_cache_acct_known::unknown);
    CHECK(k->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].estimated_us.state ==
          llama_cache_acct_known::unknown);
    // rejection evidence survives the planner fault
    CHECK(k->reason == COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);
    CHECK(rec.n_inventory == 3);
    CHECK(rec.yield.status == common_cache_plan_yield_status::fits);
    CHECK(rec.yield.plan_state ==
          common_cache_plan_yield_plan_state::not_required);
    CHECK(rec.yield.accounting_serial == 17);
}

// record-level typed-unknown discipline + per-candidate cost-term defaults: five DISTINCT
// kinds with canonical raw units — a default array would collapse to five "restore" slots
static void test_record_defaults() {
    common_cache_plan_record rec;
    CHECK(rec.schema_version == 7);
    CHECK(common_cache_plan_accounting_schema(7) == 2);
    CHECK(rec.outcome == common_cache_plan_outcome::unknown);
    CHECK(rec.n_reused_tokens.state == llama_cache_acct_known::unknown);
    CHECK(rec.ttft_us.state == llama_cache_acct_known::unknown);
    CHECK(rec.calibration_profile.empty()); // typed-unknown on the wire
    CHECK(rec.shadow_choice == -1 && rec.n_shadow_ties == 0);
    CHECK(rec.shipped_plan_candidate == -1);
    CHECK(!rec.derived_plans_incomplete);
    CHECK(rec.planner_status == common_cache_plan_planner_status::not_attempted);
    CHECK(rec.authority.policy_version == COMMON_CACHE_PLAN_AUTHORITY_POLICY_VERSION);
    CHECK(rec.authority.configured_level == common_cache_plan_authority_level::off);
    CHECK(rec.authority.state == common_cache_plan_authority_state::shadow);
    CHECK(rec.authority.legacy_plan_candidate == -1);
    CHECK(rec.yield.status == common_cache_plan_yield_status::unavailable);
    CHECK(rec.yield.plan_state ==
          common_cache_plan_yield_plan_state::unavailable);
    CHECK(rec.yield.actual_state ==
          common_cache_plan_yield_actual_state::not_observed);
    CHECK(rec.yield.actual_domains.empty());
    CHECK(rec.destruction.policy_version ==
          COMMON_CACHE_PLAN_DESTRUCTION_POLICY_VERSION);
    CHECK(rec.destruction.state ==
          common_cache_plan_destruction_state::not_required);
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::none);
    CHECK(rec.destruction_quotes.empty());

    common_cache_plan_candidate c;
    bool seen[size_t(llama_cache_acct_cost_kind::_count)] = {};
    for (const auto & term : c.cost_terms) {
        CHECK(!seen[size_t(term.kind)]);
        seen[size_t(term.kind)] = true;
        CHECK(term.raw_unit == llama_cache_acct_cost_kind_unit(term.kind));
        CHECK(term.raw.state == llama_cache_acct_known::unknown);
        CHECK(term.estimated_us.state == llama_cache_acct_known::unknown);
    }
    CHECK(c.cost_terms[size_t(llama_cache_acct_cost_kind::replay)].raw_unit ==
          llama_cache_acct_unit::tokens);
    CHECK(c.predicted_total_us.state == llama_cache_acct_known::unknown);
    CHECK(c.component_ids[0] == -1 && c.component_ids[1] == -1);
    // identity evidence starts typed-unknown across the board — never fabricated digests
    CHECK(rec.identity.model_digest.state == llama_cache_acct_known::unknown);
    CHECK(rec.identity.prefix_token_digest.state == llama_cache_acct_known::unknown);
}

// exhaustive name tables: every member names itself, no member is "invalid"
static void test_name_tables() {
    for (size_t i = 0; i < COMMON_CACHE_PLAN_REASON_MEMBER_COUNT; i++) {
        CHECK(strcmp(common_cache_plan_reason_name(common_cache_plan_reason_all[i]), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_disposition::_count); i++) {
        CHECK(strcmp(common_cache_plan_disposition_name(common_cache_plan_disposition(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_provider::_count); i++) {
        CHECK(strcmp(common_cache_plan_provider_name(common_cache_plan_provider(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_payload_kind::_count); i++) {
        CHECK(strcmp(common_cache_plan_payload_kind_name(
                         common_cache_plan_payload_kind(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_outcome::_count); i++) {
        CHECK(strcmp(common_cache_plan_outcome_name(common_cache_plan_outcome(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_selection::_count); i++) {
        CHECK(strcmp(common_cache_plan_selection_name(common_cache_plan_selection(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_authority_level::_count); i++) {
        CHECK(strcmp(common_cache_plan_authority_level_name(
                         common_cache_plan_authority_level(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_authority_state::_count); i++) {
        CHECK(strcmp(common_cache_plan_authority_state_name(
                         common_cache_plan_authority_state(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_authority_fallback::_count); i++) {
        CHECK(strcmp(common_cache_plan_authority_fallback_name(
                         common_cache_plan_authority_fallback(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_destruction_state::_count); i++) {
        CHECK(strcmp(common_cache_plan_destruction_state_name(
                         common_cache_plan_destruction_state(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_destruction_reason::_count); i++) {
        CHECK(strcmp(common_cache_plan_destruction_reason_name(
                         common_cache_plan_destruction_reason(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_destruction_effect::_count); i++) {
        CHECK(strcmp(common_cache_plan_destruction_effect_name(
                         common_cache_plan_destruction_effect(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_destruction_class::_count); i++) {
        CHECK(strcmp(common_cache_plan_destruction_class_name(
                         common_cache_plan_destruction_class(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_destruction_physical_reason::_count); i++) {
        CHECK(strcmp(common_cache_plan_destruction_physical_reason_name(
                         common_cache_plan_destruction_physical_reason(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_destruction_lease_verdict::_count); i++) {
        CHECK(strcmp(common_cache_plan_destruction_lease_verdict_name(
                         common_cache_plan_destruction_lease_verdict(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_displaced_fate::_count); i++) {
        CHECK(strcmp(common_cache_plan_displaced_fate_name(
                         common_cache_plan_displaced_fate(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_recovery_citation::_count); i++) {
        CHECK(strcmp(common_cache_plan_recovery_citation_name(
                         common_cache_plan_recovery_citation(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_destruction_comparison::_count); i++) {
        CHECK(strcmp(common_cache_plan_destruction_comparison_name(
                         common_cache_plan_destruction_comparison(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_inventory_state::_count); i++) {
        CHECK(strcmp(common_cache_plan_inventory_state_name(common_cache_plan_inventory_state(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_planner_status::_count); i++) {
        CHECK(strcmp(common_cache_plan_planner_status_name(common_cache_plan_planner_status(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_yield_status::_count); i++) {
        CHECK(strcmp(common_cache_plan_yield_status_name(
                         common_cache_plan_yield_status(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_yield_plan_state::_count); i++) {
        CHECK(strcmp(common_cache_plan_yield_plan_state_name(
                         common_cache_plan_yield_plan_state(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_yield_actual_state::_count); i++) {
        CHECK(strcmp(common_cache_plan_yield_actual_state_name(
                         common_cache_plan_yield_actual_state(i)), "invalid") != 0);
    }
    // and the closed inventory really is closed: exactly today's four providers
    CHECK(uint8_t(common_cache_plan_provider::_count) == 4);
    // schema-v5 record retains the v2 reason census + sentinel (compile-time pinned)
    CHECK(COMMON_CACHE_PLAN_REASON_MEMBER_COUNT == 30);
    CHECK(uint16_t(COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL) == 601);
}

// Couple the golden schema to the actual C++ serializer: a
// representative composed-delivery record through common_cache_plan_record_json, with
// structural assertions on every load-bearing v2 key
static void test_json_serialization() {
    common_cache_plan_record rec;
    rec.id_task = 42; rec.id_slot = 1;
    rec.calibration_profile = "test-model/test-gpu/b512";
    rec.selection = common_cache_plan_selection::similarity;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(1000);

    auto * host = rec.find_or_add(common_cache_plan_provider::host_cache_entry, 0,
                                  COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 1,
                                  common_cache_plan_selection::similarity);
    host->accept();
    host->payload_kind = common_cache_plan_payload_kind::vbr_artifact;
    host->delivered     = true;
    host->lcp_tokens    = llama_cache_acct_value::measured(500);
    host->payload_bytes = llama_cache_acct_value::measured(1000);
    auto * ckpt = rec.find_or_add(common_cache_plan_provider::live_context_checkpoint, 0,
                                  COMMON_CACHE_PLAN_PHASE_CKPT_SCAN, 1,
                                  common_cache_plan_selection::similarity);
    ckpt->accept();
    ckpt->delivered      = true;
    ckpt->component_only = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);
    rec.select(common_cache_plan_provider::live_context_checkpoint, ckpt);
    auto * chain = rec.add_chain(common_cache_plan_provider::host_cache_entry, 0, 1);
    chain->disposition = common_cache_plan_disposition::accepted;
    chain->delivered   = true;
    rec.shipped_plan_candidate = 2;
    rec.chosen  = common_cache_plan_provider::live_context_checkpoint;
    rec.outcome = common_cache_plan_outcome::restored;
    rec.planner_status = common_cache_plan_planner_status::profile_unfitted;
    // a filled term must appear on the wire; unfilled kinds must be absent
    host->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].raw =
        llama_cache_acct_value::measured(500);
    host->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].estimated_us =
        llama_cache_acct_value::measured(50000);
    host->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].estimator_version = 1;
    host->predicted_total_us = llama_cache_acct_value::measured(50000);

    // Record schema-v7 / accounting-v2 bridge: interned topology table plus per-domain
    // producer completeness, with explicit not_applicable rather than fabricated zeroes.
    llama_cache_acct_ledger ledger;
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::not_applicable);
    llama_cache_acct_shard_topology topology;
    CHECK(llama_cache_acct_build_shard_topology(
        std::vector<std::string>{ "record-test-device" }, 1, 0, nullptr, topology));
    llama_cache_acct_resource_domain device_domain;
    CHECK(ledger.make_device_domain(topology, { 0 }, device_domain));
    const llama_cache_acct_completeness_requirement requirements[] = {
        { domain, llama_cache_acct_producer::observer_init },
        { device_domain, llama_cache_acct_producer::live_memory },
    };
    CHECK(ledger.configure_required_producers(requirements, 2));
    ledger.gauge_set(llama_cache_acct_category::rolling_window_tape, domain,
                     llama_cache_acct_measure::logical_payload, 0);
    ledger.gauge_set(llama_cache_acct_category::live_attention_state, device_domain,
                     llama_cache_acct_measure::resident_allocated, 1234);
    ledger.gauge_set(llama_cache_acct_category::live_recurrent_state, device_domain,
                     llama_cache_acct_measure::resident_allocated, 567);
    ledger.gauge_set(llama_cache_acct_category::recurrent_rollback_planes, device_domain,
                     llama_cache_acct_measure::resident_allocated, 89);
    CHECK(ledger.certify_complete(domain, llama_cache_acct_producer::observer_init));
    CHECK(ledger.certify_complete(device_domain, llama_cache_acct_producer::live_memory));
    rec.acct = ledger.snapshot();
    rec.yield.status = common_cache_plan_yield_status::fits;
    rec.yield.plan_state = common_cache_plan_yield_plan_state::planned;
    rec.yield.actual_state =
        common_cache_plan_yield_actual_state::not_observed;
    rec.yield.yield_policy_version = 1;
    rec.yield.accounting_serial = rec.acct.serial;
    rec.yield.selected_attention.push_back({ 11 });
    rec.yield.selected_recurrent.push_back({ 12 });
    rec.yield.unsupported.push_back({ 13 });
    rec.yield.projected_domains.push_back({
        device_domain,
        llama_cache_acct_value::measured(1890),
        llama_cache_acct_value::measured(1922),
        llama_cache_acct_value::measured(128),
        llama_cache_acct_value::measured(0),
        llama_cache_acct_value::measured(1794),
    });
    // Synthetic serializer-coupled schema vector: production does not emit a
    // refused receipt with a resolved concrete citation. Combining them here
    // exercises both independent wire fields in the one schema-7 golden.
    rec.destruction.state = common_cache_plan_destruction_state::refused;
    rec.destruction.reason = common_cache_plan_destruction_reason::effect_drift;
    rec.destruction.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::same_target_cold_replacement) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::checkpoint_member_drop);
    rec.destruction.plan_candidate = 2;
    rec.destruction.admission_sequence = 12;
    rec.destruction.quote_duration_us = 37;
    rec.destruction.quote_accounting_serial = rec.acct.serial;
    rec.destruction.manifest_digest =
        common_cache_plan_destruction_manifest_digest::from_sha256(
            std::array<uint8_t, 32>{ 1 });
    rec.destruction.union_effect_digest =
        common_cache_plan_destruction_effect_digest::from_sha256(
            std::array<uint8_t, 32>{ 2 });
    rec.destruction.recovery_citation =
        common_cache_plan_recovery_citation::resolved;
    rec.destruction.recovery_source_artifact_id = { 21 };
    rec.destruction.recovery_source_manifest_digest =
        common_cache_plan_destruction_recovery_digest::from_sha256(
            std::array<uint8_t, 32>{ 3 });
    rec.destruction.payload_kind =
        common_cache_plan_payload_kind::vbr_artifact;
    common_cache_plan_finalize_shadow_authority(rec);

    const auto j = common_cache_plan_record_json(rec);
    // Golden regeneration door: this deliberately exercises the production
    // serializer instead of maintaining a hand-authored schema-7 facsimile.
    if (std::getenv("CACHE_PLAN_PRINT_SCHEMA7_GOLDEN")) {
        std::puts(j.dump().c_str());
    }
    CHECK(j["schema_version"] == 7);
    CHECK(j["candidates"].size() == 3);
    CHECK(j["candidates"][0]["id"] == 0);
    CHECK(j["candidates"][0]["provider"] == "host_cache_entry");
    CHECK(j["candidates"][0]["payload_kind"] == "vbr_artifact");
    CHECK(j["candidates"][1]["payload_kind"].is_null());
    CHECK(j["candidates"][2]["payload_kind"] == "vbr_artifact");
    CHECK(j["candidates"][0]["target_slot_id"] == 1);
    CHECK(j["candidates"][0]["origin_tier"] == "similarity");
    CHECK(j["candidates"][0]["cost_terms"].contains("replay"));
    CHECK(!j["candidates"][0]["cost_terms"].contains("transfer")); // absence = unavailable
    CHECK(j["candidates"][1]["component_only"] == true);
    CHECK(j["candidates"][2]["is_chain"] == true);
    CHECK(j["candidates"][2]["components"] == nlohmann::ordered_json::array({0, 1}));
    CHECK(j["chosen"] == "live_context_checkpoint");
    CHECK(j["chosen_candidate"] == 1);
    CHECK(j["shipped_plan_candidate"] == 2); // the chain, not the terminal provider row
    CHECK(j["destruction"]["payload_kind"] == "vbr_artifact");
    CHECK(j["planner_status"] == "profile_unfitted");
    CHECK(j["shadow"] == "unavailable"); // string sentinel per the acct-value convention
    CHECK(j["authority"]["policy_version"] == 1);
    CHECK(j["authority"]["configured_level"] == "off");
    CHECK(j["authority"]["legacy_tier"] == "similarity");
    CHECK(j["authority"]["decision_tier"] == "none");
    CHECK(j["authority"]["state"] == "shadow");
    CHECK(j["authority"]["legacy_plan_candidate"] == 2);
    CHECK(j["authority"]["planner_plan_candidate"] == "unavailable");
    CHECK(j["authority"]["executed_plan_candidate"] == 2);
    CHECK(j["authority"]["fallback_reason"] == "none");
    CHECK(!j["authority"]["disagreed"]);
    CHECK(j["destruction"]["state"] == "refused");
    CHECK(j["destruction"]["reason"] == "effect_drift");
    CHECK(j["destruction"]["effects"].size() == 3);
    CHECK(j["destruction"]["effects"][0]["effect"] ==
          "same_target_cold_replacement");
    CHECK(j["destruction"]["effects"][1]["effect"] ==
          "different_host_source_consumption");
    CHECK(j["destruction"]["effects"][1]["action_class"] ==
          "host_artifact_drop");
    CHECK(j["destruction"]["effects"][2]["effect"] ==
          "checkpoint_member_drop");
    CHECK(j["destruction"]["effects"][2]["action_class"] ==
          "checkpoint_drop");
    CHECK(j["destruction"]["effects"][2]["physical_reason"] ==
          "checkpoint_replace");
    CHECK(j["destruction"]["plan_candidate"] == 2);
    CHECK(j["destruction"]["admission_sequence"] == 12);
    CHECK(j["destruction"]["quote_duration_us"] == 37);
    CHECK(j["destruction"]["manifest_digest"] != "unavailable");
    CHECK(j["destruction"]["union_effect_digest"] != "unavailable");
    CHECK(j["destruction"]["recovery_citation"] == "resolved");
    CHECK(j["destruction"]["recovery_source"]["artifact_id"] == 21);
    CHECK(j["destruction"]["recovery_source"]["manifest_digest"] ==
          "0300000000000000000000000000000000000000000000000000000000000000");
    CHECK(j["destruction"]["selected"]["attention"].empty());
    CHECK(!j["destruction"].contains("projected_domains"));
    CHECK(j["inventory_states"]["host_cache_entry"] == "complete");
    CHECK(j["delivered_chain"] == nlohmann::ordered_json::array(
        {"host_cache_entry", "live_context_checkpoint"}));
    CHECK(j["accounting"]["schema_version"] == 2);
    CHECK(j["accounting"]["completeness_manifest"] == "known");
    CHECK(j["accounting"]["cells"][0]["domain"]["kind"] == "not_applicable");
    CHECK(j["accounting"]["cells"][0]["domain"]["device_ordinal"] == "not_applicable");
    CHECK(j["accounting"]["cells"][0]["certification"] == "known");
    CHECK(j["accounting"]["cells"][0]["value"] == 0);
    CHECK(j["accounting"]["cells"][1]["domain"]["device_ordinal"] == 0);
    CHECK(j["accounting"]["cells"][1]["domain"]["topology_id"] == 1);
    CHECK(j["accounting"]["topologies"].size() == 1);
    CHECK(j["accounting"]["topologies"][0]["version"] == 1);
    CHECK(j["accounting"]["topologies"][0]["digest"] ==
          "ee2a4284677b3fc301fd80997cb041cf85a5c82d59e060a7832831dffbe5414d");
    CHECK(j["accounting"]["topologies"][0]["device_identities"] ==
          nlohmann::ordered_json::array(
              {"43b772486999664b169f739107dca459818ffff40d1d3833ac9c8da18a4e5d5a"}));
    CHECK(j["accounting"]["cells"][1]["value"] == 1234);
    CHECK(j["accounting"]["completeness"][0]["producer"] == "observer_init");
    CHECK(j["accounting"]["completeness"][0]["state"] == "known");
    CHECK(j["accounting"]["completeness"][1]["domain"]["kind"] == "device_topology");
    CHECK(j["accounting"]["completeness"][1]["producer"] == "live_memory");
    CHECK(j["accounting"]["completeness"][1]["state"] == "known");
    CHECK(j["yield"]["status"] == "fits");
    CHECK(j["yield"]["plan_state"] == "planned");
    CHECK(j["yield"]["actual_state"] == "not_observed");
    CHECK(j["yield"]["yield_policy_version"] == 1);
    CHECK(j["yield"]["accounting_serial"] == rec.acct.serial);
    CHECK(j["yield"]["selected"]["attention"] ==
          nlohmann::ordered_json::array({11}));
    CHECK(j["yield"]["selected"]["recurrent"] ==
          nlohmann::ordered_json::array({12}));
    CHECK(j["yield"]["unsupported"] ==
          nlohmann::ordered_json::array({13}));
    CHECK(j["yield"]["projected_domains"].size() == 1);
    CHECK(j["yield"]["projected_domains"][0]["fit_before"] == 1922);
    CHECK(j["yield"]["projected_domains"][0]["projected_release"] == 128);
    CHECK(j["yield"]["projected_domains"][0]["projected_after"] == 1794);
    CHECK(j["yield"]["actual_domains"].empty());
}

static void test_authority_receipt_and_counters() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;
    rec.shipped_plan_candidate = 1;
    rec.shadow_choice = 0;
    rec.shadow_tie_set[0] = 0;
    rec.n_shadow_ties = 1;
    rec.planner_status = common_cache_plan_planner_status::ok;

    common_cache_plan_finalize_shadow_authority(rec);
    CHECK(rec.authority.state == common_cache_plan_authority_state::shadow);
    CHECK(rec.authority.legacy_tier == common_cache_plan_selection::similarity);
    CHECK(rec.authority.decision_tier == common_cache_plan_selection::none);
    CHECK(rec.authority.legacy_plan_candidate == 1);
    CHECK(rec.authority.planner_plan_candidate == 0);
    CHECK(rec.authority.executed_plan_candidate == 1);
    CHECK(rec.authority.disagreed);
    // Established schema-v4 semantics remain planner-owned.
    CHECK(rec.shadow_choice == 0);
    CHECK(rec.shadow_tie_set[0] == 0);

    common_cache_plan_authority_counters counters;
    counters.observe(rec.authority);
    const size_t tier = size_t(common_cache_plan_selection::similarity);
    CHECK(counters.has_receipt);
    CHECK(counters.observed[tier] == 1);
    // A shadow planner result is legible but does not claim authority eligibility.
    CHECK(counters.authority_eligible[tier] == 0);
    CHECK(counters.authority_executed[tier] == 0);
    CHECK(counters.agree[tier] == 0);
    CHECK(counters.disagree[tier] == 1);
    CHECK(counters.fallback_legacy[tier] == 0);
    CHECK(counters.last_receipt.executed_plan_candidate == 1);

    auto authoritative = rec.authority;
    authoritative.state = common_cache_plan_authority_state::authoritative;
    authoritative.decision_tier = common_cache_plan_selection::similarity;
    authoritative.executed_plan_candidate = authoritative.planner_plan_candidate;
    authoritative.fallback_reason = common_cache_plan_authority_fallback::none;
    counters.observe(authoritative);
    CHECK(counters.authority_eligible[tier] == 1);
    CHECK(counters.authority_executed[tier] == 1);

    auto fallback = rec.authority;
    fallback.state = common_cache_plan_authority_state::fallback_legacy;
    fallback.fallback_reason =
        common_cache_plan_authority_fallback::destruction_authority_required;
    counters.observe(fallback);
    CHECK(counters.fallback_legacy[tier] == 1);
    CHECK(counters.fallback_reason[size_t(
              common_cache_plan_authority_fallback::destruction_authority_required)] == 1);
}

static void test_yield_not_required_serialization() {
    common_cache_plan_record rec;
    rec.outcome = common_cache_plan_outcome::cold;
    rec.yield.status = common_cache_plan_yield_status::fits;
    rec.yield.plan_state =
        common_cache_plan_yield_plan_state::not_required;
    rec.yield.actual_state =
        common_cache_plan_yield_actual_state::not_observed;
    rec.yield.yield_policy_version = 1;
    rec.yield.accounting_serial = 23;

    const auto j = common_cache_plan_record_json(rec);
    CHECK(j["yield"]["status"] == "fits");
    CHECK(j["yield"]["plan_state"] == "not_required");
    CHECK(j["yield"]["actual_state"] == "not_observed");
    CHECK(j["yield"]["selected"]["attention"].empty());
    CHECK(j["yield"]["selected"]["recurrent"].empty());
    CHECK(j["yield"]["projected_domains"].empty());
    CHECK(j["yield"]["actual_domains"].empty());
}

static void test_actual_yield_uses_post_commit_observation() {
    common_cache_plan_yield_record yield;
    common_cache_plan_yield_domain projected;
    projected.domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    projected.current_resident_bytes = llama_cache_acct_value::measured(100);
    projected.projected_release_bytes = llama_cache_acct_value::measured(40);
    auto after = projected;
    after.current_resident_bytes = llama_cache_acct_value::measured(65);

    common_cache_plan_fill_actual_yield(yield, { projected }, { after });
    CHECK(yield.actual_state ==
          common_cache_plan_yield_actual_state::measured);
    CHECK(yield.actual_domains.size() == 1);
    CHECK(yield.actual_domains[0].before_bytes.value == 100);
    CHECK(yield.actual_domains[0].after_bytes.value == 65);
    // The actual is the observed delta (35), not the quote-time projection
    // (40) relabeled as measured evidence.
    CHECK(yield.actual_domains[0].released_bytes.value == 35);

    after.current_resident_bytes = llama_cache_acct_value::measured(101);
    common_cache_plan_fill_actual_yield(yield, { projected }, { after });
    CHECK(yield.actual_state ==
          common_cache_plan_yield_actual_state::unavailable);
    CHECK(yield.actual_domains.empty());
}

// Finalize-shaped chain composition: the tested implementation the server
// calls — simple delivery, composed delivery with sibling cost-loser chains, and the
// exact-capacity shape where the delivered pair's chain cannot be recorded
static void test_compose_chains() {
    { // simple (non-composed) delivery: shipped plan = the chosen provider's selected row
        common_cache_plan_record rec;
        auto * live = rec.find_or_add(common_cache_plan_provider::live_slot, 0,
                                      COMMON_CACHE_PLAN_PHASE_SIMILARITY);
        live->accept(); live->delivered = true;
        rec.select(common_cache_plan_provider::live_slot, live);
        rec.chosen = common_cache_plan_provider::live_slot;
        common_cache_plan_compose_chains(rec);
        CHECK(rec.shipped_plan_candidate == 0);
        CHECK(rec.n_inventory == 1); // no chains fabricated
    }
    { // composed delivery: every sibling component-only, valid ones chained, shipped = chain
        common_cache_plan_record rec;
        auto * host = rec.find_or_add(common_cache_plan_provider::host_cache_entry, 0,
                                      COMMON_CACHE_PLAN_PHASE_HOST_SCAN);
        host->accept(); host->delivered = true;
        auto * sel = rec.find_or_add(common_cache_plan_provider::live_context_checkpoint, 0,
                                     COMMON_CACHE_PLAN_PHASE_CKPT_SCAN);
        sel->accept(); sel->delivered = true; sel->component_only = true;
        sel->dependent_host_source_id = host->source_id;
        auto * sib = rec.find_or_add(common_cache_plan_provider::live_context_checkpoint, 1,
                                     COMMON_CACHE_PLAN_PHASE_CKPT_SCAN);
        sib->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL); // valid loser
        sib->component_only = true;
        sib->dependent_host_source_id = host->source_id;
        auto * bad = rec.find_or_add(common_cache_plan_provider::live_context_checkpoint, 2,
                                     COMMON_CACHE_PLAN_PHASE_CKPT_SCAN);
        bad->note_reject(COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED); // invalid
        bad->component_only = true;
        bad->dependent_host_source_id = host->source_id;
        rec.select(common_cache_plan_provider::host_cache_entry, host);
        rec.select(common_cache_plan_provider::live_context_checkpoint, sel);
        rec.chosen = common_cache_plan_provider::live_context_checkpoint;
        common_cache_plan_compose_chains(rec);
        CHECK(sel->component_only && sib->component_only && bad->component_only);
        CHECK(rec.n_inventory == 6); // 4 rows + shipped chain + sibling chain (invalid: none)
        CHECK(rec.shipped_plan_candidate >= 4);
        const auto & shipped = rec.inventory[size_t(rec.shipped_plan_candidate)];
        CHECK(shipped.is_chain() && shipped.delivered);
        CHECK(shipped.disposition == common_cache_plan_disposition::accepted);
        // the sibling's chain is a cost loser, never delivered
        bool found_sib_chain = false;
        for (uint32_t i = 4; i < rec.n_inventory; i++) {
            const auto & c = rec.inventory[i];
            if ((int32_t) i != rec.shipped_plan_candidate) {
                found_sib_chain = true;
                CHECK(c.is_chain() && !c.delivered);
                CHECK(c.disposition == common_cache_plan_disposition::valid_not_chosen_cost);
            }
        }
        CHECK(found_sib_chain);
        CHECK(!rec.derived_plans_incomplete);
    }
    { // exact capacity: the delivered pair's chain cannot be recorded — shipped plan is
      // -1 (the bare dependent checkpoint never stands in) and the planner will refuse
        common_cache_plan_record rec;
        auto * host = rec.find_or_add(common_cache_plan_provider::host_cache_entry, 0,
                                      COMMON_CACHE_PLAN_PHASE_HOST_SCAN);
        host->accept(); host->delivered = true;
        auto * sel = rec.find_or_add(common_cache_plan_provider::live_context_checkpoint, 0,
                                     COMMON_CACHE_PLAN_PHASE_CKPT_SCAN);
        sel->accept(); sel->delivered = true; sel->component_only = true;
        sel->dependent_host_source_id = host->source_id;
        for (int32_t i = 1; rec.n_inventory < COMMON_CACHE_PLAN_MAX_CANDIDATES; i++) {
            rec.find_or_add(common_cache_plan_provider::live_slot, i,
                            COMMON_CACHE_PLAN_PHASE_LRU)
                ->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);
        }
        rec.select(common_cache_plan_provider::host_cache_entry, host);
        rec.select(common_cache_plan_provider::live_context_checkpoint, sel);
        rec.chosen = common_cache_plan_provider::live_context_checkpoint;
        common_cache_plan_compose_chains(rec);
        CHECK(rec.derived_plans_incomplete);
        CHECK(rec.shipped_plan_candidate == -1);
        CHECK(sel->component_only); // the dependency fact is still recorded
    }
}

static void test_destruction_observer() {
    static_assert(
        size_t(server_cache_destruction_class::_count) == 6,
        "destruction inventory is closed at six logical classes");

    server_cache_destruction_observer observer;
    server_cache_destruction_request request;
    request.cls = server_cache_destruction_class::host_artifact_drop;
    request.reason = server_cache_destruction_reason::host_capacity;
    request.add_target(server_cache_destruction_target_kind::host_artifact, -1);

    server_cache_destruction_yield value;
    value.category = llama_cache_acct_category::full_snapshot_payload;
    value.measure = llama_cache_acct_measure::resident_allocated;
    value.domain_known = true;
    value.domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    value.value = llama_cache_acct_value::measured(4096);
    request.add_yield(value);

    const auto first_admission =
        server_cache_retention_admit(&observer, request);
    CHECK(first_admission.issued);
    CHECK(first_admission.observer_recorded);
    CHECK(first_admission.sequence == 1);
    CHECK(first_admission.covers(
        server_cache_destruction_class::host_artifact_drop,
        server_cache_destruction_reason::host_capacity));
    CHECK(observer.n_events == 1);
    CHECK(observer.totals[size_t(
        server_cache_destruction_class::host_artifact_drop)] == 1);
    CHECK(observer.events[0].verdict ==
          server_cache_destruction_verdict::admit_unleased);
    CHECK(observer.events[0].execution ==
          server_cache_destruction_execution::pass_through);
    CHECK(observer.events[0].request.n_targets == 1);
    CHECK(observer.events[0].request.n_yields == 1);
    CHECK(observer.events[0].request.yields[0].value.value == 4096);

    // Detail is bounded, while totals and sequence remain monotone.
    for (size_t i = 1; i < SERVER_CACHE_DESTRUCTION_EVENT_RING + 3; ++i) {
        CHECK(server_cache_retention_admit(&observer, request).issued);
    }
    CHECK(observer.n_events == SERVER_CACHE_DESTRUCTION_EVENT_RING + 3);
    CHECK(observer.totals[size_t(
        server_cache_destruction_class::host_artifact_drop)] ==
          SERVER_CACHE_DESTRUCTION_EVENT_RING + 3);
    const auto & newest = observer.events[size_t(
        (observer.n_events - 1) % observer.events.size())];
    CHECK(newest.sequence == observer.n_events);

    // Manifest overflow is observable but cannot block destruction execution.
    server_cache_destruction_request oversized;
    oversized.cls = server_cache_destruction_class::slot_drop;
    for (size_t i = 0; i <= SERVER_CACHE_DESTRUCTION_MAX_TARGETS; ++i) {
        oversized.add_target(
            server_cache_destruction_target_kind::live_target, int32_t(i));
    }
    CHECK(oversized.overflowed);
    const auto overflows_before = observer.overflows;
    CHECK(server_cache_retention_admit(&observer, oversized).issued);
    CHECK(observer.overflows == overflows_before + 1);

    // Disabled observer is the zero-work/pass-through branch.
    const auto unobserved = server_cache_retention_admit(nullptr, request);
    CHECK(unobserved.issued);
    CHECK(!unobserved.observer_recorded);
    CHECK(unobserved.sequence == 0);
}

int main() {
    test_precedence();
    test_valid_loser();
    test_inventory_merge();
    test_inventory_overflow();
    test_inventory_truncation();
    test_revoke_and_planner_clear();
    test_record_defaults();
    test_name_tables();
    test_authority_receipt_and_counters();
    test_json_serialization();
    test_yield_not_required_serialization();
    test_actual_yield_uses_post_commit_observation();
    test_compose_chains();
    test_destruction_observer();

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    printf("all cache-plan-record tests passed\n");
    return EXIT_SUCCESS;
}
