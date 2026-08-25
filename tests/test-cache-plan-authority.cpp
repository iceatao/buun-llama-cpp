#include "server-cache-plan-authority.h"
#include "common-cache-plan-estimate.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(COND) do { if (!(COND)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #COND); \
    std::abort(); \
} } while (0)

static common_cache_plan_candidate * add_viable(
        common_cache_plan_record & rec,
        common_cache_plan_provider provider,
        int32_t source,
        int32_t target,
        common_cache_plan_selection origin =
            common_cache_plan_selection::by_id) {
    auto * row = rec.find_or_add(provider, source, 0, target, origin);
    CHECK(row != nullptr);
    row->accept();
    return row;
}

struct host_checkpoint_chain_fixture {
    common_cache_plan_candidate * host = nullptr;
    common_cache_plan_candidate * checkpoint = nullptr;
    common_cache_plan_candidate * chain = nullptr;
};

static host_checkpoint_chain_fixture add_host_checkpoint_chain(
        common_cache_plan_record & rec,
        int32_t target,
        int32_t host_source,
        int32_t checkpoint_ordinal,
        common_cache_plan_selection origin =
            common_cache_plan_selection::by_id) {
    host_checkpoint_chain_fixture out;
    out.host = add_viable(rec, common_cache_plan_provider::host_cache_entry,
        host_source, target, origin);
    out.checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(
            host_source, checkpoint_ordinal), target, origin);
    out.checkpoint->component_only = true;
    out.checkpoint->dependent_host_source_id = host_source;
    out.chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(out.host - rec.inventory.data()),
        int32_t(out.checkpoint - rec.inventory.data()));
    CHECK(out.chain != nullptr);
    out.chain->accept();
    return out;
}

static void test_candidate_classifiers() {
    CHECK(common_cache_plan_strict_similarity(0.75, 0.50));
    CHECK(!common_cache_plan_strict_similarity(0.50, 0.50));
    CHECK(!common_cache_plan_strict_similarity(0.75, 0.0));
    CHECK(common_cache_plan_origin_in_domain(
        common_cache_plan_selection::similarity,
        common_cache_plan_selection::similarity));
    CHECK(!common_cache_plan_origin_in_domain(
        common_cache_plan_selection::route_home,
        common_cache_plan_selection::similarity));

    CHECK(common_cache_plan_authority_level_parse("off") ==
          common_cache_plan_authority_level::off);
    CHECK(common_cache_plan_authority_level_parse("route_home") ==
          common_cache_plan_authority_level::route_home);
    bool rejected = false;
    try {
        (void) common_cache_plan_authority_level_parse("future");
    } catch (...) {
        rejected = true;
    }
    CHECK(rejected);

    const auto busy = server_cache_plan_evaluate_live(true, true, 8, 16);
    CHECK(busy.reason == COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);
    const auto live = server_cache_plan_evaluate_live(false, true, 8, 16);
    CHECK(live.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(live.lcp_tokens == 8);
    common_cache_plan_candidate live_row;
    server_cache_plan_apply_live(&live_row, live);
    CHECK(!live_row.f_keep_known);
    CHECK(live_row.f_keep == -1.0f);

    const auto host = server_cache_plan_evaluate_host(
        true, true, 20, 40, 80, 1024);
    CHECK(host.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(host.f_keep == 0.25f);
    CHECK(server_cache_plan_evaluate_host(
        true, false, 20, 40, 80, 1024).reason ==
        COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);

    const auto ckpt = server_cache_plan_evaluate_checkpoint(
        true, true, true, true, 30, 39, 40, 0, 512);
    CHECK(ckpt.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(ckpt.lcp_tokens == 39);
    CHECK(server_cache_plan_evaluate_checkpoint(
        true, true, true, false, 30, 39, 40, 0, 512).reason ==
        COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);
    CHECK(server_cache_plan_viable(ckpt.reason));
    CHECK(!server_cache_plan_viable(
        COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT));
    CHECK(server_cache_plan_host_checkpoint_source_id(2, 3) == 1020003);
}

static void test_checkpoint_orientation_and_host_identity() {
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 0) == 2);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 1) == 1);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 2) == 0);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 3) == -1);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 0, 7) ==
          server_cache_plan_host_checkpoint_source_id(7, 2));
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(2) == 2);
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(
              server_cache_plan_host_checkpoint_source_id(7, 2), 7) == 2);
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(
              server_cache_plan_host_checkpoint_source_id(8, 2), 7) == -1);
    CHECK(server_cache_plan_checkpoint_reverse_position_from_source_id(
              3, server_cache_plan_host_checkpoint_source_id(7, 2), 7) == 0);

    common_cache_plan_record checkpoint_rec;
    auto * oldest = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 0, 0, 5);
    auto * middle = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 1, 0, 5);
    auto * newest = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 2, 0, 5);
    CHECK(oldest && middle && newest);
    CHECK(checkpoint_rec.find_or_add(
              common_cache_plan_provider::live_context_checkpoint,
              server_cache_plan_checkpoint_source_id_from_reverse(3, 0),
              COMMON_CACHE_PLAN_PHASE_CKPT_SCAN, 5) == newest);

    int32_t next_source = 0;
    int32_t old_prefix_instance = -1;
    int32_t survivor_instance = -1;
    int32_t replacement_instance = -1;
    int32_t source = -1;
    CHECK(server_cache_plan_assign_source_id(
        old_prefix_instance, next_source, source));
    const int32_t old_prefix = source;
    CHECK(source == 0);
    CHECK(server_cache_plan_assign_source_id(
        survivor_instance, next_source, source));
    const int32_t survivor = source;
    CHECK(source == 1);
    // Save-time prefix dedup removes old_prefix. The surviving physical node
    // must keep its identity after its list ordinal shifts from 1 to 0.
    CHECK(server_cache_plan_assign_source_id(
        survivor_instance, next_source, source));
    CHECK(source == 1);
    // The freshly saved replacement cannot inherit either removed ordinal.
    CHECK(server_cache_plan_assign_source_id(
        replacement_instance, next_source, source));
    const int32_t replacement = source;
    CHECK(source == 2);

    common_cache_plan_record host_rec;
    auto * removed_row = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, old_prefix, 0, 6);
    auto * survivor_row = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, survivor, 0, 6);
    CHECK(removed_row && survivor_row && removed_row != survivor_row);
    // After dedup shifts the survivor to list ordinal zero, its immutable id
    // still rejoins its own pre-mutation row, never removed_row.
    CHECK(host_rec.find_or_add(
              common_cache_plan_provider::host_cache_entry, survivor,
              COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 6) == survivor_row);
    CHECK(host_rec.find_or_add(
              common_cache_plan_provider::host_cache_entry, replacement,
              COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 6) != survivor_row);
}

static void test_compose_excludes_destroyed_live_checkpoint() {
    common_cache_plan_record rec;
    auto * host = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 4, 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(host != nullptr);
    host->accept();
    host->delivered = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);

    auto * destroyed_live = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 0, 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(destroyed_live != nullptr);
    destroyed_live->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);

    auto * host_checkpoint = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(4, 0), 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(host_checkpoint != nullptr);
    host_checkpoint->accept();
    host_checkpoint->delivered = true;
    host_checkpoint->component_only = true;
    host_checkpoint->dependent_host_source_id = host->source_id;
    rec.select(common_cache_plan_provider::live_context_checkpoint,
               host_checkpoint);
    rec.chosen = common_cache_plan_provider::live_context_checkpoint;

    const uint32_t before = rec.n_inventory;
    common_cache_plan_compose_chains(rec);
    CHECK(rec.n_inventory == before + 1);
    CHECK(!destroyed_live->component_only);
    CHECK(rec.inventory[before].component_ids[0] ==
          int32_t(host - rec.inventory.data()));
    CHECK(rec.inventory[before].component_ids[1] ==
          int32_t(host_checkpoint - rec.inventory.data()));
}

static void test_composed_chain_reuses_inventory_identity() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;

    const auto fixture = add_host_checkpoint_chain(
        rec, 2, 4, 0, common_cache_plan_selection::similarity);
    auto * host = fixture.host;
    auto * selected = fixture.checkpoint;
    auto * selected_chain = fixture.chain;
    host->accept();
    host->delivered = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);
    auto * sibling = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(4, 1), 0, 2,
        common_cache_plan_selection::similarity);
    auto * foreign = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(5, 0), 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(selected && sibling && foreign);
    selected->accept();
    selected->delivered = true;
    sibling->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    sibling->component_only = true;
    sibling->dependent_host_source_id = 4;
    foreign->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    foreign->component_only = true;
    foreign->dependent_host_source_id = 5;
    rec.select(common_cache_plan_provider::live_context_checkpoint, selected);
    rec.chosen = common_cache_plan_provider::live_context_checkpoint;

    auto * sibling_chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(host - rec.inventory.data()),
        int32_t(sibling - rec.inventory.data()));
    CHECK(sibling_chain);
    selected_chain->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    sibling_chain->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    const int32_t selected_chain_id =
        int32_t(selected_chain - rec.inventory.data());
    const uint32_t inventory_before = rec.n_inventory;

    // The counterfactual planner chose the inventory-time physical chain. Finalize must
    // promote that same ordinal, not append a duplicate, and must not combine the delivered
    // host with a checkpoint from host source 5.
    rec.planner_status = common_cache_plan_planner_status::ok;
    rec.shadow_choice = selected_chain_id;
    rec.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        rec, common_cache_plan_authority_level::lru,
        common_cache_plan_authority_fallback::none);
    common_cache_plan_compose_chains(rec);

    CHECK(rec.n_inventory == inventory_before);
    CHECK(rec.shipped_plan_candidate == selected_chain_id);
    CHECK(selected_chain->delivered);
    CHECK(selected_chain->disposition ==
          common_cache_plan_disposition::accepted);
    CHECK(!sibling_chain->delivered);
    CHECK(sibling_chain->disposition ==
          common_cache_plan_disposition::valid_not_chosen_cost);

    common_cache_plan_finalize_shadow_authority(rec);
    CHECK(rec.authority.legacy_plan_candidate == selected_chain_id);
    CHECK(rec.authority.planner_plan_candidate == selected_chain_id);
    CHECK(!rec.authority.disagreed);
}

static void test_inventory_saturation_refuses_qualification() {
    common_cache_plan_record rec;
    rec.calibration_profile = "test";
    rec.n_prompt_tokens = llama_cache_acct_value::measured(32);
    for (size_t i = 0; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        auto * row = rec.find_or_add(
            common_cache_plan_provider::cold_replay, int32_t(i), 0,
            int32_t(i), common_cache_plan_selection::lru);
        CHECK(row != nullptr);
        row->accept();
    }
    CHECK(!rec.inventory_saturated());
    CHECK(rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 1000, 0, 1000,
        common_cache_plan_selection::lru) == nullptr);
    CHECK(rec.inventory_saturated());
    CHECK(rec.inventory_states[size_t(
              common_cache_plan_provider::host_cache_entry)] ==
          common_cache_plan_inventory_state::overflowed);

    const common_cache_plan_calib calib = {
        "test", 1, 1.0, 1.0, 1.0,
    };
    CHECK(common_cache_plan_estimate_and_choose(rec, calib) ==
          common_cache_plan_planner_status::incomplete_evidence);
    CHECK(rec.shadow_choice == -1);
}

static void test_stale_capability_refuses_without_throwing() {
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::lru);
    common_cache_plan_record rec;
    authority.plan_before_mutation(rec, 10, 11);
    CHECK(rec.planner_precomputed);
    CHECK(!rec.authority_prequalified);
    CHECK(rec.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);
    CHECK(rec.authority.configured_level ==
          common_cache_plan_authority_level::lru);
}

static server_cache_plan_execution authorize_choice(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        int32_t choice,
        int32_t target,
        bool host_lookup_enabled = true,
        common_cache_plan_authority_level level =
            common_cache_plan_authority_level::by_id,
        common_cache_plan_selection selection =
            common_cache_plan_selection::by_id,
        bool target_identity_matches = true,
        common_cache_plan_destruction_effect_set permitted_effects = 0) {
    rec.selection = selection;
    rec.planner_status = common_cache_plan_planner_status::ok;
    rec.shadow_choice = choice;
    rec.authority_prequalified = true;
    rec.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        rec, level,
        common_cache_plan_authority_fallback::none);
    return authority.authorize(
        rec, target, host_lookup_enabled, target_identity_matches,
        permitted_effects);
}

static void test_execution_seam_fallbacks() {
    constexpr int32_t target = 4;
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);

    common_cache_plan_record coverage;
    auto * live = add_viable(
        coverage, common_cache_plan_provider::live_slot, target, target);
    auto execution = authorize_choice(
        authority, coverage,
        int32_t(live - coverage.inventory.data()), target);
    CHECK(execution.authoritative());
    CHECK(server_cache_plan_demote_for_coverage_recovery(
        authority, coverage, execution, 64, 64));
    CHECK(!execution.authoritative());
    CHECK(coverage.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(coverage.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);
    CHECK(coverage.authority_prequalified);

    // A routine dynamic-VBR low-LCP reset supersedes the armed plan after its
    // post-reclaim ownership sample. It is capability drift, not an internal
    // execution fault, and remains in the qualified denominator.
    common_cache_plan_record low_lcp;
    auto * low_lcp_live = add_viable(
        low_lcp, common_cache_plan_provider::live_slot, target, target,
        common_cache_plan_selection::route_home);
    server_cache_plan_authority route_home(
        common_cache_plan_authority_level::route_home);
    execution = authorize_choice(
        route_home, low_lcp,
        int32_t(low_lcp_live - low_lcp.inventory.data()), target, true,
        common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::route_home);
    CHECK(execution.authoritative());
    CHECK(server_cache_plan_demote_for_vbr_low_lcp_reset(
        route_home, low_lcp, execution, true));
    CHECK(!execution.authoritative());
    CHECK(low_lcp.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(low_lcp.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);
    CHECK(low_lcp.authority_prequalified);
    low_lcp.shipped_plan_candidate =
        int32_t(low_lcp_live - low_lcp.inventory.data());
    route_home.finalize_execution(low_lcp);
    CHECK(route_home.counters.authority_eligible[size_t(
              common_cache_plan_selection::route_home)] == 1);
    CHECK(route_home.counters.fallback_reason[size_t(
              common_cache_plan_authority_fallback::stale_capability)] == 1);

    server_cache_plan_execution live_execution;
    live_execution.kind =
        server_cache_plan_execution_kind::live_replay;
    CHECK(server_cache_plan_live_replay_lost_to_logits(
        live_execution, 0));
    CHECK(!server_cache_plan_live_replay_lost_to_logits(
        live_execution, 1));

    // A post-retarget coverage demotion executes legacy recovery on the
    // planner-selected slot, not on the legacy-selected slot. Preserve the
    // pre-mutation slot-A counterfactual so the receipt records disagreement.
    constexpr int32_t legacy_target = 8;
    constexpr int32_t planned_target = 9;
    common_cache_plan_record retarget;
    auto * legacy_live = add_viable(
        retarget, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target,
        common_cache_plan_selection::similarity);
    legacy_live->f_keep = 0.8;
    legacy_live->f_keep_known = true;
    auto * planned_live = add_viable(
        retarget, common_cache_plan_provider::live_slot,
        planned_target, planned_target,
        common_cache_plan_selection::similarity);
    planned_live->f_keep = 1.0;
    planned_live->f_keep_known = true;
    server_cache_plan_authority similarity(
        common_cache_plan_authority_level::similarity);
    execution = authorize_choice(
        similarity, retarget,
        int32_t(planned_live - retarget.inventory.data()), legacy_target,
        true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(execution.authoritative());
    CHECK(server_cache_plan_demote_for_coverage_recovery(
        similarity, retarget, execution, 64, 64));
    retarget.shipped_plan_candidate =
        int32_t(planned_live - retarget.inventory.data());
    similarity.finalize_execution(retarget);
    CHECK(retarget.authority.legacy_plan_candidate ==
          int32_t(legacy_live - retarget.inventory.data()));
    CHECK(retarget.authority.planner_plan_candidate ==
          int32_t(planned_live - retarget.inventory.data()));
    CHECK(retarget.authority.executed_plan_candidate ==
          int32_t(planned_live - retarget.inventory.data()));
    CHECK(retarget.authority.disagreed);
    CHECK(similarity.counters.agree[size_t(
              common_cache_plan_selection::similarity)] == 0);
    CHECK(similarity.counters.disagree[size_t(
              common_cache_plan_selection::similarity)] == 1);

    common_cache_plan_record checkpoint;
    auto * row = add_viable(
        checkpoint,
        common_cache_plan_provider::live_context_checkpoint,
        2, target);
    execution = authorize_choice(
        authority, checkpoint,
        int32_t(row - checkpoint.inventory.data()), target);
    int32_t ordinal = -1;
    CHECK(!server_cache_plan_revalidate_checkpoint_execution(
        authority, checkpoint, execution, 3, false, ordinal));
    CHECK(!execution.authoritative());
    CHECK(ordinal == -1);
    // The seam retains its legacy iterator on refusal; it never translates a
    // missing/invalid planner checkpoint into a synthetic rend/cold choice.
    CHECK(checkpoint.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);

    common_cache_plan_record armed_record;
    auto armed_plan = std::make_unique<common_cache_plan_record>(armed_record);
    auto armed_recovery = std::make_unique<int>(1);
    server_cache_plan_execution armed;
    armed.kind = server_cache_plan_execution_kind::live_replay;
    armed.target = target;
    server_cache_plan_disarm_unlaunched(
        armed, armed_plan, armed_recovery);
    CHECK(!armed.authoritative());
    CHECK(!armed_plan);
    CHECK(!armed_recovery);
}

static void test_qualified_fallback_remains_eligible() {
    constexpr int32_t target = 1;
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    common_cache_plan_record rec;
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, target, target);
    const int32_t candidate = int32_t(live - rec.inventory.data());
    auto execution = authorize_choice(
        authority, rec, candidate, target);
    CHECK(execution.authoritative());
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability);
    CHECK(rec.authority_prequalified);
    rec.shipped_plan_candidate = candidate;
    authority.finalize_execution(rec);
    CHECK(authority.counters.authority_eligible[size_t(
              common_cache_plan_selection::by_id)] == 1);
    CHECK(authority.counters.authority_executed[size_t(
              common_cache_plan_selection::by_id)] == 0);
}

static void test_by_id_execution_shapes_and_target_binding() {
    constexpr int32_t target = 7;
    common_cache_plan_record rec;
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, target, target);
    const auto fixture = add_host_checkpoint_chain(rec, target, 11, 2);
    auto * host = fixture.host;
    host->f_keep = 0.8; host->f_keep_known = true;
    host->sim = 0.8; host->sim_known = true;
    auto * checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint, 3, target);
    auto * cold = add_viable(
        rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, target);
    auto * chain = fixture.chain;

    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    struct expectation {
        common_cache_plan_candidate * candidate;
        server_cache_plan_execution_kind kind;
        int32_t host_source;
        int32_t checkpoint_source;
    } cases[] = {
        { live, server_cache_plan_execution_kind::live_replay, -1, -1 },
        { host, server_cache_plan_execution_kind::host_restore, 11, -1 },
        { checkpoint, server_cache_plan_execution_kind::checkpoint_restore, -1, 3 },
        { chain, server_cache_plan_execution_kind::host_checkpoint_restore,
          11, server_cache_plan_host_checkpoint_source_id(11, 2) },
        { cold, server_cache_plan_execution_kind::cold_replay, -1, -1 },
    };
    for (const auto & expected : cases) {
        const int32_t choice = int32_t(expected.candidate - rec.inventory.data());
        server_cache_plan_execution execution;
        CHECK(server_cache_plan_execution_from_candidate(
            rec, choice, target, execution));
        CHECK(execution.kind == expected.kind);
        CHECK(execution.target == target);
        CHECK(execution.host_source_id == expected.host_source);
        CHECK(execution.checkpoint_source_id == expected.checkpoint_source);
        if (expected.kind != server_cache_plan_execution_kind::cold_replay) {
            auto copy = rec;
            CHECK(authorize_choice(
                authority, copy, choice, target).authoritative());
        }
    }

    common_cache_plan_record cold_only;
    auto * isolated_cold = add_viable(
        cold_only, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, target);
    CHECK(authorize_choice(authority, cold_only,
              int32_t(isolated_cold - cold_only.inventory.data()), target).
          authoritative());

    // The full fixture's legacy counterfactual is the host+checkpoint chain;
    // a non-destructive live disagreement remains within non-destructive authority.
    auto copy = rec;
    CHECK(authorize_choice(authority, copy,
              int32_t(live - rec.inventory.data()), target).authoritative());

    // A planner row for another slot can never become an execution capability.
    copy = rec;
    const auto wrong_target = authorize_choice(
        authority, copy, int32_t(live - rec.inventory.data()), target + 1);
    CHECK(!wrong_target.authoritative());
    CHECK(copy.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::internal_fault);

    // Without a destruction certificate, an authority disagreement may avoid destruction but may not
    // consume a different retained host entry or turn reuse into cold.
    auto * foreign_host = add_viable(
        rec, common_cache_plan_provider::host_cache_entry, 12, target);
    foreign_host->f_keep = 0.1; foreign_host->f_keep_known = true;
    foreign_host->sim = 0.1; foreign_host->sim_known = true;
    host->f_keep = 0.8; host->f_keep_known = true;
    host->sim = 0.8; host->sim_known = true;
    copy = rec;
    CHECK(!authorize_choice(authority, copy,
              int32_t(foreign_host - rec.inventory.data()), target).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);
    copy = rec;
    CHECK(!authorize_choice(
              authority, copy, int32_t(cold - rec.inventory.data()), target).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);

    copy = rec;
    CHECK(!authorize_choice(authority, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::by_id,
              common_cache_plan_selection::similarity).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::tier_not_enabled);

    server_cache_plan_authority similarity(
        common_cache_plan_authority_level::similarity);
    copy = rec;
    CHECK(authorize_choice(similarity, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::similarity,
              common_cache_plan_selection::similarity).authoritative());

    // The final landed level keeps every earlier ratchet cumulative and now
    // admits its own LRU tier as well.
    server_cache_plan_authority future(
        common_cache_plan_authority_level::lru);
    copy = rec;
    CHECK(authorize_choice(future, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::lru,
              common_cache_plan_selection::route_home).authoritative());
    copy = rec;
    CHECK(authorize_choice(future, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::lru,
              common_cache_plan_selection::similarity).authoritative());
    copy = rec;
    CHECK(authorize_choice(future, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::lru,
              common_cache_plan_selection::lru).authoritative());

    // A disabled future tier must not blur an existing planner refusal into
    // tier_not_enabled. The latter names only an otherwise-qualified plan.
    common_cache_plan_record unfitted = rec;
    unfitted.selection = common_cache_plan_selection::route_home;
    unfitted.planner_status =
        common_cache_plan_planner_status::profile_unfitted;
    unfitted.planner_precomputed = true;
    unfitted.authority_prequalified = false;
    common_cache_plan_derive_shadow_authority(
        unfitted, common_cache_plan_authority_level::lru,
        common_cache_plan_authority_fallback::none);
    CHECK(!future.authorize(unfitted, target).authoritative());
    CHECK(unfitted.authority.fallback_reason ==
          common_cache_plan_authority_fallback::profile_unfitted);

    copy = rec;
    CHECK(!authorize_choice(authority, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::by_id,
              common_cache_plan_selection::by_id,
              false).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::incomplete_evidence);
}

static void test_legacy_counterfactual_and_authoritative_receipt() {
    constexpr int32_t target = 2;
    common_cache_plan_record rec;
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, target, target);
    live->f_keep = 0.4; live->f_keep_known = true;
    live->sim = 0.4; live->sim_known = true;
    const auto fixture = add_host_checkpoint_chain(rec, target, 5, 1);
    auto * host = fixture.host;
    host->f_keep = 0.8; host->f_keep_known = true;
    host->sim = 0.7; host->sim_known = true;
    auto * chain = fixture.chain;
    add_viable(
        rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, target);

    const int32_t legacy = server_cache_plan_legacy_candidate(rec, target);
    CHECK(legacy == int32_t(chain - rec.inventory.data()));
    CHECK(server_cache_plan_legacy_candidate(rec, target, false) ==
          int32_t(live - rec.inventory.data()));

    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    const int32_t planner = int32_t(live - rec.inventory.data());
    const auto execution = authorize_choice(authority, rec, planner, target);
    CHECK(execution.kind == server_cache_plan_execution_kind::live_replay);
    rec.shipped_plan_candidate = planner;
    authority.finalize_execution(rec);
    CHECK(rec.authority.state ==
          common_cache_plan_authority_state::authoritative);
    CHECK(rec.authority.legacy_plan_candidate == legacy);
    CHECK(rec.authority.planner_plan_candidate == planner);
    CHECK(rec.authority.executed_plan_candidate == planner);
    CHECK(rec.authority.disagreed);
    CHECK(authority.counters.authority_executed[size_t(
              common_cache_plan_selection::by_id)] == 1);
}

static void test_similarity_crossover_and_safety_envelope() {
    constexpr int32_t legacy_target = 2;
    constexpr int32_t other_target = 3;
    const common_cache_plan_calib calib = {
        "similarity-crossover", 1,
        /* replay_us_per_token */ 100.0,
        /* restore_us_per_byte */ 0.001,
        /* workspace_setup_us */  500.0,
    };

    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;
    rec.calibration_profile = calib.profile;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * live = add_viable(rec, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target, common_cache_plan_selection::similarity);
    live->lcp_tokens = llama_cache_acct_value::measured(90);
    live->f_keep = 0.9; live->f_keep_known = true;
    const auto host_checkpoint = add_host_checkpoint_chain(
        rec, legacy_target, 11, 0,
        common_cache_plan_selection::similarity);
    host_checkpoint.host->lcp_tokens = llama_cache_acct_value::measured(80);
    host_checkpoint.host->payload_bytes =
        llama_cache_acct_value::measured(5'000'000);
    host_checkpoint.host->f_keep = 0.95;
    host_checkpoint.host->f_keep_known = true;
    host_checkpoint.host->sim = 0.95;
    host_checkpoint.host->sim_known = true;
    host_checkpoint.checkpoint->lcp_tokens =
        llama_cache_acct_value::measured(99);
    host_checkpoint.checkpoint->payload_bytes =
        llama_cache_acct_value::measured(45'000'000);
    auto * cold = add_viable(rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::similarity);
    (void) cold;

    // A cheaper row below the strict threshold is route-home evidence, not a
    // similarity-tier authority candidate.
    auto * below_threshold = add_viable(
        rec, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::route_home);
    below_threshold->lcp_tokens = llama_cache_acct_value::measured(100);
    below_threshold->f_keep = 1.0; below_threshold->f_keep_known = true;

    CHECK(server_cache_plan_legacy_candidate(rec, legacy_target) ==
          int32_t(host_checkpoint.chain - rec.inventory.data()));
    rec.planner_status = common_cache_plan_estimate_and_choose(rec, calib);
    CHECK(rec.planner_status == common_cache_plan_planner_status::ok);
    CHECK(rec.shadow_choice == int32_t(live - rec.inventory.data()));

    server_cache_plan_authority authority(
        common_cache_plan_authority_level::similarity);
    rec.authority_prequalified = true;
    rec.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        rec, common_cache_plan_authority_level::similarity,
        common_cache_plan_authority_fallback::none);
    auto execution = authority.authorize(rec, legacy_target);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::live_replay);
    CHECK(execution.target == legacy_target); // crossover changes provider, not slot
    rec.shipped_plan_candidate = int32_t(live - rec.inventory.data());
    authority.finalize_execution(rec);
    CHECK(rec.authority.executed_plan_candidate ==
          int32_t(live - rec.inventory.data()));
    CHECK(!host_checkpoint.checkpoint->delivered);
    CHECK(!rec.restore_attempt_failed);

    // A replay-wins crossover against a LIVE checkpoint is structurally
    // impossible: below the coverage threshold legacy replays, while at/above
    // it replay is invalid and the coverage seam demotes. The real replay win
    // above is the host-side prompt-load/host-checkpoint chain.

    common_cache_plan_record long_suffix;
    long_suffix.selection = common_cache_plan_selection::similarity;
    long_suffix.calibration_profile = calib.profile;
    long_suffix.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * long_live = add_viable(
        long_suffix, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target,
        common_cache_plan_selection::similarity);
    long_live->lcp_tokens = llama_cache_acct_value::measured(20);
    long_live->f_keep = 0.2; long_live->f_keep_known = true;
    auto * cheap_checkpoint = add_viable(
        long_suffix, common_cache_plan_provider::live_context_checkpoint,
        0, legacy_target, common_cache_plan_selection::similarity);
    cheap_checkpoint->lcp_tokens = llama_cache_acct_value::measured(90);
    cheap_checkpoint->payload_bytes = llama_cache_acct_value::measured(1'000);
    add_viable(long_suffix, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::similarity);
    long_suffix.planner_status =
        common_cache_plan_estimate_and_choose(long_suffix, calib);
    CHECK(long_suffix.planner_status == common_cache_plan_planner_status::ok);
    CHECK(long_suffix.shadow_choice ==
          int32_t(cheap_checkpoint - long_suffix.inventory.data()));
    long_suffix.authority_prequalified = true;
    long_suffix.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        long_suffix, common_cache_plan_authority_level::similarity,
        common_cache_plan_authority_fallback::none);
    execution = authority.authorize(long_suffix, legacy_target);
    CHECK(execution.authoritative());
    CHECK(execution.kind ==
          server_cache_plan_execution_kind::checkpoint_restore);
    int32_t restore_ordinal = -1;
    CHECK(server_cache_plan_revalidate_checkpoint_execution(
        authority, long_suffix, execution, 1, true, restore_ordinal));
    CHECK(execution.authoritative());
    CHECK(restore_ordinal == 0);

    // Cross-target similarity execution remains fail-closed unless it
    // retains the target's complete live prefix (the zero-destruction case).
    common_cache_plan_record cross;
    auto * legacy_live = add_viable(
        cross, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target, common_cache_plan_selection::similarity);
    legacy_live->f_keep = 0.8; legacy_live->f_keep_known = true;
    auto * cross_live = add_viable(
        cross, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::similarity);
    cross_live->f_keep = 0.5; cross_live->f_keep_known = true;
    execution = authorize_choice(
        authority, cross, int32_t(cross_live - cross.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(!execution.authoritative());
    CHECK(cross.authority.fallback_reason ==
          common_cache_plan_authority_fallback::destruction_authority_required);

    cross_live->f_keep = 1.0;
    execution = authorize_choice(
        authority, cross, int32_t(cross_live - cross.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(execution.authoritative());
    CHECK(execution.target == other_target);

    // A route-home row cannot be smuggled into a similarity receipt even if a
    // malformed caller installs it as the planner choice.
    execution = authorize_choice(
        authority, rec,
        int32_t(below_threshold - rec.inventory.data()), legacy_target,
        true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(!execution.authoritative());
    CHECK(rec.authority.fallback_reason ==
          common_cache_plan_authority_fallback::incomplete_evidence);
}

static void test_route_home_authority_domain() {
    constexpr int32_t home_target = 4;
    constexpr int32_t other_target = 5;
    const common_cache_plan_calib calib = {
        "route-home", 1,
        /* replay_us_per_token */ 100.0,
        /* restore_us_per_byte */ 0.001,
        /* workspace_setup_us */  500.0,
    };

    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::route_home;
    rec.calibration_profile = calib.profile;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * home = add_viable(
        rec, common_cache_plan_provider::live_slot,
        home_target, home_target, common_cache_plan_selection::route_home);
    home->lcp_tokens = llama_cache_acct_value::measured(10);
    home->sim = 0.1;
    home->sim_known = true;
    home->f_keep = 1.0;
    home->f_keep_known = true;
    auto * checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint,
        0, home_target, common_cache_plan_selection::route_home);
    checkpoint->lcp_tokens = llama_cache_acct_value::measured(90);
    checkpoint->payload_bytes = llama_cache_acct_value::measured(1'000);
    add_viable(rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, home_target,
        common_cache_plan_selection::route_home);

    // LRU-origin and busy rows are each cheaper than the accepted checkpoint;
    // only their independent domain/provider exclusions keep them out.
    auto * lru_only = add_viable(
        rec, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::lru);
    lru_only->lcp_tokens = llama_cache_acct_value::measured(100);
    lru_only->sim = 1.0;
    lru_only->sim_known = true;
    lru_only->f_keep = 1.0;
    lru_only->f_keep_known = true;
    auto * busy = add_viable(
        rec, common_cache_plan_provider::live_slot,
        other_target + 1, other_target + 1,
        common_cache_plan_selection::route_home);
    busy->lcp_tokens = llama_cache_acct_value::measured(99);
    busy->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);

    rec.planner_status = common_cache_plan_estimate_and_choose(rec, calib);
    CHECK(rec.planner_status == common_cache_plan_planner_status::ok);
    CHECK(rec.shadow_choice ==
          int32_t(checkpoint - rec.inventory.data()));
    CHECK(rec.inventory[size_t(rec.shadow_choice)].target_slot_id ==
          home_target);

    server_cache_plan_authority route_home(
        common_cache_plan_authority_level::route_home);
    auto execution = authorize_choice(
        route_home, rec, rec.shadow_choice, home_target, false,
        common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::route_home);
    CHECK(execution.authoritative());
    CHECK(execution.kind ==
          server_cache_plan_execution_kind::checkpoint_restore);
    CHECK(execution.target == home_target);
    CHECK(rec.authority.decision_tier ==
          common_cache_plan_selection::route_home);

    // The previous ceiling remains a hard boundary for route-home records.
    auto lower = rec;
    server_cache_plan_authority similarity(
        common_cache_plan_authority_level::similarity);
    CHECK(!authorize_choice(
        similarity, lower, lower.shadow_choice, home_target, false,
        common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::route_home).authoritative());
    CHECK(lower.authority.fallback_reason ==
          common_cache_plan_authority_fallback::tier_not_enabled);

    // A BOS-only cross-target choice needs retention/destruction economics the
    // current fit does not carry. Dynamic VBR cannot durabilize the displaced
    // target in a host cache, so this shape remains typed legacy without a destruction certificate.
    common_cache_plan_record trivial;
    auto * legacy_bos = add_viable(
        trivial, common_cache_plan_provider::live_slot,
        home_target, home_target,
        common_cache_plan_selection::route_home);
    legacy_bos->lcp_tokens = llama_cache_acct_value::measured(1);
    legacy_bos->f_keep = 0.01;
    legacy_bos->f_keep_known = true;
    auto * alternate_bos = add_viable(
        trivial, common_cache_plan_provider::live_slot,
        other_target, other_target,
        common_cache_plan_selection::route_home);
    alternate_bos->lcp_tokens = llama_cache_acct_value::measured(1);
    alternate_bos->f_keep = 1.0;
    alternate_bos->f_keep_known = true;
    execution = authorize_choice(
        route_home, trivial,
        int32_t(alternate_bos - trivial.inventory.data()), home_target,
        false, common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::route_home);
    CHECK(!execution.authoritative());
    CHECK(trivial.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);

}

static void test_lru_authority_domain_and_eviction_fence() {
    constexpr int32_t legacy_target = 6;
    constexpr int32_t other_target = 2;
    server_cache_plan_authority lru(
        common_cache_plan_authority_level::lru);

    // Empty-slot spread: a target-qualified cold plan on the shipped LRU slot
    // has exactly the same destructive effect and is authoritative.
    common_cache_plan_record empty;
    auto * cold = add_viable(
        empty, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::lru);
    auto execution = authorize_choice(
        lru, empty, int32_t(cold - empty.inventory.data()), legacy_target,
        true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::cold_replay);
    CHECK(execution.target == legacy_target);
    // Exercise the two server seam predicates that follow authorize(): an LRU
    // same-target cold plan must skip cross-target currency lookup (there is no
    // viable live row on an empty target) and pass construction-empty recheck.
    CHECK(!server_cache_plan_retarget_currency_required(
        server_cache_plan_selection_admits_retarget(
            common_cache_plan_authority_level::lru,
            common_cache_plan_selection::lru),
        execution.target, legacy_target));
    // and the positive direction: a genuine cross-target plan still requires
    // the currency lookup (a predicate degraded to `return false` must fail)
    CHECK(server_cache_plan_retarget_currency_required(
        server_cache_plan_selection_admits_retarget(
            common_cache_plan_authority_level::lru,
            common_cache_plan_selection::lru),
        execution.target + 1, legacy_target));
    CHECK(server_cache_plan_cold_target_current(execution, true, true));
    CHECK(!server_cache_plan_cold_target_current(execution, false, true));

    // Occupied same-target reuse remains within non-destructive authority.
    common_cache_plan_record occupied;
    auto * live = add_viable(
        occupied, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target, common_cache_plan_selection::lru);
    live->f_keep = 1.0;
    live->f_keep_known = true;
    execution = authorize_choice(
        lru, occupied, int32_t(live - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::live_replay);

    // Cross-target replacement has no certified eviction/retention evidence
    // without a destruction certificate. Schema 5 uses its existing availability spelling rather than
    // adding a new wire enum value.
    auto * foreign = add_viable(
        occupied, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::lru);
    foreign->f_keep = 1.0;
    foreign->f_keep_known = true;
    execution = authorize_choice(
        lru, occupied, int32_t(foreign - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(!execution.authoritative());
    CHECK(occupied.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              budget_or_lease_unavailable);
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
            occupied, int32_t(foreign - occupied.inventory.data()),
            occupied.authority.legacy_plan_candidate),
        common_cache_plan_destruction_effect::cross_target_displacement));

    // Same-target cold replacement requires eviction evidence and remains closed
    // without a destruction certificate.
    auto * occupied_cold = add_viable(
        occupied, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::lru);
    execution = authorize_choice(
        lru, occupied, int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(!execution.authoritative());
    CHECK(occupied.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              budget_or_lease_unavailable);
    const auto lifecycle_effects =
        server_cache_plan_nonconsuming_host_effects(true);
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
              occupied,
              int32_t(occupied_cold - occupied.inventory.data()),
              occupied.authority.legacy_plan_candidate),
        common_cache_plan_destruction_effect::same_target_cold_replacement));
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
            occupied,
            int32_t(occupied_cold - occupied.inventory.data()),
            occupied.authority.legacy_plan_candidate,
            lifecycle_effects),
        common_cache_plan_destruction_effect::same_target_cold_replacement));

    // Lifecycle tracking introduces that class but keeps it closed until the
    // selected effect is certified.
    execution = authorize_choice(
        lru, occupied, int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true, lifecycle_effects);
    CHECK(!execution.authoritative());
    CHECK(occupied.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              budget_or_lease_unavailable);

    // The certificate opens exactly its effect bit. The same occupied cold
    // replacement remains refused above with no capability evidence.
    const auto certified_cold = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::same_target_cold_replacement);
    execution = authorize_choice(
        lru, occupied,
        int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true,
        certified_cold | lifecycle_effects);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::cold_replay);

    // A partial certificate never opens a plural destructive plan.
    auto plural = occupied;
    plural.inventory[size_t(occupied_cold - occupied.inventory.data())]
        .provider = common_cache_plan_provider::host_cache_entry;
    plural.inventory[size_t(occupied_cold - occupied.inventory.data())]
        .source_id = 77;
    execution = authorize_choice(
        lru, plural,
        int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true, certified_cold);
    CHECK(!execution.authoritative());

    // Host-consumption authority is not eviction evidence: choosing another
    // retained host source on the same target keeps the established reason.
    common_cache_plan_record hosts;
    auto * legacy_host = add_viable(
        hosts, common_cache_plan_provider::host_cache_entry,
        41, legacy_target, common_cache_plan_selection::lru);
    legacy_host->f_keep = 0.8; legacy_host->f_keep_known = true;
    legacy_host->sim = 0.8; legacy_host->sim_known = true;
    auto * other_host = add_viable(
        hosts, common_cache_plan_provider::host_cache_entry,
        42, legacy_target, common_cache_plan_selection::lru);
    other_host->f_keep = 0.7; other_host->f_keep_known = true;
    other_host->sim = 0.9; other_host->sim_known = true;
    execution = authorize_choice(
        lru, hosts, int32_t(other_host - hosts.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(!execution.authoritative());
    CHECK(hosts.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
            hosts, int32_t(other_host - hosts.inventory.data()),
            hosts.authority.legacy_plan_candidate),
        common_cache_plan_destruction_effect::
            different_host_source_consumption));
    const auto nonconsuming_host =
        server_cache_plan_nonconsuming_host_effects(true);
    CHECK(server_cache_destruction_effects_for(
              hosts, int32_t(other_host - hosts.inventory.data()),
              hosts.authority.legacy_plan_candidate,
              nonconsuming_host) == 0);
    execution = authorize_choice(
        lru, hosts, int32_t(other_host - hosts.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true, nonconsuming_host);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::host_restore);

    // The previous ceiling still cannot flip an LRU selection.
    common_cache_plan_record lower;
    auto * lower_cold = add_viable(
        lower, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::lru);
    server_cache_plan_authority route_home(
        common_cache_plan_authority_level::route_home);
    CHECK(!authorize_choice(
        route_home, lower,
        int32_t(lower_cold - lower.inventory.data()), legacy_target, true,
        common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::lru).authoritative());
    CHECK(lower.authority.fallback_reason ==
          common_cache_plan_authority_fallback::tier_not_enabled);

    // Raising the configured ceiling leaves all prior ratchets cumulative.
    for (const auto selection : {
            common_cache_plan_selection::by_id,
            common_cache_plan_selection::similarity,
            common_cache_plan_selection::route_home }) {
        common_cache_plan_record prior;
        auto * prior_live = add_viable(
            prior, common_cache_plan_provider::live_slot,
            legacy_target, legacy_target, selection);
        CHECK(authorize_choice(
            lru, prior, int32_t(prior_live - prior.inventory.data()),
            legacy_target, true, common_cache_plan_authority_level::lru,
            selection).authoritative());
    }
}

static void test_typed_planner_fallbacks() {
    const common_cache_plan_planner_status statuses[] = {
        common_cache_plan_planner_status::no_profile,
        common_cache_plan_planner_status::profile_unfitted,
        common_cache_plan_planner_status::incomplete_evidence,
        common_cache_plan_planner_status::internal_fault,
    };
    const common_cache_plan_authority_fallback reasons[] = {
        common_cache_plan_authority_fallback::no_profile,
        common_cache_plan_authority_fallback::profile_unfitted,
        common_cache_plan_authority_fallback::incomplete_evidence,
        common_cache_plan_authority_fallback::internal_fault,
    };
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    for (size_t i = 0; i < std::size(statuses); ++i) {
        common_cache_plan_record rec;
        rec.selection = common_cache_plan_selection::by_id;
        rec.planner_status = statuses[i];
        rec.planner_precomputed = true;
        common_cache_plan_derive_shadow_authority(
            rec, common_cache_plan_authority_level::by_id,
            common_cache_plan_authority_fallback::none);
        const auto execution = authority.authorize(rec, 0);
        CHECK(!execution.authoritative());
        CHECK(rec.authority.state ==
              common_cache_plan_authority_state::fallback_legacy);
        CHECK(rec.authority.fallback_reason == reasons[i]);
    }
}

static void test_off_stays_shadow_and_failed_delivery_not_counted() {
    common_cache_plan_record base;
    auto * live = add_viable(
        base, common_cache_plan_provider::live_slot, 0, 0);
    const int32_t live_id = int32_t(live - base.inventory.data());

    server_cache_plan_authority off(common_cache_plan_authority_level::off);
    auto shadow = base;
    shadow.selection = common_cache_plan_selection::by_id;
    shadow.planner_status = common_cache_plan_planner_status::ok;
    shadow.shadow_choice = live_id;
    shadow.authority_prequalified = true;
    shadow.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        shadow, common_cache_plan_authority_level::off,
        common_cache_plan_authority_fallback::none);
    CHECK(!off.authorize(shadow, 0).authoritative());
    CHECK(shadow.authority.state == common_cache_plan_authority_state::shadow);

    common_cache_plan_record exception;
    exception.selection = common_cache_plan_selection::by_id;
    server_cache_plan_authority flipped(
        common_cache_plan_authority_level::by_id);
    flipped.fail_closed(exception);
    CHECK(exception.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(exception.authority.fallback_reason ==
          common_cache_plan_authority_fallback::internal_fault);

    server_cache_plan_authority on(
        common_cache_plan_authority_level::by_id);
    auto failed = base;
    CHECK(authorize_choice(on, failed, live_id, 0).authoritative());
    auto * cold = add_viable(
        failed, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, 0);
    failed.shipped_plan_candidate = int32_t(cold - failed.inventory.data());
    on.finalize_execution(failed);
    CHECK(failed.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(failed.authority.fallback_reason ==
          common_cache_plan_authority_fallback::internal_fault);
    CHECK(on.counters.authority_executed[size_t(
              common_cache_plan_selection::by_id)] == 0);
}

static void test_eligible_and_executed_index_different_tiers() {
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::lru);
    common_cache_plan_record rec;
    rec.planner_precomputed = true;
    rec.authority_prequalified = true;
    rec.selection = common_cache_plan_selection::similarity;
    rec.authority.configured_level = common_cache_plan_authority_level::lru;
    rec.authority.decision_tier = common_cache_plan_selection::route_home;
    rec.authority.planner_plan_candidate = 2;
    rec.shipped_plan_candidate = 1;
    authority.finalize_execution(rec);

    CHECK(authority.counters.observed[size_t(
              common_cache_plan_selection::similarity)] == 1);
    CHECK(authority.counters.disagree[size_t(
              common_cache_plan_selection::similarity)] == 1);
    CHECK(authority.counters.authority_eligible[size_t(
              common_cache_plan_selection::route_home)] == 1);
    CHECK(authority.counters.authority_executed[size_t(
              common_cache_plan_selection::route_home)] == 0);
    CHECK(rec.authority.executed_plan_candidate == 1);
}

int main() {
    test_candidate_classifiers();
    test_checkpoint_orientation_and_host_identity();
    test_compose_excludes_destroyed_live_checkpoint();
    test_composed_chain_reuses_inventory_identity();
    test_inventory_saturation_refuses_qualification();
    test_stale_capability_refuses_without_throwing();
    test_execution_seam_fallbacks();
    test_qualified_fallback_remains_eligible();
    test_by_id_execution_shapes_and_target_binding();
    test_legacy_counterfactual_and_authoritative_receipt();
    test_similarity_crossover_and_safety_envelope();
    test_route_home_authority_domain();
    test_lru_authority_domain_and_eviction_fence();
    test_typed_planner_fallbacks();
    test_off_stays_shadow_and_failed_delivery_not_counted();
    test_eligible_and_executed_index_different_tiers();
    return 0;
}
