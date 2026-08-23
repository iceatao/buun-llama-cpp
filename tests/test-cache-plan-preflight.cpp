#include "server-cache-plan-preflight-internal.h"
#include "server-cache-plan-authority.h"
#include "server-http.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

#define CHECK(COND) do { if (!(COND)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
        __FILE__, __LINE__, #COND); \
    std::abort(); \
} } while (0)

static common_cache_plan_record fitted_live_record() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;
    rec.planner_status = common_cache_plan_planner_status::ok;
    rec.authority.configured_level = common_cache_plan_authority_level::lru;
    rec.authority.fallback_reason =
        common_cache_plan_authority_fallback::none;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * live = rec.find_or_add(
        common_cache_plan_provider::live_slot, 7,
        COMMON_CACHE_PLAN_PHASE_SIMILARITY, 7,
        common_cache_plan_selection::similarity);
    CHECK(live != nullptr);
    live->accept();
    live->cost_terms[size_t(llama_cache_acct_cost_kind::restore)].raw =
        llama_cache_acct_value::measured(0);
    live->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].raw =
        llama_cache_acct_value::measured(4);
    live->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].estimated_us =
        llama_cache_acct_value::measured(40);
    live->cost_terms[size_t(llama_cache_acct_cost_kind::replay)]
        .estimator_version = 7;
    live->predicted_total_us = llama_cache_acct_value::measured(40);
    rec.shadow_choice = int32_t(live - rec.inventory.data());
    rec.authority.planner_plan_candidate = rec.shadow_choice;
    return rec;
}

static void test_expected_path_closed_set() {
    auto rec = fitted_live_record();
    CHECK(server_cache_plan_preflight_derive_expected_path(rec) ==
          server_cache_plan_preflight_expected_path::
              planner_if_still_current);

    rec.authority.configured_level = common_cache_plan_authority_level::by_id;
    CHECK(server_cache_plan_preflight_derive_expected_path(rec) ==
          server_cache_plan_preflight_expected_path::legacy);
    server_cache_plan_preflight_view lower_view;
    CHECK(server_cache_plan_preflight_build_view(
        rec, 7, true, lower_view));
    CHECK(lower_view.fallback_reason ==
          common_cache_plan_authority_fallback::tier_not_enabled);
    rec.authority.configured_level = common_cache_plan_authority_level::lru;

    rec.destruction.plan_candidate = rec.shadow_choice;
    rec.destruction.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::cross_target_displacement);
    rec.destruction.state = common_cache_plan_destruction_state::quoted;
    CHECK(server_cache_plan_preflight_derive_expected_path(rec) ==
          server_cache_plan_preflight_expected_path::
              conditional_on_destruction_certification);
    rec.destruction.state = common_cache_plan_destruction_state::refused;
    rec.destruction.reason =
        common_cache_plan_destruction_reason::lifecycle_disabled;
    CHECK(server_cache_plan_preflight_derive_expected_path(rec) ==
          server_cache_plan_preflight_expected_path::legacy);
    rec.destruction.state = common_cache_plan_destruction_state::quoted;
    CHECK(server_cache_plan_preflight_derive_expected_path(rec, false) ==
          server_cache_plan_preflight_expected_path::legacy);
    server_cache_plan_preflight_view stale_view;
    CHECK(server_cache_plan_preflight_build_view(
        rec, 7, false, stale_view));
    CHECK(stale_view.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);

    rec.planner_status = common_cache_plan_planner_status::profile_unfitted;
    CHECK(server_cache_plan_preflight_derive_expected_path(rec) ==
          server_cache_plan_preflight_expected_path::legacy);
}

static void test_destruction_view_mapping() {
    auto rec = fitted_live_record();
    auto & receipt = rec.destruction;
    receipt.state = common_cache_plan_destruction_state::quoted;
    receipt.reason = common_cache_plan_destruction_reason::none;
    receipt.plan_candidate = rec.shadow_choice;
    receipt.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::cross_target_displacement);
    receipt.lease_verdict =
        common_cache_plan_destruction_lease_verdict::soft_leased;
    receipt.displaced_fate =
        common_cache_plan_displaced_fate::retained_host;
    receipt.recovery_citation =
        common_cache_plan_recovery_citation::prospective;
    common_cache_plan_destruction_quote quote;
    quote.receipt = receipt;
    common_cache_plan_yield_domain domain;
    domain.projected_release_bytes = llama_cache_acct_value::measured(64);
    quote.projected_domains.push_back(domain);
    rec.destruction_quotes.push_back(quote);
    rec.inventory[size_t(rec.shadow_choice)]
        .cost_terms[size_t(llama_cache_acct_cost_kind::eviction)]
        .estimated_us = llama_cache_acct_value::measured(123);

    server_cache_plan_preflight_view view;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, true, view));
    CHECK(view.expected_path ==
          server_cache_plan_preflight_expected_path::
              conditional_on_destruction_certification);
    CHECK(view.destruction.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(view.destruction.effects == receipt.effects);
    CHECK(view.destruction.protection == receipt.lease_verdict);
    CHECK(view.destruction.displaced_fate == receipt.displaced_fate);
    CHECK(view.destruction.recovery == receipt.recovery_citation);
    CHECK(view.destruction.projected_release_bytes.value == 64);
    CHECK(view.destruction.estimated_destruction_us.value == 123);

    receipt.state = common_cache_plan_destruction_state::refused;
    receipt.reason = common_cache_plan_destruction_reason::lifecycle_disabled;
    rec.destruction = receipt;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, true, view));
    CHECK(view.expected_path ==
          server_cache_plan_preflight_expected_path::legacy);
    CHECK(view.destruction.state ==
          common_cache_plan_destruction_state::refused);
    CHECK(view.destruction.reason ==
          common_cache_plan_destruction_reason::lifecycle_disabled);

    rec.destruction.plan_candidate = rec.shadow_choice + 1;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, true, view));
    CHECK(view.destruction.state ==
          common_cache_plan_destruction_state::failed);
    CHECK(view.destruction.reason ==
          common_cache_plan_destruction_reason::
              release_evidence_unavailable);
}

static void test_gcp_dispatch_excludes_preflight() {
    CHECK(!server_http_gcp_predict_dispatch_allowed("/cache/plan"));
    CHECK(server_http_gcp_predict_dispatch_allowed("/completion"));
    CHECK(server_http_gcp_predict_dispatch_allowed("cachePlan"));
}

static void test_saturated_inventory_refuses_typed() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::lru;
    rec.calibration_profile =
        "qwen35-2b-q4-k---medium/nvidia-geforce-rtx-3090-ngl99/"
        "b512/kf16-vf16";
    rec.n_prompt_tokens = llama_cache_acct_value::measured(128);
    for (uint32_t i = 0; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        auto * row = rec.find_or_add(
            common_cache_plan_provider::cold_replay, int32_t(i),
            COMMON_CACHE_PLAN_PHASE_LRU, int32_t(i),
            common_cache_plan_selection::lru);
        CHECK(row != nullptr);
        row->accept();
    }
    CHECK(rec.find_or_add(
              common_cache_plan_provider::cold_replay, 999,
              COMMON_CACHE_PLAN_PHASE_LRU, 999,
              common_cache_plan_selection::lru) == nullptr);
    CHECK(rec.inventory_saturated());
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::lru);
    authority.plan_before_mutation(rec, 7, 7);
    CHECK(rec.planner_status ==
          common_cache_plan_planner_status::incomplete_evidence);
    server_cache_plan_preflight_view view;
    CHECK(server_cache_plan_preflight_build_view(rec, 0, true, view));
    CHECK(view.expected_path ==
          server_cache_plan_preflight_expected_path::legacy);
    CHECK(!view.provider_available);
}

static void test_as_if_completion_semantics() {
    const auto native = server_cache_plan_stage1_semantics_for(
        false, true, true, true, true);
    const auto literal_preflight = server_cache_plan_stage1_semantics_for(
        false, false, true, true, true);
    const auto as_if_preflight = server_cache_plan_stage1_semantics_for(
        true, false, true, true, true);
    CHECK(native.completion_semantics);
    CHECK(native.host_lookup_enabled);
    CHECK(native.recovery_citation ==
          common_cache_plan_recovery_citation::prospective);
    CHECK(!literal_preflight.completion_semantics);
    CHECK(!literal_preflight.host_lookup_enabled);
    CHECK(literal_preflight.recovery_citation ==
          common_cache_plan_recovery_citation::unavailable);
    CHECK(as_if_preflight.completion_semantics ==
          native.completion_semantics);
    CHECK(as_if_preflight.host_lookup_enabled == native.host_lookup_enabled);
    CHECK(as_if_preflight.recovery_citation == native.recovery_citation);
}

static void test_view_and_oracles() {
    auto rec = fitted_live_record();
    auto * rejected = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 3,
        COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 7,
        common_cache_plan_selection::similarity);
    CHECK(rejected != nullptr);
    rejected->note_reject(COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);
    auto * rejected_again = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 4,
        COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 7,
        common_cache_plan_selection::similarity);
    CHECK(rejected_again != nullptr);
    rejected_again->note_reject(
        COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);

    server_cache_plan_preflight_view view;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, true, view));
    CHECK(view.status == server_cache_plan_preflight_status::ok);
    CHECK(view.provider_available);
    CHECK(view.provider == common_cache_plan_provider::live_slot);
    CHECK(view.target_relation ==
          server_cache_plan_preflight_target_relation::same_as_legacy);
    CHECK(view.cache_hit == server_cache_plan_preflight_cache_hit::partial);
    CHECK(view.predicted_reuse_tokens.state == llama_cache_acct_known::known);
    CHECK(view.predicted_reuse_tokens.value == 96);
    CHECK(view.predicted_replay_tokens.value == 4);
    CHECK(view.predicted_ttft_us.value == 40);
    CHECK(view.miss_reasons.size() == 1);
    CHECK(view.miss_reasons[0].provider ==
          common_cache_plan_provider::host_cache_entry);
    CHECK(view.miss_reasons[0].reason ==
          COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);
    CHECK(view.miss_reasons[0].count == 2);

    rec.planner_status = common_cache_plan_planner_status::profile_unfitted;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, true, view));
    CHECK(view.planner_status ==
          common_cache_plan_planner_status::profile_unfitted);
    CHECK(view.expected_path ==
          server_cache_plan_preflight_expected_path::legacy);
    CHECK(!view.provider_available);
}

static void test_local_source_registry() {
    server_cache_plan_local_source_registry registry;
    int32_t first = -1;
    int32_t second = -1;
    int32_t repeated = -1;
    CHECK(registry.get_or_assign(0x1000, first));
    CHECK(registry.get_or_assign(0x2000, second));
    CHECK(registry.get_or_assign(0x1000, repeated));
    CHECK(first == 0);
    CHECK(second == 1);
    CHECK(repeated == first);
    CHECK(registry.size() == 2);
    int32_t found = -1;
    CHECK(registry.find(0x2000, found));
    CHECK(found == second);
    CHECK(!registry.find(0x3000, found));
    CHECK(found == -1);

    int32_t publish_next = 0;
    int32_t published_a = -1;
    int32_t published_b = -1;
    int32_t publish_a = -1;
    int32_t publish_b = -1;
    CHECK(server_cache_plan_assign_source_id(
        published_a, publish_next, publish_a));
    CHECK(server_cache_plan_assign_source_id(
        published_b, publish_next, publish_b));
    CHECK(publish_a == first);
    CHECK(publish_b == second);
}

static void assert_redacted_keys(const nlohmann::ordered_json & value) {
    // Canonical exhaustive E0 private-key oracle. The contract scan and live
    // driver carry deliberate security-critical subsets and point back here.
    static const std::set<std::string> forbidden = {
        "target_slot_id", "source_id", "candidate_id", "component_ids",
        "checkpoint_ordinal", "artifact_id", "victim_ids",
        "recovery_source", "manifest_digest", "effect_digest",
        "recovery_digest", "accounting_serial", "admission_sequence",
        "memo_key", "lease_holder", "lease_scope", "lease_expiry",
        "main_family", "device_ordinal", "topology_id", "domains",
        "journal_id", "re_request",
    };
    if (value.is_object()) {
        for (const auto & item : value.items()) {
            CHECK(forbidden.count(item.key()) == 0);
            assert_redacted_keys(item.value());
        }
    } else if (value.is_array()) {
        for (const auto & item : value) {
            assert_redacted_keys(item);
        }
    }
}

static void test_wire_serializer_and_golden() {
    auto rec = fitted_live_record();
    auto & selected = rec.inventory[size_t(rec.shadow_choice)];
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::restore)].raw =
        llama_cache_acct_value::measured(16);
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::restore)]
        .estimated_us = llama_cache_acct_value::measured(3);
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::restore)]
        .estimator_version = 7;
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::eviction)].raw =
        llama_cache_acct_value::measured(64);
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::eviction)]
        .estimated_us = llama_cache_acct_value::measured(9);
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::eviction)]
        .estimator_version = 7;
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::workspace)].raw =
        llama_cache_acct_value::measured(2);
    selected.cost_terms[size_t(llama_cache_acct_cost_kind::workspace)]
        .raw_unit = llama_cache_acct_unit::operations;
    selected.predicted_total_us = llama_cache_acct_value::measured(52);

    auto * rejected = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 91,
        COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 7,
        common_cache_plan_selection::similarity);
    CHECK(rejected != nullptr);
    rejected->note_reject(COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);

    rec.destruction.state = common_cache_plan_destruction_state::quoted;
    rec.destruction.reason = common_cache_plan_destruction_reason::none;
    rec.destruction.plan_candidate = rec.shadow_choice;
    rec.destruction.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::cross_target_displacement) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
    rec.destruction.lease_verdict =
        common_cache_plan_destruction_lease_verdict::soft_leased;
    rec.destruction.displaced_fate =
        common_cache_plan_displaced_fate::retained_host;
    rec.destruction.recovery_citation =
        common_cache_plan_recovery_citation::prospective;
    rec.destruction.selected_attention.push_back({ 987654321 });
    rec.destruction.recovery_source_artifact_id = { 876543210 };
    rec.destruction.manifest_digest =
        common_cache_plan_destruction_manifest_digest::from_sha256(
            std::array<uint8_t, 32>{ 0xab });
    common_cache_plan_destruction_quote quote;
    quote.receipt = rec.destruction;
    common_cache_plan_yield_domain domain;
    domain.projected_release_bytes = llama_cache_acct_value::measured(64);
    quote.projected_domains.push_back(domain);
    rec.destruction_quotes.push_back(std::move(quote));

    server_cache_plan_preflight_view view;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, true, view));
    const auto wire = server_cache_plan_preflight_json(view);
    CHECK(wire["object"] == "cache_plan_preflight");
    CHECK(wire["schema_version"] == 1);
    CHECK(wire["cache_plan_schema_version"] == 7);
    CHECK(wire["authoritative"] == false);
    CHECK(wire["reservation"] == "none");
    CHECK(wire["valid_until"].is_null());
    CHECK(wire["planner"]["expected_path"] ==
          "conditional_on_destruction_certification");
    CHECK(wire["planner"]["estimate_scope"] == "cache_path_only");
    CHECK(wire["planner"]["estimator_version"] == 7);
    CHECK(wire["planner"]["cost_terms"]["workspace"]["operations"] == 2);
    CHECK(!wire["planner"]["cost_terms"]["workspace"].contains("bytes"));
    CHECK(wire["destruction"]["effects"].size() == 2);
    CHECK(wire["limitations"].size() == 4);
    assert_redacted_keys(wire);
    const std::string encoded = wire.dump(2) + "\n";
    CHECK(encoded.find("987654321") == std::string::npos);
    CHECK(encoded.find("876543210") == std::string::npos);
    CHECK(encoded.find("ab000000") == std::string::npos);

    if (std::getenv("CACHE_PLAN_PRINT_PREFLIGHT_GOLDEN")) {
        std::fputs(encoded.c_str(), stdout);
        std::fflush(stdout);
        std::exit(EXIT_SUCCESS);
    }
#ifdef CACHE_PLAN_PREFLIGHT_GOLDEN_PATH
    std::ifstream golden(CACHE_PLAN_PREFLIGHT_GOLDEN_PATH);
    CHECK(golden.good());
    std::ostringstream expected;
    expected << golden.rdbuf();
    CHECK(encoded == expected.str());
#endif
}

static void test_exposure_gate() {
    CHECK(server_cache_plan_preflight_exposure_allowed("127.0.0.1", 0));
    CHECK(server_cache_plan_preflight_exposure_allowed("localhost", 1));
    CHECK(server_cache_plan_preflight_exposure_allowed("::1", 0));
    CHECK(server_cache_plan_preflight_exposure_allowed(
        "/tmp/llama.sock", 1));
    CHECK(!server_cache_plan_preflight_exposure_allowed("0.0.0.0", 0));
    CHECK(!server_cache_plan_preflight_exposure_allowed("127.0.0.1", 2));

    CHECK(server_cache_plan_preflight_request_field_allowed("prompt"));
    CHECK(server_cache_plan_preflight_request_field_allowed("id_slot"));
    CHECK(server_cache_plan_preflight_request_field_allowed("cache_prompt"));
    CHECK(server_cache_plan_preflight_request_field_allowed("lora"));
    CHECK(server_cache_plan_preflight_request_field_allowed(
        "message_delimiters"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("sampling"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("ticket"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("claim"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("preview_id"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("nonce"));
    CHECK(!server_cache_plan_preflight_request_field_allowed(
        "manifest_digest"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("artifact_id"));
}

int main() {
    test_expected_path_closed_set();
    test_view_and_oracles();
    test_destruction_view_mapping();
    test_saturated_inventory_refuses_typed();
    test_as_if_completion_semantics();
    test_local_source_registry();
    test_wire_serializer_and_golden();
    test_exposure_gate();
    test_gcp_dispatch_excludes_preflight();
    std::puts("test-cache-plan-preflight: PASS");
    return 0;
}
