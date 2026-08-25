// Policy-neutral budget arithmetic: closed category classification,
// durable+reserved accounting, hierarchical constraints, and fail-closed plans.

#include "llama-cache-budget.h"

#include <cstdio>
#include <limits>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static llama_cache_acct_resource_domain device_domain(uint16_t ordinal, uint32_t topology) {
    llama_cache_acct_resource_domain out;
    out.residency = llama_cache_acct_residency::device;
    out.kind = llama_cache_acct_domain_kind::device_topology;
    out.device_ordinal = { ordinal };
    out.topology = { topology };
    return out;
}

static void add_cell(
        llama_cache_acct_snapshot & snapshot,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        uint64_t resident,
        uint64_t reserved = 0,
        uint64_t transient_peak = 0,
        llama_cache_acct_known state = llama_cache_acct_known::known) {
    llama_cache_acct_cell_row row;
    row.category = category;
    row.domain = domain;
    row.certification = state;
    row.cell.measures[size_t(llama_cache_acct_measure::resident_allocated)] =
        { resident, state };
    row.cell.measures[size_t(llama_cache_acct_measure::reserved)] =
        { reserved, state };
    row.cell.measures[size_t(llama_cache_acct_measure::transient_peak)] =
        { transient_peak, state };
    snapshot.cells.push_back(row);
}

static void add_domain(
        llama_cache_acct_snapshot & snapshot,
        const llama_cache_acct_resource_domain & domain) {
    snapshot.completeness.push_back({
        domain, llama_cache_acct_producer::observer_init,
        llama_cache_acct_known::known,
    });
}

static llama_cache_acct_snapshot base_snapshot() {
    const auto d0 = device_domain(0, 1);
    const auto d1 = device_domain(0, 2);
    const auto pageable = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);

    llama_cache_acct_snapshot snapshot;
    snapshot.serial = 17;
    snapshot.completeness_manifest = llama_cache_acct_known::known;
    for (const auto & domain : { d0, d1, pageable }) {
        add_domain(snapshot, domain);
    }
    add_cell(snapshot, llama_cache_acct_category::live_attention_state,
             d0, 100, 77, 900);
    add_cell(snapshot, llama_cache_acct_category::live_recurrent_state,
             d1, 50);
    add_cell(snapshot, llama_cache_acct_category::full_snapshot_payload,
             pageable, 20, 5);
    return snapshot;
}

static llama_cache_budget_config base_config() {
    const void * device = reinterpret_cast<const void *>(uintptr_t(1));
    llama_cache_budget_device_input first;
    first.backend_device = device;
    first.domain = device_domain(0, 1);
    first.physical_total = 1000;
    first.physical_free = 400;
    first.phys_state = llama_cache_budget_capacity_state::known;
    first.current_compute_allocated = 100;
    first.configured_compute_reserve = 150;
    first.compute_state = llama_cache_budget_capacity_state::known;
    first.cache_cap_state = llama_cache_budget_capacity_state::unbounded;

    llama_cache_budget_device_input second = first;
    second.domain = device_domain(0, 2);

    llama_cache_budget_config config;
    config.devices = { first, second };
    config.host.pageable_cap = 100;
    config.host.pageable_state = llama_cache_budget_capacity_state::known;
    config.host.pinned_cap = 10;
    config.host.pinned_state = llama_cache_budget_capacity_state::known;
    return config;
}

static const llama_cache_budget_row * find_group(
        const llama_cache_budget_result & result,
        llama_cache_budget_resource_kind kind,
        llama_cache_acct_residency residency =
            llama_cache_acct_residency::not_applicable) {
    for (const auto & row : result.groups) {
        if (row.resource.kind == kind &&
            (kind != llama_cache_budget_resource_kind::host_residency ||
             row.resource.residency == residency)) {
            return &row;
        }
    }
    return nullptr;
}

static const llama_cache_budget_row * find_domain_row(
        const llama_cache_budget_result & result,
        const llama_cache_acct_resource_domain & domain) {
    for (const auto & row : result.domains) {
        if (row.resource.domain == domain) {
            return &row;
        }
    }
    return nullptr;
}

static void test_baseline_and_group_rollup() {
    auto snapshot = base_snapshot();
    auto config = base_config();
    llama_cache_budget_coordinator coordinator;
    CHECK(coordinator.reset(snapshot, config));

    llama_cache_budget_plan plan;
    plan.accounting_serial = snapshot.serial;
    auto result = coordinator.fits(plan);
    CHECK(result.state == llama_cache_budget_fit_state::fits);
    const auto * device = find_group(
        result, llama_cache_budget_resource_kind::physical_device);
    CHECK(device != nullptr);
    if (device) {
        CHECK(device->current_resident.value == 150);
        CHECK(device->before.value == 150);
        // total 1000 - other(600 used - cache 150 - compute 100) - reserve 150
        CHECK(device->ceiling.value == 500);
        CHECK(device->headroom_after.value == 350);
    }
    const auto * d0 = find_domain_row(result, device_domain(0, 1));
    CHECK(d0 && d0->state == llama_cache_budget_fit_state::fits);
    CHECK(d0 && d0->after.value == 100);
    CHECK(d0 && d0->headroom_after.value == 400);
    const auto * pageable = find_group(
        result, llama_cache_budget_resource_kind::host_residency,
        llama_cache_acct_residency::pageable_host);
    CHECK(pageable && pageable->before.value == 25);
    // Historical staging high-water (900) is not current occupancy.
    CHECK(device && device->before.value != 1050);

    plan.entries.push_back({ device_domain(0, 1), 351, 0 });
    result = coordinator.fits(plan);
    device = find_group(result, llama_cache_budget_resource_kind::physical_device);
    CHECK(device && device->state == llama_cache_budget_fit_state::exceeds);
    // The member alone remains below its group ceiling. This proves group roll-up,
    // rather than a coincidental leaf failure, caused the aggregate rejection.
    CHECK(!result.domains.empty() &&
          result.domains[0].state == llama_cache_budget_fit_state::fits);
    d0 = find_domain_row(result, device_domain(0, 1));
    CHECK(d0 && d0->after.value == 451);
    CHECK(d0 && d0->headroom_after.value == 49);
    CHECK(result.state == llama_cache_budget_fit_state::exceeds);

    plan.entries[0].reserve_bytes = 401;
    result = coordinator.fits(plan);
    d0 = find_domain_row(result, device_domain(0, 1));
    CHECK(d0 && d0->state == llama_cache_budget_fit_state::exceeds);
    CHECK(d0 && d0->after.value == 501);
    CHECK(d0 && d0->headroom_state ==
          llama_cache_budget_capacity_state::unavailable);
}

static void test_optional_hierarchy() {
    auto snapshot = base_snapshot();
    auto config = base_config();
    config.host.total_cap = 30;
    config.host.total_state = llama_cache_budget_capacity_state::known;
    config.administrative_global_cap = 525;
    config.global_cap_state = llama_cache_budget_capacity_state::known;

    llama_cache_budget_coordinator coordinator;
    CHECK(coordinator.reset(snapshot, config));
    llama_cache_budget_plan plan { snapshot.serial, {} };
    auto result = coordinator.fits(plan);
    const auto * host_total = find_group(
        result, llama_cache_budget_resource_kind::host_total);
    const auto * global = find_group(
        result, llama_cache_budget_resource_kind::administrative_global);
    CHECK(host_total && host_total->before.value == 25);
    CHECK(global && global->before.value == 175);
    CHECK(result.state == llama_cache_budget_fit_state::fits);

    config.host.total_state = llama_cache_budget_capacity_state::unbounded;
    config.global_cap_state = llama_cache_budget_capacity_state::unbounded;
    CHECK(coordinator.reset(snapshot, config));
    result = coordinator.fits(plan);
    CHECK(find_group(result, llama_cache_budget_resource_kind::host_total) == nullptr);
    CHECK(find_group(result,
          llama_cache_budget_resource_kind::administrative_global) == nullptr);
}

static void test_fail_closed_inputs() {
    auto snapshot = base_snapshot();
    auto config = base_config();
    CHECK(config.devices[0].reserve_provenance ==
          llama_cache_budget_reserve_provenance::configured);
    llama_cache_budget_coordinator coordinator;
    CHECK(coordinator.reset(snapshot, config));

    llama_cache_budget_plan stale { snapshot.serial + 1, {} };
    CHECK(coordinator.fits(stale).state ==
          llama_cache_budget_fit_state::unavailable);

    llama_cache_budget_plan duplicate { snapshot.serial, {
        { device_domain(0, 1), 0, 0 },
        { device_domain(0, 1), 0, 0 },
    } };
    CHECK(coordinator.fits(duplicate).state ==
          llama_cache_budget_fit_state::unavailable);

    llama_cache_budget_plan underflow { snapshot.serial, {
        { device_domain(0, 1), 0, 101 },
    } };
    CHECK(coordinator.fits(underflow).state ==
          llama_cache_budget_fit_state::unavailable);

    auto unknown = snapshot;
    unknown.cells[0].cell.measures[
        size_t(llama_cache_acct_measure::resident_allocated)] =
            { 0, llama_cache_acct_known::unknown };
    CHECK(coordinator.reset(unknown, config));
    llama_cache_budget_plan empty { unknown.serial, {} };
    CHECK(coordinator.fits(empty).state ==
          llama_cache_budget_fit_state::unavailable);
    const auto unknown_result = coordinator.fits(empty);
    const auto * unknown_d0 =
        find_domain_row(unknown_result, device_domain(0, 1));
    CHECK(unknown_d0 &&
          unknown_d0->state == llama_cache_budget_fit_state::unavailable);

    auto overflow = snapshot;
    overflow.cells[0].cell.measures[
        size_t(llama_cache_acct_measure::resident_allocated)] =
            llama_cache_acct_value::measured(
                std::numeric_limits<uint64_t>::max());
    add_cell(overflow, llama_cache_acct_category::rolling_window_tape,
             device_domain(0, 1), 1);
    CHECK(coordinator.reset(overflow, config));
    empty.accounting_serial = overflow.serial;
    CHECK(coordinator.fits(empty).state ==
          llama_cache_budget_fit_state::unavailable);

    config = base_config();
    for (auto & input : config.devices) {
        input.configured_compute_reserve =
            input.current_compute_allocated - 1;
    }
    CHECK(coordinator.reset(snapshot, config));
    empty.accounting_serial = snapshot.serial;
    CHECK(coordinator.fits(empty).state ==
          llama_cache_budget_fit_state::unavailable);

    config = base_config();
    for (auto & input : config.devices) {
        input.reserve_provenance =
            llama_cache_budget_reserve_provenance::measured;
    }
    CHECK(coordinator.reset(snapshot, config));
    CHECK(coordinator.fits(empty).state ==
          llama_cache_budget_fit_state::unavailable);

    config = base_config();
    for (auto & input : config.devices) {
        input.phys_state = llama_cache_budget_capacity_state::unavailable;
    }
    CHECK(coordinator.reset(snapshot, config));
    CHECK(coordinator.fits(empty).state ==
          llama_cache_budget_fit_state::unavailable);

    config = base_config();
    for (auto & input : config.devices) {
        input.configured_cache_cap = 140;
        input.cache_cap_state = llama_cache_budget_capacity_state::known;
    }
    CHECK(coordinator.reset(snapshot, config));
    CHECK(coordinator.fits(empty).state ==
          llama_cache_budget_fit_state::exceeds);
}

static void test_capacity_states_and_classification() {
    auto snapshot = base_snapshot();
    auto config = base_config();
    config.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    llama_cache_budget_coordinator coordinator;
    CHECK(coordinator.reset(snapshot, config));
    llama_cache_budget_plan plan { snapshot.serial, {
        { llama_cache_acct_resource_domain::non_device(
              llama_cache_acct_residency::pageable_host),
          std::numeric_limits<uint64_t>::max() - 25, 0 },
    } };
    CHECK(coordinator.fits(plan).state ==
          llama_cache_budget_fit_state::fits);

    config.host.pageable_state =
        llama_cache_budget_capacity_state::unavailable;
    CHECK(coordinator.reset(snapshot, config));
    CHECK(coordinator.fits({ snapshot.serial, {} }).state ==
          llama_cache_budget_fit_state::unavailable);

    CHECK(llama_cache_budget_classify(
              llama_cache_acct_category::live_attention_state).participation ==
          llama_cache_budget_capacity_participation::participating);
    for (const auto category : {
            llama_cache_acct_category::unit_version_payload,
            llama_cache_acct_category::clean_stash_payload,
            llama_cache_acct_category::artifact_descriptor_metadata,
            llama_cache_acct_category::artifact_reference_metadata,
            llama_cache_acct_category::transfer_staging,
            llama_cache_acct_category::codec_workspace,
            llama_cache_acct_category::pinned_preimage_ring }) {
        const auto row = llama_cache_budget_classify(category);
        CHECK(row.participation ==
              llama_cache_budget_capacity_participation::participating);
        CHECK(row.scope ==
              llama_cache_budget_residency_scope::by_domain);
    }
    CHECK(llama_cache_budget_classify(
              llama_cache_acct_category::unit_version_payload).scope ==
          llama_cache_budget_residency_scope::by_domain);
    CHECK(llama_cache_budget_classify(
              llama_cache_acct_category::full_snapshot_payload).mode ==
          llama_cache_budget_accounting_mode::transactional);
}

static void test_f2_capacity_activation_is_inert_until_observed() {
    auto baseline_snapshot = base_snapshot();
    const auto config = base_config();
    llama_cache_budget_coordinator coordinator;
    CHECK(coordinator.reset(baseline_snapshot, config));
    const llama_cache_budget_plan baseline_plan {
        baseline_snapshot.serial, {},
    };
    const auto baseline = coordinator.fits(baseline_plan);

    auto dormant_snapshot = baseline_snapshot;
    for (const auto category : {
            llama_cache_acct_category::unit_version_payload,
            llama_cache_acct_category::clean_stash_payload,
            llama_cache_acct_category::artifact_descriptor_metadata,
            llama_cache_acct_category::artifact_reference_metadata,
            llama_cache_acct_category::transfer_staging,
            llama_cache_acct_category::codec_workspace,
            llama_cache_acct_category::pinned_preimage_ring }) {
        add_cell(
            dormant_snapshot, category, device_domain(0, 1),
            0, 0, 0, llama_cache_acct_known::unknown);
    }
    CHECK(coordinator.reset(dormant_snapshot, config));
    const auto dormant = coordinator.fits({
        dormant_snapshot.serial, {},
    });
    const auto * baseline_device = find_group(
        baseline, llama_cache_budget_resource_kind::physical_device);
    const auto * dormant_device = find_group(
        dormant, llama_cache_budget_resource_kind::physical_device);
    CHECK(dormant.state == baseline.state);
    CHECK(dormant.domains.size() == baseline.domains.size());
    CHECK(dormant.groups.size() == baseline.groups.size());
    CHECK(baseline_device && dormant_device);
    CHECK(baseline_device && dormant_device &&
          dormant_device->before.value == baseline_device->before.value &&
          dormant_device->before.state == baseline_device->before.state);
    CHECK(baseline_device && dormant_device &&
          dormant_device->headroom_after.value ==
              baseline_device->headroom_after.value &&
          dormant_device->headroom_after.state ==
              baseline_device->headroom_after.state);

    auto active_snapshot = dormant_snapshot;
    auto & active = active_snapshot.cells[
        baseline_snapshot.cells.size()];
    active.certification = llama_cache_acct_known::known;
    active.cell.measures[
        size_t(llama_cache_acct_measure::resident_allocated)] =
            llama_cache_acct_value::measured(7);
    active.cell.measures[
        size_t(llama_cache_acct_measure::reserved)] =
            llama_cache_acct_value::measured(3);
    CHECK(coordinator.reset(active_snapshot, config));
    const auto active_result = coordinator.fits({
        active_snapshot.serial, {},
    });
    const auto * active_device = find_group(
        active_result, llama_cache_budget_resource_kind::physical_device);
    CHECK(active_result.state == llama_cache_budget_fit_state::fits);
    CHECK(active_device && baseline_device &&
          active_device->before.value == baseline_device->before.value + 10);

    active.cell.measures[
        size_t(llama_cache_acct_measure::reserved)] = {};
    CHECK(coordinator.reset(active_snapshot, config));
    CHECK(coordinator.fits({ active_snapshot.serial, {} }).state ==
          llama_cache_budget_fit_state::unavailable);

    active.cell.measures[
        size_t(llama_cache_acct_measure::resident_allocated)] = {};
    active.cell.measures[
        size_t(llama_cache_acct_measure::reserved)] =
            llama_cache_acct_value::measured(3);
    CHECK(coordinator.reset(active_snapshot, config));
    CHECK(coordinator.fits({ active_snapshot.serial, {} }).state ==
          llama_cache_budget_fit_state::unavailable);

    auto pinned_snapshot = baseline_snapshot;
    const auto pinned = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pinned_host);
    add_domain(pinned_snapshot, pinned);
    add_cell(
        pinned_snapshot,
        llama_cache_acct_category::pinned_preimage_ring,
        pinned, 11);
    CHECK(coordinator.reset(pinned_snapshot, config));
    CHECK(coordinator.fits({ pinned_snapshot.serial, {} }).state ==
          llama_cache_budget_fit_state::exceeds);

    const auto disk = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::disk);
    llama_cache_budget_plan unsupported {
        baseline_snapshot.serial, { { disk, 1, 0 } },
    };
    CHECK(coordinator.reset(baseline_snapshot, config));
    CHECK(coordinator.fits(unsupported).state ==
          llama_cache_budget_fit_state::unavailable);
}

int main() {
    test_baseline_and_group_rollup();
    test_optional_hierarchy();
    test_fail_closed_inputs();
    test_capacity_states_and_classification();
    test_f2_capacity_activation_is_inert_until_observed();
    if (failures) {
        std::fprintf(stderr, "%d cache-budget checks failed\n", failures);
        return 1;
    }
    std::puts("cache-budget checks passed");
    return 0;
}
