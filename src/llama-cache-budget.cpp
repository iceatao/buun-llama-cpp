#include "llama-cache-budget.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace {

bool add_checked(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool sub_checked(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > a) {
        return false;
    }
    out = a - b;
    return true;
}

bool valid_capacity_state(llama_cache_budget_capacity_state state) {
    return state >= llama_cache_budget_capacity_state::known &&
           state <  llama_cache_budget_capacity_state::_count;
}

bool valid_domain_for_budget(const llama_cache_acct_resource_domain & domain) {
    if (!llama_cache_acct_resource_domain_valid(domain)) {
        return false;
    }
    switch (domain.residency) {
        case llama_cache_acct_residency::device:
        case llama_cache_acct_residency::pinned_host:
        case llama_cache_acct_residency::pageable_host:
            return true;
        case llama_cache_acct_residency::disk:
        case llama_cache_acct_residency::remote:
        case llama_cache_acct_residency::not_applicable:
        case llama_cache_acct_residency::_count:
            return false;
    }
    return false;
}

llama_cache_acct_value unavailable_value() {
    return { 0, llama_cache_acct_known::unavailable };
}

llama_cache_budget_fit_state combine_state(
        llama_cache_budget_fit_state a,
        llama_cache_budget_fit_state b) {
    if (a == llama_cache_budget_fit_state::unavailable ||
        b == llama_cache_budget_fit_state::unavailable) {
        return llama_cache_budget_fit_state::unavailable;
    }
    if (a == llama_cache_budget_fit_state::exceeds ||
        b == llama_cache_budget_fit_state::exceeds) {
        return llama_cache_budget_fit_state::exceeds;
    }
    return llama_cache_budget_fit_state::fits;
}

struct domain_calc {
    llama_cache_acct_resource_domain domain;
    const void * backend_device = nullptr;
    uint64_t resident = 0;
    uint64_t outstanding_reserved = 0;
    uint64_t plan_reserved = 0;
    uint64_t plan_released = 0;
    bool available = true;
    bool saw_cell = false;
    bool completeness_known = false;
};

domain_calc * find_domain(
        std::vector<domain_calc> & domains,
        const llama_cache_acct_resource_domain & domain) {
    auto it = std::find_if(domains.begin(), domains.end(),
            [&](const domain_calc & row) { return row.domain == domain; });
    return it == domains.end() ? nullptr : &*it;
}

bool same_device_numbers(
        const llama_cache_budget_device_input & a,
        const llama_cache_budget_device_input & b) {
    return a.physical_total == b.physical_total &&
           a.physical_free == b.physical_free &&
           a.phys_state == b.phys_state &&
           a.current_compute_allocated == b.current_compute_allocated &&
           a.configured_compute_reserve == b.configured_compute_reserve &&
           a.reserve_provenance == b.reserve_provenance &&
           a.compute_state == b.compute_state &&
           a.configured_cache_cap == b.configured_cache_cap &&
           a.cache_cap_state == b.cache_cap_state;
}

void set_ceiling(
        llama_cache_budget_row & row,
        llama_cache_budget_capacity_state state,
        uint64_t value) {
    row.ceiling_state = state;
    if (state == llama_cache_budget_capacity_state::known) {
        row.ceiling = llama_cache_acct_value::measured(value);
    } else {
        row.ceiling = unavailable_value();
    }
}

void populate_values(
        llama_cache_budget_row & row,
        bool inputs_available,
        uint64_t current_resident,
        uint64_t before,
        uint64_t released,
        uint64_t reserved) {
    if (!inputs_available) {
        row.current_resident = unavailable_value();
        row.before = unavailable_value();
        row.released = unavailable_value();
        row.reserved = unavailable_value();
        row.after = unavailable_value();
        row.headroom_after = unavailable_value();
        row.headroom_state = llama_cache_budget_capacity_state::unavailable;
        row.state = llama_cache_budget_fit_state::unavailable;
        return;
    }
    row.current_resident = llama_cache_acct_value::measured(current_resident);
    row.before           = llama_cache_acct_value::measured(before);
    row.released         = llama_cache_acct_value::measured(released);
    row.reserved         = llama_cache_acct_value::measured(reserved);

    uint64_t after_release = 0;
    uint64_t after = 0;
    if (!sub_checked(before, released, after_release) ||
        !add_checked(after_release, reserved, after)) {
        row.after = unavailable_value();
        row.headroom_after = unavailable_value();
        row.headroom_state = llama_cache_budget_capacity_state::unavailable;
        row.state = llama_cache_budget_fit_state::unavailable;
        return;
    }
    row.after = llama_cache_acct_value::measured(after);
}

void apply_ceiling(llama_cache_budget_row & row) {
    if (row.after.state != llama_cache_acct_known::known) {
        row.headroom_after = unavailable_value();
        row.headroom_state = llama_cache_budget_capacity_state::unavailable;
        row.state = llama_cache_budget_fit_state::unavailable;
        return;
    }
    switch (row.ceiling_state) {
        case llama_cache_budget_capacity_state::known:
            if (row.after.value > row.ceiling.value) {
                row.headroom_after = unavailable_value();
                row.headroom_state = llama_cache_budget_capacity_state::unavailable;
                row.state = llama_cache_budget_fit_state::exceeds;
            } else {
                row.headroom_after =
                    llama_cache_acct_value::measured(
                        row.ceiling.value - row.after.value);
                row.headroom_state = llama_cache_budget_capacity_state::known;
                row.state = llama_cache_budget_fit_state::fits;
            }
            return;
        case llama_cache_budget_capacity_state::unbounded:
            row.headroom_after = unavailable_value();
            row.headroom_state = llama_cache_budget_capacity_state::unbounded;
            row.state = llama_cache_budget_fit_state::fits;
            return;
        case llama_cache_budget_capacity_state::unavailable:
        case llama_cache_budget_capacity_state::_count:
            row.headroom_after = unavailable_value();
            row.headroom_state = llama_cache_budget_capacity_state::unavailable;
            row.state = llama_cache_budget_fit_state::unavailable;
            return;
    }
}

struct rollup_values {
    uint64_t current_resident = 0;
    uint64_t before = 0;
    uint64_t released = 0;
    uint64_t reserved = 0;
    bool available = true;
};

rollup_values sum_row_values(
        const std::vector<llama_cache_budget_row *> & rows,
        bool additional_available = true) {
    rollup_values out;
    out.available = additional_available;
    for (const llama_cache_budget_row * row : rows) {
        if (!row ||
            row->current_resident.state != llama_cache_acct_known::known ||
            row->before.state != llama_cache_acct_known::known ||
            row->released.state != llama_cache_acct_known::known ||
            row->reserved.state != llama_cache_acct_known::known) {
            out.available = false;
            continue;
        }
        if (!add_checked(out.current_resident, row->current_resident.value,
                         out.current_resident) ||
            !add_checked(out.before, row->before.value, out.before) ||
            !add_checked(out.released, row->released.value, out.released) ||
            !add_checked(out.reserved, row->reserved.value, out.reserved)) {
            out.available = false;
        }
    }
    return out;
}

llama_cache_budget_row & add_rollup_group(
        std::vector<llama_cache_budget_row> & groups,
        llama_cache_budget_resource_kind kind,
        llama_cache_acct_residency residency,
        const void * backend_device,
        const llama_cache_acct_resource_domain & domain,
        const rollup_values & values,
        llama_cache_budget_capacity_state ceiling_state,
        uint64_t ceiling) {
    llama_cache_budget_row row;
    row.resource.kind = kind;
    row.resource.residency = residency;
    row.resource.backend_device = backend_device;
    row.resource.domain = domain;
    set_ceiling(row, ceiling_state, ceiling);
    populate_values(row, values.available, values.current_resident,
                    values.before, values.released, values.reserved);
    apply_ceiling(row);
    groups.push_back(std::move(row));
    return groups.back();
}

} // namespace

llama_cache_budget_category_classification llama_cache_budget_classify(
        llama_cache_acct_category category) noexcept {
    switch (category) {
#define LLAMA_CACHE_BUDGET_CLASSIFY_CASE(name, participation, mode, scope)       \
        case llama_cache_acct_category::name:                                    \
            return { llama_cache_budget_capacity_participation::participation,   \
                     llama_cache_budget_accounting_mode::mode,                   \
                     llama_cache_budget_residency_scope::scope };
        LLAMA_CACHE_BUDGET_CATEGORY_TABLE(LLAMA_CACHE_BUDGET_CLASSIFY_CASE)
#undef LLAMA_CACHE_BUDGET_CLASSIFY_CASE
        case llama_cache_acct_category::_count:
            break;
    }
    return { llama_cache_budget_capacity_participation::_count,
             llama_cache_budget_accounting_mode::_count,
             llama_cache_budget_residency_scope::_count };
}

bool llama_cache_budget_coordinator::reset(
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_budget_config & config) noexcept {
    try {
        return reset(llama_cache_acct_snapshot(snapshot), config);
    } catch (...) {
        configured_ = false;
        snapshot_ = {};
        config_ = {};
        return false;
    }
}

bool llama_cache_budget_coordinator::reset(
        llama_cache_acct_snapshot && snapshot,
        const llama_cache_budget_config & config) noexcept {
    configured_ = false;
    try {
        snapshot_ = std::move(snapshot);
        config_ = config;
        configured_ = true;
    } catch (...) {
        snapshot_ = {};
        config_ = {};
    }
    return configured_;
}

llama_cache_budget_result llama_cache_budget_coordinator::fits(
        const llama_cache_budget_plan & plan) const noexcept {
    llama_cache_budget_result out;
    out.accounting_serial = snapshot_.serial;

    try {
        if (!configured_ ||
            snapshot_.schema_version != LLAMA_CACHE_ACCT_SCHEMA_VERSION ||
            plan.accounting_serial != snapshot_.serial ||
            snapshot_.completeness_manifest != llama_cache_acct_known::known) {
            return out;
        }

        std::vector<domain_calc> domains;
        domains.reserve(config_.devices.size() + 2);

        for (const auto & input : config_.devices) {
            if (!input.backend_device || !valid_domain_for_budget(input.domain) ||
                input.domain.residency != llama_cache_acct_residency::device ||
                !valid_capacity_state(input.phys_state) ||
                !valid_capacity_state(input.compute_state) ||
                !valid_capacity_state(input.cache_cap_state)) {
                return out;
            }
            domain_calc * existing = find_domain(domains, input.domain);
            if (existing) {
                return out;
            }
            domains.push_back({ input.domain, input.backend_device });
        }

        for (const auto & cell : snapshot_.cells) {
            if (!valid_domain_for_budget(cell.domain)) {
                continue;
            }
            domain_calc * domain = find_domain(domains, cell.domain);
            if (!domain) {
                domains.push_back({ cell.domain });
                domain = &domains.back();
            }

            const auto classification = llama_cache_budget_classify(cell.category);
            if (classification.participation ==
                    llama_cache_budget_capacity_participation::_count) {
                return out;
            }
            if (classification.participation !=
                    llama_cache_budget_capacity_participation::participating) {
                continue;
            }
            const bool by_domain =
                classification.scope ==
                    llama_cache_budget_residency_scope::by_domain;
            const bool device_scope =
                (classification.scope ==
                     llama_cache_budget_residency_scope::device ||
                 by_domain) &&
                cell.domain.residency ==
                    llama_cache_acct_residency::device;
            const bool host_scope =
                (classification.scope ==
                     llama_cache_budget_residency_scope::host ||
                 by_domain) &&
                (cell.domain.residency ==
                    llama_cache_acct_residency::pinned_host ||
                 cell.domain.residency ==
                    llama_cache_acct_residency::pageable_host);
            if (!device_scope && !host_scope) {
                continue;
            }
            // Artifact-capacity rows are dormant until their producer observes at
            // least one capacity measure. This preserves the pre-artifact budget
            // surface when artifact machinery is not allocated, while a
            // partially-observed active row still fails closed below.
            const auto resident =
                cell.cell.measures[size_t(llama_cache_acct_measure::resident_allocated)];
            const auto reserved =
                cell.cell.measures[size_t(llama_cache_acct_measure::reserved)];
            if (by_domain &&
                resident.state == llama_cache_acct_known::unknown &&
                (classification.mode !=
                     llama_cache_budget_accounting_mode::transactional ||
                 reserved.state == llama_cache_acct_known::unknown)) {
                continue;
            }
            domain->saw_cell = true;
            if (cell.certification != llama_cache_acct_known::known) {
                domain->available = false;
                continue;
            }
            if (resident.state != llama_cache_acct_known::known ||
                !add_checked(domain->resident, resident.value, domain->resident)) {
                domain->available = false;
            }
            if (classification.mode == llama_cache_budget_accounting_mode::transactional) {
                if (reserved.state != llama_cache_acct_known::known ||
                    !add_checked(domain->outstanding_reserved, reserved.value,
                                 domain->outstanding_reserved)) {
                    domain->available = false;
                }
            }
        }

        for (auto & domain : domains) {
            bool saw_requirement = false;
            bool all_known = true;
            for (const auto & row : snapshot_.completeness) {
                if (row.domain != domain.domain) {
                    continue;
                }
                saw_requirement = true;
                all_known = all_known && row.state == llama_cache_acct_known::known;
            }
            domain.completeness_known = saw_requirement && all_known;
            domain.available = domain.available &&
                               domain.saw_cell &&
                               domain.completeness_known;
        }

        for (const auto & entry : plan.entries) {
            if (!valid_domain_for_budget(entry.domain)) {
                return out;
            }
            domain_calc * domain = find_domain(domains, entry.domain);
            if (!domain) {
                return out;
            }
            if (domain->plan_reserved != 0 || domain->plan_released != 0) {
                return out;
            }
            // A zero/zero entry still owns the domain key and therefore participates
            // in duplicate rejection.
            domain->plan_reserved = entry.reserve_bytes;
            domain->plan_released = entry.release_bytes;
        }
        for (size_t i = 0; i < plan.entries.size(); ++i) {
            for (size_t j = i + 1; j < plan.entries.size(); ++j) {
                if (plan.entries[i].domain == plan.entries[j].domain) {
                    return out;
                }
            }
        }

        out.domains.reserve(domains.size());
        out.groups.reserve(config_.devices.size() + 4);
        for (const auto & calc : domains) {
            llama_cache_budget_row row;
            row.resource.kind = llama_cache_budget_resource_kind::accounting_domain;
            row.resource.domain = calc.domain;
            row.resource.backend_device = calc.backend_device;
            row.resource.residency = calc.domain.residency;

            uint64_t before = 0;
            bool available = calc.available &&
                add_checked(calc.resident, calc.outstanding_reserved, before);
            if (calc.domain.residency != llama_cache_acct_residency::device) {
                // Host caps constrain their residency roll-up, not this 1:1
                // accounting-domain leaf a second time.
                set_ceiling(row, llama_cache_budget_capacity_state::unbounded, 0);
            }
            populate_values(row, available, calc.resident, before,
                            calc.plan_released, calc.plan_reserved);
            if (calc.domain.residency != llama_cache_acct_residency::device) {
                apply_ceiling(row);
            }
            out.domains.push_back(std::move(row));
        }

        // Physical device groups derive one cache ceiling from the point-in-time
        // capacity sample, then constrain the sum of all topology-qualified domains
        // bound to that physical device.
        for (size_t i = 0; i < config_.devices.size(); ++i) {
            const auto & input = config_.devices[i];
            bool already = false;
            for (size_t j = 0; j < i; ++j) {
                already = already ||
                    config_.devices[j].backend_device == input.backend_device;
            }
            if (already) {
                continue;
            }

            std::vector<llama_cache_budget_row *> members;
            bool consistent = true;
            for (size_t j = i; j < config_.devices.size(); ++j) {
                const auto & candidate = config_.devices[j];
                if (candidate.backend_device != input.backend_device) {
                    continue;
                }
                consistent = consistent && same_device_numbers(input, candidate);
                auto it = std::find_if(out.domains.begin(), out.domains.end(),
                        [&](const llama_cache_budget_row & row) {
                            return row.resource.domain == candidate.domain;
                        });
                if (it == out.domains.end()) {
                    consistent = false;
                } else {
                    members.push_back(&*it);
                }
            }

            const rollup_values values =
                sum_row_values(members, consistent);

            llama_cache_budget_capacity_state ceiling_state =
                llama_cache_budget_capacity_state::unavailable;
            uint64_t ceiling = 0;
            if (values.available &&
                input.phys_state == llama_cache_budget_capacity_state::known &&
                input.compute_state == llama_cache_budget_capacity_state::known &&
                input.reserve_provenance ==
                    llama_cache_budget_reserve_provenance::configured &&
                input.physical_free <= input.physical_total &&
                input.configured_compute_reserve >=
                    input.current_compute_allocated) {
                uint64_t used = 0;
                uint64_t cache_and_compute = 0;
                uint64_t other_noncache = 0;
                uint64_t before_reserve = 0;
                uint64_t derived = 0;
                if (sub_checked(input.physical_total, input.physical_free, used) &&
                    add_checked(values.current_resident,
                                input.current_compute_allocated,
                                cache_and_compute) &&
                    sub_checked(used, cache_and_compute, other_noncache) &&
                    sub_checked(input.physical_total, other_noncache,
                                before_reserve) &&
                    sub_checked(before_reserve,
                                input.configured_compute_reserve, derived)) {
                    ceiling = derived;
                    ceiling_state = llama_cache_budget_capacity_state::known;
                    if (input.cache_cap_state ==
                            llama_cache_budget_capacity_state::known) {
                        ceiling = std::min(ceiling, input.configured_cache_cap);
                    } else if (input.cache_cap_state ==
                            llama_cache_budget_capacity_state::unavailable) {
                        ceiling_state =
                            llama_cache_budget_capacity_state::unavailable;
                    }
                }
            }

            for (auto * member : members) {
                set_ceiling(*member, ceiling_state, ceiling);
                apply_ceiling(*member);
            }

            add_rollup_group(
                out.groups,
                llama_cache_budget_resource_kind::physical_device,
                llama_cache_acct_residency::device,
                input.backend_device,
                input.domain,
                values,
                ceiling_state,
                ceiling);
        }

        // Pinned and pageable host are independently constrained.
        std::vector<llama_cache_budget_row *> host_groups;
        for (const auto residency : {
                llama_cache_acct_residency::pinned_host,
                llama_cache_acct_residency::pageable_host }) {
            std::vector<llama_cache_budget_row *> members;
            for (auto & row : out.domains) {
                if (row.resource.residency == residency) {
                    members.push_back(&row);
                }
            }
            const rollup_values values = sum_row_values(members);
            llama_cache_budget_capacity_state ceiling_state;
            uint64_t ceiling;
            if (residency == llama_cache_acct_residency::pinned_host) {
                ceiling_state = config_.host.pinned_state;
                ceiling = config_.host.pinned_cap;
            } else {
                ceiling_state = config_.host.pageable_state;
                ceiling = config_.host.pageable_cap;
            }
            host_groups.push_back(&add_rollup_group(
                out.groups,
                llama_cache_budget_resource_kind::host_residency,
                residency,
                nullptr,
                {},
                values,
                ceiling_state,
                ceiling));
        }

        if (config_.host.total_state !=
                llama_cache_budget_capacity_state::unbounded) {
            add_rollup_group(
                out.groups,
                llama_cache_budget_resource_kind::host_total,
                llama_cache_acct_residency::not_applicable,
                nullptr,
                {},
                sum_row_values(host_groups),
                config_.host.total_state,
                config_.host.total_cap);
        }

        if (config_.global_cap_state !=
                llama_cache_budget_capacity_state::unbounded) {
            std::vector<llama_cache_budget_row *> top_level;
            for (auto & row : out.groups) {
                if (row.resource.kind ==
                        llama_cache_budget_resource_kind::physical_device ||
                    row.resource.kind ==
                        llama_cache_budget_resource_kind::host_residency) {
                    top_level.push_back(&row);
                }
            }
            add_rollup_group(
                out.groups,
                llama_cache_budget_resource_kind::administrative_global,
                llama_cache_acct_residency::not_applicable,
                nullptr,
                {},
                sum_row_values(top_level),
                config_.global_cap_state,
                config_.administrative_global_cap);
        }

        out.state = llama_cache_budget_fit_state::fits;
        for (const auto & row : out.domains) {
            out.state = combine_state(out.state, row.state);
        }
        for (const auto & row : out.groups) {
            out.state = combine_state(out.state, row.state);
        }
        return out;
    } catch (...) {
        out.domains.clear();
        out.groups.clear();
        out.state = llama_cache_budget_fit_state::unavailable;
        return out;
    }
}

const char * llama_cache_budget_capacity_state_name(
        llama_cache_budget_capacity_state state) noexcept {
    switch (state) {
        case llama_cache_budget_capacity_state::known:       return "known";
        case llama_cache_budget_capacity_state::unbounded:   return "unbounded";
        case llama_cache_budget_capacity_state::unavailable: return "unavailable";
        case llama_cache_budget_capacity_state::_count:      return "invalid";
    }
    return "invalid";
}

const char * llama_cache_budget_reserve_provenance_name(
        llama_cache_budget_reserve_provenance provenance) noexcept {
    switch (provenance) {
        case llama_cache_budget_reserve_provenance::configured: return "configured";
        case llama_cache_budget_reserve_provenance::measured:   return "measured";
        case llama_cache_budget_reserve_provenance::_count:     return "invalid";
    }
    return "invalid";
}

const char * llama_cache_budget_fit_state_name(
        llama_cache_budget_fit_state state) noexcept {
    switch (state) {
        case llama_cache_budget_fit_state::fits:        return "fits";
        case llama_cache_budget_fit_state::exceeds:     return "exceeds";
        case llama_cache_budget_fit_state::unavailable: return "unavailable";
        case llama_cache_budget_fit_state::_count:      return "invalid";
    }
    return "invalid";
}
