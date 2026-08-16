#include "server-cache-yield.h"
#include "../../common/common-cache-plan.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace {

bool op_union_add(
        std::vector<llama_cache_acct_op_id> & selected,
        const std::vector<llama_cache_acct_op_id> & added) {
    for (const auto op : added) {
        if (!op) {
            return false;
        }
        if (std::find(selected.begin(), selected.end(), op) ==
                selected.end()) {
            if (selected.size() == selected.max_size()) {
                return false;
            }
            selected.push_back(op);
        }
    }
    return true;
}

}

bool server_cache_yield_release_plan(
        const llama_cache_acct_release_set_preview & release,
        uint64_t accounting_serial,
        llama_cache_budget_plan & plan) noexcept {
    if (release.accounting_serial != accounting_serial) {
        return false;
    }
    plan = {};
    plan.accounting_serial = accounting_serial;
    try {
        plan.entries.reserve(release.rows.size());
        for (const auto & row : release.rows) {
            plan.entries.push_back({
                row.domain, 0, row.resident_allocated,
            });
        }
        return true;
    } catch (...) {
        plan = {};
        return false;
    }
}

bool server_cache_yield_lower_domain(
        const llama_cache_budget_row & row,
        common_cache_plan_yield_domain & out) noexcept {
    if (row.resource.kind !=
            llama_cache_budget_resource_kind::accounting_domain) {
        return false;
    }
    out = {
        row.resource.domain,
        row.current_resident,
        row.before,
        row.released,
        row.reserved,
        row.after,
    };
    return true;
}

const char * server_cache_yield_status_name(
        server_cache_yield_status status) noexcept {
    switch (status) {
        case server_cache_yield_status::fits:
            return "fits";
        case server_cache_yield_status::insufficient_yield:
            return "insufficient_yield";
        case server_cache_yield_status::unsupported_required:
            return "unsupported_required";
        case server_cache_yield_status::unavailable:
            return "unavailable";
        case server_cache_yield_status::_count:
            return "invalid";
    }
    return "invalid";
}

bool server_cache_yield_assemble(
        const std::vector<server_retention_candidate> & catalog,
        server_cache_lease_table & leases,
        const server_cache_yield_candidate_resolver & resolver,
        std::vector<server_cache_yield_candidate> & out) noexcept {
    out.clear();
    if (!resolver || catalog.size() > SERVER_CACHE_YIELD_MAX_CANDIDATES) {
        return false;
    }
    try {
        out.reserve(catalog.size());
        for (const auto & source : catalog) {
            server_cache_yield_candidate candidate;
            candidate.artifact_id = source.artifact_id;
            candidate.record = source.record;
            candidate.lineage = source.lineage;
            candidate.availability = source.avail;
            candidate.release_ops = source.release_ops;
            server_cache_lease_identity identity;
            bool identity_known = false;
            resolver(source, candidate, identity, identity_known);
            candidate.identity_known = identity_known;
            candidate.lease = leases.inspect(
                candidate.artifact_id, identity);
            out.push_back(std::move(candidate));
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

server_retention_shadow_projection server_retention_shadow_project(
        const std::vector<server_cache_yield_candidate> & candidates,
        uint64_t competition_epoch,
        const llama_cache_acct_resource_domain & pressured_domain,
        uint64_t accounting_serial,
        const server_cache_yield_preview_callback & preview,
        const common_retention_frequency_config & config) noexcept {
    server_retention_shadow_projection result;
    if (competition_epoch == 0 ||
        candidates.size() > SERVER_CACHE_YIELD_MAX_CANDIDATES ||
        !preview || !config.valid()) {
        return result;
    }

    struct member {
        const server_cache_yield_candidate * candidate = nullptr;
        bool releasable = false;
    };
    struct group {
        common_retention_lineage_record lineage;
        std::vector<member> members;
    };

    try {
        std::vector<group> groups;
        groups.reserve(candidates.size());
        std::unordered_map<uint64_t, size_t> group_index;
        group_index.reserve(candidates.size());
        bool complete = true;
        for (const auto & candidate : candidates) {
            // DF1 values whole prompt payloads. A checkpoint is neither a
            // substitute for a live/host prefix nor a frequency victim.
            if (candidate.record.kind ==
                    common_retention_artifact_kind::checkpoint) {
                continue;
            }
            if (!candidate.artifact_id.v ||
                candidate.availability ==
                    server_retention_candidate_availability::
                        backing_missing_or_stale ||
                !candidate.identity_known ||
                candidate.lease.state != server_cache_lease_eval_state::known ||
                !candidate.record.valid() ||
                candidate.record.stamp.state !=
                    common_retention_score_state::known ||
                candidate.external_shared_coverage_tokens >
                    candidate.record.stamp.coverage_tokens ||
                !candidate.lineage.valid(competition_epoch) ||
                candidate.record.stamp.pool != candidate.lineage.pool ||
                candidate.record.stamp.lineage_id !=
                    candidate.lineage.lineage_id) {
                complete = false;
                continue;
            }

            const uint64_t group_key =
                (candidate.lineage.lineage_id << 1) |
                uint64_t(candidate.lineage.pool);
            const auto indexed = group_index.find(group_key);
            group * found = nullptr;
            if (indexed == group_index.end()) {
                groups.push_back({ candidate.lineage, {} });
                const size_t position = groups.size() - 1;
                group_index.emplace(group_key, position);
                found = &groups[position];
            } else {
                found = &groups[indexed->second];
            }
            if (found->lineage != candidate.lineage) {
                complete = false;
                continue;
            }

            const bool releasable =
                candidate.availability ==
                    server_retention_candidate_availability::available &&
                candidate.lease.eligibility ==
                    server_cache_lease_eligibility::eligible &&
                !server_cache_lease_is_hard(candidate.lease) &&
                !candidate.release_ops.empty();
            found->members.push_back({ &candidate, releasable });
        }

        result.alternatives.reserve(groups.size());
        size_t compound_evaluations = 0;
        for (auto & group : groups) {
            // Append supersession keeps the normal case tiny. This sort also
            // makes the representative deterministic when many exact aliases
            // share one frontier.
            std::sort(group.members.begin(), group.members.end(),
                [](const member & a, const member & b) {
                    const auto & as = a.candidate->record.stamp;
                    const auto & bs = b.candidate->record.stamp;
                    return std::tie(
                               as.coverage_tokens, as.recency_ordinal,
                               as.stable_id, a.candidate->artifact_id.v) <
                           std::tie(
                               bs.coverage_tokens, bs.recency_ordinal,
                               bs.stable_id, b.candidate->artifact_id.v);
                });

            server_retention_shadow_alternative best;
            bool have_best = false;
            bool saw_zero_single = false;
            const auto evaluate = [&](
                    const std::vector<size_t> & removed) {
                if (removed.empty() ||
                    removed.size() > SERVER_RETENTION_SHADOW_MAX_COMPOUND) {
                    return;
                }
                std::vector<llama_cache_acct_op_id> ops;
                std::vector<llama_cache_acct_artifact_id> artifacts;
                uint64_t removed_coverage = 0;
                uint64_t external_retained_coverage = 0;
                uint64_t recency = 0;
                std::vector<bool> is_removed(group.members.size(), false);
                for (const size_t index : removed) {
                    if (index >= group.members.size() ||
                        !group.members[index].releasable) {
                        return;
                    }
                    is_removed[index] = true;
                    const auto & candidate = *group.members[index].candidate;
                    artifacts.push_back(candidate.artifact_id);
                    ops.insert(ops.end(), candidate.release_ops.begin(),
                               candidate.release_ops.end());
                    const uint64_t coverage =
                        candidate.record.stamp.coverage_tokens;
                    removed_coverage = std::max(removed_coverage, coverage);
                    external_retained_coverage = std::max(
                        external_retained_coverage,
                        candidate.external_shared_coverage_tokens);
                    recency = std::max(
                        recency, candidate.record.stamp.recency_ordinal);
                }
                std::sort(ops.begin(), ops.end());
                ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
                std::sort(artifacts.begin(), artifacts.end(),
                    [](auto a, auto b) { return a.v < b.v; });

                llama_cache_acct_release_set_preview release;
                if (!preview(ops, accounting_serial, release) ||
                    release.accounting_serial != accounting_serial) {
                    complete = false;
                    return;
                }
                uint64_t marginal = 0;
                for (const auto & row : release.rows) {
                    if (row.domain != pressured_domain) {
                        continue;
                    }
                    if (row.resident_allocated > UINT64_MAX - marginal) {
                        complete = false;
                        return;
                    }
                    marginal += row.resident_allocated;
                }
                if (marginal == 0) {
                    if (removed.size() == 1) {
                        saw_zero_single = true;
                    }
                    return;
                }
                uint64_t retained_coverage = 0;
                for (size_t i = 0; i < group.members.size(); ++i) {
                    if (!is_removed[i]) {
                        retained_coverage = std::max(
                            retained_coverage,
                            group.members[i].candidate->record.stamp.
                                coverage_tokens);
                    }
                }
                retained_coverage = std::max(
                    retained_coverage, external_retained_coverage);
                const uint64_t lost_work = removed_coverage > retained_coverage
                    ? removed_coverage - retained_coverage : 0;
                common_retention_shadow_value quote;
                if (!common_retention_shadow_quote(
                        group.lineage, competition_epoch, lost_work,
                        marginal, recency,
                        config, quote)) {
                    complete = false;
                    return;
                }
                server_retention_shadow_alternative proposed {
                    group.lineage.pool,
                    group.lineage.lineage_id,
                    std::move(artifacts),
                    lost_work,
                    quote,
                };
                if (!have_best ||
                    common_retention_shadow_compare(
                        proposed.value, best.value) < 0) {
                    best = proposed;
                    have_best = true;
                }
            };

            std::vector<size_t> releasable;
            for (size_t i = 0; i < group.members.size(); ++i) {
                if (!group.members[i].releasable) {
                    continue;
                }
                releasable.push_back(i);
                evaluate({ i });
            }

            // A zero-yield singleton proves that refcounted allocation
            // dependencies exist. Enumerate the entire bounded powerset so
            // unequal-frontier aliases and mixed descendants receive their
            // exact marginal release and retained coverage. Never substitute
            // a heuristic compound for a larger dependency graph.
            if (saw_zero_single) {
                if (releasable.size() >
                        SERVER_RETENTION_SHADOW_MAX_COMPOUND) {
                    complete = false;
                } else {
                    const uint64_t n_subsets = uint64_t(1) << releasable.size();
                    for (uint64_t mask = 1; mask < n_subsets; ++mask) {
                        if ((mask & (mask - 1)) == 0) {
                            continue; // singletons were evaluated above
                        }
                        if (compound_evaluations ==
                                SERVER_RETENTION_SHADOW_MAX_COMPOUND_EVALUATIONS) {
                            complete = false;
                            break;
                        }
                        compound_evaluations++;
                        std::vector<size_t> removed;
                        removed.reserve(releasable.size());
                        for (size_t bit = 0; bit < releasable.size(); ++bit) {
                            if (mask & (uint64_t(1) << bit)) {
                                removed.push_back(releasable[bit]);
                            }
                        }
                        evaluate(removed);
                    }
                }
            }
            if (have_best) {
                result.alternatives.push_back(std::move(best));
            }
        }
        std::sort(result.alternatives.begin(), result.alternatives.end(),
            [](const auto & a, const auto & b) {
                const int value = common_retention_shadow_compare(
                    a.value, b.value);
                if (value != 0) {
                    return value < 0;
                }
                const uint64_t a_id = a.artifact_ids.empty()
                    ? 0 : a.artifact_ids.front().v;
                const uint64_t b_id = b.artifact_ids.empty()
                    ? 0 : b.artifact_ids.front().v;
                return std::tie(a.pool, a.lineage_id, a_id) <
                       std::tie(b.pool, b.lineage_id, b_id);
            });
        result.complete = complete;
        return result;
    } catch (...) {
        result = {};
        return result;
    }
}

server_cache_yield_result server_cache_yield_plan(
        const std::vector<server_cache_yield_candidate> & candidates,
        uint64_t accounting_serial,
        const server_cache_yield_preview_callback & preview,
        const server_cache_yield_fits_callback & fits,
        uint32_t policy_version) noexcept {
    server_cache_yield_result result;
    result.accounting_serial = accounting_serial;
    result.yield_policy_version = policy_version;
    const auto mark_unavailable = [&]() {
        result.status = server_cache_yield_status::unavailable;
        result.selected = {};
        result.plan.clear();
        result.unsupported.clear();
        result.projected_fit = {};
    };
    if (policy_version != SERVER_CACHE_YIELD_POLICY_VERSION ||
        candidates.size() > SERVER_CACHE_YIELD_MAX_CANDIDATES ||
        !preview || !fits) {
        mark_unavailable();
        return result;
    }

    try {
        std::array<std::vector<const server_cache_yield_candidate *>,
                   size_t(common_retention_pool::_count)> pools;
        bool unavailable_evidence = false;
        for (const auto & candidate : candidates) {
            if (candidate.has_unsupported_host_spill) {
                result.unsupported.push_back(candidate.artifact_id);
            }
            if (!candidate.artifact_id.v ||
                candidate.availability >=
                    server_retention_candidate_availability::_count ||
                candidate.availability !=
                    server_retention_candidate_availability::available ||
                !candidate.identity_known ||
                candidate.lease.state != server_cache_lease_eval_state::known ||
                candidate.lease.cls >= server_cache_lease_class::_count ||
                candidate.lease.eligibility >=
                    server_cache_lease_eligibility::_count ||
                !candidate.record.valid() ||
                candidate.record.stamp.state !=
                    common_retention_score_state::known ||
                candidate.record.stamp.soft_leased ||
                candidate.record.stamp.pool >= common_retention_pool::_count) {
                unavailable_evidence = true;
                continue;
            }
            if (candidate.record.stamp.mandatory_anchor ||
                server_cache_lease_is_hard(candidate.lease)) {
                continue;
            }
            if (candidate.release_ops.empty()) {
                unavailable_evidence = true;
                continue;
            }

            // Validate the operation citations before they enter the order. The byte
            // result is deliberately discarded: only the selected UNION is priced.
            llama_cache_acct_release_set_preview validation;
            if (!preview(
                    candidate.release_ops, accounting_serial, validation)) {
                unavailable_evidence = true;
                continue;
            }
            pools[size_t(candidate.record.stamp.pool)].push_back(&candidate);
        }

        for (auto & pool : pools) {
            std::sort(pool.begin(), pool.end(),
                [](const auto & a, const auto & b) {
                    const auto & as = a->record.stamp;
                    const auto & bs = b->record.stamp;
                    const bool a_soft =
                        a->lease.cls == server_cache_lease_class::soft;
                    const bool b_soft =
                        b->lease.cls == server_cache_lease_class::soft;
                    return std::tie(
                               a_soft, as.anchor_rank,
                               as.recency_ordinal, as.coverage_tokens,
                               as.stable_id) <
                           std::tie(
                               b_soft, bs.anchor_rank,
                               bs.recency_ordinal, bs.coverage_tokens,
                               bs.stable_id);
                });
        }

        llama_cache_budget_plan budget_plan;
        budget_plan.accounting_serial = accounting_serial;
        auto fit = fits(budget_plan);
        if (fit.accounting_serial != accounting_serial ||
            fit.state == llama_cache_budget_fit_state::unavailable) {
            mark_unavailable();
            return result;
        }
        if (fit.state == llama_cache_budget_fit_state::fits) {
            // The winning fit and selected-union evidence are serial-bound to
            // the same accounting snapshot; D-S7 projects these rows verbatim.
            result.projected_fit = std::move(fit);
            result.status = server_cache_yield_status::fits;
            return result;
        }

        std::vector<llama_cache_acct_op_id> selected_ops;
        for (const auto pool_kind : {
                common_retention_pool::attention,
                common_retention_pool::recurrent }) {
            for (const auto * candidate : pools[size_t(pool_kind)]) {
                const size_t n_ops_before = selected_ops.size();
                if (!op_union_add(
                        selected_ops, candidate->release_ops)) {
                    mark_unavailable();
                    return result;
                }
                if (selected_ops.size() == n_ops_before) {
                    continue;
                }
                llama_cache_acct_release_set_preview released;
                if (!preview(selected_ops, accounting_serial, released) ||
                    !server_cache_yield_release_plan(
                        released, accounting_serial, budget_plan)) {
                    mark_unavailable();
                    return result;
                }
                fit = fits(budget_plan);
                if (fit.accounting_serial != accounting_serial ||
                    fit.state == llama_cache_budget_fit_state::unavailable) {
                    mark_unavailable();
                    return result;
                }
                result.selected[size_t(pool_kind)].push_back(
                    candidate->artifact_id);
                result.plan = budget_plan.entries;
                if (fit.state == llama_cache_budget_fit_state::fits) {
                    result.projected_fit = std::move(fit);
                    result.status = server_cache_yield_status::fits;
                    return result;
                }
            }
        }

        // These known terminals require an exclusively priceable/eligible
        // catalog. A host-entry-only catalog can reach insufficient_yield;
        // common mixed live catalogs remain unavailable while slots/checkpoints
        // lack exact operation ownership. unsupported_required additionally
        // awaits an available spill producer (milestone F).
        if (unavailable_evidence) {
            mark_unavailable();
            return result;
        } else if (!result.unsupported.empty()) {
            result.status =
                server_cache_yield_status::unsupported_required;
        } else {
            result.status =
                server_cache_yield_status::insufficient_yield;
        }
        return result;
    } catch (...) {
        result.status = server_cache_yield_status::unavailable;
        result.selected = {};
        result.plan.clear();
        result.unsupported.clear();
        result.projected_fit = {};
        return result;
    }
}
