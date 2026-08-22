#include "llama-vbr-downward.h"

#include "llama-sha256.h"
#include "llama-vbr-identity-digest.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

namespace {

constexpr std::array<ggml_type, 6> TIERS = {
    GGML_TYPE_F16,
    GGML_TYPE_TURBO8_0,
    GGML_TYPE_TURBO4_0,
    GGML_TYPE_TURBO3_TCQ,
    GGML_TYPE_TURBO2_TCQ,
    GGML_TYPE_TURBO1_TCQ,
};

int tier_rank(ggml_type type) noexcept {
    const auto it = std::find(TIERS.begin(), TIERS.end(), type);
    return it == TIERS.end() ? -1 : int(it - TIERS.begin());
}

} // namespace

vbr_repr_domain vbr_downward_tier_domain(ggml_type type) noexcept {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_TURBO8_0
        ? vbr_repr_domain::full : vbr_repr_domain::tapped;
}

const char * vbr_downward_recipe_status_name(vbr_downward_recipe_status status) noexcept {
    switch (status) {
        case vbr_downward_recipe_status::resolved: return "resolved";
        case vbr_downward_recipe_status::equal_tier: return "equal_tier";
        case vbr_downward_recipe_status::upward_forbidden: return "upward_forbidden";
        case vbr_downward_recipe_status::unsupported_type: return "unsupported_type";
        case vbr_downward_recipe_status::below_floor: return "below_floor";
        case vbr_downward_recipe_status::nonmovable: return "nonmovable";
        case vbr_downward_recipe_status::invalid_argument: return "invalid_argument";
        case vbr_downward_recipe_status::_count: break;
    }
    return "invalid";
}

vbr_downward_recipe_status vbr_downward_resolve_recipe(
        ggml_type source_type,
        ggml_type target_type,
        ggml_type floor_type,
        bool movable,
        vbr_downward_recipe & out) noexcept {
    out = {};
    out.source_type = source_type;
    out.target_type = target_type;
    if (!movable) {
        return vbr_downward_recipe_status::nonmovable;
    }
    const int source = tier_rank(source_type);
    const int target = tier_rank(target_type);
    const int floor = tier_rank(floor_type);
    if (source < 0 || target < 0 || floor < 0) {
        return vbr_downward_recipe_status::unsupported_type;
    }
    if (source == target) {
        return vbr_downward_recipe_status::equal_tier;
    }
    if (target < source) {
        return vbr_downward_recipe_status::upward_forbidden;
    }
    if (target > floor) {
        return vbr_downward_recipe_status::below_floor;
    }
    for (int i = source; i < target; ++i) {
        const ggml_type a = TIERS[size_t(i)];
        const ggml_type b = TIERS[size_t(i + 1)];
        out.edges[out.n_edges++] = { a, b, vbr_downward_tier_domain(a), vbr_downward_tier_domain(b),
            vbr_downward_tier_domain(a) == vbr_repr_domain::tapped };
    }
    return vbr_downward_recipe_status::resolved;
}

std::array<uint8_t, 32> vbr_downward_build_identity(
        const vbr_downward_recipe & recipe,
        int32_t meansub_model_id,
        const std::array<uint8_t, 32> & meansub_digest,
        const std::array<uint8_t, 32> & policy_digest,
        const std::array<uint8_t, 32> & tree_policy_digest) noexcept {
    try {
        if (meansub_model_id < 0 || recipe.n_edges == 0 ||
            recipe.n_edges > recipe.edges.size()) {
            return {};
        }
        llama_sha256_writer writer;
        static constexpr char domain_label[] =
            "buun.vbr.downward/build-identity/v1";
        writer.string(domain_label, sizeof(domain_label) - 1);
        writer.u32(recipe.version);
        writer.u32(uint32_t(recipe.source_type));
        writer.u32(uint32_t(recipe.target_type));
        writer.u64(recipe.n_edges);
        for (size_t i = 0; i < recipe.n_edges; ++i) {
            const auto & edge = recipe.edges[i];
            writer.u32(uint32_t(edge.source_type));
            writer.u32(uint32_t(edge.target_type));
            writer.u32(uint32_t(edge.source_domain));
            writer.u32(uint32_t(edge.target_domain));
            writer.u32(edge.capture_stash_before);
        }
        writer.u32(uint32_t(meansub_model_id));
        writer.bytes(meansub_digest.data(), meansub_digest.size());
        writer.bytes(policy_digest.data(), policy_digest.size());
        writer.bytes(tree_policy_digest.data(), tree_policy_digest.size());
        return writer.finish();
    } catch (...) {
        return {};
    }
}

const char * vbr_downward_policy_status_name(vbr_downward_policy_status status) noexcept {
    switch (status) {
        case vbr_downward_policy_status::coherent: return "coherent";
        case vbr_downward_policy_status::incoherent: return "incoherent";
        case vbr_downward_policy_status::exhausted: return "exhausted";
        case vbr_downward_policy_status::invalid: return "invalid";
        case vbr_downward_policy_status::overflow: return "overflow";
        case vbr_downward_policy_status::_count: break;
    }
    return "invalid";
}

const char * vbr_import_destination_status_name(
        vbr_import_destination_status status) noexcept {
    switch (status) {
        case vbr_import_destination_status::feasible_current:
            return "feasible_current";
        case vbr_import_destination_status::feasible_degraded:
            return "feasible_degraded";
        case vbr_import_destination_status::exhausted: return "exhausted";
        case vbr_import_destination_status::invalid: return "invalid";
        case vbr_import_destination_status::overflow: return "overflow";
        case vbr_import_destination_status::_count: break;
    }
    return "invalid";
}

vbr_import_destination_projection vbr_select_import_destination(
        const std::vector<vbr_import_destination_child> & children,
        void * context,
        vbr_import_destination_measure_fn measure) noexcept {
    vbr_import_destination_projection result;
    try {
        if (children.empty() || measure == nullptr) {
            return result;
        }
        result.final_types.reserve(children.size());
        result.final_cursors.reserve(children.size());
        std::vector<llama_vbr_policy::child> policies;
        policies.reserve(children.size());
        for (const auto & child : children) {
            if (child.initial_types.empty() ||
                child.watermark_cells == 0) {
                return result;
            }
            result.final_types.push_back(child.initial_types);
            result.final_cursors.push_back(child.initial_cursor);
            policies.push_back(child.policy);
        }
        bool measure_ok = true;
        const auto fits = [&](const llama_vbr_policy::selection * selected) {
            vbr_import_destination_evidence evidence;
            if (!measure(
                    context, result.final_types, selected, evidence) ||
                !evidence.active) {
                measure_ok = false;
                return false;
            }
            result.pools = evidence.pools;
            result.logical_bytes_needed = evidence.logical_bytes_needed;
            result.logical_bytes_available = evidence.logical_bytes_available;
            result.physical_growth_needed = evidence.physical_growth_needed;
            result.physical_growth_available =
                evidence.physical_growth_available;
            result.max_deficit = evidence.max_deficit;
            return evidence.fits;
        };
        const bool current_fits = fits(nullptr);
        if (!measure_ok) {
            return result;
        }
        if (current_fits) {
            result.status =
                vbr_import_destination_status::feasible_current;
        } else {
            llama_vbr_policy::shortest_prefix_stream stream(
                std::move(policies));
            llama_vbr_policy::selection selected;
            for (;;) {
                const auto status = stream.next(selected);
                if (status != llama_vbr_policy::result::selected) {
                    result.status = status ==
                            llama_vbr_policy::result::exhausted
                        ? vbr_import_destination_status::exhausted
                        : status == llama_vbr_policy::result::overflow
                            ? vbr_import_destination_status::overflow
                            : vbr_import_destination_status::invalid;
                    return result;
                }
                if (selected.child_index >= result.final_types.size() ||
                    selected.value.slot >= result.final_types[
                        selected.child_index].size() ||
                    result.final_types[selected.child_index][
                        selected.value.slot] !=
                            static_cast<ggml_type>(selected.value.type_a)) {
                    return {};
                }
                result.final_types[selected.child_index][
                    selected.value.slot] = static_cast<ggml_type>(
                        selected.value.type_b);
                result.final_cursors[selected.child_index] =
                    std::max<uint64_t>(
                        result.final_cursors[selected.child_index],
                        selected.value.order_index + 1);
                result.prefix.push_back(selected);
                const bool selected_fits = fits(&selected);
                if (!measure_ok) {
                    return {};
                }
                if (selected_fits) {
                    result.status =
                        vbr_import_destination_status::feasible_degraded;
                    break;
                }
            }
        }
        result.child_type_digests.reserve(result.final_types.size());
        for (const auto & types : result.final_types) {
            result.child_type_digests.push_back(
                vbr_type_vector_digest(types));
        }
        result.tree_digest =
            vbr_type_tree_digest(
                result.child_type_digests,
                VBR_DOWNWARD_RECIPE_VERSION);
        return result;
    } catch (...) {
        return {};
    }
}

vbr_downward_policy_projection vbr_downward_project_policy_prefix(
        const std::vector<vbr_downward_policy_child> & children) noexcept {
    vbr_downward_policy_projection out;
    try {
        if (children.empty()) {
            return out;
        }
        size_t mismatch_count = 0;
        out.final_types.reserve(children.size());
        out.final_cursors.reserve(children.size());
        for (const auto & child : children) {
            if (child.initial_types.empty() ||
                child.initial_types.size() != child.target_types.size()) {
                return out;
            }
            out.final_types.push_back(child.initial_types);
            out.final_cursors.push_back(child.initial_cursor);
            mismatch_count += std::inner_product(
                child.initial_types.begin(), child.initial_types.end(),
                child.target_types.begin(), size_t(0), std::plus<size_t>(),
                std::not_equal_to<ggml_type>());
        }
        if (mismatch_count != 0) {
            std::vector<llama_vbr_policy::child> policies;
            policies.reserve(children.size());
            for (const auto & child : children) {
                policies.push_back(child.policy);
            }
            llama_vbr_policy::shortest_prefix_stream stream(std::move(policies));
            bool incoherent = false;
            const auto status = stream.shortest_prefix(
                [&](const std::vector<llama_vbr_policy::selection> & prefix) {
                const auto & selected = prefix.back();
                // Selection-coherence rule mirrors the live prefix apply in
                // llama_kv_cache::vbr_tx_reprice — keep the two in sync.
                if (selected.child_index >= out.final_types.size() ||
                    selected.value.slot >= out.final_types[selected.child_index].size() ||
                    out.final_types[selected.child_index][selected.value.slot] !=
                        static_cast<ggml_type>(selected.value.type_a)) {
                    incoherent = true;
                    return true;
                }
                auto & current = out.final_types[selected.child_index][selected.value.slot];
                if (selected.value.order_index == SIZE_MAX) {
                    incoherent = true;
                    return true;
                }
                out.final_cursors[selected.child_index] =
                    std::max<uint64_t>(
                        out.final_cursors[selected.child_index],
                        selected.value.order_index + 1);
                const auto target = children[selected.child_index].target_types[selected.value.slot];
                const bool matched_before = current == target;
                current = static_cast<ggml_type>(selected.value.type_b);
                const bool matched_after = current == target;
                if (matched_before != matched_after) {
                    if (matched_after) {
                        --mismatch_count;
                    } else {
                        ++mismatch_count;
                    }
                }
                return mismatch_count == 0;
            }, out.prefix);
            if (incoherent) {
                out.status = vbr_downward_policy_status::incoherent;
                return out;
            }
            if (status != llama_vbr_policy::result::selected) {
                if (status == llama_vbr_policy::result::exhausted) {
                    out.status = vbr_downward_policy_status::exhausted;
                } else if (status == llama_vbr_policy::result::overflow) {
                    out.status = vbr_downward_policy_status::overflow;
                } else {
                    out.status = vbr_downward_policy_status::invalid;
                }
                return out;
            }
        }
        out.child_type_digests.reserve(out.final_types.size());
        for (const auto & types : out.final_types) {
            out.child_type_digests.push_back(vbr_type_vector_digest(types));
        }
        out.tree_digest = vbr_type_tree_digest(
            out.child_type_digests, VBR_DOWNWARD_RECIPE_VERSION);
        out.status = vbr_downward_policy_status::coherent;
        return out;
    } catch (...) {
        out.status = vbr_downward_policy_status::invalid;
        return out;
    }
}

vbr_downward_resource_receipts::vbr_downward_resource_receipts(
        llama_cache_acct_ledger & ledger) noexcept : ledger_(&ledger) {}

void vbr_downward_resource_receipts::release_ops() noexcept {
    if (ledger_ != nullptr) {
        for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) {
            ledger_->release(*it);
        }
    }
    ops_.clear();
}

vbr_downward_resource_receipts::~vbr_downward_resource_receipts() {
    release_ops();
}

vbr_downward_resource_receipts::vbr_downward_resource_receipts(
        vbr_downward_resource_receipts && other) noexcept
    : ledger_(other.ledger_),
      records_(std::move(other.records_)),
      ops_(std::move(other.ops_)) {
    other.ledger_ = nullptr;
    other.records_.clear();
    other.ops_.clear();
}

vbr_downward_resource_receipts & vbr_downward_resource_receipts::operator=(
        vbr_downward_resource_receipts && other) noexcept {
    if (this != &other) {
        release_ops();
        ledger_ = other.ledger_;
        records_ = std::move(other.records_);
        ops_ = std::move(other.ops_);
        other.ledger_ = nullptr;
        other.records_.clear();
        other.ops_.clear();
    }
    return *this;
}

vbr_downward_resource_receipts::record *
vbr_downward_resource_receipts::find(const endpoint_key & key) noexcept {
    const auto it = std::find_if(records_.begin(), records_.end(),
        [&](const record & candidate) { return candidate.key == key; });
    return it == records_.end() ? nullptr : &*it;
}

const char * vbr_downward_reserve_status_name(vbr_downward_reserve_status status) noexcept {
    switch (status) {
        case vbr_downward_reserve_status::not_attempted: return "not_attempted";
        case vbr_downward_reserve_status::reserved: return "reserved";
        case vbr_downward_reserve_status::reserved_stashless: return "reserved_stashless";
        case vbr_downward_reserve_status::projection_unavailable: return "projection_unavailable";
        case vbr_downward_reserve_status::accounting_refused: return "accounting_refused";
        case vbr_downward_reserve_status::workspace_reserve_failed: return "workspace_reserve_failed";
        case vbr_downward_reserve_status::internal_error: return "internal_error";
        case vbr_downward_reserve_status::_count: break;
    }
    return "invalid";
}

vbr_downward_reserve_result vbr_downward_resource_receipts::reserve_resources(
        const llama_cache_budget_config & budget,
        const std::vector<vbr_downward_workspace_endpoint> & workspaces,
        const std::vector<vbr_downward_stash_endpoint> & stashes) noexcept {
    vbr_downward_reserve_result out;
    struct projected {
        endpoint_key key;
        llama_cache_acct_attribution attribution;
        uint64_t endpoint = 0;
        uint64_t growth = 0;
        llama_cache_acct_op_id committed;
        bool is_new = false;
    };
    try {
        if (ledger_ == nullptr) {
            return out;
        }
        std::vector<projected> projected_rows;
        projected_rows.reserve(workspaces.size() + stashes.size());
        const auto duplicate_key = [&](const endpoint_key & key) {
            return std::any_of(projected_rows.begin(), projected_rows.end(),
                [&](const projected & row) { return row.key == key; });
        };
        const auto project_row = [&](const endpoint_key & key,
                const llama_cache_acct_attribution & attribution,
                uint64_t now, uint64_t endpoint, uint64_t & growth_total) {
            if (duplicate_key(key) || endpoint < now) {
                return false;
            }
            const auto * existing = find(key);
            const uint64_t accounted = existing ? existing->endpoint : now;
            if (accounted > endpoint) {
                return false;
            }
            const uint64_t growth = endpoint - accounted;
            projected_rows.push_back({
                key, attribution, endpoint, growth, {}, existing == nullptr,
            });
            return llama_vbr_transaction::add_u64(growth_total, growth);
        };

        for (const auto & workspace : workspaces) {
            // backend must be live here: projection alone blesses null (pre-side-
            // backend), but reserve_resources ends in a physical reserve whose
            // backend assert would abort AFTER the C row committed.
            if (workspace.owner == nullptr || workspace.iface == nullptr ||
                workspace.iface->kv_transcode_workspace_memory == nullptr ||
                workspace.iface->kv_transcode_workspace_reserve == nullptr ||
                workspace.backend == nullptr ||
                workspace.device < 0 || workspace.requests.empty()) {
                out.status = vbr_downward_reserve_status::projection_unavailable;
                return out;
            }
            const endpoint_key key { workspace.owner,
                llama_cache_acct_category::codec_workspace, workspace.domain };
            uint64_t now = 0;
            uint64_t endpoint = 0;
            const bool ok = llama_vbr_transaction::workspace_endpoint(
                workspace.requests,
                [&](const llama_vbr_transaction::workspace_request & request,
                        uint64_t & current, uint64_t & reserved) {
                    size_t c = 0;
                    size_t r = 0;
                    if (!workspace.iface->kv_transcode_workspace_memory(
                            workspace.backend, workspace.device, request.n_cells,
                            request.ne0, request.stash_rows, &c, &r)) {
                        return false;
                    }
                    current = c;
                    reserved = r;
                    return true;
                }, now, endpoint);
            if (!ok || !project_row(key, workspace.attribution, now, endpoint,
                    out.workspace_growth)) {
                out.status = vbr_downward_reserve_status::projection_unavailable;
                return out;
            }
        }

        for (const auto & stash : stashes) {
            if (stash.owner == nullptr || stash.unit_ids.empty() ||
                std::any_of(stash.unit_ids.begin(), stash.unit_ids.end(),
                    [](uint64_t unit_id) { return unit_id == 0; }) ||
                stash.context == nullptr ||
                stash.memory == nullptr || stash.reserve == nullptr) {
                out.status = vbr_downward_reserve_status::projection_unavailable;
                return out;
            }
            const endpoint_key key { stash.owner,
                llama_cache_acct_category::clean_stash_payload, stash.domain };
            uint64_t now = 0;
            uint64_t endpoint = 0;
            if (!stash.memory(stash.context, now, endpoint) ||
                !project_row(key, stash.attribution, now, endpoint, out.stash_growth)) {
                out.status = vbr_downward_reserve_status::projection_unavailable;
                return out;
            }
        }

        size_t new_records = 0;
        for (const auto & row : projected_rows) {
            new_records += row.is_new;
        }
        records_.reserve(records_.size() + new_records);
        std::vector<llama_cache_transaction_leaf> leaves;
        leaves.reserve(projected_rows.size());
        for (auto & row : projected_rows) {
            if (row.growth == 0) {
                continue;
            }
            llama_cache_transaction_leaf leaf;
            leaf.category = row.key.category;
            leaf.domain = row.key.domain;
            leaf.attribution = row.attribution;
            leaf.expected_logical = row.growth;
            leaf.reserve_resident = row.growth;
            leaf.stage_resident = row.growth;
            leaf.committed_op = &row.committed;
            leaves.push_back(leaf);
        }
        ops_.reserve(ops_.size() + leaves.size());
        llama_cache_transaction_result tx;
        if (!leaves.empty()) {
            tx = llama_cache_execute_reservation_transaction(
                *ledger_, budget, leaves);
            out.transaction_status = tx.status;
            out.admission_status = tx.admission_status;
            if (tx.status != llama_cache_transaction_status::committed) {
                out.status = vbr_downward_reserve_status::accounting_refused;
                return out;
            }
        } else {
            out.transaction_status = llama_cache_transaction_status::committed;
            out.admission_status = llama_cache_admission_status::admitted;
        }

        // Publish the receipts before invoking grow-only physical reserves. A
        // partial/failed grow remains conservatively charged and a retry sees
        // the same endpoint without minting a duplicate C operation.
        for (auto & row : projected_rows) {
            auto * record = find(row.key);
            if (record == nullptr) {
                records_.push_back({ row.key, row.endpoint });
                record = &records_.back();
            } else {
                record->endpoint = row.endpoint;
            }
            if (row.committed.v != 0) {
                ops_.push_back(row.committed);
            }
        }

        for (const auto & workspace : workspaces) {
            for (const auto & request : workspace.requests) {
                if (!workspace.iface->kv_transcode_workspace_reserve(
                        workspace.backend, request.n_cells, request.ne0, request.stash_rows)) {
                    out.status = vbr_downward_reserve_status::workspace_reserve_failed;
                    return out;
                }
            }
        }
        for (const auto & stash : stashes) {
            if (!stash.reserve(stash.context)) {
                out.stashless_units.insert(out.stashless_units.end(),
                    stash.unit_ids.begin(), stash.unit_ids.end());
            }
        }
        out.status = out.stashless_units.empty()
            ? vbr_downward_reserve_status::reserved
            : vbr_downward_reserve_status::reserved_stashless;
        return out;
    } catch (...) {
        out.status = vbr_downward_reserve_status::internal_error;
        return out;
    }
}

vbr_downward_transform_status vbr_downward_execute_edges(
        const vbr_downward_recipe & recipe,
        const vbr_downward_edge_driver & driver,
        bool & stash_regenerated,
        uint32_t * edge_reached) noexcept {
    stash_regenerated = false;
    if (edge_reached) {
        *edge_reached = UINT32_MAX;
    }
    try {
        if (recipe.version != VBR_DOWNWARD_RECIPE_VERSION || recipe.n_edges == 0 ||
            recipe.n_edges > recipe.edges.size() ||
            recipe.edges[0].source_type != recipe.source_type ||
            recipe.edges[recipe.n_edges - 1].target_type != recipe.target_type ||
            driver.transcode == nullptr) {
            return vbr_downward_transform_status::invalid_recipe;
        }
        for (size_t i = 0; i < recipe.n_edges; ++i) {
            const auto & edge = recipe.edges[i];
            if (edge_reached) {
                *edge_reached = uint32_t(i);
            }
            // Re-derive the domain/stash fields instead of trusting them: a recipe
            // value with a valid type chain but lying capture_stash_before would
            // otherwise run tapped edges stashless through a permissive adapter.
            if ((i > 0 && recipe.edges[i - 1].target_type != edge.source_type) ||
                tier_rank(edge.source_type) < 0 ||
                tier_rank(edge.target_type) != tier_rank(edge.source_type) + 1 ||
                edge.source_domain != vbr_downward_tier_domain(edge.source_type) ||
                edge.target_domain != vbr_downward_tier_domain(edge.target_type) ||
                edge.capture_stash_before !=
                    (vbr_downward_tier_domain(edge.source_type) == vbr_repr_domain::tapped)) {
                return vbr_downward_transform_status::invalid_recipe;
            }
            const bool stash_available = driver.stash_available != nullptr &&
                driver.stash_available(driver.context);
            if (edge.capture_stash_before && !stash_available) {
                if (driver.capture_stash == nullptr ||
                    !driver.capture_stash(driver.context, edge)) {
                    return vbr_downward_transform_status::stash_unavailable;
                }
                stash_regenerated = true;
            }
            if (!driver.transcode(driver.context, edge)) {
                return vbr_downward_transform_status::transform_failed;
            }
        }
        return vbr_downward_transform_status::transformed;
    } catch (...) {
        return vbr_downward_transform_status::internal_error;
    }
}

vbr_downward_transform_result vbr_downward_execute_recipe(
        const vbr_downward_recipe & recipe,
        const std::vector<uint8_t> & source,
        const std::vector<uint8_t> * authorized_stash,
        const vbr_downward_transform_iface & iface) noexcept {
    vbr_downward_transform_result out;
    struct byte_state {
        const vbr_downward_transform_iface * iface = nullptr;
        std::vector<uint8_t> current;
        std::vector<uint8_t> next;
        std::vector<uint8_t> stash;
    } state;
    try {
        if (source.empty() || iface.transcode == nullptr) {
            out.status = vbr_downward_transform_status::invalid_recipe;
            return out;
        }
        state.iface = &iface;
        state.current = source;
        if (authorized_stash != nullptr) {
            state.stash = *authorized_stash;
        }
        vbr_downward_edge_driver driver;
        driver.context = &state;
        driver.stash_available = [](void * context) noexcept {
            return !static_cast<byte_state *>(context)->stash.empty();
        };
        driver.capture_stash = [](void * context,
                                  const vbr_downward_edge & edge) noexcept {
            auto & value = *static_cast<byte_state *>(context);
            return value.iface->capture_stash != nullptr &&
                value.iface->capture_stash(
                    value.iface->context, edge.source_type,
                    value.current, value.stash);
        };
        driver.transcode = [](void * context,
                              const vbr_downward_edge & edge) noexcept {
            auto & value = *static_cast<byte_state *>(context);
            value.next.clear();
            const auto * stash = value.stash.empty() ? nullptr : &value.stash;
            if (!value.iface->transcode(
                    value.iface->context, edge, value.current,
                    stash, value.next) || value.next.empty()) {
                return false;
            }
            value.current.swap(value.next);
            return true;
        };
        out.status = vbr_downward_execute_edges(
            recipe, driver, out.stash_regenerated);
        if (out.status == vbr_downward_transform_status::transformed) {
            out.bytes = std::move(state.current);
            out.stash = std::move(state.stash);
        }
        return out;
    } catch (...) {
        out.status = vbr_downward_transform_status::internal_error;
        return out;
    }
}
