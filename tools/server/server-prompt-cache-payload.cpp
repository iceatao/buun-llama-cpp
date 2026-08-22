#include "server-prompt-cache-payload.h"

#include "../../src/llama-vbr-artifact-catalog.h"
#include "../../src/llama-vbr-downward.h"

#include <algorithm>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <utility>
#include <vector>

struct server_prompt_cache_vbr_payload::impl {
    vbr_artifact_package_view package;
    llama_cache_acct_artifact_id reference;
    uint64_t logical = 0;
    uint64_t resident = 0;
    size_t allocations = 0;
};

struct server_prompt_cache_vbr_variant_set::impl {
    server_prompt_cache_vbr_owner compact;
    server_prompt_cache_vbr_owner anchor;
    uint64_t logical = 0;
    uint64_t resident = 0;
    size_t allocations = 0;
};

namespace {

bool allocation_equal(
        const vbr_artifact_allocation_view & a,
        const vbr_artifact_allocation_view & b) noexcept {
    return a.category == b.category &&
           a.domain == b.domain &&
           a.logical == b.logical &&
           a.resident == b.resident &&
           a.allocation == b.allocation &&
           a.artifact == b.artifact &&
           a.content == b.content &&
           a.lineage == b.lineage;
}

bool identity_equal(
        const vbr_artifact_identity_block & a,
        const vbr_artifact_identity_block & b) noexcept {
    return a.execution_identity == b.execution_identity &&
           a.adapter_config_identity == b.adapter_config_identity &&
           a.media_content_identity == b.media_content_identity &&
           a.sequence_epoch == b.sequence_epoch &&
           a.token_count == b.token_count &&
           a.next_position == b.next_position;
}

bool companion_equal(
        const vbr_artifact_companion_payload & a,
        const vbr_artifact_companion_payload & b) noexcept {
    return a.kind == b.kind &&
           a.format_version == b.format_version &&
           a.build_identity_digest == b.build_identity_digest &&
           a.domain == b.domain &&
           a.payload_digest == b.payload_digest &&
           a.payload_bytes == b.payload_bytes &&
           a.section_checksum == b.section_checksum;
}

bool same_frontier(
        const vbr_artifact_reference_manifest & a,
        const vbr_artifact_reference_manifest & b) noexcept {
    if (a.identity_policy_order_digest != b.identity_policy_order_digest ||
        !identity_equal(a.identity, b.identity) ||
        a.token_block.codec_version != b.token_block.codec_version ||
        a.token_block.digest != b.token_block.digest ||
        a.token_block.tokens != b.token_block.tokens ||
        a.companions.size() != b.companions.size()) {
        return false;
    }
    for (size_t i = 0; i < a.companions.size(); ++i) {
        if (!companion_equal(a.companions[i], b.companions[i])) {
            return false;
        }
    }
    return true;
}

bool same_logical_unit_geometry(
        const vbr_artifact_unit_descriptor & a,
        const vbr_artifact_unit_descriptor & b) noexcept {
    return a.child_id == b.child_id &&
           a.logical_unit_id == b.logical_unit_id &&
           a.side == b.side &&
           a.layout == b.layout &&
           a.n_stream == b.n_stream &&
           a.unified == b.unified &&
           a.wm_cells == b.wm_cells &&
           a.rank == b.rank &&
           a.dimensions == b.dimensions;
}

bool logical_unit_less(
        const vbr_artifact_unit_view * a,
        const vbr_artifact_unit_view * b) noexcept {
    const auto & lhs = a->descriptor;
    const auto & rhs = b->descriptor;
    if (lhs.child_id != rhs.child_id) {
        return lhs.child_id < rhs.child_id;
    }
    if (lhs.logical_unit_id != rhs.logical_unit_id) {
        return lhs.logical_unit_id < rhs.logical_unit_id;
    }
    return uint8_t(lhs.side) < uint8_t(rhs.side);
}

bool quality_anchor_dominates(
        const vbr_artifact_package_view & compact,
        const vbr_artifact_package_view & anchor) {
    const auto & compact_units = compact.units();
    const auto & anchor_units = anchor.units();
    if (compact_units.empty() || compact_units.size() != anchor_units.size()) {
        return false;
    }

    std::vector<const vbr_artifact_unit_view *> ordered_compact;
    std::vector<const vbr_artifact_unit_view *> ordered_anchor;
    ordered_compact.reserve(compact_units.size());
    ordered_anchor.reserve(anchor_units.size());
    for (const auto & unit : compact_units) {
        ordered_compact.push_back(&unit);
    }
    for (const auto & unit : anchor_units) {
        ordered_anchor.push_back(&unit);
    }
    std::sort(
        ordered_compact.begin(), ordered_compact.end(), logical_unit_less);
    std::sort(
        ordered_anchor.begin(), ordered_anchor.end(), logical_unit_less);

    bool strictly_better = false;
    for (size_t i = 0; i < ordered_compact.size(); ++i) {
        const auto & compact_descriptor = ordered_compact[i]->descriptor;
        const auto & anchor_descriptor = ordered_anchor[i]->descriptor;
        if (!same_logical_unit_geometry(
                compact_descriptor, anchor_descriptor)) {
            return false;
        }
        if (anchor_descriptor.representation.source_loss_history >
                compact_descriptor.representation.source_loss_history ||
            anchor_descriptor.representation.checkpoint_codec_hops >
                compact_descriptor.representation.checkpoint_codec_hops) {
            return false;
        }
        strictly_better |=
            anchor_descriptor.representation.source_loss_history <
                compact_descriptor.representation.source_loss_history ||
            anchor_descriptor.representation.checkpoint_codec_hops <
                compact_descriptor.representation.checkpoint_codec_hops;

        vbr_downward_recipe recipe;
        const auto quality = vbr_downward_resolve_recipe(
            ggml_type(anchor_descriptor.current_type),
            ggml_type(compact_descriptor.current_type),
            ggml_type(compact_descriptor.current_type), true, recipe);
        if (quality == vbr_downward_recipe_status::resolved) {
            strictly_better = true;
        } else if (quality != vbr_downward_recipe_status::equal_tier) {
            return false;
        }
    }
    return strictly_better;
}

bool allocation_row_count(
        const vbr_artifact_package_view & package,
        size_t & count) noexcept {
    count = package.reference_allocations().size();
    for (const auto & unit : package.units()) {
        if (unit.payload_allocations.size() >
                std::numeric_limits<size_t>::max() - count) {
            return false;
        }
        count += unit.payload_allocations.size();
        if (unit.stash_allocations.size() >
                std::numeric_limits<size_t>::max() - count) {
            return false;
        }
        count += unit.stash_allocations.size();
    }
    return true;
}

void append_allocations(
        const vbr_artifact_package_view & package,
        std::vector<vbr_artifact_allocation_view> & allocations) {
    allocations.insert(
        allocations.end(), package.reference_allocations().begin(),
        package.reference_allocations().end());
    for (const auto & unit : package.units()) {
        allocations.insert(
            allocations.end(), unit.payload_allocations.begin(),
            unit.payload_allocations.end());
        allocations.insert(
            allocations.end(), unit.stash_allocations.begin(),
            unit.stash_allocations.end());
    }
}

} // namespace

bool server_prompt_cache_summarize_vbr_allocations(
        std::vector<vbr_artifact_allocation_view> allocations,
        server_prompt_cache_vbr_accounting_summary & summary) noexcept {
    summary = {};
    try {
        server_prompt_cache_vbr_accounting_summary result;
        std::sort(
            allocations.begin(), allocations.end(),
            [](const auto & a, const auto & b) {
                return a.allocation.v < b.allocation.v;
            });

        const vbr_artifact_allocation_view * previous = nullptr;
        for (const auto & value : allocations) {
            if (!value.allocation) {
                if (value.logical != 0 || value.resident != 0) {
                    return false;
                }
                continue;
            }
            if (previous && previous->allocation == value.allocation) {
                if (!allocation_equal(*previous, value)) {
                    return false;
                }
                continue;
            }
            if (value.logical > std::numeric_limits<uint64_t>::max() -
                                    result.logical_bytes ||
                value.resident > std::numeric_limits<uint64_t>::max() -
                                     result.resident_bytes) {
                return false;
            }
            result.logical_bytes += value.logical;
            result.resident_bytes += value.resident;
            ++result.allocation_count;
            previous = &value;
        }
        if (result.allocation_count == 0 ||
            result.resident_bytes > SIZE_MAX) {
            return false;
        }
        summary = result;
        return true;
    } catch (...) {
        return false;
    }
}

std::shared_ptr<const server_prompt_cache_vbr_payload>
server_prompt_cache_vbr_payload::adopt(
        vbr_artifact_package_view && package) noexcept {
    if (!package || package.reference_artifact().v == 0) {
        return {};
    }
    try {
        size_t allocation_rows = 0;
        if (!allocation_row_count(package, allocation_rows)) {
            return {};
        }

        std::vector<vbr_artifact_allocation_view> allocations;
        allocations.reserve(allocation_rows);
        append_allocations(package, allocations);
        server_prompt_cache_vbr_accounting_summary summary;
        if (!server_prompt_cache_summarize_vbr_allocations(
                std::move(allocations), summary)) {
            return {};
        }

        auto state = std::unique_ptr<impl>(new impl);
        state->reference = package.reference_artifact();
        state->logical = summary.logical_bytes;
        state->resident = summary.resident_bytes;
        state->allocations = summary.allocation_count;
        state->package = std::move(package);
        return std::shared_ptr<const server_prompt_cache_vbr_payload>(
            new server_prompt_cache_vbr_payload(std::move(state)));
    } catch (...) {
        return {};
    }
}

std::shared_ptr<const server_prompt_cache_vbr_payload>
server_prompt_cache_vbr_payload::adopt_owned(
        vbr_artifact_package_view && package) noexcept {
    if (!package.claim_host_ownership()) {
        return {};
    }
    return adopt(std::move(package));
}

server_prompt_cache_vbr_payload::server_prompt_cache_vbr_payload(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

server_prompt_cache_vbr_payload::~server_prompt_cache_vbr_payload() = default;

llama_cache_acct_artifact_id
server_prompt_cache_vbr_payload::reference_artifact() const noexcept {
    return impl_ ? impl_->reference : llama_cache_acct_artifact_id {};
}

uint64_t server_prompt_cache_vbr_payload::logical_bytes() const noexcept {
    return impl_ ? impl_->logical : 0;
}

uint64_t server_prompt_cache_vbr_payload::resident_bytes() const noexcept {
    return impl_ ? impl_->resident : 0;
}

size_t server_prompt_cache_vbr_payload::allocation_count() const noexcept {
    return impl_ ? impl_->allocations : 0;
}

const vbr_artifact_package_view &
server_prompt_cache_vbr_payload::package() const noexcept {
    static const vbr_artifact_package_view empty;
    return impl_ ? impl_->package : empty;
}

bool server_prompt_cache_vbr_payload::accounted_by(
        const llama_cache_acct_ledger * ledger) const noexcept {
    return impl_ && impl_->package.accounted_by(ledger);
}

bool server_prompt_cache_vbr_payload::retirement_owned() const noexcept {
    return impl_ && impl_->package.host_owned();
}

std::shared_ptr<const server_prompt_cache_vbr_variant_set>
server_prompt_cache_vbr_variant_set::create(
        server_prompt_cache_vbr_owner compact_current,
        server_prompt_cache_vbr_owner quality_anchor) noexcept {
    if (!compact_current ||
        (quality_anchor && quality_anchor == compact_current)) {
        return {};
    }
    try {
        const auto & compact_package = compact_current->package();
        if (!compact_package) {
            return {};
        }
        if (!quality_anchor) {
            auto state = std::unique_ptr<impl>(new impl);
            state->compact = std::move(compact_current);
            state->logical = state->compact->logical_bytes();
            state->resident = state->compact->resident_bytes();
            state->allocations = state->compact->allocation_count();
            return std::shared_ptr<const server_prompt_cache_vbr_variant_set>(
                new server_prompt_cache_vbr_variant_set(std::move(state)));
        }

        size_t allocation_rows = 0;
        if (!allocation_row_count(compact_package, allocation_rows)) {
            return {};
        }
        const auto & anchor_package = quality_anchor->package();
        size_t anchor_rows = 0;
        if (!anchor_package ||
            !anchor_package.same_catalog(compact_package) ||
            !same_frontier(
                compact_package.manifest(), anchor_package.manifest()) ||
            !quality_anchor_dominates(compact_package, anchor_package) ||
            !allocation_row_count(anchor_package, anchor_rows) ||
            anchor_rows >
                std::numeric_limits<size_t>::max() - allocation_rows) {
            return {};
        }
        allocation_rows += anchor_rows;

        std::vector<vbr_artifact_allocation_view> allocations;
        allocations.reserve(allocation_rows);
        append_allocations(compact_package, allocations);
        append_allocations(anchor_package, allocations);
        server_prompt_cache_vbr_accounting_summary summary;
        if (!server_prompt_cache_summarize_vbr_allocations(
                std::move(allocations), summary)) {
            return {};
        }

        auto state = std::unique_ptr<impl>(new impl);
        state->compact = std::move(compact_current);
        state->anchor = std::move(quality_anchor);
        state->logical = summary.logical_bytes;
        state->resident = summary.resident_bytes;
        state->allocations = summary.allocation_count;
        return std::shared_ptr<const server_prompt_cache_vbr_variant_set>(
            new server_prompt_cache_vbr_variant_set(std::move(state)));
    } catch (...) {
        return {};
    }
}

server_prompt_cache_vbr_variant_set::server_prompt_cache_vbr_variant_set(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

server_prompt_cache_vbr_variant_set::~server_prompt_cache_vbr_variant_set() =
    default;

const server_prompt_cache_vbr_owner &
server_prompt_cache_vbr_variant_set::compact_current() const noexcept {
    static const server_prompt_cache_vbr_owner empty;
    return impl_ ? impl_->compact : empty;
}

const server_prompt_cache_vbr_owner &
server_prompt_cache_vbr_variant_set::quality_anchor() const noexcept {
    static const server_prompt_cache_vbr_owner empty;
    return impl_ ? impl_->anchor : empty;
}

uint64_t server_prompt_cache_vbr_variant_set::logical_bytes() const noexcept {
    return impl_ ? impl_->logical : 0;
}

uint64_t server_prompt_cache_vbr_variant_set::resident_bytes() const noexcept {
    return impl_ ? impl_->resident : 0;
}

uint64_t server_prompt_cache_vbr_variant_set::anchor_resident_bytes()
        const noexcept {
    return impl_ && impl_->compact &&
            impl_->compact->resident_bytes() <= impl_->resident
        ? impl_->resident - impl_->compact->resident_bytes() : 0;
}

size_t server_prompt_cache_vbr_variant_set::allocation_count() const noexcept {
    return impl_ ? impl_->allocations : 0;
}

bool server_prompt_cache_vbr_variant_set::accounted_by(
        const llama_cache_acct_ledger * ledger) const noexcept {
    return impl_ && impl_->compact && impl_->compact->accounted_by(ledger) &&
           (!impl_->anchor || impl_->anchor->accounted_by(ledger));
}

bool server_prompt_cache_vbr_variant_set::retirement_owned() const noexcept {
    return impl_ && impl_->compact && impl_->compact->retirement_owned() &&
           (!impl_->anchor || impl_->anchor->retirement_owned());
}

bool server_prompt_cache_vbr_variant_set::retirement_exclusive() const noexcept {
    return retirement_owned() && impl_->compact.use_count() == 1 &&
           (!impl_->anchor || impl_->anchor.use_count() == 1);
}

bool server_prompt_cache_vbr_variant_set::logical_erase_preserves_storage()
        const noexcept {
    return retirement_owned() && impl_->compact.use_count() > 1 &&
           (!impl_->anchor || impl_->anchor.use_count() > 1);
}

bool server_prompt_cache_vbr_variant_set::has_quality_anchor() const noexcept {
    return impl_ && bool(impl_->anchor);
}

bool server_prompt_cache_vbr_variant_set::prepare_anchor_retire_owned(
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept {
    out.reset();
    if (!impl_ || !impl_->anchor ||
        !impl_->anchor->retirement_owned()) {
        return false;
    }
    try {
        std::vector<const vbr_artifact_package_view *> packages {
            &impl_->anchor->package(),
        };
        return packages.front()->prepare_owned_retire(
            packages, expected_serial, out);
    } catch (...) {
        out.reset();
        return false;
    }
}

std::shared_ptr<const server_prompt_cache_vbr_variant_set>
server_prompt_cache_vbr_variant_set::compact_only() const noexcept {
    return impl_ && impl_->compact
        ? create(impl_->compact) : nullptr;
}

bool server_prompt_cache_vbr_variant_set::preview_retire(
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept {
    out = {};
    if (!retirement_exclusive()) {
        return false;
    }
    try {
        std::vector<const vbr_artifact_package_view *> packages;
        packages.reserve(impl_->anchor ? 2 : 1);
        packages.push_back(&impl_->compact->package());
        if (impl_->anchor) {
            packages.push_back(&impl_->anchor->package());
        }
        return packages.front()->preview_owned_retire(
            packages, expected_serial, out);
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_prompt_cache_vbr_variant_set::prepare_retire(
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept {
    out.reset();
    if (!retirement_exclusive()) {
        return false;
    }
    try {
        std::vector<const vbr_artifact_package_view *> packages;
        packages.reserve(impl_->anchor ? 2 : 1);
        packages.push_back(&impl_->compact->package());
        if (impl_->anchor) {
            packages.push_back(&impl_->anchor->package());
        }
        return packages.front()->prepare_owned_retire(
            packages, expected_serial, out);
    } catch (...) {
        return false;
    }
}

bool server_prompt_cache_vbr_variant_set::preview_retire_union(
        const std::vector<const server_prompt_cache_vbr_variant_set *> & variants,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) noexcept {
    out = {};
    if (variants.empty()) {
        return false;
    }
    try {
        std::vector<const vbr_artifact_package_view *> packages;
        for (const auto * variant : variants) {
            if (!variant || !variant->retirement_owned() ||
                !variant->impl_ || !variant->impl_->compact) {
                return false;
            }
            packages.push_back(&variant->impl_->compact->package());
            if (variant->impl_->anchor) {
                packages.push_back(&variant->impl_->anchor->package());
            }
        }
        std::sort(packages.begin(), packages.end(), std::less<>());
        packages.erase(
            std::unique(packages.begin(), packages.end()), packages.end());
        return packages.front()->preview_owned_retire(
            packages, expected_serial, out);
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_prompt_cache_vbr_variant_set::preview_retire_resident_batch(
        const std::vector<const server_prompt_cache_vbr_variant_set *> & variants,
        uint64_t expected_serial,
        std::vector<vbr_artifact_retire_resident_preview> & out) noexcept {
    out.clear();
    if (variants.empty()) {
        return false;
    }
    try {
        if (variants.size() > SIZE_MAX/2) {
            return false;
        }
        std::vector<const vbr_artifact_package_view *> package_arena;
        std::vector<vbr_artifact_package_set_view> groups;
        package_arena.reserve(variants.size()*2);
        groups.reserve(variants.size());
        for (const auto * variant : variants) {
            if (!variant || !variant->retirement_exclusive() ||
                !variant->impl_ || !variant->impl_->compact) {
                return false;
            }
            const size_t first = package_arena.size();
            package_arena.push_back(&variant->impl_->compact->package());
            if (variant->impl_->anchor) {
                package_arena.push_back(&variant->impl_->anchor->package());
            }
            groups.push_back({
                package_arena.data() + first,
                package_arena.size() - first,
            });
        }
        return package_arena.front()->preview_owned_retire_resident_batch(
            groups, expected_serial, out);
    } catch (...) {
        out.clear();
        return false;
    }
}

server_prompt_cache_payload server_prompt_cache_payload::from_vbr(
        vbr_owner owner) noexcept {
    return from_vbr_variants(
        server_prompt_cache_vbr_variant_set::create(std::move(owner)));
}

server_prompt_cache_payload server_prompt_cache_payload::from_vbr_variants(
        vbr_variant_owner variants) noexcept {
    server_prompt_cache_payload result;
    result.storage_ = std::move(variants);
    return result;
}

server_prompt_cache_payload_kind
server_prompt_cache_payload::kind() const noexcept {
    return std::holds_alternative<server_prompt_data>(storage_)
        ? server_prompt_cache_payload_kind::fixed_state
        : server_prompt_cache_payload_kind::vbr_artifact;
}

server_prompt_data * server_prompt_cache_payload::fixed_state() noexcept {
    return std::get_if<server_prompt_data>(&storage_);
}

const server_prompt_data *
server_prompt_cache_payload::fixed_state() const noexcept {
    return std::get_if<server_prompt_data>(&storage_);
}

const server_prompt_cache_vbr_payload *
server_prompt_cache_payload::vbr_artifact() const noexcept {
    const auto * variants = vbr_variants();
    return variants && variants->compact_current()
        ? variants->compact_current().get()
        : nullptr;
}

server_prompt_cache_payload::vbr_owner
server_prompt_cache_payload::vbr_compact_owner() const noexcept {
    const auto * variants = vbr_variants();
    return variants ? variants->compact_current() : vbr_owner {};
}

const server_prompt_cache_vbr_variant_set *
server_prompt_cache_payload::vbr_variants() const noexcept {
    const auto * owner = std::get_if<vbr_variant_owner>(&storage_);
    return owner && *owner ? owner->get() : nullptr;
}

bool server_prompt_cache_payload::valid() const noexcept {
    const auto * fixed = fixed_state();
    return fixed ? !fixed->main.empty() : vbr_artifact() != nullptr;
}

bool server_prompt_cache_payload::publishable() const noexcept {
    return valid();
}

bool server_prompt_cache_payload::restorable() const noexcept {
    const auto * fixed = fixed_state();
    return fixed && !fixed->main.empty();
}

bool server_prompt_cache_payload::accounted_by(
        const llama_cache_acct_ledger * ledger) const noexcept {
    const auto * variants = vbr_variants();
    return variants && variants->accounted_by(ledger);
}

bool server_prompt_cache_payload::vbr_retirement_owned() const noexcept {
    const auto * variants = vbr_variants();
    return variants && variants->retirement_owned();
}

bool server_prompt_cache_payload::vbr_retirement_exclusive() const noexcept {
    const auto * owner = std::get_if<vbr_variant_owner>(&storage_);
    return owner && *owner && owner->use_count() == 1 &&
           (*owner)->retirement_exclusive();
}

bool server_prompt_cache_payload::vbr_logical_erase_only() const noexcept {
    const auto * owner = std::get_if<vbr_variant_owner>(&storage_);
    return owner && *owner && (*owner)->retirement_owned() &&
           (owner->use_count() > 1 ||
            (*owner)->logical_erase_preserves_storage());
}

bool server_prompt_cache_payload::vbr_has_quality_anchor() const noexcept {
    const auto * variants = vbr_variants();
    return variants && variants->has_quality_anchor();
}

uint64_t server_prompt_cache_payload::vbr_anchor_resident_bytes()
        const noexcept {
    const auto * variants = vbr_variants();
    return variants ? variants->anchor_resident_bytes() : 0;
}

bool server_prompt_cache_payload::prepare_vbr_refresh(
        vbr_owner incoming,
        server_prompt_cache_payload & replacement,
        bool allow_anchor,
        bool & unchanged) const noexcept {
    replacement = {};
    unchanged = false;
    const auto * variants = vbr_variants();
    if (!incoming || !variants || !variants->compact_current()) {
        return false;
    }
    const auto best = variants->quality_anchor()
        ? variants->quality_anchor() : variants->compact_current();
    try {
        if (!same_frontier(
                incoming->package().manifest(),
                best->package().manifest())) {
            return false;
        }
        if (!quality_anchor_dominates(
                incoming->package(), best->package())) {
            unchanged = true;
            return false;
        }
    } catch (...) {
        return false;
    }
    if (!allow_anchor && !variants->quality_anchor()) {
        replacement = from_vbr(std::move(incoming));
        return replacement.valid();
    }
    auto refreshed = server_prompt_cache_vbr_variant_set::create(
        std::move(incoming), std::move(best));
    if (!refreshed) {
        return false;
    }
    replacement = from_vbr_variants(std::move(refreshed));
    return replacement.valid() && replacement.vbr_has_quality_anchor();
}

void server_prompt_cache_payload::swap_vbr_storage(
        server_prompt_cache_payload & other) noexcept {
    auto * lhs = std::get_if<vbr_variant_owner>(&storage_);
    auto * rhs = std::get_if<vbr_variant_owner>(&other.storage_);
    GGML_ASSERT(lhs && rhs);
    lhs->swap(*rhs);
}

void server_prompt_cache_payload::reset_vbr_storage() noexcept {
    auto * owner = std::get_if<vbr_variant_owner>(&storage_);
    GGML_ASSERT(owner);
    owner->reset();
}

bool server_prompt_cache_payload::prepare_vbr_compact_only(
        server_prompt_cache_payload & out) const noexcept {
    out = {};
    const auto * variants = vbr_variants();
    if (!variants || !variants->has_quality_anchor()) {
        return false;
    }
    auto compact = variants->compact_only();
    if (!compact) {
        return false;
    }
    out = from_vbr_variants(std::move(compact));
    return out.valid() && !out.vbr_has_quality_anchor();
}

bool server_prompt_cache_payload::prepare_vbr_anchor_retire_batch(
        const std::vector<const server_prompt_cache_payload *> & selected,
        uint64_t expected_serial,
        std::vector<vbr_artifact_prepared_retire> & out) noexcept {
    out.clear();
    if (selected.empty() || expected_serial == 0) {
        return false;
    }
    try {
        struct variant_state {
            const vbr_variant_owner * owner = nullptr;
            size_t selected_refs = 0;
        };
        std::map<const server_prompt_cache_vbr_variant_set *, variant_state>
            variants;
        std::set<const server_prompt_cache_payload *> unique_payloads;
        for (const auto * payload : selected) {
            if (!payload || !unique_payloads.insert(payload).second) {
                return false;
            }
            const auto * owner = std::get_if<vbr_variant_owner>(
                &payload->storage_);
            if (!owner || !*owner ||
                !(*owner)->has_quality_anchor()) {
                return false;
            }
            auto & state = variants[owner->get()];
            if (!state.owner) {
                // Two shared_ptr objects may name the same immutable variant;
                // either one is a valid use-count witness.
                state.owner = owner;
            }
            state.selected_refs++;
        }

        struct anchor_state {
            const server_prompt_cache_vbr_owner * owner = nullptr;
            const server_prompt_cache_vbr_variant_set * representative =
                nullptr;
            size_t destroyed_variants = 0;
        };
        std::map<const server_prompt_cache_vbr_payload *, anchor_state>
            anchors;
        for (const auto & entry : variants) {
            const auto & state = entry.second;
            if (!state.owner || state.selected_refs == 0 ||
                state.selected_refs > size_t(state.owner->use_count())) {
                return false;
            }
            if (state.selected_refs != size_t(state.owner->use_count())) {
                continue;
            }
            const auto & anchor = entry.first->quality_anchor();
            if (!anchor || !anchor->retirement_owned()) {
                return false;
            }
            auto & group = anchors[anchor.get()];
            if (!group.owner) {
                group.owner = &anchor;
                group.representative = entry.first;
            }
            group.destroyed_variants++;
        }

        size_t physical_groups = 0;
        for (const auto & entry : anchors) {
            const auto & group = entry.second;
            if (!group.owner || !*group.owner ||
                group.destroyed_variants >
                    size_t(group.owner->use_count())) {
                return false;
            }
            physical_groups += group.destroyed_variants ==
                size_t(group.owner->use_count());
        }
        out.reserve(physical_groups);
        for (const auto & entry : anchors) {
            const auto & group = entry.second;
            if (group.destroyed_variants !=
                    size_t(group.owner->use_count())) {
                continue;
            }
            vbr_artifact_prepared_retire prepared;
            if (!group.representative->prepare_anchor_retire_owned(
                    expected_serial, prepared)) {
                out.clear();
                return false;
            }
            out.push_back(std::move(prepared));
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_prompt_cache_payload::preview_vbr_retire(
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept {
    out = {};
    const auto * variants = vbr_variants();
    return vbr_retirement_exclusive() &&
           variants->preview_retire(expected_serial, out);
}

bool server_prompt_cache_payload::prepare_vbr_retire(
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept {
    out.reset();
    const auto * variants = vbr_variants();
    return vbr_retirement_exclusive() &&
           variants->prepare_retire(expected_serial, out);
}

bool server_prompt_cache_payload::preview_vbr_retire_union(
        const std::vector<const server_prompt_cache_payload *> & payloads,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) noexcept {
    out = {};
    if (payloads.empty()) {
        return false;
    }
    try {
        std::vector<const server_prompt_cache_vbr_variant_set *> variants;
        variants.reserve(payloads.size());
        for (const auto * payload : payloads) {
            if (!payload) {
                return false;
            }
            const auto * owner = std::get_if<vbr_variant_owner>(
                &payload->storage_);
            if (!owner || !*owner || !(*owner)->retirement_owned()) {
                return false;
            }
            variants.push_back(owner->get());
        }
        std::sort(variants.begin(), variants.end(), std::less<>());
        variants.erase(
            std::unique(variants.begin(), variants.end()), variants.end());
        return server_prompt_cache_vbr_variant_set::preview_retire_union(
            variants, expected_serial, out);
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_prompt_cache_payload::preview_vbr_retire_resident_batch(
        const std::vector<const server_prompt_cache_payload *> & payloads,
        uint64_t expected_serial,
        std::vector<vbr_artifact_retire_resident_preview> & out) noexcept {
    out.clear();
    if (payloads.empty()) {
        return false;
    }
    try {
        std::vector<const server_prompt_cache_vbr_variant_set *> variants;
        variants.reserve(payloads.size());
        for (const auto * payload : payloads) {
            const auto * owner = payload
                ? std::get_if<vbr_variant_owner>(&payload->storage_)
                : nullptr;
            if (!owner || !*owner) {
                return false;
            }
            variants.push_back(owner->get());
        }
        return server_prompt_cache_vbr_variant_set::
            preview_retire_resident_batch(variants, expected_serial, out);
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_prompt_cache_payload::summarize_vbr_budgets(
        const std::vector<const server_prompt_cache_payload *> & payloads,
        server_prompt_cache_vbr_budget_summary & out) noexcept {
    out = {};
    if (payloads.empty()) {
        return true;
    }
    try {
        size_t compact_rows = 0;
        size_t all_rows = 0;
        for (const auto * payload : payloads) {
            const auto * variants = payload ? payload->vbr_variants() : nullptr;
            if (!variants || !variants->compact_current()) {
                return false;
            }
            size_t rows = 0;
            if (!allocation_row_count(
                    variants->compact_current()->package(), rows) ||
                rows > std::numeric_limits<size_t>::max() - compact_rows) {
                return false;
            }
            compact_rows += rows;
            if (variants->quality_anchor()) {
                if (!allocation_row_count(
                        variants->quality_anchor()->package(), rows) ||
                    rows > std::numeric_limits<size_t>::max() - all_rows) {
                    return false;
                }
                all_rows += rows;
            }
        }
        const size_t anchor_rows = all_rows;
        if (compact_rows > std::numeric_limits<size_t>::max() - all_rows) {
            return false;
        }
        all_rows += compact_rows;

        std::vector<vbr_artifact_allocation_view> compact;
        std::vector<vbr_artifact_allocation_view> all;
        compact.reserve(compact_rows);
        if (anchor_rows != 0) {
            all.reserve(all_rows);
        }
        for (const auto * payload : payloads) {
            const auto * variants = payload->vbr_variants();
            append_allocations(variants->compact_current()->package(), compact);
            if (anchor_rows != 0) {
                append_allocations(
                    variants->compact_current()->package(), all);
                if (variants->quality_anchor()) {
                    append_allocations(
                        variants->quality_anchor()->package(), all);
                }
            }
        }
        server_prompt_cache_vbr_accounting_summary compact_summary;
        if (!server_prompt_cache_summarize_vbr_allocations(
                std::move(compact), compact_summary)) {
            return false;
        }
        if (anchor_rows == 0) {
            out.compact_resident_bytes = compact_summary.resident_bytes;
            out.compact_allocations = compact_summary.allocation_count;
            return true;
        }
        server_prompt_cache_vbr_accounting_summary all_summary;
        if (!server_prompt_cache_summarize_vbr_allocations(
                std::move(all), all_summary) ||
            compact_summary.resident_bytes > all_summary.resident_bytes ||
            compact_summary.allocation_count > all_summary.allocation_count) {
            return false;
        }
        out.compact_resident_bytes = compact_summary.resident_bytes;
        out.anchor_resident_bytes =
            all_summary.resident_bytes - compact_summary.resident_bytes;
        out.compact_allocations = compact_summary.allocation_count;
        out.anchor_allocations =
            all_summary.allocation_count - compact_summary.allocation_count;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_prompt_cache_plan_vbr_anchor_releases(
        const std::vector<server_prompt_cache_vbr_anchor_plan_candidate> & candidates,
        uint64_t current_anchor_bytes,
        uint64_t limit_anchor_bytes,
        std::vector<llama_cache_acct_artifact_id> & selected) noexcept {
    selected.clear();
    if (current_anchor_bytes <= limit_anchor_bytes) {
        return true;
    }
    if (candidates.empty() || candidates.size() >
            SERVER_PROMPT_CACHE_VBR_ANCHOR_MAX_CANDIDATES) {
        return false;
    }
    try {
        struct tagged_allocation {
            const vbr_artifact_allocation_view * value = nullptr;
            size_t candidate = 0;
            bool compact = false;
        };
        std::vector<tagged_allocation> tagged;
        std::vector<llama_cache_acct_artifact_id> artifacts;
        artifacts.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto & candidate = candidates[i];
            const auto * variants = candidate.payload
                ? candidate.payload->vbr_variants() : nullptr;
            if (!candidate.artifact_id.v || !variants ||
                !variants->compact_current() ||
                (candidate.eligible && !variants->quality_anchor())) {
                return false;
            }
            artifacts.push_back(candidate.artifact_id);
            const auto append = [&](const vbr_artifact_package_view & package,
                                    bool compact) {
                const auto add = [&](const auto & rows) {
                    for (const auto & row : rows) {
                        tagged.push_back({ &row, i, compact });
                    }
                };
                add(package.reference_allocations());
                for (const auto & unit : package.units()) {
                    add(unit.payload_allocations);
                    add(unit.stash_allocations);
                }
            };
            append(variants->compact_current()->package(), true);
            if (variants->quality_anchor()) {
                append(variants->quality_anchor()->package(), false);
            }
        }
        std::sort(artifacts.begin(), artifacts.end(),
            [](auto a, auto b) { return a.v < b.v; });
        if (std::adjacent_find(artifacts.begin(), artifacts.end()) !=
                artifacts.end()) {
            return false;
        }
        std::sort(tagged.begin(), tagged.end(), [](const auto & a,
                                                   const auto & b) {
            return std::tie(a.value->allocation.v, a.compact, a.candidate) <
                   std::tie(b.value->allocation.v, b.compact, b.candidate);
        });

        struct allocation_state {
            uint64_t resident = 0;
            uint32_t references = 0;
            std::vector<size_t> candidates;
        };
        std::vector<allocation_state> allocations;
        std::vector<std::vector<size_t>> candidate_allocations(
            candidates.size());
        uint64_t measured_anchor_bytes = 0;
        for (size_t first = 0; first < tagged.size();) {
            size_t last = first + 1;
            const auto * canonical = tagged[first].value;
            while (last < tagged.size() &&
                   tagged[last].value->allocation ==
                       canonical->allocation) {
                if (!allocation_equal(*canonical, *tagged[last].value)) {
                    return false;
                }
                last++;
            }
            if (!canonical->allocation) {
                if (canonical->logical != 0 || canonical->resident != 0) {
                    return false;
                }
                first = last;
                continue;
            }
            const bool compact = std::any_of(
                tagged.begin() + first, tagged.begin() + last,
                [](const auto & row) { return row.compact; });
            if (!compact) {
                allocation_state state;
                state.resident = canonical->resident;
                for (size_t i = first; i < last; ++i) {
                    if (tagged[i].compact ||
                        (!state.candidates.empty() &&
                         state.candidates.back() == tagged[i].candidate)) {
                        continue;
                    }
                    state.candidates.push_back(tagged[i].candidate);
                }
                if (state.candidates.size() > UINT32_MAX ||
                    state.resident > UINT64_MAX - measured_anchor_bytes) {
                    return false;
                }
                state.references = uint32_t(state.candidates.size());
                if (state.references != 0) {
                    const size_t index = allocations.size();
                    for (const size_t candidate : state.candidates) {
                        candidate_allocations[candidate].push_back(index);
                    }
                    measured_anchor_bytes += state.resident;
                    allocations.push_back(std::move(state));
                }
            }
            first = last;
        }
        if (measured_anchor_bytes != current_anchor_bytes) {
            return false;
        }

        std::vector<size_t> priority;
        priority.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].eligible) {
                priority.push_back(i);
            }
        }
        std::sort(priority.begin(), priority.end(), [&](size_t a, size_t b) {
            const auto & lhs = candidates[a];
            const auto & rhs = candidates[b];
            return std::tie(lhs.parent_value_q, lhs.recency_ordinal,
                            lhs.pool, lhs.lineage_id, lhs.artifact_id.v) <
                   std::tie(rhs.parent_value_q, rhs.recency_ordinal,
                            rhs.pool, rhs.lineage_id, rhs.artifact_id.v);
        });

        struct choice {
            uint64_t bytes = 0;
            size_t candidate = 0;
        };
        struct choice_less {
            const std::vector<server_prompt_cache_vbr_anchor_plan_candidate> *
                candidates = nullptr;
            bool operator()(const choice & a, const choice & b) const {
                if (a.bytes != b.bytes) {
                    return a.bytes > b.bytes;
                }
                const auto & lhs = (*candidates)[a.candidate];
                const auto & rhs = (*candidates)[b.candidate];
                return std::tie(lhs.pool, lhs.lineage_id,
                                lhs.artifact_id.v) <
                       std::tie(rhs.pool, rhs.lineage_id,
                                rhs.artifact_id.v);
            }
        };
        std::vector<uint64_t> marginal(candidates.size(), 0);
        std::vector<bool> remaining(candidates.size(), true);
        for (size_t first = 0; first < priority.size() &&
                measured_anchor_bytes > limit_anchor_bytes;) {
            size_t last = first + 1;
            const auto & group = candidates[priority[first]];
            while (last < priority.size()) {
                const auto & next = candidates[priority[last]];
                if (next.parent_value_q != group.parent_value_q ||
                    next.recency_ordinal != group.recency_ordinal) {
                    break;
                }
                last++;
            }
            std::set<choice, choice_less> choices(
                choice_less { &candidates });
            for (size_t i = first; i < last; ++i) {
                const size_t candidate = priority[i];
                uint64_t bytes = 0;
                for (const size_t allocation :
                        candidate_allocations[candidate]) {
                    if (allocations[allocation].references == 1) {
                        if (allocations[allocation].resident >
                                UINT64_MAX - bytes) {
                            return false;
                        }
                        bytes += allocations[allocation].resident;
                    }
                }
                marginal[candidate] = bytes;
                choices.insert({ bytes, candidate });
            }
            while (!choices.empty() &&
                   measured_anchor_bytes > limit_anchor_bytes) {
                const choice current = *choices.begin();
                choices.erase(choices.begin());
                const size_t candidate = current.candidate;
                remaining[candidate] = false;
                selected.push_back(candidates[candidate].artifact_id);
                for (const size_t allocation :
                        candidate_allocations[candidate]) {
                    auto & state = allocations[allocation];
                    if (state.references == 0) {
                        return false;
                    }
                    state.references--;
                    if (state.references == 0) {
                        if (state.resident > measured_anchor_bytes) {
                            return false;
                        }
                        measured_anchor_bytes -= state.resident;
                    } else if (state.references == 1) {
                        size_t survivor = candidates.size();
                        for (const size_t value : state.candidates) {
                            if (remaining[value]) {
                                survivor = value;
                                break;
                            }
                        }
                        if (survivor != candidates.size() &&
                            candidates[survivor].eligible &&
                            candidates[survivor].parent_value_q ==
                                group.parent_value_q &&
                            candidates[survivor].recency_ordinal ==
                                group.recency_ordinal) {
                            choices.erase({ marginal[survivor], survivor });
                            if (state.resident >
                                    UINT64_MAX - marginal[survivor]) {
                                return false;
                            }
                            marginal[survivor] += state.resident;
                            choices.insert({ marginal[survivor], survivor });
                        }
                    }
                }
            }
            first = last;
        }
        if (measured_anchor_bytes > limit_anchor_bytes) {
            selected.clear();
            return false;
        }
        return true;
    } catch (...) {
        selected.clear();
        return false;
    }
}

size_t server_prompt_cache_payload::size() const noexcept {
    const auto * fixed = fixed_state();
    if (fixed) {
        return fixed->size();
    }
    const auto * variants = vbr_variants();
    return variants ? size_t(variants->resident_bytes()) : 0;
}

bool server_prompt_cache_payload::same_storage(
        const server_prompt_cache_payload & other) const noexcept {
    if (kind() != other.kind()) {
        return false;
    }
    if (const auto * fixed = fixed_state()) {
        const auto * rhs = other.fixed_state();
        return rhs && fixed->main == rhs->main && fixed->drft == rhs->drft;
    }
    const auto * lhs = std::get_if<vbr_variant_owner>(&storage_);
    const auto * rhs = std::get_if<vbr_variant_owner>(&other.storage_);
    if (!lhs || !rhs || !*lhs || !*rhs) {
        return false;
    }
    // Artifact IDs are catalog-local. The immutable retained owners are the
    // storage identity; independently assembled sets over those same owners
    // are therefore exact aliases as well.
    return (*lhs)->compact_current() == (*rhs)->compact_current() &&
           (*lhs)->quality_anchor() == (*rhs)->quality_anchor();
}
