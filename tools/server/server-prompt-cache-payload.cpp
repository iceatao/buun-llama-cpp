#include "server-prompt-cache-payload.h"

#include "../../src/llama-vbr-artifact-catalog.h"
#include "../../src/llama-vbr-downward.h"

#include <algorithm>
#include <limits>
#include <new>
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
