#include "server-prompt-cache-payload.h"

#include "../../src/llama-vbr-artifact-catalog.h"

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
        size_t allocation_rows = package.reference_allocations().size();
        for (const auto & unit : package.units()) {
            if (unit.payload_allocations.size() >
                    std::numeric_limits<size_t>::max() - allocation_rows) {
                return {};
            }
            allocation_rows += unit.payload_allocations.size();
            if (unit.stash_allocations.size() >
                    std::numeric_limits<size_t>::max() - allocation_rows) {
                return {};
            }
            allocation_rows += unit.stash_allocations.size();
        }

        std::vector<vbr_artifact_allocation_view> allocations;
        allocations.reserve(allocation_rows);
        const auto append = [&](const auto & values) {
            allocations.insert(
                allocations.end(), values.begin(), values.end());
        };
        append(package.reference_allocations());
        for (const auto & unit : package.units()) {
            append(unit.payload_allocations);
            append(unit.stash_allocations);
        }
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

server_prompt_cache_payload server_prompt_cache_payload::from_vbr(
        vbr_owner owner) noexcept {
    server_prompt_cache_payload result;
    result.storage_ = std::move(owner);
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
    const auto * owner = std::get_if<vbr_owner>(&storage_);
    return owner && *owner ? owner->get() : nullptr;
}

bool server_prompt_cache_payload::valid() const noexcept {
    const auto * fixed = fixed_state();
    return fixed ? !fixed->main.empty() : vbr_artifact() != nullptr;
}

bool server_prompt_cache_payload::publishable() const noexcept {
    // H1 has not yet installed the VBR restore/admission transaction. Keeping
    // this fixed-only prevents a sealed lease from being mistaken for a state
    // image while allowing the real backend owner to exist behind the type.
    const auto * fixed = fixed_state();
    return fixed && !fixed->main.empty();
}

size_t server_prompt_cache_payload::size() const noexcept {
    const auto * fixed = fixed_state();
    if (fixed) {
        return fixed->size();
    }
    const auto * vbr = vbr_artifact();
    return vbr ? size_t(vbr->resident_bytes()) : 0;
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
    const auto * lhs = std::get_if<vbr_owner>(&storage_);
    const auto * rhs = std::get_if<vbr_owner>(&other.storage_);
    // Artifact IDs are catalog-local. Only aliases of this exact retained
    // owner are known to share immutable backing storage in H1.
    return lhs && rhs && *lhs && *lhs == *rhs;
}
