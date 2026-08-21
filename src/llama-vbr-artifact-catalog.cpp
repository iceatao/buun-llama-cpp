#include "llama-vbr-artifact-catalog.h"

#include "llama-sha256.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>

namespace {

using digest_key = std::array<uint8_t, 32>;

enum class intern_purpose : uint8_t {
    unit = 0,
    stash,
    manifest,
    logical_unit,
    _count,
};

using intern_key = std::pair<intern_purpose, digest_key>;

struct configured_cell {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
};

bool operator<(const configured_cell & a, const configured_cell & b) {
    if (a.category != b.category) {
        return a.category < b.category;
    }
    if (a.domain.residency != b.domain.residency) {
        return a.domain.residency < b.domain.residency;
    }
    if (a.domain.kind != b.domain.kind) {
        return a.domain.kind < b.domain.kind;
    }
    if (a.domain.topology.v != b.domain.topology.v) {
        return a.domain.topology.v < b.domain.topology.v;
    }
    return a.domain.device_ordinal.v < b.domain.device_ordinal.v;
}

} // namespace

struct llama_vbr_artifact_catalog::impl {
    struct allocation {
        llama_cache_acct_category category =
            llama_cache_acct_category::container_overhead;
        llama_cache_acct_resource_domain domain;
        uint64_t logical = 0;
        uint64_t resident = 0;
        llama_cache_acct_alloc_id alloc;
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest content;
        llama_cache_acct_lineage_id lineage;
    };

    struct blob {
        vbr_unit_version_id id;
        vbr_payload_digest payload_digest;
        vbr_stash_payload_id stash_id;
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest content;
        llama_cache_acct_lineage_id lineage;
        vbr_artifact_unit_descriptor descriptor;
        std::vector<std::shared_ptr<const artifact_segment_chain>>
            payload_shards;
        std::vector<allocation> allocations;
    };

    struct stash {
        vbr_stash_payload_id id;
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest content;
        llama_cache_acct_lineage_id lineage;
        vbr_artifact_clean_stash descriptor;
        std::vector<std::shared_ptr<const artifact_segment_chain>>
            shards;
        std::vector<allocation> allocations;
    };

    struct reference {
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest unit_content;
        llama_cache_acct_lineage_id lineage;
        std::vector<vbr_unit_version_id> unit_ids;
        std::vector<vbr_stash_payload_id> stash_ids;
        vbr_artifact_reference_manifest manifest;
        std::vector<std::shared_ptr<const artifact_segment_chain>>
            companion_payloads;
        std::vector<llama_cache_acct_op_id> operations;
        std::vector<allocation> allocations;
        uint64_t borrow_count = 0;
        bool host_owned = false;
        bool retire_pending = false;
        uint64_t prepared_retire_token = 0;
    };

    struct txn_leaf {
        allocation binding;
        uint64_t reserve_resident = 0;
        bool existing = false;
    };

    explicit impl(llama_cache_acct_ledger & ledger_) : ledger(ledger_) {}

    bool resolve_domain(
            const vbr_artifact_portable_domain & portable,
            llama_cache_acct_resource_domain & out) const {
        out = {};
        if (portable.residency == llama_cache_acct_residency::device) {
            const auto it = std::find_if(
                domains.begin(), domains.end(),
                [&](const llama_vbr_artifact_domain_binding & binding) {
                    return binding.topology_index == portable.topology_index &&
                           binding.device_ordinal == portable.device_ordinal;
                });
            if (it == domains.end()) {
                return false;
            }
            out = it->domain;
            return true;
        }
        if (portable.kind != llama_cache_acct_domain_kind::not_applicable ||
            portable.topology_index != UINT32_MAX ||
            portable.device_ordinal != UINT16_MAX ||
            portable.residency >= llama_cache_acct_residency::_count ||
            portable.residency == llama_cache_acct_residency::not_applicable) {
            return false;
        }
        out = llama_cache_acct_resource_domain::non_device(
            portable.residency);
        return true;
    }

    bool issue(uint64_t & next, uint64_t & out) {
        if (next == 0 || next == std::numeric_limits<uint64_t>::max()) {
            return false;
        }
        out = next++;
        return true;
    }

    bool intern_content(
            intern_purpose purpose,
            const digest_key & key,
            llama_cache_acct_content_digest & out) {
        const intern_key typed { purpose, key };
        const auto found = content_ids.find(typed);
        if (found != content_ids.end()) {
            out = { found->second };
            return true;
        }
        uint64_t id;
        if (!issue(next_content, id)) {
            return false;
        }
        content_ids.emplace(typed, id);
        out = { id };
        return true;
    }

    bool intern_lineage(
            intern_purpose purpose,
            const digest_key & key,
            llama_cache_acct_lineage_id & out) {
        const intern_key typed { purpose, key };
        const auto found = lineage_ids.find(typed);
        if (found != lineage_ids.end()) {
            out = { found->second };
            return true;
        }
        uint64_t id;
        if (!issue(next_lineage, id)) {
            return false;
        }
        lineage_ids.emplace(typed, id);
        out = { id };
        return true;
    }

    bool issue_artifact(llama_cache_acct_artifact_id & out) {
        uint64_t id;
        if (!issue(next_artifact, id)) {
            return false;
        }
        out = { id };
        return true;
    }

    const allocation * find_allocation(
            const std::vector<allocation> & values,
            llama_cache_acct_category category,
            const llama_cache_acct_resource_domain & domain,
            uint64_t logical,
            uint64_t resident) const {
        const auto it = std::find_if(
            values.begin(), values.end(),
            [&](const allocation & value) {
                return value.category == category &&
                       value.domain == domain &&
                       value.logical == logical &&
                       value.resident == resident;
            });
        return it == values.end() ? nullptr : &*it;
    }

    llama_cache_acct_ledger & ledger;
    mutable std::mutex mutex;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<llama_vbr_artifact_domain_binding> domains;
    std::set<configured_cell> configured;
    std::map<digest_key, blob> blobs;
    std::map<digest_key, stash> stashes;
    std::map<uint64_t, reference> references;
    std::map<intern_key, uint64_t> content_ids;
    std::map<intern_key, uint64_t> lineage_ids;
    uint64_t next_artifact = 1;
    uint64_t next_content = 1;
    uint64_t next_lineage = 1;
    uint64_t next_retire_token = 1;
    uint64_t n_published = 0;
    uint64_t n_adopted = 0;
    uint64_t n_refusals = 0;
    uint64_t n_staging_overlap_refusals = 0;

    void erase_orphan_storage(
            const std::vector<vbr_unit_version_id> & affected_units,
            const std::vector<vbr_stash_payload_id> & affected_stashes)
            noexcept {
        for (const auto & affected : affected_units) {
            const auto key = affected.bytes();
            const bool live = std::any_of(
                references.begin(), references.end(), [&](const auto & row) {
                    return std::any_of(
                        row.second.unit_ids.begin(), row.second.unit_ids.end(),
                        [&](const auto & id) {
                            return id.bytes() == key;
                        });
                });
            if (!live) {
                blobs.erase(key);
            }
        }
        for (const auto & affected : affected_stashes) {
            if (!affected.valid()) {
                continue;
            }
            const auto key = affected.bytes();
            const bool live = std::any_of(
                references.begin(), references.end(), [&](const auto & row) {
                    return std::any_of(
                        row.second.stash_ids.begin(),
                        row.second.stash_ids.end(),
                        [&](const auto & id) {
                            return id.valid() && id.bytes() == key;
                        });
                });
            if (!live) {
                stashes.erase(key);
            }
        }
    }
};

struct vbr_artifact_package_view::storage {
    llama_cache_acct_artifact_id reference;
    std::vector<vbr_artifact_portable_topology> topologies;
    vbr_artifact_reference_manifest manifest;
    std::vector<vbr_artifact_unit_view> units;
    std::vector<vbr_artifact_companion_view> companions;
    std::vector<vbr_artifact_allocation_view> reference_allocations;
};

struct vbr_artifact_prepared_retire::impl {
    llama_vbr_artifact_catalog * owner = nullptr;
    uint64_t token = 0;
    std::vector<llama_cache_acct_artifact_id> references;
    std::vector<vbr_unit_version_id> unit_ids;
    std::vector<vbr_stash_payload_id> stash_ids;
    llama_cache_prepared_release_set release;
};

namespace {

bool digest_nonzero(const std::array<uint8_t, 32> & digest) {
    return std::any_of(
        digest.begin(), digest.end(),
        [](uint8_t byte) { return byte != 0; });
}

struct catalog_stream_state {
    // Borrowed by the move-only build: every build/unit handle must be
    // destroyed before its catalog and ledger.
    llama_vbr_artifact_catalog * catalog = nullptr;
    llama_cache_acct_ledger * ledger = nullptr;
    vbr_artifact_package package;
    llama_cache_budget_config budget;
    llama_cache_transaction_fault fault;
    bool charge_transfer_staging = true;
    std::vector<llama_cache_acct_artifact_id> blob_artifacts;
    std::vector<llama_cache_acct_artifact_id> stash_artifacts;
    llama_cache_acct_artifact_id reference_artifact;
    std::vector<llama_cache_transaction_leaf>
        durable_prepared_leaves;
    std::vector<llama_cache_transaction_leaf> staging_leaves;
    std::vector<llama_cache_acct_op_id> durable_ops;
    std::vector<llama_cache_acct_alloc_id> durable_allocs;
    std::vector<llama_cache_acct_op_id> staging_ops;
    std::vector<llama_cache_acct_alloc_id> staging_allocs;
    llama_cache_prepared_claim_group durable_prepared;
    llama_cache_prepared_claim_group staging_prepared;
    bool staging_committed = false;
    std::vector<vbr_verified_segment> segments;
    std::vector<vbr_verified_companion> companions;
    std::vector<bool> unit_open;
    std::vector<bool> unit_sealed;
    bool published = false;
    vbr_capture_stream_status failed =
        vbr_capture_stream_status::ok;

    ~catalog_stream_state() {
        if (ledger && staging_committed) {
            for (const auto op : staging_ops) {
                if (op) {
                    const bool released = ledger->release(op);
                    GGML_ASSERT(released);
                }
            }
        }
    }

    bool commit_staging() noexcept {
        if (!charge_transfer_staging ||
            staging_committed) {
            return true;
        }
        const auto result =
            staging_prepared.materialize_and_commit(
                staging_leaves);
        if (result.status !=
                llama_cache_transaction_status::committed) {
            failed = result.status ==
                    llama_cache_transaction_status::stage_failed
                ? vbr_capture_stream_status::stage_failed
                : result.status ==
                      llama_cache_transaction_status::
                          commit_failed
                    ? vbr_capture_stream_status::commit_failed
                    : vbr_capture_stream_status::
                        accounting_refused;
            return false;
        }
        staging_committed = true;
        return true;
    }
};

class catalog_unit_build final : public vbr_unit_build {
public:
    explicit catalog_unit_build(
            std::shared_ptr<catalog_stream_state> state,
            uint32_t unit_index)
        : state_(std::move(state)), unit_index_(unit_index) {}

    ~catalog_unit_build() override {
        if (state_ && unit_index_ < state_->unit_sealed.size() &&
            !state_->unit_sealed[unit_index_] &&
            state_->failed == vbr_capture_stream_status::ok) {
            state_->failed =
                vbr_capture_stream_status::missing_segment;
        }
    }

    vbr_capture_stream_status accept_verified_segment(
            const vbr_verified_segment & segment) noexcept override {
        try {
            if (!state_ || state_->published ||
                unit_index_ >= state_->unit_sealed.size() ||
                state_->unit_sealed[unit_index_]) {
                return vbr_capture_stream_status::late_segment;
            }
            if (state_->failed !=
                    vbr_capture_stream_status::ok) {
                return state_->failed;
            }
            if (!state_->commit_staging()) {
                return state_->failed;
            }
            if (segment.unit_index != unit_index_ ||
                !segment.bytes ||
                segment.bytes->size() == 0 ||
                !digest_nonzero(segment.streaming_digest) ||
                vbr_capture_stream_digest(*segment.bytes) !=
                    segment.streaming_digest) {
                state_->failed =
                    vbr_capture_stream_status::hash_mismatch;
                return state_->failed;
            }
            const auto duplicate = std::find_if(
                state_->segments.begin(),
                state_->segments.end(),
                [&](const vbr_verified_segment & current) {
                    return current.unit_index ==
                               segment.unit_index &&
                           current.shard_index ==
                               segment.shard_index &&
                           current.clean_stash ==
                               segment.clean_stash;
                });
            if (duplicate != state_->segments.end()) {
                state_->failed =
                    vbr_capture_stream_status::duplicate_segment;
                return state_->failed;
            }
            state_->segments.push_back(segment);
            return vbr_capture_stream_status::ok;
        } catch (...) {
            if (state_) {
                state_->failed =
                    vbr_capture_stream_status::internal_error;
            }
            return vbr_capture_stream_status::internal_error;
        }
    }

    vbr_capture_stream_status seal_unit() noexcept override {
        try {
            if (!state_ || state_->published ||
                unit_index_ >= state_->unit_sealed.size() ||
                state_->unit_sealed[unit_index_]) {
                return vbr_capture_stream_status::late_segment;
            }
            if (state_->failed !=
                    vbr_capture_stream_status::ok) {
                return state_->failed;
            }
            const auto & descriptor =
                state_->package.unit_blobs[unit_index_].descriptor;
            const size_t expected =
                descriptor.shards.size() +
                (descriptor.clean_stash_state ==
                     vbr_artifact_clean_stash_state::present
                     ? descriptor.clean_stash.shards.size()
                     : 0);
            const size_t actual = std::count_if(
                state_->segments.begin(), state_->segments.end(),
                [&](const auto & segment) {
                    return segment.unit_index == unit_index_;
                });
            if (actual != expected) {
                state_->failed =
                    vbr_capture_stream_status::missing_segment;
                return state_->failed;
            }
            for (const auto & shard : descriptor.shards) {
                const auto found = std::find_if(
                    state_->segments.begin(),
                    state_->segments.end(),
                    [&](const vbr_verified_segment & value) {
                        return value.unit_index == unit_index_ &&
                               !value.clean_stash &&
                               value.shard_index ==
                                   shard.shard_index &&
                               value.bytes->size() ==
                                   shard.payload_bytes;
                    });
                if (found == state_->segments.end()) {
                    state_->failed =
                        vbr_capture_stream_status::
                            missing_segment;
                    return state_->failed;
                }
            }
            if (descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present) {
                for (const auto & shard :
                     descriptor.clean_stash.shards) {
                    const auto found = std::find_if(
                        state_->segments.begin(),
                        state_->segments.end(),
                        [&](const vbr_verified_segment & value) {
                            return value.unit_index == unit_index_ &&
                                   value.clean_stash &&
                                   value.shard_index ==
                                       shard.shard_index &&
                                   value.bytes->size() ==
                                       shard.payload_bytes;
                        });
                    if (found == state_->segments.end()) {
                        state_->failed =
                            vbr_capture_stream_status::
                                missing_segment;
                        return state_->failed;
                    }
                }
            }
            state_->unit_sealed[unit_index_] = true;
            return vbr_capture_stream_status::ok;
        } catch (...) {
            if (state_) {
                state_->failed =
                    vbr_capture_stream_status::internal_error;
            }
            return vbr_capture_stream_status::internal_error;
        }
    }

private:
    std::shared_ptr<catalog_stream_state> state_;
    uint32_t unit_index_ = UINT32_MAX;
};

} // namespace

class llama_vbr_artifact_catalog_stream_build final
        : public vbr_capture_build {
public:
    explicit llama_vbr_artifact_catalog_stream_build(
            std::shared_ptr<catalog_stream_state> state)
        : state_(std::move(state)) {}

    std::unique_ptr<vbr_unit_build> begin_unit(
            uint32_t unit_index,
            vbr_capture_stream_status & status) noexcept override {
        status = vbr_capture_stream_status::invalid_argument;
        try {
            if (!state_ || state_->published ||
                unit_index >= state_->package.unit_blobs.size() ||
                unit_index >= state_->unit_open.size() ||
                state_->unit_open[unit_index]) {
                return nullptr;
            }
            // Canonical construction is forward-only: a later unit cannot be
            // opened while an earlier one remains unsealed.
            for (uint32_t i = 0; i < unit_index; ++i) {
                if (!state_->unit_sealed[i]) {
                    return nullptr;
                }
            }
            state_->unit_open[unit_index] = true;
            status = vbr_capture_stream_status::ok;
            return std::unique_ptr<vbr_unit_build>(
                new catalog_unit_build(state_, unit_index));
        } catch (...) {
            status = vbr_capture_stream_status::internal_error;
            return nullptr;
        }
    }

    vbr_capture_stream_status accept_verified_companion(
            const vbr_verified_companion & companion) noexcept override {
        try {
            if (!state_ || state_->published ||
                companion.companion_index >=
                    state_->package.companions.size() ||
                !companion.bytes ||
                companion.bytes->size() !=
                    state_->package.companions[
                        companion.companion_index].payload_bytes ||
                vbr_capture_stream_digest(*companion.bytes) !=
                    companion.streaming_digest) {
                return vbr_capture_stream_status::hash_mismatch;
            }
            const auto duplicate = std::find_if(
                state_->companions.begin(),
                state_->companions.end(),
                [&](const auto & current) {
                    return current.companion_index ==
                           companion.companion_index;
                });
            if (duplicate != state_->companions.end()) {
                return vbr_capture_stream_status::duplicate_segment;
            }
            if (!state_->commit_staging()) {
                return state_->failed;
            }
            state_->companions.push_back(companion);
            return vbr_capture_stream_status::ok;
        } catch (...) {
            if (state_) {
                state_->failed =
                    vbr_capture_stream_status::internal_error;
            }
            return vbr_capture_stream_status::internal_error;
        }
    }

    vbr_capture_sink_result publish_reference() noexcept override {
        vbr_capture_sink_result out;
        if (!state_ || state_->published ||
            state_->unit_sealed.empty() ||
            !std::all_of(state_->unit_sealed.begin(),
                         state_->unit_sealed.end(),
                         [](bool sealed) { return sealed; }) ||
            state_->companions.size() !=
                state_->package.companions.size() ||
            state_->failed != vbr_capture_stream_status::ok) {
            out.status = state_ &&
                    state_->failed !=
                        vbr_capture_stream_status::ok
                ? state_->failed
                : vbr_capture_stream_status::missing_segment;
            return out;
        }
        state_->published = true;
        const auto result = state_->catalog->publish_stream(
            state_->package, state_->segments,
            state_->budget, state_->fault,
            state_.get());
        out.reference_artifact = result.reference_artifact;
        out.unit_content = result.unit_content;
        out.reference_lineage = result.reference_lineage;
        out.adopted =
            result.status ==
                llama_vbr_artifact_publish_status::adopted;
        switch (result.status) {
            case llama_vbr_artifact_publish_status::published:
            case llama_vbr_artifact_publish_status::adopted:
                out.status = vbr_capture_stream_status::ok;
                break;
            case llama_vbr_artifact_publish_status::invalid_argument:
                out.status =
                    vbr_capture_stream_status::invalid_argument;
                break;
            case llama_vbr_artifact_publish_status::shard_failed:
                out.status =
                    vbr_capture_stream_status::transfer_failed;
                break;
            case llama_vbr_artifact_publish_status::duplicate_completion:
                out.status =
                    vbr_capture_stream_status::duplicate_segment;
                break;
            case llama_vbr_artifact_publish_status::missing_completion:
                out.status =
                    vbr_capture_stream_status::missing_segment;
                break;
            case llama_vbr_artifact_publish_status::format_rejected:
                out.status =
                    vbr_capture_stream_status::format_rejected;
                break;
            case llama_vbr_artifact_publish_status::accounting_unavailable:
                out.status =
                    vbr_capture_stream_status::
                        accounting_unavailable;
                break;
            case llama_vbr_artifact_publish_status::admission_refused:
                out.status =
                    vbr_capture_stream_status::accounting_refused;
                break;
            case llama_vbr_artifact_publish_status::stage_failed:
                out.status =
                    vbr_capture_stream_status::stage_failed;
                break;
            case llama_vbr_artifact_publish_status::commit_failed:
                out.status =
                    vbr_capture_stream_status::commit_failed;
                break;
            case llama_vbr_artifact_publish_status::publication_failed:
                out.status =
                    vbr_capture_stream_status::publication_failed;
                break;
            case llama_vbr_artifact_publish_status::internal_error:
            case llama_vbr_artifact_publish_status::_count:
                out.status =
                    vbr_capture_stream_status::internal_error;
                break;
        }
        return out;
    }

private:
    std::shared_ptr<catalog_stream_state> state_;
};

llama_vbr_artifact_catalog::llama_vbr_artifact_catalog(
        llama_cache_acct_ledger & ledger)
    : impl_(new impl(ledger)) {}

vbr_artifact_prepared_retire::vbr_artifact_prepared_retire() noexcept =
    default;

vbr_artifact_prepared_retire::vbr_artifact_prepared_retire(
        vbr_artifact_prepared_retire && other) noexcept
    : impl_(std::move(other.impl_)) {}

vbr_artifact_prepared_retire &
vbr_artifact_prepared_retire::operator=(
        vbr_artifact_prepared_retire && other) noexcept {
    if (this != &other) {
        reset();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

vbr_artifact_prepared_retire::~vbr_artifact_prepared_retire() {
    reset();
}

bool vbr_artifact_prepared_retire::ready() const noexcept {
    return impl_ && impl_->owner && impl_->token != 0 &&
           impl_->release.ready();
}

const llama_cache_acct_release_set_preview &
vbr_artifact_prepared_retire::preview() const noexcept {
    static const llama_cache_acct_release_set_preview empty;
    return ready() ? impl_->release.preview() : empty;
}

vbr_artifact_prepared_retire_status
vbr_artifact_prepared_retire::commit() noexcept {
    if (!ready()) {
        return vbr_artifact_prepared_retire_status::unavailable;
    }
    auto state = std::move(impl_);
    const auto status = state->owner->commit_owned_retire(
        state->token, state->references, state->unit_ids,
        state->stash_ids, state->release);
    if (status == vbr_artifact_prepared_retire_status::unavailable) {
        state->owner->cancel_owned_retire(
            state->token, state->references, state->unit_ids,
            state->stash_ids, state->release);
    }
    return status;
}

void vbr_artifact_prepared_retire::reset() noexcept {
    if (impl_ && impl_->owner && impl_->token != 0) {
        impl_->owner->cancel_owned_retire(
            impl_->token, impl_->references, impl_->unit_ids,
            impl_->stash_ids, impl_->release);
    }
    impl_.reset();
}

vbr_artifact_package_view::vbr_artifact_package_view(
        vbr_artifact_package_view && other) noexcept
    : owner_(other.owner_),
      storage_(std::move(other.storage_)),
      host_owned_(other.host_owned_) {
    other.owner_ = nullptr;
    other.host_owned_ = false;
}

vbr_artifact_package_view & vbr_artifact_package_view::operator=(
        vbr_artifact_package_view && other) noexcept {
    if (this != &other) {
        reset();
        owner_ = other.owner_;
        storage_ = std::move(other.storage_);
        host_owned_ = other.host_owned_;
        other.owner_ = nullptr;
        other.host_owned_ = false;
    }
    return *this;
}

vbr_artifact_package_view::~vbr_artifact_package_view() {
    reset();
}

void vbr_artifact_package_view::reset() noexcept {
    auto * owner = owner_;
    const auto reference = reference_artifact();
    const bool host_owned = host_owned_;
    owner_ = nullptr;
    host_owned_ = false;
    storage_.reset();
    if (owner != nullptr) {
        owner->release_reference_lease(reference, host_owned);
    }
}

llama_cache_acct_artifact_id
vbr_artifact_package_view::reference_artifact() const noexcept {
    return storage_ ? storage_->reference : llama_cache_acct_artifact_id{};
}

bool vbr_artifact_package_view::accounted_by(
        const llama_cache_acct_ledger * ledger) const noexcept {
    return owner_ != nullptr && storage_ && owner_->accounted_by(ledger);
}

bool vbr_artifact_package_view::claim_host_ownership() noexcept {
    if (owner_ == nullptr || !storage_ || host_owned_) {
        return false;
    }
    if (!owner_->claim_host_ownership(storage_->reference)) {
        return false;
    }
    host_owned_ = true;
    return true;
}

bool vbr_artifact_package_view::prepare_owned_retire(
        const std::vector<const vbr_artifact_package_view *> & packages,
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept {
    out.reset();
    if (owner_ == nullptr || !storage_ || packages.empty()) {
        return false;
    }
    try {
        std::vector<llama_cache_acct_artifact_id> references;
        references.reserve(packages.size());
        for (const auto * package : packages) {
            if (!package || package->owner_ != owner_ ||
                !package->storage_ || !package->host_owned_) {
                return false;
            }
            references.push_back(package->storage_->reference);
        }
        return owner_->prepare_owned_retire(
            references, expected_serial, out);
    } catch (...) {
        return false;
    }
}

bool vbr_artifact_package_view::preview_owned_retire(
        const std::vector<const vbr_artifact_package_view *> & packages,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept {
    out = {};
    if (owner_ == nullptr || !storage_ || packages.empty()) {
        return false;
    }
    try {
        std::vector<llama_cache_acct_artifact_id> references;
        references.reserve(packages.size());
        for (const auto * package : packages) {
            if (!package || package->owner_ != owner_ ||
                !package->storage_ || !package->host_owned_) {
                return false;
            }
            references.push_back(package->storage_->reference);
        }
        return owner_->preview_owned_retire(
            references, expected_serial, out);
    } catch (...) {
        out = {};
        return false;
    }
}

const std::vector<vbr_artifact_portable_topology> &
vbr_artifact_package_view::topologies() const noexcept {
    static const std::vector<vbr_artifact_portable_topology> empty;
    return storage_ ? storage_->topologies : empty;
}

const vbr_artifact_reference_manifest &
vbr_artifact_package_view::manifest() const noexcept {
    static const vbr_artifact_reference_manifest empty;
    return storage_ ? storage_->manifest : empty;
}

const std::vector<vbr_artifact_unit_view> &
vbr_artifact_package_view::units() const noexcept {
    static const std::vector<vbr_artifact_unit_view> empty;
    return storage_ ? storage_->units : empty;
}

const std::vector<vbr_artifact_companion_view> &
vbr_artifact_package_view::companions() const noexcept {
    static const std::vector<vbr_artifact_companion_view> empty;
    return storage_ ? storage_->companions : empty;
}

const std::vector<vbr_artifact_allocation_view> &
vbr_artifact_package_view::reference_allocations() const noexcept {
    static const std::vector<vbr_artifact_allocation_view> empty;
    return storage_ ? storage_->reference_allocations : empty;
}

vbr_artifact_status vbr_artifact_package_view::validate() const noexcept {
    if (!storage_) {
        return vbr_artifact_status::invalid_argument;
    }
    try {
        vbr_artifact_package package;
        package.version = storage_->manifest.version;
        package.topologies = storage_->topologies;
        package.manifest = storage_->manifest;
        package.unit_blobs.reserve(storage_->units.size());
        for (const auto & view : storage_->units) {
            vbr_artifact_unit_blob blob;
            blob.unit_version_id = view.unit_version_id;
            blob.payload_digest = view.payload_digest;
            blob.descriptor = view.descriptor;
            if (blob.descriptor.shards.size() != view.payload_shards.size() ||
                blob.descriptor.clean_stash.shards.size() !=
                    view.stash_shards.size()) {
                return vbr_artifact_status::malformed;
            }
            for (size_t i = 0; i < view.payload_shards.size(); ++i) {
                if (!view.payload_shards[i]) {
                    return vbr_artifact_status::malformed;
                }
                blob.descriptor.shards[i].payload =
                    view.payload_shards[i]->source();
            }
            for (size_t i = 0; i < view.stash_shards.size(); ++i) {
                if (!view.stash_shards[i]) {
                    return vbr_artifact_status::malformed;
                }
                blob.descriptor.clean_stash.shards[i].payload =
                    view.stash_shards[i]->source();
            }
            package.unit_blobs.push_back(std::move(blob));
        }
        package.companions.reserve(storage_->companions.size());
        for (const auto & view : storage_->companions) {
            if (!view.payload) {
                return vbr_artifact_status::malformed;
            }
            auto companion = view.descriptor;
            companion.payload = view.payload->source();
            package.companions.push_back(std::move(companion));
        }
        return vbr_artifact_validate_prepared_package(package);
    } catch (...) {
        return vbr_artifact_status::internal_error;
    }
}

vbr_artifact_resolve_status vbr_artifact_package_view::retain(
        vbr_artifact_package_view & output) const noexcept {
    if (owner_ == nullptr || !storage_) {
        output.reset();
        return vbr_artifact_resolve_status::not_found;
    }
    return owner_->resolve_reference(storage_->reference, output);
}

llama_vbr_artifact_catalog::~llama_vbr_artifact_catalog() {
    while (impl_ && !impl_->references.empty()) {
        const llama_cache_acct_artifact_id reference {
            impl_->references.begin()->first,
        };
        const auto status = retire(reference);
        if (status != vbr_artifact_retire_status::retired) {
            GGML_ABORT(
                "VBR artifact catalog teardown could not release reference %" PRIu64
                " status=%u (live package view outlived catalog)",
                reference.v, unsigned(status));
        }
    }
}

bool llama_vbr_artifact_catalog::bind_topologies(
        const std::vector<vbr_artifact_portable_topology> & topologies,
        std::vector<llama_vbr_artifact_domain_binding> & bindings) noexcept {
    bindings.clear();
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (topologies.empty() || !impl_->references.empty()) {
            return false;
        }
        if (!impl_->topologies.empty()) {
            if (impl_->topologies != topologies) {
                return false;
            }
            bindings = impl_->domains;
            return true;
        }

        std::vector<llama_vbr_artifact_domain_binding> built;
        for (uint32_t topology_index = 0;
             topology_index < topologies.size(); ++topology_index) {
            const auto & topology = topologies[topology_index];
            if (!topology.digest.valid() ||
                topology.digest !=
                    llama_cache_acct_compute_topology_digest(topology)) {
                return false;
            }
            for (uint16_t ordinal = 0;
                 ordinal < topology.device_count; ++ordinal) {
                llama_cache_acct_resource_domain domain;
                if (!impl_->ledger.make_device_domain(
                        topology,
                        llama_cache_acct_device_ordinal { ordinal },
                        domain)) {
                    return false;
                }
                built.push_back({ topology_index, ordinal, domain });
            }
        }
        impl_->topologies = topologies;
        impl_->domains = built;
        bindings = std::move(built);
        return true;
    } catch (...) {
        bindings.clear();
        return false;
    }
}

bool llama_vbr_artifact_catalog::configure_accounting(
        const vbr_artifact_package & package) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->topologies != package.topologies) {
            return false;
        }

        std::set<configured_cell> needed;
        needed.insert({
            llama_cache_acct_category::transfer_staging,
            llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host),
        });
        for (const auto & row : package.manifest.accounting) {
            llama_cache_acct_resource_domain domain;
            const auto category =
                vbr_artifact_accounting_category(row.role);
            if (category == llama_cache_acct_category::_count ||
                !impl_->resolve_domain(row.domain, domain)) {
                return false;
            }
            needed.insert({ category, domain });
            // Kept for the landed F2/F0 capacity tests and future
            // device-local codec staging. F3's pageable segment image uses
            // the single host-domain transfer_staging cell inserted above.
            needed.insert({
                llama_cache_acct_category::transfer_staging,
                domain,
            });
            needed.insert({
                llama_cache_acct_category::codec_workspace, domain,
            });
            if (domain.residency ==
                    llama_cache_acct_residency::pinned_host) {
                needed.insert({
                    llama_cache_acct_category::pinned_preimage_ring, domain,
                });
            }
        }

        const auto before = impl_->ledger.snapshot();
        std::vector<configured_cell> added;
        for (const auto & cell : needed) {
            if (impl_->configured.count(cell)) {
                continue;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                impl_->ledger.gauge_set(
                    cell.category, cell.domain, measure, 0);
            }
            added.push_back(cell);
        }

        const auto snapshot = impl_->ledger.snapshot();
        if (snapshot.faults_overflow != before.faults_overflow ||
            snapshot.faults_invalid_transition !=
                before.faults_invalid_transition ||
            snapshot.faults_allocation != before.faults_allocation) {
            return false;
        }
        for (const auto & cell : needed) {
            const auto found = std::find_if(
                snapshot.cells.begin(), snapshot.cells.end(),
                [&](const llama_cache_acct_cell_row & row) {
                    return row.category == cell.category &&
                           row.domain == cell.domain;
                });
            if (found == snapshot.cells.end()) {
                return false;
            }
        }
        impl_->configured.insert(added.begin(), added.end());
        return true;
    } catch (...) {
        return false;
    }
}

bool llama_vbr_artifact_catalog::prepare_capture_package(
        const vbr_artifact_package & package) noexcept {
    try {
        std::vector<llama_vbr_artifact_domain_binding> ignored;
        bool needs_bind = false;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (!impl_->topologies.empty() &&
                impl_->topologies != package.topologies) {
                return false;
            }
            needs_bind = impl_->topologies.empty();
        }
        if (needs_bind &&
            !bind_topologies(package.topologies, ignored)) {
            return false;
        }
        return configure_accounting(package);
    } catch (...) {
        return false;
    }
}

std::unique_ptr<vbr_capture_build>
llama_vbr_artifact_catalog::begin_capture(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        vbr_capture_stream_status & status,
        vbr_capture_begin_diagnostics * diagnostics) noexcept {
    return begin_capture_impl(
        package, budget, fault, true, status, diagnostics);
}

std::unique_ptr<vbr_capture_build>
llama_vbr_artifact_catalog::begin_capture_impl(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        bool charge_transfer_staging,
        vbr_capture_stream_status & status,
        vbr_capture_begin_diagnostics * diagnostics) noexcept {
    status = vbr_capture_stream_status::invalid_argument;
    if (diagnostics) {
        *diagnostics = {};
    }
    try {
        if (package.unit_blobs.empty() ||
            package.manifest.unit_references.size() !=
                package.unit_blobs.size()) {
            return nullptr;
        }
        auto state = std::make_shared<catalog_stream_state>();
        state->catalog = this;
        state->ledger = &impl_->ledger;
        state->package = package;
        state->budget = budget;
        state->fault = fault;
        state->charge_transfer_staging =
            charge_transfer_staging;
        state->unit_open.assign(package.unit_blobs.size(), false);
        state->unit_sealed.assign(package.unit_blobs.size(), false);
        if (charge_transfer_staging) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            state->blob_artifacts.resize(package.unit_blobs.size());
            state->stash_artifacts.resize(package.unit_blobs.size());
            bool issued = impl_->topologies == package.topologies;
            for (size_t i = 0; issued && i < package.unit_blobs.size(); ++i) {
                issued = impl_->issue_artifact(state->blob_artifacts[i]);
                if (issued &&
                    package.unit_blobs[i].descriptor.clean_stash_state ==
                        vbr_artifact_clean_stash_state::present) {
                    issued = impl_->issue_artifact(state->stash_artifacts[i]);
                }
            }
            if (!issued ||
                !impl_->issue_artifact(state->reference_artifact)) {
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
            llama_cache_acct_content_digest staging_content;
            llama_cache_acct_lineage_id staging_lineage;
            llama_sha256_writer staging_hash;
            static constexpr char STAGING_DOMAIN[] =
                "buun.vbr.capture.transfer-staging";
            staging_hash.string(
                STAGING_DOMAIN,
                sizeof(STAGING_DOMAIN) - 1);
            staging_hash.u64(
                package.manifest.accounting.size());
            for (const auto & row :
                 package.manifest.accounting) {
                staging_hash.u32(uint32_t(row.role));
                staging_hash.u32(
                    uint32_t(row.domain.residency));
                staging_hash.u32(
                    uint32_t(row.domain.kind));
                staging_hash.u32(
                    row.domain.topology_index);
                staging_hash.u32(
                    row.domain.device_ordinal);
                staging_hash.u64(row.logical_bytes);
                staging_hash.u64(row.resident_bytes);
            }
            const auto staging_digest =
                staging_hash.finish();
            if (!impl_->intern_content(
                    intern_purpose::manifest,
                    staging_digest,
                    staging_content) ||
                !impl_->intern_lineage(
                    intern_purpose::manifest,
                    staging_digest,
                    staging_lineage)) {
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }

            const size_t count =
                package.manifest.accounting.size();
            state->durable_ops.resize(count);
            state->durable_allocs.resize(count);
            state->staging_ops.resize(1);
            state->staging_allocs.resize(1);
            state->durable_prepared_leaves.reserve(count);
            state->staging_leaves.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                const auto & row =
                    package.manifest.accounting[i];
                llama_cache_acct_resource_domain domain;
                const auto category =
                    vbr_artifact_accounting_category(
                        row.role);
                if (!impl_->resolve_domain(
                        row.domain, domain) ||
                    !impl_->configured.count({
                        category, domain })) {
                    status =
                        vbr_capture_stream_status::
                            accounting_refused;
                    return nullptr;
                }
                const auto artifact =
                    package.unit_blobs.size() == 1 &&
                    row.role ==
                        vbr_artifact_accounting_role::
                            clean_stash_payload
                        ? state->stash_artifacts[0]
                        : package.unit_blobs.size() == 1 &&
                          row.role !=
                              vbr_artifact_accounting_role::
                                  reference_metadata
                            ? state->blob_artifacts[0]
                            : state->reference_artifact;
                llama_cache_transaction_leaf durable;
                durable.category = category;
                durable.domain = domain;
                durable.attribution = {
                    llama_cache_acct_attr_kind::artifact,
                    -1, artifact,
                };
                durable.expected_logical =
                    row.logical_bytes;
                durable.reserve_resident =
                    row.resident_bytes;
                durable.stage_resident =
                    row.resident_bytes;
                durable.artifact = artifact;
                durable.committed_op =
                    &state->durable_ops[i];
                durable.allocation_out =
                    &state->durable_allocs[i];
                state->durable_prepared_leaves.push_back(
                    durable);

            }
            uint64_t staging_bytes = 0;
            for (const auto & row :
                 package.manifest.accounting) {
                if (row.resident_bytes >
                        UINT64_MAX - staging_bytes) {
                    status =
                        vbr_capture_stream_status::
                            accounting_refused;
                    return nullptr;
                }
                staging_bytes += row.resident_bytes;
            }
            const auto staging_domain =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::
                        pageable_host);
            if (staging_bytes == 0 ||
                !impl_->configured.count({
                    llama_cache_acct_category::
                        transfer_staging,
                    staging_domain })) {
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
            llama_cache_transaction_leaf staging;
            staging.category =
                llama_cache_acct_category::transfer_staging;
            staging.domain = staging_domain;
            staging.attribution = {
                llama_cache_acct_attr_kind::artifact,
                -1, package.unit_blobs.size() == 1
                    ? state->blob_artifacts[0]
                    : state->reference_artifact,
            };
            staging.expected_logical = staging_bytes;
            staging.reserve_resident = staging_bytes;
            staging.stage_resident = staging_bytes;
            staging.artifact = package.unit_blobs.size() == 1
                ? state->blob_artifacts[0]
                : state->reference_artifact;
            staging.content = staging_content;
            staging.lineage = staging_lineage;
            staging.committed_op = &state->staging_ops[0];
            staging.allocation_out =
                &state->staging_allocs[0];
            state->staging_leaves.push_back(staging);
            state->staging_prepared =
                llama_cache_prepare_reservation_transaction(
                    impl_->ledger, budget,
                    state->staging_leaves);
            if (!state->staging_prepared.ready()) {
                if (diagnostics) {
                    diagnostics->reservation_group =
                        vbr_capture_reservation_group::
                            transfer_staging;
                    diagnostics->prepare_status =
                        state->staging_prepared.preparation().status;
                    diagnostics->admission_status =
                        state->staging_prepared.preparation().
                            admission_status;
                    diagnostics->failed_leaf =
                        state->staging_prepared.preparation().
                            failed_leaf;
                }
                impl_->n_staging_overlap_refusals++;
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
            state->durable_prepared =
                llama_cache_prepare_reservation_transaction(
                    impl_->ledger, budget,
                    state->durable_prepared_leaves);
            if (!state->durable_prepared.ready()) {
                if (diagnostics) {
                    diagnostics->reservation_group =
                        vbr_capture_reservation_group::
                            durable_artifact;
                    diagnostics->prepare_status =
                        state->durable_prepared.preparation().status;
                    diagnostics->admission_status =
                        state->durable_prepared.preparation().
                            admission_status;
                    diagnostics->failed_leaf =
                        state->durable_prepared.preparation().
                            failed_leaf;
                }
                impl_->n_staging_overlap_refusals++;
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
        }
        status = vbr_capture_stream_status::ok;
        return std::unique_ptr<vbr_capture_build>(
            new llama_vbr_artifact_catalog_stream_build(
                std::move(state)));
    } catch (...) {
        status = vbr_capture_stream_status::internal_error;
        return nullptr;
    }
}

llama_vbr_artifact_publish_result llama_vbr_artifact_catalog::publish(
        const vbr_artifact_package & package,
        const std::vector<llama_vbr_artifact_fake_shard_completion> & completions,
        const llama_cache_budget_config & budget,
        const llama_vbr_artifact_publish_fault & fault) noexcept {
    llama_vbr_artifact_publish_result out;
    try {
        const uint64_t expected =
            package.unit_blobs.size() == 1
                ? package.unit_blobs[0].descriptor.shards.size() +
                      (package.unit_blobs[0].descriptor
                                   .clean_stash_state ==
                               vbr_artifact_clean_stash_state::present
                           ? package.unit_blobs[0].descriptor
                                 .clean_stash.shards.size()
                           : 0)
                : 0;
        if (completions.size() != expected) {
            out.status = completions.size() < expected
                ? llama_vbr_artifact_publish_status::
                    missing_completion
                : llama_vbr_artifact_publish_status::
                    duplicate_completion;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        vbr_capture_stream_status stream_status;
        // F2's fake completions are already caller-owned resident vectors;
        // unlike the F3 D2H stream they do not allocate a pageable transfer
        // image. They still drive the exact same seal/dedup/publication core.
        auto build = begin_capture_impl(
            package, budget, fault, false, stream_status,
            nullptr);
        if (!build) {
            out.status =
                llama_vbr_artifact_publish_status::
                    invalid_argument;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        auto unit = build->begin_unit(0, stream_status);
        if (!unit) {
            out.status =
                llama_vbr_artifact_publish_status::
                    invalid_argument;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        for (const auto & completion : completions) {
            if (completion.unit_index != 0) {
                out.status =
                    llama_vbr_artifact_publish_status::
                        invalid_argument;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
            if (!completion.success) {
                out.status =
                    llama_vbr_artifact_publish_status::
                        shard_failed;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
            auto chain =
                std::make_shared<artifact_segment_chain>();
            if (!chain->append(
                    completion.bytes.data(),
                    completion.bytes.size())) {
                out.status =
                    llama_vbr_artifact_publish_status::
                        internal_error;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
            vbr_verified_segment segment;
            segment.unit_index = completion.unit_index;
            segment.shard_index = completion.shard_index;
            segment.clean_stash = completion.clean_stash;
            segment.bytes = std::move(chain);
            segment.streaming_digest =
                vbr_capture_stream_digest(*segment.bytes);
            stream_status =
                unit->accept_verified_segment(segment);
            if (stream_status !=
                    vbr_capture_stream_status::ok) {
                out.status =
                    stream_status ==
                        vbr_capture_stream_status::
                            duplicate_segment
                    ? llama_vbr_artifact_publish_status::
                        duplicate_completion
                    : llama_vbr_artifact_publish_status::
                        invalid_argument;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
        }
        stream_status = unit->seal_unit();
        if (stream_status != vbr_capture_stream_status::ok) {
            out.status =
                stream_status ==
                    vbr_capture_stream_status::missing_segment
                ? llama_vbr_artifact_publish_status::
                    missing_completion
                : llama_vbr_artifact_publish_status::
                    invalid_argument;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        const auto streamed = build->publish_reference();
        out.reference_artifact = streamed.reference_artifact;
        out.unit_content = streamed.unit_content;
        out.reference_lineage = streamed.reference_lineage;
        if (streamed.status != vbr_capture_stream_status::ok) {
            switch (streamed.status) {
                case vbr_capture_stream_status::format_rejected:
                case vbr_capture_stream_status::hash_mismatch:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            format_rejected;
                    break;
                case vbr_capture_stream_status::
                        accounting_unavailable:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            accounting_unavailable;
                    break;
                case vbr_capture_stream_status::accounting_refused:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            admission_refused;
                    break;
                case vbr_capture_stream_status::stage_failed:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            stage_failed;
                    break;
                case vbr_capture_stream_status::commit_failed:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            commit_failed;
                    break;
                case vbr_capture_stream_status::publication_failed:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            publication_failed;
                    break;
                case vbr_capture_stream_status::duplicate_segment:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            duplicate_completion;
                    break;
                case vbr_capture_stream_status::missing_segment:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            missing_completion;
                    break;
                case vbr_capture_stream_status::transfer_failed:
                case vbr_capture_stream_status::short_read:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            shard_failed;
                    break;
                case vbr_capture_stream_status::invalid_argument:
                case vbr_capture_stream_status::ring_unavailable:
                case vbr_capture_stream_status::late_segment:
                case vbr_capture_stream_status::projection_invalid:
                case vbr_capture_stream_status::snapshot_unavailable:
                case vbr_capture_stream_status::snapshot_changed:
                case vbr_capture_stream_status::internal_error:
                case vbr_capture_stream_status::_count:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            internal_error;
                    break;
                case vbr_capture_stream_status::ok:
                    break;
            }
            return out;
        }
        out.status = streamed.adopted
            ? llama_vbr_artifact_publish_status::adopted
            : llama_vbr_artifact_publish_status::published;
        return out;
    } catch (...) {
        out.status =
            llama_vbr_artifact_publish_status::internal_error;
        if (impl_) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
        }
        return out;
    }
}

llama_vbr_artifact_publish_result
llama_vbr_artifact_catalog::publish_stream(
        const vbr_artifact_package & package,
        const std::vector<vbr_verified_segment> & segments,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        void * prepared_stream_state) noexcept {
    llama_vbr_artifact_publish_result result;
    try {
        auto * stream_state =
            static_cast<catalog_stream_state *>(
                prepared_stream_state);
        if (package.unit_blobs.size() != 1 ||
            !package.companions.empty()) {
            return publish_stream_complete(
                package, segments, budget, fault,
                prepared_stream_state);
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->topologies != package.topologies ||
            package.unit_blobs.size() != 1 ||
            package.manifest.unit_references.size() != 1 ||
            !package.companions.empty()) {
            result.status =
                llama_vbr_artifact_publish_status::invalid_argument;
            impl_->n_refusals++;
            return result;
        }

        vbr_artifact_package working = package;
        auto & descriptor = working.unit_blobs[0].descriptor;
        const uint64_t expected =
            descriptor.shards.size() +
            (descriptor.clean_stash_state ==
                 vbr_artifact_clean_stash_state::present
                 ? descriptor.clean_stash.shards.size() : 0);
        if (segments.size() != expected) {
            result.status =
                segments.size() < expected
                    ? llama_vbr_artifact_publish_status::missing_completion
                    : llama_vbr_artifact_publish_status::duplicate_completion;
            impl_->n_refusals++;
            return result;
        }

        std::set<std::pair<bool, uint32_t>> seen;
        for (const auto & segment : segments) {
            if (segment.unit_index != 0 || !segment.bytes) {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }
            // The accepting pass hashes each completed D2H segment. Re-read
            // its immutable backing at the publication boundary so a source
            // mutation/corruption between completion and final artifact
            // encoding cannot silently mint a different content address.
            if (vbr_capture_stream_digest(*segment.bytes) !=
                    segment.streaming_digest) {
                result.status =
                    llama_vbr_artifact_publish_status::
                        format_rejected;
                impl_->n_refusals++;
                return result;
            }
            if (!seen.insert({
                    segment.clean_stash,
                    segment.shard_index }).second) {
                result.status =
                    llama_vbr_artifact_publish_status::duplicate_completion;
                impl_->n_refusals++;
                return result;
            }
            auto & shards = segment.clean_stash
                ? descriptor.clean_stash.shards
                : descriptor.shards;
            const auto shard = std::find_if(
                shards.begin(), shards.end(),
                [&](const vbr_artifact_shard_descriptor & candidate) {
                    return candidate.shard_index ==
                           segment.shard_index;
                });
            if (shard == shards.end()) {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }
            shard->payload = segment.bytes->source();
            shard->payload_bytes = segment.bytes->size();
        }

        const auto prepared = vbr_artifact_prepare(working);
        if (prepared != vbr_artifact_status::ok) {
            result.status =
                llama_vbr_artifact_publish_status::format_rejected;
            impl_->n_refusals++;
            return result;
        }

        const auto unit_key =
            working.unit_blobs[0].unit_version_id.bytes();
        const auto stash_key =
            working.unit_blobs[0].descriptor.clean_stash.payload_id.bytes();
        const bool has_stash =
            working.unit_blobs[0].descriptor.clean_stash_state ==
                vbr_artifact_clean_stash_state::present;
        const auto blob_it = impl_->blobs.find(unit_key);
        const bool blob_exists = blob_it != impl_->blobs.end();
        auto stash_it = has_stash
            ? impl_->stashes.find(stash_key) : impl_->stashes.end();
        const bool stash_exists =
            has_stash && stash_it != impl_->stashes.end();
        if (blob_exists &&
            (blob_it->second.stash_id.valid() != has_stash ||
             (has_stash &&
              blob_it->second.stash_id.bytes() != stash_key))) {
            result.status =
                llama_vbr_artifact_publish_status::publication_failed;
            impl_->n_refusals++;
            return result;
        }

        impl::blob pending_blob;
        impl::stash pending_stash;
        impl::reference pending_reference;
        if (blob_exists) {
            pending_blob = blob_it->second;
        } else {
            pending_blob.id =
                working.unit_blobs[0].unit_version_id;
            pending_blob.payload_digest =
                working.unit_blobs[0].payload_digest;
            pending_blob.stash_id =
                working.unit_blobs[0].descriptor.clean_stash.payload_id;
            pending_blob.descriptor =
                working.unit_blobs[0].descriptor;
            for (auto & shard : pending_blob.descriptor.shards) {
                shard.payload = {};
            }
            for (auto & shard :
                 pending_blob.descriptor.clean_stash.shards) {
                shard.payload = {};
            }
            pending_blob.artifact =
                stream_state &&
                    stream_state->charge_transfer_staging &&
                    !stream_state->blob_artifacts.empty()
                ? stream_state->blob_artifacts[0]
                : llama_cache_acct_artifact_id {};
            if ((pending_blob.artifact.v == 0 &&
                 !impl_->issue_artifact(
                     pending_blob.artifact)) ||
                !impl_->intern_content(
                    intern_purpose::unit, unit_key,
                    pending_blob.content) ||
                !impl_->intern_lineage(
                    intern_purpose::logical_unit,
                    vbr_artifact_logical_unit_digest(descriptor),
                    pending_blob.lineage)) {
                result.status =
                    llama_vbr_artifact_publish_status::internal_error;
                impl_->n_refusals++;
                return result;
            }
        }

        if (has_stash) {
            if (stash_exists) {
                pending_stash = stash_it->second;
            } else {
                pending_stash.id =
                    descriptor.clean_stash.payload_id;
                pending_stash.descriptor =
                    descriptor.clean_stash;
                for (auto & shard : pending_stash.descriptor.shards) {
                    shard.payload = {};
                }
                pending_stash.artifact =
                    stream_state &&
                        stream_state->charge_transfer_staging &&
                        !stream_state->stash_artifacts.empty()
                    ? stream_state->stash_artifacts[0]
                    : llama_cache_acct_artifact_id {};
                if ((pending_stash.artifact.v == 0 &&
                     !impl_->issue_artifact(
                         pending_stash.artifact)) ||
                    !impl_->intern_content(
                        intern_purpose::stash, stash_key,
                        pending_stash.content) ||
                    !impl_->intern_lineage(
                        intern_purpose::stash, stash_key,
                        pending_stash.lineage)) {
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    impl_->n_refusals++;
                    return result;
                }
            }
        }

        pending_reference.artifact =
            stream_state &&
                stream_state->charge_transfer_staging
            ? stream_state->reference_artifact
            : llama_cache_acct_artifact_id {};
        if ((pending_reference.artifact.v == 0 &&
             !impl_->issue_artifact(
                 pending_reference.artifact)) ||
            !impl_->intern_lineage(
                intern_purpose::manifest,
                working.manifest.manifest_digest.bytes(),
                pending_reference.lineage)) {
            result.status =
                llama_vbr_artifact_publish_status::internal_error;
            impl_->n_refusals++;
            return result;
        }
        pending_reference.unit_content = pending_blob.content;
        pending_reference.unit_ids = { pending_blob.id };
        pending_reference.stash_ids = { pending_blob.stash_id };
        pending_reference.manifest = working.manifest;
        for (size_t i = 0; i < working.companions.size(); ++i) {
            const auto & source = working.companions[i].payload;
            if (!source.valid() || source.size == 0) {
                result.status =
                    llama_vbr_artifact_publish_status::format_rejected;
                impl_->n_refusals++;
                return result;
            }
            auto chain = std::make_shared<artifact_segment_chain>();
            std::vector<uint8_t> chunk(
                size_t(std::min<uint64_t>(
                    source.size, 1024ull*1024)));
            uint64_t offset = 0;
            while (offset < source.size) {
                const size_t size = size_t(std::min<uint64_t>(
                    chunk.size(), source.size - offset));
                if (!source.read(
                        source.context, offset,
                        chunk.data(), size) ||
                    !chain->append(chunk.data(), size)) {
                    result.status =
                        llama_vbr_artifact_publish_status::
                            format_rejected;
                    impl_->n_refusals++;
                    return result;
                }
                offset += size;
            }
            pending_reference.companion_payloads.push_back(chain);
            pending_reference.manifest.companions[i].payload =
                chain->source();
        }

        std::vector<impl::txn_leaf> leaves;
        leaves.reserve(working.manifest.accounting.size());
        for (const auto & row : working.manifest.accounting) {
            llama_cache_acct_resource_domain domain;
            const auto category =
                vbr_artifact_accounting_category(row.role);
            if (!impl_->resolve_domain(row.domain, domain) ||
                !impl_->configured.count({ category, domain })) {
                result.status =
                    llama_vbr_artifact_publish_status::accounting_unavailable;
                impl_->n_refusals++;
                return result;
            }

            impl::txn_leaf leaf;
            leaf.binding.category = category;
            leaf.binding.domain = domain;
            leaf.binding.logical = row.logical_bytes;
            leaf.binding.resident = row.resident_bytes;
            const std::vector<impl::allocation> * existing = nullptr;
            if (row.role ==
                    vbr_artifact_accounting_role::unit_payload ||
                row.role ==
                    vbr_artifact_accounting_role::descriptor_metadata) {
                leaf.binding.artifact = pending_blob.artifact;
                leaf.binding.content = pending_blob.content;
                leaf.binding.lineage = pending_blob.lineage;
                existing = blob_exists
                    ? &blob_it->second.allocations : nullptr;
            } else if (row.role ==
                    vbr_artifact_accounting_role::clean_stash_payload) {
                if (!has_stash) {
                    result.status =
                        llama_vbr_artifact_publish_status::format_rejected;
                    impl_->n_refusals++;
                    return result;
                }
                leaf.binding.artifact = pending_stash.artifact;
                leaf.binding.content = pending_stash.content;
                leaf.binding.lineage = pending_stash.lineage;
                existing = stash_exists
                    ? &stash_it->second.allocations : nullptr;
            } else if (row.role ==
                    vbr_artifact_accounting_role::reference_metadata) {
                leaf.binding.artifact =
                    pending_reference.artifact;
                if (!impl_->intern_content(
                        intern_purpose::manifest,
                        working.manifest.manifest_digest.bytes(),
                        leaf.binding.content)) {
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    impl_->n_refusals++;
                    return result;
                }
                leaf.binding.lineage =
                    pending_reference.lineage;
            } else {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }

            if (existing) {
                const auto * allocation = impl_->find_allocation(
                    *existing, category, domain,
                    row.logical_bytes, row.resident_bytes);
                if (!allocation) {
                    result.status =
                        llama_vbr_artifact_publish_status::publication_failed;
                    impl_->n_refusals++;
                    return result;
                }
                leaf.binding = *allocation;
                leaf.existing = true;
                leaf.reserve_resident = 0;
            } else {
                leaf.reserve_resident = row.resident_bytes;
            }
            leaves.push_back(leaf);
        }

        std::vector<llama_cache_acct_op_id> committed(
            leaves.size());
        std::vector<llama_cache_acct_alloc_id> allocations(
            leaves.size());
        std::vector<llama_cache_transaction_leaf>
            transaction_leaves;
        transaction_leaves.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            llama_cache_transaction_leaf leaf;
            leaf.category = leaves[i].binding.category;
            leaf.domain = leaves[i].binding.domain;
            leaf.attribution = {
                llama_cache_acct_attr_kind::artifact,
                -1,
                leaves[i].binding.artifact,
            };
            leaf.expected_logical =
                leaves[i].binding.logical;
            leaf.reserve_resident =
                leaves[i].reserve_resident;
            leaf.stage_resident =
                leaves[i].binding.resident;
            leaf.artifact = leaves[i].binding.artifact;
            leaf.content = leaves[i].binding.content;
            leaf.lineage = leaves[i].binding.lineage;
            leaf.existing_allocation =
                leaves[i].existing
                    ? leaves[i].binding.alloc
                    : llama_cache_acct_alloc_id {};
            leaf.committed_op = &committed[i];
            leaf.allocation_out = &allocations[i];
            transaction_leaves.push_back(leaf);
        }

        // A real stream prepared both capacity groups before its first D2H
        // allocation.  The staging group is committed on first acceptance
        // and remains live through this terminal.  A content-addressed
        // adoption changes the allocation shape, so its deliberately
        // conservative fresh durable claims are aborted and re-priced while
        // staging is still live.
        llama_cache_prepared_claim_group local_durable;
        llama_cache_prepared_claim_group * prepared_durable =
            nullptr;
        const bool use_stream_preparation =
            stream_state &&
            stream_state->charge_transfer_staging &&
            !blob_exists && !stash_exists;
        if (use_stream_preparation) {
            prepared_durable =
                &stream_state->durable_prepared;
        } else {
            if (stream_state &&
                stream_state->charge_transfer_staging) {
                stream_state->durable_prepared = {};
            }
            local_durable =
                llama_cache_prepare_reservation_transaction(
                    impl_->ledger, budget,
                    transaction_leaves);
            prepared_durable = &local_durable;
        }
        if (!prepared_durable->ready()) {
            result.status =
                llama_vbr_artifact_publish_status::
                    admission_refused;
            if (stream_state &&
                stream_state->charge_transfer_staging) {
                impl_->n_staging_overlap_refusals++;
            }
            impl_->n_refusals++;
            return result;
        }

        struct materialize_context {
            impl::blob * blob = nullptr;
            impl::stash * stash = nullptr;
            const vbr_artifact_unit_descriptor * descriptor =
                nullptr;
            const std::vector<vbr_verified_segment> *
                    segments = nullptr;
            bool blob_exists = false;
            bool stash_exists = false;
            bool has_stash = false;
        } materialize {
            &pending_blob,
            &pending_stash,
            &descriptor,
            &segments,
            blob_exists,
            stash_exists,
            has_stash,
        };
        const auto materialize_storage = [](void * opaque) -> bool {
            auto * context =
                static_cast<materialize_context *>(opaque);
            if (!context || !context->blob ||
                !context->stash || !context->descriptor ||
                !context->segments) {
                return false;
            }
            if (!context->blob_exists) {
                context->blob->payload_shards.reserve(
                    context->descriptor->shards.size());
                for (const auto & shard :
                     context->descriptor->shards) {
                    const auto completion = std::find_if(
                        context->segments->begin(),
                        context->segments->end(),
                        [&](const auto & candidate) {
                            return !candidate.clean_stash &&
                                   candidate.shard_index ==
                                       shard.shard_index;
                        });
                    if (completion ==
                            context->segments->end() ||
                        !completion->bytes) {
                        return false;
                    }
                    context->blob->payload_shards.push_back(
                        completion->bytes);
                }
            }
            if (context->has_stash &&
                !context->stash_exists) {
                context->stash->shards.reserve(
                    context->descriptor->clean_stash.shards.size());
                for (const auto & shard :
                     context->descriptor->clean_stash.shards) {
                    const auto completion = std::find_if(
                        context->segments->begin(),
                        context->segments->end(),
                        [&](const auto & candidate) {
                            return candidate.clean_stash &&
                                   candidate.shard_index ==
                                       shard.shard_index;
                        });
                    if (completion ==
                            context->segments->end() ||
                        !completion->bytes) {
                        return false;
                    }
                    context->stash->shards.push_back(
                        completion->bytes);
                }
            }
            return true;
        };
        const llama_cache_transaction_after_admit after_admit {
            &materialize, materialize_storage,
        };
        const auto transaction =
            prepared_durable->materialize_and_commit(
                transaction_leaves, fault, after_admit);
        if (transaction.status !=
                llama_cache_transaction_status::committed) {
            switch (transaction.status) {
                case llama_cache_transaction_status::admission_refused:
                    result.status =
                        llama_vbr_artifact_publish_status::admission_refused;
                    break;
                case llama_cache_transaction_status::stage_failed:
                    result.status =
                        llama_vbr_artifact_publish_status::stage_failed;
                    break;
                case llama_cache_transaction_status::commit_failed:
                    result.status =
                        llama_vbr_artifact_publish_status::commit_failed;
                    break;
                case llama_cache_transaction_status::post_commit_fault:
                    result.status =
                        llama_vbr_artifact_publish_status::publication_failed;
                    break;
                case llama_cache_transaction_status::invalid_argument:
                case llama_cache_transaction_status::after_admit_failed:
                case llama_cache_transaction_status::internal_fault:
                case llama_cache_transaction_status::_count:
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    break;
                case llama_cache_transaction_status::committed:
                    break;
            }
            impl_->n_refusals++;
            return result;
        }

        struct rollback_guard {
            llama_cache_acct_ledger * ledger = nullptr;
            std::vector<llama_cache_acct_op_id> * ops = nullptr;
            bool keep = false;
            ~rollback_guard() {
                if (!keep && ledger && ops) {
                    for (const auto op : *ops) {
                        ledger->release(op);
                    }
                }
            }
        } rollback { &impl_->ledger, &committed, false };

        for (size_t i = 0; i < leaves.size(); ++i) {
            leaves[i].binding.alloc = allocations[i];
            if (leaves[i].binding.artifact ==
                    pending_reference.artifact) {
                pending_reference.allocations.push_back(
                    leaves[i].binding);
            }
            if (!leaves[i].existing) {
                if (leaves[i].binding.category ==
                        llama_cache_acct_category::clean_stash_payload) {
                    pending_stash.allocations.push_back(
                        leaves[i].binding);
                } else if (leaves[i].binding.category !=
                        llama_cache_acct_category::artifact_reference_metadata) {
                    pending_blob.allocations.push_back(
                        leaves[i].binding);
                }
            }
        }

        pending_reference.operations = committed;
        bool inserted_blob = false;
        bool inserted_stash = false;
        try {
            if (!blob_exists) {
                inserted_blob =
                    impl_->blobs.emplace(
                        unit_key, std::move(pending_blob)).second;
                if (!inserted_blob) {
                    throw 0;
                }
            }
            if (has_stash && !stash_exists) {
                inserted_stash =
                    impl_->stashes.emplace(
                        stash_key, std::move(pending_stash)).second;
                if (!inserted_stash) {
                    throw 0;
                }
            }
            const auto inserted_reference =
                impl_->references.emplace(
                    pending_reference.artifact.v,
                    std::move(pending_reference)).second;
            if (!inserted_reference) {
                throw 0;
            }
        } catch (...) {
            if (inserted_blob) {
                impl_->blobs.erase(unit_key);
            }
            if (inserted_stash) {
                impl_->stashes.erase(stash_key);
            }
            result.status =
                llama_vbr_artifact_publish_status::publication_failed;
            impl_->n_refusals++;
            return result;
        }

        rollback.keep = true;
        result.status = blob_exists
            ? llama_vbr_artifact_publish_status::adopted
            : llama_vbr_artifact_publish_status::published;
        const auto & stored =
            impl_->references.find(
                pending_reference.artifact.v)->second;
        result.reference_artifact = stored.artifact;
        result.unit_content = stored.unit_content;
        result.reference_lineage = stored.lineage;
        if (blob_exists) {
            impl_->n_adopted++;
        } else {
            impl_->n_published++;
        }
        return result;
    } catch (...) {
        result.status =
            llama_vbr_artifact_publish_status::internal_error;
        if (impl_) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
        }
        return result;
    }
}

llama_vbr_artifact_publish_result
llama_vbr_artifact_catalog::publish_stream_complete(
        const vbr_artifact_package & package,
        const std::vector<vbr_verified_segment> & segments,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        void * prepared_stream_state) noexcept {
    llama_vbr_artifact_publish_result result;
    try {
        auto * stream_state =
            static_cast<catalog_stream_state *>(prepared_stream_state);
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->topologies != package.topologies ||
            package.unit_blobs.empty() ||
            package.manifest.unit_references.size() !=
                package.unit_blobs.size()) {
            result.status =
                llama_vbr_artifact_publish_status::invalid_argument;
            impl_->n_refusals++;
            return result;
        }

        vbr_artifact_package working = package;
        size_t expected_segments = 0;
        for (const auto & blob : working.unit_blobs) {
            expected_segments += blob.descriptor.shards.size();
            if (blob.descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present) {
                expected_segments +=
                    blob.descriptor.clean_stash.shards.size();
            }
        }
        if (segments.size() != expected_segments) {
            result.status = segments.size() < expected_segments
                ? llama_vbr_artifact_publish_status::missing_completion
                : llama_vbr_artifact_publish_status::duplicate_completion;
            impl_->n_refusals++;
            return result;
        }

        std::set<std::tuple<uint32_t, bool, uint32_t>> seen;
        for (const auto & segment : segments) {
            if (segment.unit_index >= working.unit_blobs.size() ||
                !segment.bytes ||
                vbr_capture_stream_digest(*segment.bytes) !=
                    segment.streaming_digest ||
                !seen.emplace(
                    segment.unit_index,
                    segment.clean_stash,
                    segment.shard_index).second) {
                result.status =
                    llama_vbr_artifact_publish_status::format_rejected;
                impl_->n_refusals++;
                return result;
            }
            auto & descriptor =
                working.unit_blobs[segment.unit_index].descriptor;
            auto & shards = segment.clean_stash
                ? descriptor.clean_stash.shards
                : descriptor.shards;
            const auto shard = std::find_if(
                shards.begin(), shards.end(),
                [&](const auto & candidate) {
                    return candidate.shard_index ==
                           segment.shard_index;
                });
            if (shard == shards.end() ||
                shard->payload_bytes != segment.bytes->size()) {
                result.status =
                    llama_vbr_artifact_publish_status::format_rejected;
                impl_->n_refusals++;
                return result;
            }
            shard->payload = segment.bytes->source();
        }
        if (stream_state == nullptr ||
            stream_state->companions.size() !=
                working.companions.size()) {
            result.status =
                llama_vbr_artifact_publish_status::missing_completion;
            impl_->n_refusals++;
            return result;
        }
        for (const auto & companion :
             stream_state->companions) {
            if (companion.companion_index >=
                    working.companions.size() ||
                !companion.bytes) {
                result.status =
                    llama_vbr_artifact_publish_status::format_rejected;
                impl_->n_refusals++;
                return result;
            }
            working.companions[
                companion.companion_index].payload =
                    companion.bytes->source();
        }
        if (vbr_artifact_prepare(working) !=
                vbr_artifact_status::ok) {
            result.status =
                llama_vbr_artifact_publish_status::format_rejected;
            impl_->n_refusals++;
            return result;
        }

        std::vector<impl::blob> pending_blobs(
            working.unit_blobs.size());
        std::vector<impl::stash> pending_stashes(
            working.unit_blobs.size());
        std::vector<bool> blob_exists(
            working.unit_blobs.size(), false);
        std::vector<bool> stash_exists(
            working.unit_blobs.size(), false);
        std::vector<size_t> stash_alias(
            working.unit_blobs.size(), SIZE_MAX);
        std::map<digest_key, size_t> package_stashes;
        impl::reference pending_reference;
        pending_reference.artifact =
            stream_state && stream_state->charge_transfer_staging
                ? stream_state->reference_artifact
                : llama_cache_acct_artifact_id {};
        if ((pending_reference.artifact.v == 0 &&
             !impl_->issue_artifact(pending_reference.artifact)) ||
            !impl_->intern_lineage(
                intern_purpose::manifest,
                working.manifest.manifest_digest.bytes(),
                pending_reference.lineage)) {
            result.status =
                llama_vbr_artifact_publish_status::internal_error;
            impl_->n_refusals++;
            return result;
        }
        pending_reference.manifest = working.manifest;
        pending_reference.companion_payloads.reserve(
            stream_state->companions.size());
        for (const auto & companion : stream_state->companions) {
            pending_reference.companion_payloads.push_back(
                companion.bytes);
            pending_reference.manifest.companions[
                companion.companion_index].payload =
                    companion.bytes->source();
        }

        for (size_t u = 0; u < working.unit_blobs.size(); ++u) {
            const auto unit_key =
                working.unit_blobs[u].unit_version_id.bytes();
            const auto found = impl_->blobs.find(unit_key);
            blob_exists[u] = found != impl_->blobs.end();
            if (blob_exists[u]) {
                pending_blobs[u] = found->second;
            } else {
                auto & pending = pending_blobs[u];
                pending.id = working.unit_blobs[u].unit_version_id;
                pending.payload_digest =
                    working.unit_blobs[u].payload_digest;
                pending.descriptor =
                    working.unit_blobs[u].descriptor;
                pending.stash_id =
                    pending.descriptor.clean_stash.payload_id;
                for (auto & shard : pending.descriptor.shards) {
                    shard.payload = {};
                }
                for (auto & shard :
                     pending.descriptor.clean_stash.shards) {
                    shard.payload = {};
                }
                pending.artifact =
                    stream_state &&
                    u < stream_state->blob_artifacts.size()
                        ? stream_state->blob_artifacts[u]
                        : llama_cache_acct_artifact_id {};
                if ((pending.artifact.v == 0 &&
                     !impl_->issue_artifact(pending.artifact)) ||
                    !impl_->intern_content(
                        intern_purpose::unit, unit_key,
                        pending.content) ||
                    !impl_->intern_lineage(
                        intern_purpose::logical_unit,
                        vbr_artifact_logical_unit_digest(
                            pending.descriptor),
                        pending.lineage)) {
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    impl_->n_refusals++;
                    return result;
                }
            }
            pending_reference.unit_ids.push_back(
                pending_blobs[u].id);
            pending_reference.stash_ids.push_back(
                pending_blobs[u].stash_id);

            const bool has_stash =
                working.unit_blobs[u].descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present;
            if (!has_stash) {
                continue;
            }
            const auto stash_key =
                working.unit_blobs[u].descriptor.clean_stash
                    .payload_id.bytes();
            const auto package_stash =
                package_stashes.find(stash_key);
            if (package_stash != package_stashes.end()) {
                stash_alias[u] = package_stash->second;
                stash_exists[u] =
                    stash_exists[package_stash->second];
                pending_stashes[u] =
                    pending_stashes[package_stash->second];
                continue;
            }
            package_stashes.emplace(stash_key, u);
            const auto found_stash = impl_->stashes.find(stash_key);
            stash_exists[u] =
                found_stash != impl_->stashes.end();
            if (stash_exists[u]) {
                pending_stashes[u] = found_stash->second;
            } else {
                auto & pending = pending_stashes[u];
                pending.id =
                    working.unit_blobs[u].descriptor.clean_stash
                        .payload_id;
                pending.descriptor =
                    working.unit_blobs[u].descriptor.clean_stash;
                for (auto & shard : pending.descriptor.shards) {
                    shard.payload = {};
                }
                pending.artifact =
                    stream_state &&
                    u < stream_state->stash_artifacts.size()
                        ? stream_state->stash_artifacts[u]
                        : llama_cache_acct_artifact_id {};
                if ((pending.artifact.v == 0 &&
                     !impl_->issue_artifact(pending.artifact)) ||
                    !impl_->intern_content(
                        intern_purpose::stash, stash_key,
                        pending.content) ||
                    !impl_->intern_lineage(
                        intern_purpose::stash, stash_key,
                        pending.lineage)) {
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    impl_->n_refusals++;
                    return result;
                }
            }
        }
        pending_reference.unit_content =
            pending_blobs.front().content;

        // Build exact content-addressed leaves. Per-unit payload/stash rows
        // sum to the portable aggregate manifest but retain charge-once
        // allocation identity across references.
        std::vector<impl::txn_leaf> leaves;
        auto append_leaf = [&](impl::allocation binding,
                               const std::vector<impl::allocation> * existing) {
            impl::txn_leaf leaf;
            leaf.binding = binding;
            if (existing) {
                const auto * allocation = impl_->find_allocation(
                    *existing, binding.category, binding.domain,
                    binding.logical, binding.resident);
                if (!allocation) {
                    return false;
                }
                leaf.binding = *allocation;
                leaf.existing = true;
            } else {
                leaf.reserve_resident = binding.resident;
            }
            leaves.push_back(leaf);
            return true;
        };
        for (size_t u = 0; u < working.unit_blobs.size(); ++u) {
            std::map<std::pair<uint32_t, uint16_t>, uint64_t>
                payload_by_domain;
            for (const auto & shard :
                 working.unit_blobs[u].descriptor.shards) {
                payload_by_domain[{
                    shard.topology_index,
                    shard.device_ordinal }] += shard.payload_bytes;
            }
            for (const auto & row : payload_by_domain) {
                llama_cache_acct_resource_domain domain;
                const vbr_artifact_portable_domain portable {
                    llama_cache_acct_residency::device,
                    llama_cache_acct_domain_kind::device_topology,
                    row.first.first, row.first.second,
                };
                if (!impl_->resolve_domain(portable, domain)) {
                    result.status =
                        llama_vbr_artifact_publish_status::
                            accounting_unavailable;
                    impl_->n_refusals++;
                    return result;
                }
                impl::allocation binding;
                binding.category =
                    llama_cache_acct_category::unit_version_payload;
                binding.domain = domain;
                binding.logical = row.second;
                binding.resident = row.second;
                binding.artifact = pending_blobs[u].artifact;
                binding.content = pending_blobs[u].content;
                binding.lineage = pending_blobs[u].lineage;
                if (!append_leaf(
                        binding, blob_exists[u]
                            ? &impl_->blobs.find(
                                working.unit_blobs[u]
                                    .unit_version_id.bytes())
                                  ->second.allocations
                            : nullptr)) {
                    result.status =
                        llama_vbr_artifact_publish_status::
                            publication_failed;
                    impl_->n_refusals++;
                    return result;
                }
            }
            if (working.unit_blobs[u].descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present) {
                if (stash_alias[u] != SIZE_MAX) {
                    continue;
                }
                std::map<std::pair<uint32_t, uint16_t>, uint64_t>
                    stash_by_domain;
                for (const auto & shard :
                     working.unit_blobs[u].descriptor.clean_stash.shards) {
                    stash_by_domain[{
                        shard.topology_index,
                        shard.device_ordinal }] += shard.payload_bytes;
                }
                for (const auto & row : stash_by_domain) {
                    llama_cache_acct_resource_domain domain;
                    const vbr_artifact_portable_domain portable {
                        llama_cache_acct_residency::device,
                        llama_cache_acct_domain_kind::device_topology,
                        row.first.first, row.first.second,
                    };
                    if (!impl_->resolve_domain(portable, domain)) {
                        result.status =
                            llama_vbr_artifact_publish_status::
                                accounting_unavailable;
                        impl_->n_refusals++;
                        return result;
                    }
                    impl::allocation binding;
                    binding.category =
                        llama_cache_acct_category::
                            clean_stash_payload;
                    binding.domain = domain;
                    binding.logical = row.second;
                    binding.resident = row.second;
                    binding.artifact = pending_stashes[u].artifact;
                    binding.content = pending_stashes[u].content;
                    binding.lineage = pending_stashes[u].lineage;
                    if (!append_leaf(
                            binding, stash_exists[u]
                                ? &impl_->stashes.find(
                                    working.unit_blobs[u].descriptor
                                        .clean_stash.payload_id.bytes())
                                      ->second.allocations
                                : nullptr)) {
                        result.status =
                            llama_vbr_artifact_publish_status::
                                publication_failed;
                        impl_->n_refusals++;
                        return result;
                    }
                }
            }
        }

        llama_cache_acct_content_digest manifest_content;
        if (!impl_->intern_content(
                intern_purpose::manifest,
                working.manifest.manifest_digest.bytes(),
                manifest_content)) {
            result.status =
                llama_vbr_artifact_publish_status::internal_error;
            impl_->n_refusals++;
            return result;
        }
        for (const auto & row : working.manifest.accounting) {
            if (row.role !=
                    vbr_artifact_accounting_role::descriptor_metadata &&
                row.role !=
                    vbr_artifact_accounting_role::reference_metadata &&
                row.role !=
                    vbr_artifact_accounting_role::recurrent_payload &&
                row.role !=
                    vbr_artifact_accounting_role::
                        typed_accelerator_payload) {
                continue;
            }
            llama_cache_acct_resource_domain domain;
            if (!impl_->resolve_domain(row.domain, domain)) {
                result.status =
                    llama_vbr_artifact_publish_status::
                        accounting_unavailable;
                impl_->n_refusals++;
                return result;
            }
            impl::allocation binding;
            binding.category =
                vbr_artifact_accounting_category(row.role);
            binding.domain = domain;
            binding.logical = row.logical_bytes;
            binding.resident = row.resident_bytes;
            binding.artifact = pending_reference.artifact;
            binding.content = manifest_content;
            binding.lineage = pending_reference.lineage;
            if (!append_leaf(binding, nullptr)) {
                result.status =
                    llama_vbr_artifact_publish_status::internal_error;
                impl_->n_refusals++;
                return result;
            }
        }

        std::vector<llama_cache_acct_op_id> committed(
            leaves.size());
        std::vector<llama_cache_acct_alloc_id> allocations(
            leaves.size());
        std::vector<llama_cache_transaction_leaf> transaction_leaves;
        transaction_leaves.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            llama_cache_transaction_leaf leaf;
            leaf.category = leaves[i].binding.category;
            leaf.domain = leaves[i].binding.domain;
            leaf.attribution = {
                llama_cache_acct_attr_kind::artifact, -1,
                leaves[i].binding.artifact,
            };
            leaf.expected_logical = leaves[i].binding.logical;
            leaf.reserve_resident = leaves[i].existing
                ? 0 : leaves[i].binding.resident;
            leaf.stage_resident = leaves[i].binding.resident;
            leaf.artifact = leaves[i].binding.artifact;
            leaf.content = leaves[i].binding.content;
            leaf.lineage = leaves[i].binding.lineage;
            leaf.existing_allocation = leaves[i].existing
                ? leaves[i].binding.alloc
                : llama_cache_acct_alloc_id {};
            leaf.committed_op = &committed[i];
            leaf.allocation_out = &allocations[i];
            transaction_leaves.push_back(leaf);
        }
        if (stream_state && stream_state->charge_transfer_staging) {
            stream_state->durable_prepared = {};
        }
        auto prepared =
            llama_cache_prepare_reservation_transaction(
                impl_->ledger, budget, transaction_leaves);
        if (!prepared.ready()) {
            result.status =
                llama_vbr_artifact_publish_status::admission_refused;
            impl_->n_staging_overlap_refusals++;
            impl_->n_refusals++;
            return result;
        }

        struct materialize_context {
            std::vector<impl::blob> * blobs;
            std::vector<impl::stash> * stashes;
            const std::vector<bool> * blob_exists;
            const std::vector<bool> * stash_exists;
            const std::vector<size_t> * stash_alias;
            const std::vector<vbr_verified_segment> * segments;
        } materialize {
            &pending_blobs, &pending_stashes,
            &blob_exists, &stash_exists, &stash_alias, &segments,
        };
        const auto materialize_storage = [](void * opaque) -> bool {
            auto * context =
                static_cast<materialize_context *>(opaque);
            for (size_t u = 0; u < context->blobs->size(); ++u) {
                auto & blob = (*context->blobs)[u];
                if (!(*context->blob_exists)[u]) {
                    for (const auto & shard : blob.descriptor.shards) {
                        const auto segment = std::find_if(
                            context->segments->begin(),
                            context->segments->end(),
                            [&](const auto & candidate) {
                                return candidate.unit_index == u &&
                                       !candidate.clean_stash &&
                                       candidate.shard_index ==
                                           shard.shard_index;
                            });
                        if (segment == context->segments->end()) {
                            return false;
                        }
                        blob.payload_shards.push_back(segment->bytes);
                    }
                }
                if (blob.stash_id.valid() &&
                    !(*context->stash_exists)[u] &&
                    (*context->stash_alias)[u] == SIZE_MAX) {
                    auto & stash = (*context->stashes)[u];
                    for (const auto & shard :
                         stash.descriptor.shards) {
                        const auto segment = std::find_if(
                            context->segments->begin(),
                            context->segments->end(),
                            [&](const auto & candidate) {
                                return candidate.unit_index == u &&
                                       candidate.clean_stash &&
                                       candidate.shard_index ==
                                           shard.shard_index;
                            });
                        if (segment == context->segments->end()) {
                            return false;
                        }
                        stash.shards.push_back(segment->bytes);
                    }
                }
            }
            return true;
        };
        const auto transaction = prepared.materialize_and_commit(
            transaction_leaves, fault,
            { &materialize, materialize_storage });
        if (transaction.status !=
                llama_cache_transaction_status::committed) {
            result.status =
                transaction.status ==
                    llama_cache_transaction_status::stage_failed
                ? llama_vbr_artifact_publish_status::stage_failed
                : transaction.status ==
                      llama_cache_transaction_status::commit_failed
                    ? llama_vbr_artifact_publish_status::commit_failed
                    : llama_vbr_artifact_publish_status::
                        admission_refused;
            impl_->n_refusals++;
            return result;
        }

        struct rollback_guard {
            llama_cache_acct_ledger * ledger;
            std::vector<llama_cache_acct_op_id> * ops;
            bool keep = false;
            ~rollback_guard() {
                if (!keep) {
                    for (const auto op : *ops) {
                        if (op) {
                            ledger->release(op);
                        }
                    }
                }
            }
        } rollback { &impl_->ledger, &committed, false };
        for (size_t i = 0; i < leaves.size(); ++i) {
            leaves[i].binding.alloc = allocations[i];
            if (leaves[i].binding.artifact ==
                    pending_reference.artifact) {
                pending_reference.allocations.push_back(
                    leaves[i].binding);
            }
            if (leaves[i].existing) {
                continue;
            }
            bool assigned = false;
            for (size_t u = 0; u < pending_blobs.size(); ++u) {
                if (leaves[i].binding.artifact ==
                        pending_blobs[u].artifact) {
                    pending_blobs[u].allocations.push_back(
                        leaves[i].binding);
                    assigned = true;
                    break;
                }
                if (pending_stashes[u].artifact.v != 0 &&
                    leaves[i].binding.artifact ==
                        pending_stashes[u].artifact) {
                    pending_stashes[u].allocations.push_back(
                        leaves[i].binding);
                    assigned = true;
                    break;
                }
            }
            (void) assigned;
        }
        pending_reference.operations = committed;
        std::vector<digest_key> inserted_blobs;
        std::vector<digest_key> inserted_stashes;
        try {
            for (size_t u = 0; u < pending_blobs.size(); ++u) {
                if (!blob_exists[u]) {
                    const auto key = pending_blobs[u].id.bytes();
                    if (!impl_->blobs.emplace(
                            key, std::move(pending_blobs[u])).second) {
                        throw 0;
                    }
                    inserted_blobs.push_back(key);
                }
                if (pending_stashes[u].id.valid() &&
                    !stash_exists[u] &&
                    stash_alias[u] == SIZE_MAX) {
                    const auto key = pending_stashes[u].id.bytes();
                    if (!impl_->stashes.emplace(
                            key, std::move(pending_stashes[u])).second) {
                        throw 0;
                    }
                    inserted_stashes.push_back(key);
                }
            }
            if (!impl_->references.emplace(
                    pending_reference.artifact.v,
                    std::move(pending_reference)).second) {
                throw 0;
            }
        } catch (...) {
            for (const auto & key : inserted_blobs) {
                impl_->blobs.erase(key);
            }
            for (const auto & key : inserted_stashes) {
                impl_->stashes.erase(key);
            }
            result.status =
                llama_vbr_artifact_publish_status::publication_failed;
            impl_->n_refusals++;
            return result;
        }
        rollback.keep = true;
        const bool all_adopted =
            std::all_of(blob_exists.begin(), blob_exists.end(),
                [](bool value) { return value; });
        result.status = all_adopted
            ? llama_vbr_artifact_publish_status::adopted
            : llama_vbr_artifact_publish_status::published;
        const auto & stored = impl_->references.find(
            pending_reference.artifact.v)->second;
        result.reference_artifact = stored.artifact;
        result.unit_content = stored.unit_content;
        result.reference_lineage = stored.lineage;
        if (all_adopted) {
            impl_->n_adopted++;
        } else {
            impl_->n_published++;
        }
        return result;
    } catch (...) {
        result.status =
            llama_vbr_artifact_publish_status::internal_error;
        if (impl_) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
        }
        return result;
    }
}

vbr_artifact_resolve_status llama_vbr_artifact_catalog::resolve_reference(
        llama_cache_acct_artifact_id reference,
        vbr_artifact_package_view & out) noexcept {
    out.reset();
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->references.find(reference.v);
        if (it == impl_->references.end()) {
            return vbr_artifact_resolve_status::not_found;
        }
        if (it->second.host_owned || it->second.retire_pending ||
            it->second.prepared_retire_token != 0 ||
            it->second.borrow_count == UINT64_MAX) {
            return vbr_artifact_resolve_status::busy;
        }

        auto state = std::make_shared<vbr_artifact_package_view::storage>();
        const auto allocation_view = [](const impl::allocation & value) {
            return vbr_artifact_allocation_view {
                value.category,
                value.domain,
                value.logical,
                value.resident,
                value.alloc,
                value.artifact,
                value.content,
                value.lineage,
            };
        };
        state->reference = reference;
        state->topologies = impl_->topologies;
        state->manifest = it->second.manifest;
        state->units.reserve(it->second.unit_ids.size());
        for (const auto & id : it->second.unit_ids) {
            const auto found = impl_->blobs.find(id.bytes());
            if (found == impl_->blobs.end()) {
                return vbr_artifact_resolve_status::unavailable;
            }
            vbr_artifact_unit_view unit;
            unit.unit_version_id = found->second.id;
            unit.payload_digest = found->second.payload_digest;
            unit.descriptor = found->second.descriptor;
            for (auto & shard : unit.descriptor.shards) {
                shard.payload = {};
            }
            for (auto & shard : unit.descriptor.clean_stash.shards) {
                shard.payload = {};
            }
            unit.payload_shards = found->second.payload_shards;
            for (const auto & allocation : found->second.allocations) {
                unit.payload_allocations.push_back(
                    allocation_view(allocation));
            }
            if (found->second.stash_id.valid()) {
                const auto stash = impl_->stashes.find(
                    found->second.stash_id.bytes());
                if (stash == impl_->stashes.end()) {
                    return vbr_artifact_resolve_status::unavailable;
                }
                unit.stash_shards = stash->second.shards;
                for (const auto & allocation : stash->second.allocations) {
                    unit.stash_allocations.push_back(
                        allocation_view(allocation));
                }
            }
            state->units.push_back(std::move(unit));
        }
        if (it->second.companion_payloads.size() !=
            it->second.manifest.companions.size()) {
            return vbr_artifact_resolve_status::unavailable;
        }
        state->companions.reserve(it->second.companion_payloads.size());
        for (size_t i = 0; i < it->second.companion_payloads.size(); ++i) {
            vbr_artifact_companion_view companion;
            companion.descriptor = it->second.manifest.companions[i];
            companion.descriptor.payload = {};
            companion.payload = it->second.companion_payloads[i];
            state->companions.push_back(std::move(companion));
        }
        for (const auto & allocation : it->second.allocations) {
            state->reference_allocations.push_back(
                allocation_view(allocation));
        }

        ++it->second.borrow_count;
        out.owner_ = this;
        out.storage_ = std::move(state);
        return vbr_artifact_resolve_status::ok;
    } catch (...) {
        return vbr_artifact_resolve_status::internal_error;
    }
}

bool llama_vbr_artifact_catalog::claim_host_ownership(
        llama_cache_acct_artifact_id reference) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->references.find(reference.v);
        if (it == impl_->references.end() ||
            it->second.borrow_count != 1 || it->second.host_owned ||
            it->second.retire_pending ||
            it->second.prepared_retire_token != 0) {
            return false;
        }
        std::sort(
            it->second.operations.begin(), it->second.operations.end());
        if ((!it->second.operations.empty() &&
             !it->second.operations.front()) ||
            std::adjacent_find(
                it->second.operations.begin(),
                it->second.operations.end()) !=
                it->second.operations.end()) {
            return false;
        }
        it->second.host_owned = true;
        return true;
    } catch (...) {
        return false;
    }
}

bool llama_vbr_artifact_catalog::prepare_owned_retire(
        const std::vector<llama_cache_acct_artifact_id> & references,
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) noexcept {
    out.reset();
    try {
        if (references.empty() || expected_serial == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        std::vector<llama_cache_acct_artifact_id> canonical = references;
        std::sort(canonical.begin(), canonical.end(),
            [](const auto & a, const auto & b) { return a.v < b.v; });
        if (canonical.front().v == 0 ||
            std::adjacent_find(
                canonical.begin(), canonical.end(),
                [](const auto & a, const auto & b) { return a == b; }) !=
                    canonical.end()) {
            return false;
        }

        std::vector<llama_cache_acct_op_id> operations;
        std::vector<vbr_unit_version_id> unit_ids;
        std::vector<vbr_stash_payload_id> stash_ids;
        for (const auto reference : canonical) {
            const auto it = impl_->references.find(reference.v);
            if (it == impl_->references.end() || !it->second.host_owned ||
                it->second.retire_pending ||
                it->second.prepared_retire_token != 0 ||
                it->second.borrow_count != 1) {
                return false;
            }
            if (it->second.operations.size() > SIZE_MAX - operations.size()) {
                return false;
            }
            operations.insert(
                operations.end(), it->second.operations.begin(),
                it->second.operations.end());
            unit_ids.insert(
                unit_ids.end(), it->second.unit_ids.begin(),
                it->second.unit_ids.end());
            stash_ids.insert(
                stash_ids.end(), it->second.stash_ids.begin(),
                it->second.stash_ids.end());
        }
        const auto digest_less = [](const auto & a, const auto & b) {
            return a.bytes() < b.bytes();
        };
        const auto digest_equal = [](const auto & a, const auto & b) {
            return a.bytes() == b.bytes();
        };
        std::sort(unit_ids.begin(), unit_ids.end(), digest_less);
        unit_ids.erase(
            std::unique(unit_ids.begin(), unit_ids.end(), digest_equal),
            unit_ids.end());
        std::sort(stash_ids.begin(), stash_ids.end(), digest_less);
        stash_ids.erase(
            std::unique(stash_ids.begin(), stash_ids.end(), digest_equal),
            stash_ids.end());
        auto release = llama_cache_prepare_release_set(
            impl_->ledger, operations, expected_serial);
        if (!release.ready()) {
            return false;
        }

        if (impl_->next_retire_token == 0 ||
            impl_->next_retire_token == UINT64_MAX) {
            return false;
        }
        const uint64_t token = impl_->next_retire_token++;
        auto state = std::unique_ptr<vbr_artifact_prepared_retire::impl>(
            new vbr_artifact_prepared_retire::impl);
        state->owner = this;
        state->token = token;
        state->references = std::move(canonical);
        state->unit_ids = std::move(unit_ids);
        state->stash_ids = std::move(stash_ids);
        state->release = std::move(release);
        for (const auto reference : state->references) {
            impl_->references.find(reference.v)->second.prepared_retire_token =
                token;
        }
        out.impl_ = std::move(state);
        return true;
    } catch (...) {
        out.reset();
        return false;
    }
}

bool llama_vbr_artifact_catalog::preview_owned_retire(
        const std::vector<llama_cache_acct_artifact_id> & references,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept {
    out = {};
    try {
        if (references.empty() || expected_serial == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        std::vector<llama_cache_acct_artifact_id> canonical = references;
        std::sort(canonical.begin(), canonical.end(),
            [](const auto & a, const auto & b) { return a.v < b.v; });
        if (canonical.front().v == 0 ||
            std::adjacent_find(
                canonical.begin(), canonical.end(),
                [](const auto & a, const auto & b) { return a == b; }) !=
                    canonical.end()) {
            return false;
        }

        std::vector<llama_cache_acct_op_id> operations;
        for (const auto reference : canonical) {
            const auto it = impl_->references.find(reference.v);
            if (it == impl_->references.end() || !it->second.host_owned ||
                it->second.retire_pending ||
                it->second.prepared_retire_token != 0 ||
                it->second.borrow_count != 1 ||
                it->second.operations.size() >
                    SIZE_MAX - operations.size()) {
                return false;
            }
            operations.insert(
                operations.end(), it->second.operations.begin(),
                it->second.operations.end());
        }
        return impl_->ledger.preview_release_set(
            operations, expected_serial, out);
    } catch (...) {
        out = {};
        return false;
    }
}

vbr_artifact_prepared_retire_status
llama_vbr_artifact_catalog::commit_owned_retire(
        uint64_t token,
        const std::vector<llama_cache_acct_artifact_id> & references,
        const std::vector<vbr_unit_version_id> & unit_ids,
        const std::vector<vbr_stash_payload_id> & stash_ids,
        llama_cache_prepared_release_set & release) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (token == 0 || references.empty() || !release.ready()) {
            return vbr_artifact_prepared_retire_status::unavailable;
        }
        for (const auto reference : references) {
            const auto it = impl_->references.find(reference.v);
            if (it == impl_->references.end() || !it->second.host_owned ||
                !it->second.retire_pending || it->second.borrow_count != 0 ||
                it->second.prepared_retire_token != token) {
                return vbr_artifact_prepared_retire_status::unavailable;
            }
        }
        const auto status = release.commit();
        if (status != llama_cache_conditional_release_status::released) {
            if (impl_->ledger.release_set_current(release.ops()) !=
                    llama_cache_conditional_release_status::released) {
                return vbr_artifact_prepared_retire_status::unavailable;
            }
            for (const auto reference : references) {
                impl_->references.erase(reference.v);
            }
            impl_->erase_orphan_storage(unit_ids, stash_ids);
            return vbr_artifact_prepared_retire_status::
                retired_projection_stale;
        }
        for (const auto reference : references) {
            impl_->references.erase(reference.v);
        }
        impl_->erase_orphan_storage(unit_ids, stash_ids);
        return vbr_artifact_prepared_retire_status::retired;
    } catch (...) {
        return vbr_artifact_prepared_retire_status::unavailable;
    }
}

void llama_vbr_artifact_catalog::cancel_owned_retire(
        uint64_t token,
        const std::vector<llama_cache_acct_artifact_id> & references,
        const std::vector<vbr_unit_version_id> & unit_ids,
        const std::vector<vbr_stash_payload_id> & stash_ids,
        llama_cache_prepared_release_set & release) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    bool all_gone = !references.empty();
    for (const auto reference : references) {
        const auto it = impl_->references.find(reference.v);
        if (it == impl_->references.end() ||
            it->second.prepared_retire_token != token) {
            all_gone = false;
            continue;
        }
        all_gone &= it->second.host_owned && it->second.retire_pending &&
                    it->second.borrow_count == 0;
    }
    if (!all_gone) {
        bool erased = false;
        for (const auto reference : references) {
            const auto it = impl_->references.find(reference.v);
            if (it != impl_->references.end() &&
                it->second.prepared_retire_token == token) {
                it->second.prepared_retire_token = 0;
                if (it->second.host_owned && it->second.retire_pending &&
                    it->second.borrow_count == 0) {
                    if (impl_->ledger.release_set_current(
                            it->second.operations) !=
                            llama_cache_conditional_release_status::released) {
                        return;
                    }
                    impl_->references.erase(it);
                    erased = true;
                }
            }
        }
        if (erased) {
            impl_->erase_orphan_storage(unit_ids, stash_ids);
        }
        return;
    }

    auto status = release.ready()
        ? release.commit()
        : llama_cache_conditional_release_status::serial_conflict;
    if (status != llama_cache_conditional_release_status::released) {
        // `release.ops()` remains the canonical sorted union after a failed
        // conditional commit, so retrying at the current serial needs no
        // allocation or per-operation ledger lock.
        if (impl_->ledger.release_set_current(release.ops()) !=
                llama_cache_conditional_release_status::released) {
            return;
        }
    }
    for (const auto reference : references) {
        impl_->references.erase(reference.v);
    }
    impl_->erase_orphan_storage(unit_ids, stash_ids);
}

void llama_vbr_artifact_catalog::release_reference_lease(
        llama_cache_acct_artifact_id reference,
        bool host_owned) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->references.find(reference.v);
    GGML_ASSERT(it != impl_->references.end() &&
                it->second.borrow_count > 0 &&
                (!host_owned || it->second.host_owned));
    --it->second.borrow_count;
    if (host_owned) {
        it->second.retire_pending = true;
    }
    if (it->second.borrow_count != 0 ||
        !it->second.retire_pending ||
        it->second.prepared_retire_token != 0) {
        return;
    }
    if (impl_->ledger.release_set_current(it->second.operations) !=
            llama_cache_conditional_release_status::released) {
        return;
    }
    auto unit_ids = std::move(it->second.unit_ids);
    auto stash_ids = std::move(it->second.stash_ids);
    impl_->references.erase(it);
    impl_->erase_orphan_storage(unit_ids, stash_ids);
}

bool llama_vbr_artifact_catalog::accounted_by(
        const llama_cache_acct_ledger * ledger) const noexcept {
    return ledger != nullptr && &impl_->ledger == ledger;
}

vbr_artifact_retire_status llama_vbr_artifact_catalog::retire(
        llama_cache_acct_artifact_id reference) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->references.find(reference.v);
        if (it == impl_->references.end()) {
            return vbr_artifact_retire_status::not_found;
        }
        if (it->second.borrow_count != 0 || it->second.host_owned ||
            it->second.retire_pending ||
            it->second.prepared_retire_token != 0) {
            return vbr_artifact_retire_status::busy;
        }
        auto prepared = llama_cache_prepare_release_set(
            impl_->ledger, it->second.operations, impl_->ledger.serial());
        if (!prepared.ready() || prepared.commit() !=
                llama_cache_conditional_release_status::released) {
            return vbr_artifact_retire_status::internal_error;
        }
        auto units = std::move(it->second.unit_ids);
        auto stashes = std::move(it->second.stash_ids);
        impl_->references.erase(it);
        impl_->erase_orphan_storage(units, stashes);
        return vbr_artifact_retire_status::retired;
    } catch (...) {
        return vbr_artifact_retire_status::internal_error;
    }
}

bool llama_vbr_artifact_catalog::reference_tokens(
        llama_cache_acct_artifact_id reference,
        llama_vbr_artifact_reference_tokens & out) const noexcept {
    out = {};
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->references.find(reference.v);
        if (it == impl_->references.end()) {
            return false;
        }
        out = {
            it->second.artifact,
            it->second.unit_content,
            it->second.lineage,
        };
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

llama_vbr_artifact_catalog_snapshot
llama_vbr_artifact_catalog::snapshot() const noexcept {
    llama_vbr_artifact_catalog_snapshot out;
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        out.blobs = impl_->blobs.size();
        out.stashes = impl_->stashes.size();
        out.references = impl_->references.size();
        out.published = impl_->n_published;
        out.adopted = impl_->n_adopted;
        out.refusals = impl_->n_refusals;
        out.staging_overlap_refusals =
            impl_->n_staging_overlap_refusals;
    } catch (...) {
        out = {};
    }
    return out;
}

const char * llama_vbr_artifact_publish_status_name(
        llama_vbr_artifact_publish_status status) noexcept {
    switch (status) {
        case llama_vbr_artifact_publish_status::published:              return "published";
        case llama_vbr_artifact_publish_status::adopted:                return "adopted";
        case llama_vbr_artifact_publish_status::invalid_argument:       return "invalid_argument";
        case llama_vbr_artifact_publish_status::shard_failed:           return "shard_failed";
        case llama_vbr_artifact_publish_status::duplicate_completion:   return "duplicate_completion";
        case llama_vbr_artifact_publish_status::missing_completion:     return "missing_completion";
        case llama_vbr_artifact_publish_status::format_rejected:        return "format_rejected";
        case llama_vbr_artifact_publish_status::accounting_unavailable: return "accounting_unavailable";
        case llama_vbr_artifact_publish_status::admission_refused:      return "admission_refused";
        case llama_vbr_artifact_publish_status::stage_failed:           return "stage_failed";
        case llama_vbr_artifact_publish_status::commit_failed:          return "commit_failed";
        case llama_vbr_artifact_publish_status::publication_failed:     return "publication_failed";
        case llama_vbr_artifact_publish_status::internal_error:         return "internal_error";
        case llama_vbr_artifact_publish_status::_count:                 break;
    }
    return "invalid";
}
