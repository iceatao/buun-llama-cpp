#pragma once

#include "chat.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

constexpr uint32_t COMMON_RETENTION_SIDECAR_VERSION = 2;
constexpr uint32_t COMMON_RETENTION_TURN_TABLE_VERSION = 1;
// Stable ids are shifted left by one to form the pool-qualified durable key.
// Keep the ceiling beside the codec/allocator contract so the server cannot drift.
constexpr uint64_t COMMON_RETENTION_MAX_POOL_COUNTER =
    (UINT64_MAX >> 1) - 1;
constexpr uint64_t COMMON_RETENTION_FREQUENCY_ONE = uint64_t(1) << 20;
constexpr uint32_t COMMON_RETENTION_MAX_PRIOR_MILLI = 4000;
constexpr size_t COMMON_RETENTION_MAX_TURN_BOUNDARIES = 8192;

// Fidelity of the durable turn-table source.
enum class common_retention_source_state : uint8_t {
    known = 0,
    unavailable,
    _count,
};

enum class common_retention_pool : uint8_t {
    attention = 0,
    recurrent,
    _count,
};

enum class common_retention_artifact_kind : uint8_t {
    live_slot = 0,
    host_entry,
    checkpoint,
    _count,
};

// Availability of the derived retention score. This is intentionally distinct
// from source_state: future scoring policy may refuse an otherwise valid table.
enum class common_retention_score_state : uint8_t {
    known = 0,
    unavailable,
    _count,
};

enum class common_retention_frequency_state : uint8_t {
    unavailable = 0,
    probation,
    promoted,
    _count,
};

// Shadow-policy constants remain explicit inputs until DF1 trace selection
// freezes them for DF2. Arithmetic is integer-only and saturating.
struct common_retention_frequency_config {
    uint64_t credit_q = COMMON_RETENTION_FREQUENCY_ONE;
    uint64_t maximum_q = COMMON_RETENTION_FREQUENCY_ONE*16;
    uint64_t decay_interval_epochs = 8;
    uint32_t promotion_hits = 2;

    bool valid() const noexcept;
};

// One record belongs to one logical lineage, never to one artifact/alias.
// Artifact records carry lineage_id references; cloning cannot copy this
// record or multiply its value.
struct common_retention_lineage_record {
    common_retention_pool pool = common_retention_pool::attention;
    common_retention_frequency_state state =
        common_retention_frequency_state::probation;
    uint64_t lineage_id = 0;
    uint64_t reuse_hits = 0;
    uint64_t frequency_q = 0;
    uint64_t admission_epoch = 0;
    uint64_t frequency_epoch = 0;
    uint64_t last_credit_epoch = 0;
    uint32_t prior_milli = 1000;

    bool valid(uint64_t competition_epoch) const noexcept;
};

inline bool operator==(
        const common_retention_lineage_record & a,
        const common_retention_lineage_record & b) noexcept {
    return a.pool == b.pool &&
           a.state == b.state &&
           a.lineage_id == b.lineage_id &&
           a.reuse_hits == b.reuse_hits &&
           a.frequency_q == b.frequency_q &&
           a.admission_epoch == b.admission_epoch &&
           a.frequency_epoch == b.frequency_epoch &&
           a.last_credit_epoch == b.last_credit_epoch &&
           a.prior_milli == b.prior_milli;
}

inline bool operator!=(
        const common_retention_lineage_record & a,
        const common_retention_lineage_record & b) noexcept {
    return !(a == b);
}

enum class common_retention_credit_result : uint8_t {
    credited = 0,
    coalesced,
    unavailable,
    _count,
};

bool common_retention_frequency_normalize(
        common_retention_lineage_record & lineage,
        uint64_t competition_epoch,
        const common_retention_frequency_config & config) noexcept;

common_retention_credit_result common_retention_frequency_credit(
        common_retention_lineage_record & lineage,
        uint64_t competition_epoch,
        const common_retention_frequency_config & config) noexcept;

enum class common_retention_shadow_value_state : uint8_t {
    known = 0,
    unavailable,
    _count,
};

struct common_retention_shadow_value {
    common_retention_shadow_value_state state =
        common_retention_shadow_value_state::unavailable;
    common_retention_frequency_state frequency_state =
        common_retention_frequency_state::unavailable;
    uint64_t lineage_id = 0;
    uint64_t normalized_frequency_q = 0;
    uint64_t lost_value_q = 0;
    uint64_t marginal_resource = 0;
    uint64_t recency_ordinal = 0;
};

bool common_retention_shadow_quote(
        common_retention_lineage_record lineage,
        uint64_t competition_epoch,
        uint64_t avoided_work_units,
        uint64_t marginal_resource,
        uint64_t recency_ordinal,
        const common_retention_frequency_config & config,
        common_retention_shadow_value & out) noexcept;

// Negative means a is the preferred victim, positive means b, zero means the
// quotes are identical. This is a shadow-only DF1 comparator.
int common_retention_shadow_compare(
        const common_retention_shadow_value & a,
        const common_retention_shadow_value & b) noexcept;

struct common_retention_turn_boundary {
    uint64_t ordinal  = 0;
    uint64_t token_pos = 0;
    uint64_t token_end = 0;
};

struct common_retention_turn_table {
    uint32_t version = COMMON_RETENTION_TURN_TABLE_VERSION;
    common_retention_source_state source = common_retention_source_state::unavailable;
    uint64_t token_count = 0;
    std::vector<common_retention_turn_boundary> boundaries;

    bool valid() const noexcept;
};

struct common_retention_stamp {
    common_retention_score_state state = common_retention_score_state::unavailable;
    common_retention_pool pool = common_retention_pool::attention;
    bool soft_leased = false;
    // Serialized scoring output: consumers must not have to reimplement the
    // geometry policy to recover the non-evictable result.
    bool mandatory_anchor = false;
    uint64_t stable_id = 0;
    uint64_t lineage_id = 0;
    uint64_t recency_ordinal = 0;
    uint64_t mapped_turn_ordinal = 0;
    uint64_t anchor_rank = 0;
    uint64_t coverage_tokens = 0;

    bool valid() const noexcept;
};

struct common_retention_artifact_record {
    common_retention_artifact_kind kind = common_retention_artifact_kind::live_slot;
    // Immutable prefix geometry is shared across physical aliases and
    // checkpoints. A copied artifact record must not duplicate an 8K-entry
    // boundary table.
    std::shared_ptr<const common_retention_turn_table> turns;
    common_retention_stamp stamp;

    bool valid() const noexcept;
};

struct common_retention_sidecar_snapshot {
    uint32_t version = COMMON_RETENTION_SIDECAR_VERSION;
    std::array<uint64_t, size_t(common_retention_pool::_count)> recency_high_water = {};
    std::array<uint64_t, size_t(common_retention_pool::_count)> stable_high_water = {};
    std::array<uint64_t, size_t(common_retention_pool::_count)> lineage_high_water = {};
    uint64_t competition_epoch = 1;
    std::vector<common_retention_lineage_record> lineages;
    std::vector<common_retention_artifact_record> artifacts;

    bool valid() const noexcept;
};

bool common_retention_build_turn_table(
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t token_count,
        common_retention_turn_table & out) noexcept;

bool common_retention_score(
        const common_retention_turn_table & turns,
        uint64_t frontier,
        common_retention_stamp & stamp) noexcept;

bool common_retention_sidecar_encode(
        const common_retention_sidecar_snapshot & snapshot,
        std::vector<uint8_t> & out) noexcept;

// Exact one-record envelope size used by the in-memory catalog's C charge.
// This is arithmetic only: no payload allocation and no checksum computation.
bool common_retention_sidecar_artifact_encoded_size(
        const common_retention_artifact_record & artifact,
        uint64_t & out) noexcept;

bool common_retention_sidecar_decode(
        const uint8_t * data,
        size_t size,
        common_retention_sidecar_snapshot & out) noexcept;

class common_retention_allocator {
public:
    bool issue(common_retention_pool pool, common_retention_stamp & stamp) noexcept;
    bool issue_lineage(
        common_retention_pool pool,
        uint64_t competition_epoch,
        common_retention_lineage_record & lineage) noexcept;
    bool import_snapshot(const common_retention_sidecar_snapshot & snapshot) noexcept;
    uint64_t recency_high_water(common_retention_pool pool) const noexcept;
    uint64_t stable_high_water(common_retention_pool pool) const noexcept;
    uint64_t lineage_high_water(common_retention_pool pool) const noexcept;

private:
    std::array<uint64_t, size_t(common_retention_pool::_count)> next_recency = { 1, 1 };
    std::array<uint64_t, size_t(common_retention_pool::_count)> next_stable = { 1, 1 };
    std::array<uint64_t, size_t(common_retention_pool::_count)> next_lineage = { 1, 1 };
};
