#include "common-retention-sidecar.h"

#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <utility>

namespace {

constexpr uint32_t SIDECAR_MAGIC = 0x44533352; // "DS3R", canonical little-endian
constexpr size_t SIDECAR_HEADER_SIZE = 4 + 4 + 8 + 32;
constexpr size_t SIDECAR_SNAPSHOT_PREFIX_SIZE = 8*7 + 4*2;
constexpr size_t SIDECAR_LINEAGE_FIXED_SIZE = 2 + 8*6 + 4;
constexpr size_t SIDECAR_ARTIFACT_FIXED_SIZE = 6 + 4 + 8*7 + 4;
constexpr size_t SIDECAR_BOUNDARY_SIZE = 8*3;
constexpr uint64_t MAX_SIDECAR_BYTES = 64ull * 1024 * 1024;
constexpr uint32_t MAX_ARTIFACTS = 8192;
constexpr uint32_t MAX_LINEAGES = 8192;
constexpr uint32_t MAX_TURNS_PER_ARTIFACT =
    COMMON_RETENTION_MAX_TURN_BOUNDARIES;

void put_u8(std::vector<uint8_t> & out, uint8_t value) {
    out.push_back(value);
}

void put_u32(std::vector<uint8_t> & out, uint32_t value) {
    uint8_t data[4];
    llama_store_le_u32(data, value);
    out.insert(out.end(), std::begin(data), std::end(data));
}

void put_u64(std::vector<uint8_t> & out, uint64_t value) {
    uint8_t data[8];
    llama_store_le_u64(data, value);
    out.insert(out.end(), std::begin(data), std::end(data));
}

struct reader {
    const uint8_t * data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    bool bytes(void * dst, size_t n) {
        if (n > size - pos) {
            return false;
        }
        memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }

    bool u8(uint8_t & value) {
        return bytes(&value, sizeof(value));
    }

    bool u32(uint32_t & value) {
        uint8_t raw[4];
        if (!bytes(raw, sizeof(raw))) {
            return false;
        }
        value = 0;
        for (size_t i = 0; i < sizeof(raw); ++i) {
            value |= uint32_t(raw[i]) << (8*i);
        }
        return true;
    }

    bool u64(uint64_t & value) {
        uint8_t raw[8];
        if (!bytes(raw, sizeof(raw))) {
            return false;
        }
        value = 0;
        for (size_t i = 0; i < sizeof(raw); ++i) {
            value |= uint64_t(raw[i]) << (8*i);
        }
        return true;
    }
};

uint64_t ceil_log2(uint64_t value) {
    uint64_t result = 0;
    uint64_t power = 1;
    while (power < value) {
        if (power > std::numeric_limits<uint64_t>::max()/2) {
            return 64;
        }
        power <<= 1;
        result++;
    }
    return result;
}

struct uint128_product {
    uint64_t hi = 0;
    uint64_t lo = 0;
};

uint128_product multiply_u64(uint64_t a, uint64_t b) noexcept {
    const uint64_t a0 = uint32_t(a);
    const uint64_t a1 = a >> 32;
    const uint64_t b0 = uint32_t(b);
    const uint64_t b1 = b >> 32;
    const uint64_t p0 = a0*b0;
    const uint64_t p1 = a0*b1;
    const uint64_t p2 = a1*b0;
    const uint64_t p3 = a1*b1;
    const uint64_t middle =
        (p0 >> 32) + uint32_t(p1) + uint32_t(p2);
    return {
        p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32),
        (middle << 32) | uint32_t(p0),
    };
}

int compare_product(
        const uint128_product & a,
        const uint128_product & b) noexcept {
    if (a.hi != b.hi) {
        return a.hi < b.hi ? -1 : 1;
    }
    if (a.lo != b.lo) {
        return a.lo < b.lo ? -1 : 1;
    }
    return 0;
}

bool encode_payload(
        const common_retention_sidecar_snapshot & snapshot,
        std::vector<uint8_t> & payload) {
    put_u64(payload, snapshot.recency_high_water[0]);
    put_u64(payload, snapshot.recency_high_water[1]);
    put_u64(payload, snapshot.stable_high_water[0]);
    put_u64(payload, snapshot.stable_high_water[1]);
    put_u64(payload, snapshot.lineage_high_water[0]);
    put_u64(payload, snapshot.lineage_high_water[1]);
    put_u64(payload, snapshot.competition_epoch);
    put_u32(payload, uint32_t(snapshot.lineages.size()));
    for (const auto & lineage : snapshot.lineages) {
        put_u8(payload, uint8_t(lineage.pool));
        put_u8(payload, uint8_t(lineage.state));
        put_u64(payload, lineage.lineage_id);
        put_u64(payload, lineage.reuse_hits);
        put_u64(payload, lineage.frequency_q);
        put_u64(payload, lineage.admission_epoch);
        put_u64(payload, lineage.frequency_epoch);
        put_u64(payload, lineage.last_credit_epoch);
        put_u32(payload, lineage.prior_milli);
    }
    put_u32(payload, uint32_t(snapshot.artifacts.size()));

    for (const auto & artifact : snapshot.artifacts) {
        if (!artifact.turns) {
            return false;
        }
        const auto & turns = *artifact.turns;
        put_u8(payload, uint8_t(artifact.kind));
        put_u8(payload, uint8_t(turns.source));
        put_u8(payload, uint8_t(artifact.stamp.state));
        put_u8(payload, uint8_t(artifact.stamp.pool));
        put_u8(payload, artifact.stamp.soft_leased ? 1 : 0);
        put_u8(payload, artifact.stamp.mandatory_anchor ? 1 : 0);
        put_u32(payload, turns.version);
        put_u64(payload, turns.token_count);
        put_u64(payload, artifact.stamp.stable_id);
        put_u64(payload, artifact.stamp.lineage_id);
        put_u64(payload, artifact.stamp.recency_ordinal);
        put_u64(payload, artifact.stamp.mapped_turn_ordinal);
        put_u64(payload, artifact.stamp.anchor_rank);
        put_u64(payload, artifact.stamp.coverage_tokens);
        put_u32(payload, uint32_t(turns.boundaries.size()));
        for (const auto & boundary : turns.boundaries) {
            put_u64(payload, boundary.ordinal);
            put_u64(payload, boundary.token_pos);
            put_u64(payload, boundary.token_end);
        }
    }
    return payload.size() <= MAX_SIDECAR_BYTES - SIDECAR_HEADER_SIZE;
}

bool decode_lineage(
        reader & in,
        uint64_t competition_epoch,
        common_retention_lineage_record & lineage) {
    uint8_t pool;
    uint8_t state;
    if (!in.u8(pool) ||
        !in.u8(state) ||
        !in.u64(lineage.lineage_id) ||
        !in.u64(lineage.reuse_hits) ||
        !in.u64(lineage.frequency_q) ||
        !in.u64(lineage.admission_epoch) ||
        !in.u64(lineage.frequency_epoch) ||
        !in.u64(lineage.last_credit_epoch) ||
        !in.u32(lineage.prior_milli) ||
        pool >= uint8_t(common_retention_pool::_count) ||
        state >= uint8_t(common_retention_frequency_state::_count)) {
        return false;
    }
    lineage.pool = common_retention_pool(pool);
    lineage.state = common_retention_frequency_state(state);
    return lineage.valid(competition_epoch);
}

bool decode_artifact(reader & in, common_retention_artifact_record & artifact) {
    auto turns = std::make_shared<common_retention_turn_table>();
    uint8_t kind;
    uint8_t source;
    uint8_t score_state;
    uint8_t pool;
    uint8_t soft;
    uint8_t mandatory;
    uint32_t n_turns;
    if (!in.u8(kind) ||
        !in.u8(source) ||
        !in.u8(score_state) ||
        !in.u8(pool) ||
        !in.u8(soft) ||
        !in.u8(mandatory) ||
        !in.u32(turns->version) ||
        !in.u64(turns->token_count) ||
        !in.u64(artifact.stamp.stable_id) ||
        !in.u64(artifact.stamp.lineage_id) ||
        !in.u64(artifact.stamp.recency_ordinal) ||
        !in.u64(artifact.stamp.mapped_turn_ordinal) ||
        !in.u64(artifact.stamp.anchor_rank) ||
        !in.u64(artifact.stamp.coverage_tokens) ||
        !in.u32(n_turns)) {
        return false;
    }
    if (kind >= uint8_t(common_retention_artifact_kind::_count) ||
        source >= uint8_t(common_retention_source_state::_count) ||
        score_state >= uint8_t(common_retention_score_state::_count) ||
        pool >= uint8_t(common_retention_pool::_count) ||
        soft > 1 || mandatory > 1 ||
        n_turns > MAX_TURNS_PER_ARTIFACT) {
        return false;
    }
    artifact.kind = common_retention_artifact_kind(kind);
    turns->source = common_retention_source_state(source);
    artifact.stamp.state = common_retention_score_state(score_state);
    artifact.stamp.pool = common_retention_pool(pool);
    artifact.stamp.soft_leased = soft != 0;
    artifact.stamp.mandatory_anchor = mandatory != 0;
    turns->boundaries.resize(n_turns);
    for (auto & boundary : turns->boundaries) {
        if (!in.u64(boundary.ordinal) ||
            !in.u64(boundary.token_pos) ||
            !in.u64(boundary.token_end)) {
            return false;
        }
    }
    artifact.turns = std::move(turns);
    return artifact.valid();
}

} // namespace

bool common_retention_frequency_config::valid() const noexcept {
    return credit_q != 0 &&
           maximum_q >= credit_q &&
           decay_interval_epochs != 0 &&
           promotion_hits >= 2;
}

bool common_retention_lineage_record::valid(
        uint64_t competition_epoch) const noexcept {
    if (pool >= common_retention_pool::_count ||
        state >= common_retention_frequency_state::_count ||
        lineage_id == 0 ||
        lineage_id > COMMON_RETENTION_MAX_POOL_COUNTER ||
        prior_milli == 0 ||
        prior_milli > COMMON_RETENTION_MAX_PRIOR_MILLI ||
        admission_epoch == 0 ||
        admission_epoch > competition_epoch ||
        frequency_epoch < admission_epoch ||
        frequency_epoch > competition_epoch ||
        (last_credit_epoch != 0 &&
         last_credit_epoch < admission_epoch) ||
        last_credit_epoch > competition_epoch ||
        ((reuse_hits == 0) != (last_credit_epoch == 0))) {
        return false;
    }
    if (state == common_retention_frequency_state::unavailable) {
        return reuse_hits == 0 && frequency_q == 0 &&
               last_credit_epoch == 0;
    }
    return state != common_retention_frequency_state::promoted ||
           reuse_hits >= 2;
}

bool common_retention_frequency_normalize(
        common_retention_lineage_record & lineage,
        uint64_t competition_epoch,
        const common_retention_frequency_config & config) noexcept {
    if (!config.valid() || !lineage.valid(competition_epoch)) {
        return false;
    }
    if (lineage.state == common_retention_frequency_state::unavailable) {
        lineage.frequency_epoch = competition_epoch;
        return false;
    }
    const uint64_t elapsed = competition_epoch - lineage.frequency_epoch;
    const uint64_t shifts = elapsed/config.decay_interval_epochs;
    lineage.frequency_q = shifts >= 64 ? 0 : lineage.frequency_q >> shifts;
    // Preserve the incomplete interval. Advancing all the way to the query
    // epoch when shifts == 0 would let a frequently inspected candidate avoid
    // decay forever. The cursor denotes the last applied decay boundary, not
    // the last observation.
    lineage.frequency_epoch += shifts*config.decay_interval_epochs;
    return true;
}

common_retention_credit_result common_retention_frequency_credit(
        common_retention_lineage_record & lineage,
        uint64_t competition_epoch,
        const common_retention_frequency_config & config) noexcept {
    if (!common_retention_frequency_normalize(
            lineage, competition_epoch, config) ||
        lineage.state == common_retention_frequency_state::unavailable) {
        return common_retention_credit_result::unavailable;
    }
    if (lineage.last_credit_epoch == competition_epoch) {
        return common_retention_credit_result::coalesced;
    }
    lineage.frequency_q = lineage.frequency_q >
            config.maximum_q - std::min(config.credit_q, config.maximum_q)
        ? config.maximum_q
        : lineage.frequency_q + config.credit_q;
    if (lineage.reuse_hits != UINT64_MAX) {
        lineage.reuse_hits++;
    }
    lineage.last_credit_epoch = competition_epoch;
    if (lineage.reuse_hits >= config.promotion_hits) {
        lineage.state = common_retention_frequency_state::promoted;
    }
    return common_retention_credit_result::credited;
}

bool common_retention_shadow_quote(
        common_retention_lineage_record lineage,
        uint64_t competition_epoch,
        uint64_t avoided_work_units,
        uint64_t marginal_resource,
        uint64_t recency_ordinal,
        const common_retention_frequency_config & config,
        common_retention_shadow_value & out) noexcept {
    out = {};
    if (marginal_resource == 0 || recency_ordinal == 0 ||
        !common_retention_frequency_normalize(
            lineage, competition_epoch, config) ||
        lineage.state == common_retention_frequency_state::unavailable) {
        return false;
    }
    const uint64_t prior_age = competition_epoch - lineage.admission_epoch;
    const uint64_t prior_shifts =
        prior_age/config.decay_interval_epochs;
    const uint64_t raw_prior_q =
        COMMON_RETENTION_FREQUENCY_ONE*lineage.prior_milli/1000;
    const uint64_t prior_q = prior_shifts >= 64
        ? 0 : raw_prior_q >> prior_shifts;
    if (lineage.frequency_q > UINT64_MAX - prior_q) {
        return false;
    }
    const uint64_t signal_q = lineage.frequency_q + prior_q;
    if (avoided_work_units != 0 &&
        signal_q > UINT64_MAX/avoided_work_units) {
        return false;
    }
    out.state = common_retention_shadow_value_state::known;
    out.frequency_state = lineage.state;
    out.lineage_id = lineage.lineage_id;
    out.normalized_frequency_q = lineage.frequency_q;
    out.lost_value_q = signal_q*avoided_work_units;
    out.marginal_resource = marginal_resource;
    out.recency_ordinal = recency_ordinal;
    return true;
}

int common_retention_shadow_compare(
        const common_retention_shadow_value & a,
        const common_retention_shadow_value & b) noexcept {
    const bool a_known =
        a.state == common_retention_shadow_value_state::known &&
        a.marginal_resource != 0;
    const bool b_known =
        b.state == common_retention_shadow_value_state::known &&
        b.marginal_resource != 0;
    if (a_known != b_known) {
        return a_known ? -1 : 1;
    }
    if (!a_known) {
        return 0;
    }
    const int density = compare_product(
        multiply_u64(a.lost_value_q, b.marginal_resource),
        multiply_u64(b.lost_value_q, a.marginal_resource));
    if (density != 0) {
        return density;
    }
    if (a.recency_ordinal != b.recency_ordinal) {
        return a.recency_ordinal < b.recency_ordinal ? -1 : 1;
    }
    if (a.lineage_id != b.lineage_id) {
        return a.lineage_id < b.lineage_id ? -1 : 1;
    }
    return 0;
}

bool common_retention_turn_table::valid() const noexcept {
    if (version != COMMON_RETENTION_TURN_TABLE_VERSION ||
        source >= common_retention_source_state::_count) {
        return false;
    }
    if (source == common_retention_source_state::unavailable) {
        return boundaries.empty();
    }
    if (boundaries.empty() ||
        boundaries.front().ordinal != 0 ||
        boundaries.front().token_pos != 0) {
        return false;
    }
    for (size_t i = 0; i < boundaries.size(); ++i) {
        const auto & cur = boundaries[i];
        if (cur.ordinal != i ||
            cur.token_pos > cur.token_end ||
            cur.token_end > token_count) {
            return false;
        }
        if (i > 0 && boundaries[i - 1].token_pos >= cur.token_pos) {
            return false;
        }
    }
    return true;
}

bool common_retention_stamp::valid() const noexcept {
    if (state >= common_retention_score_state::_count ||
        pool >= common_retention_pool::_count ||
        stable_id == 0 ||
        lineage_id == 0 ||
        recency_ordinal == 0 ||
        stable_id > COMMON_RETENTION_MAX_POOL_COUNTER ||
        lineage_id > COMMON_RETENTION_MAX_POOL_COUNTER ||
        recency_ordinal > COMMON_RETENTION_MAX_POOL_COUNTER) {
        return false;
    }
    if (state == common_retention_score_state::unavailable) {
        return !mandatory_anchor && mapped_turn_ordinal == 0 && anchor_rank == 0;
    }
    return true;
}

bool common_retention_artifact_record::valid() const noexcept {
    if (kind >= common_retention_artifact_kind::_count ||
        !turns || !turns->valid() ||
        !stamp.valid()) {
        return false;
    }
    return stamp.state == common_retention_score_state::unavailable ||
           (turns->source == common_retention_source_state::known &&
            stamp.coverage_tokens <= turns->token_count &&
            stamp.mapped_turn_ordinal < turns->boundaries.size());
}

bool common_retention_sidecar_snapshot::valid() const noexcept {
    if (version != COMMON_RETENTION_SIDECAR_VERSION ||
        competition_epoch == 0 ||
        lineages.size() > MAX_LINEAGES ||
        artifacts.size() > MAX_ARTIFACTS) {
        return false;
    }
    std::array<uint64_t, size_t(common_retention_pool::_count)> max_recency = {};
    std::array<uint64_t, size_t(common_retention_pool::_count)> max_stable = {};
    std::array<uint64_t, size_t(common_retention_pool::_count)> max_lineage = {};
    std::array<std::vector<uint64_t>, size_t(common_retention_pool::_count)> ids;
    std::array<std::vector<uint64_t>, size_t(common_retention_pool::_count)> lineage_ids;
    for (const auto & lineage : lineages) {
        if (!lineage.valid(competition_epoch)) {
            return false;
        }
        const size_t pool = size_t(lineage.pool);
        max_lineage[pool] = std::max(max_lineage[pool], lineage.lineage_id);
        lineage_ids[pool].push_back(lineage.lineage_id);
    }
    for (auto & pool_ids : lineage_ids) {
        std::sort(pool_ids.begin(), pool_ids.end());
        if (std::adjacent_find(pool_ids.begin(), pool_ids.end()) !=
                pool_ids.end()) {
            return false;
        }
    }
    for (const auto & artifact : artifacts) {
        if (!artifact.valid()) {
            return false;
        }
        const size_t pool = size_t(artifact.stamp.pool);
        if (!std::binary_search(
                lineage_ids[pool].begin(), lineage_ids[pool].end(),
                artifact.stamp.lineage_id)) {
            return false;
        }
        max_recency[pool] = std::max(max_recency[pool], artifact.stamp.recency_ordinal);
        max_stable[pool] = std::max(max_stable[pool], artifact.stamp.stable_id);
        ids[pool].push_back(artifact.stamp.stable_id);
    }
    for (size_t pool = 0; pool < max_recency.size(); ++pool) {
        if (recency_high_water[pool] < max_recency[pool] ||
            stable_high_water[pool] < max_stable[pool] ||
            lineage_high_water[pool] < max_lineage[pool] ||
            recency_high_water[pool] > COMMON_RETENTION_MAX_POOL_COUNTER ||
            stable_high_water[pool] > COMMON_RETENTION_MAX_POOL_COUNTER ||
            lineage_high_water[pool] > COMMON_RETENTION_MAX_POOL_COUNTER) {
            return false;
        }
        std::sort(ids[pool].begin(), ids[pool].end());
        if (std::adjacent_find(ids[pool].begin(), ids[pool].end()) != ids[pool].end()) {
            return false;
        }
    }
    return true;
}

bool common_retention_build_turn_table(
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t token_count,
        common_retention_turn_table & out) noexcept {
    out = {};
    try {
        common_retention_turn_table built;
        built.token_count = token_count;
        if (!source_known || spans.spans.empty()) {
            out = std::move(built);
            return true;
        }

        uint64_t prior_end = 0;
        size_t n_boundaries = 0;
        for (const auto & span : spans.spans) {
            if (!span.valid() ||
                span.pos > token_count ||
                span.len > token_count - span.pos ||
                span.pos < prior_end) {
                return false;
            }
            const uint64_t end = uint64_t(span.pos) + uint64_t(span.len);
            prior_end = end;
            if (span.role == COMMON_CHAT_ROLE_USER) {
                if (n_boundaries == 0) {
                    n_boundaries++;
                }
                if (span.pos != 0) {
                    n_boundaries++;
                }
                if (n_boundaries >
                        COMMON_RETENTION_MAX_TURN_BOUNDARIES) {
                    return false;
                }
            }
        }
        if (n_boundaries == 0) {
            out = std::move(built);
            return true;
        }

        built.source = common_retention_source_state::known;
        built.boundaries.reserve(n_boundaries);
        built.boundaries.push_back({ 0, 0, 0 });
        for (const auto & span : spans.spans) {
            if (span.role != COMMON_CHAT_ROLE_USER) {
                continue;
            }
            const uint64_t end = uint64_t(span.pos) + uint64_t(span.len);
            if (span.pos == 0) {
                built.boundaries.front().token_end = end;
                continue;
            }
            if (built.boundaries.back().token_pos >= span.pos) {
                return false;
            }
            built.boundaries.push_back({
                uint64_t(built.boundaries.size()),
                uint64_t(span.pos),
                end,
            });
        }
        if (!built.valid()) {
            return false;
        }
        out = std::move(built);
        return true;
    } catch (...) {
        return false;
    }
}

bool common_retention_score(
        const common_retention_turn_table & turns,
        uint64_t frontier,
        common_retention_stamp & stamp) noexcept {
    stamp.state = common_retention_score_state::unavailable;
    stamp.mandatory_anchor = false;
    stamp.mapped_turn_ordinal = 0;
    stamp.anchor_rank = 0;
    if (!turns.valid() ||
        turns.source != common_retention_source_state::known ||
        frontier > turns.token_count) {
        return false;
    }

    const auto upper = std::upper_bound(
        turns.boundaries.begin(), turns.boundaries.end(), frontier,
        [](uint64_t pos, const common_retention_turn_boundary & boundary) {
            return pos < boundary.token_pos;
        });
    const size_t mapped =
        size_t(std::distance(turns.boundaries.begin(), upper) - 1);
    const uint64_t n = turns.boundaries.size() - 1;
    stamp.state = common_retention_score_state::known;
    stamp.mapped_turn_ordinal = mapped;
    stamp.mandatory_anchor = mapped == 0 || mapped == n;
    if (stamp.mandatory_anchor) {
        return true;
    }

    const uint64_t k_max = ceil_log2(std::max<uint64_t>(n, 1));
    for (uint64_t k = 0; k <= k_max && k < 64; ++k) {
        const uint64_t distance = uint64_t(1) << k;
        const uint64_t index = distance >= n ? 0 : n - distance;
        if (index == mapped) {
            stamp.anchor_rank = k_max + 1 - k;
            break;
        }
        if (index == 0) {
            break;
        }
    }
    return true;
}

bool common_retention_sidecar_encode(
        const common_retention_sidecar_snapshot & snapshot,
        std::vector<uint8_t> & out) noexcept {
    try {
        if (!snapshot.valid()) {
            return false;
        }
        std::vector<uint8_t> payload;
        if (!encode_payload(snapshot, payload)) {
            return false;
        }
        llama_sha256 hash;
        hash.update(payload.data(), payload.size());
        const auto digest = hash.finish();

        std::vector<uint8_t> encoded;
        encoded.reserve(SIDECAR_HEADER_SIZE + payload.size());
        put_u32(encoded, SIDECAR_MAGIC);
        put_u32(encoded, COMMON_RETENTION_SIDECAR_VERSION);
        put_u64(encoded, SIDECAR_HEADER_SIZE + payload.size());
        encoded.insert(encoded.end(), digest.begin(), digest.end());
        encoded.insert(encoded.end(), payload.begin(), payload.end());
        out = std::move(encoded);
        return true;
    } catch (...) {
        return false;
    }
}

bool common_retention_sidecar_artifact_encoded_size(
        const common_retention_artifact_record & artifact,
        uint64_t & out) noexcept {
    out = 0;
    if (!artifact.valid() ||
        artifact.turns->boundaries.size() > MAX_TURNS_PER_ARTIFACT) {
        return false;
    }
    const uint64_t n_boundaries = artifact.turns->boundaries.size();
    const uint64_t fixed =
        SIDECAR_HEADER_SIZE +
        SIDECAR_SNAPSHOT_PREFIX_SIZE +
        SIDECAR_LINEAGE_FIXED_SIZE +
        SIDECAR_ARTIFACT_FIXED_SIZE;
    if (n_boundaries >
        (MAX_SIDECAR_BYTES - fixed)/SIDECAR_BOUNDARY_SIZE) {
        return false;
    }
    out = fixed + n_boundaries*SIDECAR_BOUNDARY_SIZE;
    return out <= MAX_SIDECAR_BYTES;
}

bool common_retention_sidecar_decode(
        const uint8_t * data,
        size_t size,
        common_retention_sidecar_snapshot & out) noexcept {
    // Fail closed even when the caller reuses an object that previously held valid
    // evidence: no decode failure may leave that prior record observable.
    out.version = 0;
    out.recency_high_water = {};
    out.stable_high_water = {};
    out.lineage_high_water = {};
    out.competition_epoch = 0;
    out.lineages.clear();
    out.artifacts.clear();
    try {
        if (!data || size < SIDECAR_HEADER_SIZE || size > MAX_SIDECAR_BYTES) {
            return false;
        }
        reader header { data, size, 0 };
        uint32_t magic;
        uint32_t version;
        uint64_t declared_size;
        std::array<uint8_t, 32> expected;
        if (!header.u32(magic) ||
            !header.u32(version) ||
            !header.u64(declared_size) ||
            !header.bytes(expected.data(), expected.size()) ||
            magic != SIDECAR_MAGIC ||
            version != COMMON_RETENTION_SIDECAR_VERSION ||
            declared_size != size) {
            return false;
        }

        llama_sha256 hash;
        hash.update(data + SIDECAR_HEADER_SIZE, size - SIDECAR_HEADER_SIZE);
        if (hash.finish() != expected) {
            return false;
        }

        reader payload {
            data + SIDECAR_HEADER_SIZE,
            size - SIDECAR_HEADER_SIZE,
            0,
        };
        common_retention_sidecar_snapshot decoded;
        uint32_t n_lineages;
        uint32_t n_artifacts;
        if (!payload.u64(decoded.recency_high_water[0]) ||
            !payload.u64(decoded.recency_high_water[1]) ||
            !payload.u64(decoded.stable_high_water[0]) ||
            !payload.u64(decoded.stable_high_water[1]) ||
            !payload.u64(decoded.lineage_high_water[0]) ||
            !payload.u64(decoded.lineage_high_water[1]) ||
            !payload.u64(decoded.competition_epoch) ||
            !payload.u32(n_lineages) ||
            n_lineages > MAX_LINEAGES) {
            return false;
        }
        decoded.lineages.resize(n_lineages);
        for (auto & lineage : decoded.lineages) {
            if (!decode_lineage(
                    payload, decoded.competition_epoch, lineage)) {
                return false;
            }
        }
        if (!payload.u32(n_artifacts) ||
            n_artifacts > MAX_ARTIFACTS) {
            return false;
        }
        decoded.artifacts.resize(n_artifacts);
        for (auto & artifact : decoded.artifacts) {
            if (!decode_artifact(payload, artifact)) {
                return false;
            }
        }
        if (payload.pos != payload.size || !decoded.valid()) {
            return false;
        }
        out = std::move(decoded);
        return true;
    } catch (...) {
        return false;
    }
}

bool common_retention_allocator::issue(
        common_retention_pool pool,
        common_retention_stamp & stamp) noexcept {
    const size_t index = size_t(pool);
    if (index >= next_recency.size() ||
        next_recency[index] == 0 ||
        next_stable[index] == 0 ||
        next_recency[index] > COMMON_RETENTION_MAX_POOL_COUNTER ||
        next_stable[index] > COMMON_RETENTION_MAX_POOL_COUNTER) {
        return false;
    }
    stamp.pool = pool;
    stamp.recency_ordinal = next_recency[index]++;
    stamp.stable_id = next_stable[index]++;
    return true;
}

bool common_retention_allocator::issue_lineage(
        common_retention_pool pool,
        uint64_t competition_epoch,
        common_retention_lineage_record & lineage) noexcept {
    const size_t index = size_t(pool);
    if (index >= next_lineage.size() || competition_epoch == 0 ||
        next_lineage[index] == 0 ||
        next_lineage[index] > COMMON_RETENTION_MAX_POOL_COUNTER) {
        return false;
    }
    lineage = {};
    lineage.pool = pool;
    lineage.lineage_id = next_lineage[index]++;
    lineage.admission_epoch = competition_epoch;
    lineage.frequency_epoch = competition_epoch;
    return lineage.valid(competition_epoch);
}

bool common_retention_allocator::import_snapshot(
        const common_retention_sidecar_snapshot & imported) noexcept {
    if (!imported.valid()) {
        return false;
    }
    for (size_t i = 0; i < next_recency.size(); ++i) {
        if (imported.recency_high_water[i] >=
                COMMON_RETENTION_MAX_POOL_COUNTER ||
            imported.stable_high_water[i] >=
                COMMON_RETENTION_MAX_POOL_COUNTER ||
            imported.lineage_high_water[i] >=
                COMMON_RETENTION_MAX_POOL_COUNTER) {
            return false;
        }
    }
    for (size_t i = 0; i < next_recency.size(); ++i) {
        next_recency[i] =
            std::max(next_recency[i], imported.recency_high_water[i] + 1);
        next_stable[i] =
            std::max(next_stable[i], imported.stable_high_water[i] + 1);
        next_lineage[i] =
            std::max(next_lineage[i], imported.lineage_high_water[i] + 1);
    }
    return true;
}

uint64_t common_retention_allocator::recency_high_water(
        common_retention_pool pool) const noexcept {
    const size_t index = size_t(pool);
    return index < next_recency.size() ? next_recency[index] - 1 : 0;
}

uint64_t common_retention_allocator::stable_high_water(
        common_retention_pool pool) const noexcept {
    const size_t index = size_t(pool);
    return index < next_stable.size() ? next_stable[index] - 1 : 0;
}

uint64_t common_retention_allocator::lineage_high_water(
        common_retention_pool pool) const noexcept {
    const size_t index = size_t(pool);
    return index < next_lineage.size() ? next_lineage[index] - 1 : 0;
}
