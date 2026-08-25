#pragma once

#include "llama-sha256.h"
#include "llama-vbr-checkpoint-types.h"
#include "llama-vbr-generation-types.h"

#include "ggml.h"

#include <array>
#include <vector>

// Canonical checkpoint/artifact identity-policy row. Keeping the hash recipe
// here makes the checkpoint bridge and explicit capture share one
// source; neither the server nor a codec may invent a weaker ordering digest.
struct vbr_identity_policy_digest_row {
    uint32_t child_id = 0;
    checkpoint_child_dependency_mode mode =
        checkpoint_child_dependency_mode::absent;
    vbr_lineage_uuid lineage_uuid = {};
};

inline bool vbr_digest_nonzero(const std::array<uint8_t, 32> & digest) {
    for (const uint8_t value : digest) {
        if (value != 0) {
            return true;
        }
    }
    return false;
}

// Canonical capture/controller type-vector identity. One shared recipe keeps
// capture, downward validation, and late rechecks from drifting.
inline std::array<uint8_t, 32> vbr_type_vector_digest(
        const ggml_type * types, size_t count) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] = "buun.vbr.capture/type-vector";
    writer.string(domain_label, sizeof(domain_label) - 1);
    for (size_t i = 0; i < count; ++i) {
        writer.u32(static_cast<uint32_t>(types[i]));
    }
    return writer.finish();
}

inline std::array<uint8_t, 32> vbr_type_vector_digest(
        const std::vector<ggml_type> & types) {
    return vbr_type_vector_digest(types.data(), types.size());
}

inline std::array<uint8_t, 32> vbr_type_tree_digest(
        const std::vector<std::array<uint8_t, 32>> & children,
        uint32_t recipe_version) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] =
        "buun.vbr.downward/tree-policy";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u32(recipe_version);
    writer.u64(children.size());
    for (const auto & digest : children) {
        writer.bytes(digest.data(), digest.size());
    }
    return writer.finish();
}

inline std::array<uint8_t, 32> vbr_identity_policy_digest(
        const vbr_checkpoint_frontier_fields & frontier,
        const std::vector<vbr_identity_policy_digest_row> & policy) {
    llama_sha256_writer hash;
    static const char domain[] =
        "vbr checkpoint identity/policy/order digest v1";
    hash.bytes(domain, sizeof(domain) - 1);
    hash.string(
        frontier.execution_identity,
        frontier.execution_identity_len);
    hash.string(
        frontier.adapter_config_identity,
        frontier.adapter_config_identity_len);
    hash.string(
        frontier.media_content_identity,
        frontier.media_content_identity_len);
    hash.u64(frontier.sequence_epoch);
    hash.u64(uint64_t(frontier.token_count));
    hash.u64(uint64_t(int64_t(frontier.next_position)));
    hash.u64(policy.size());
    for (const auto & row : policy) {
        hash.u64(row.child_id);
        hash.u64(uint64_t(static_cast<uint8_t>(row.mode)));
        hash.u64(row.lineage_uuid.hi);
        hash.u64(row.lineage_uuid.lo);
    }
    return hash.finish();
}
