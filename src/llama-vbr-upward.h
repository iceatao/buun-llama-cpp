#pragma once

#include "llama-vbr-generation-types.h"

#include "ggml-vbr.h"

#include <array>
#include <cstddef>
#include <cstdint>

constexpr uint32_t VBR_UPWARD_RECIPE_VERSION = 1;
constexpr uint32_t VBR_UPWARD_RECIPE_ID = 1;

enum class vbr_upward_recipe_status : uint8_t {
    resolved = 0,
    equal_tier,
    cross_domain_unsupported,
    tapped_domain_unsupported,
    unsupported_type,
    invalid_argument,
    _count,
};

const char * vbr_upward_recipe_status_name(
    vbr_upward_recipe_status status) noexcept;

struct vbr_upward_edge {
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type target_type = GGML_TYPE_COUNT;
    vbr_repr_domain source_domain = vbr_repr_domain::full;
    vbr_repr_domain target_domain = vbr_repr_domain::full;

    bool operator==(const vbr_upward_edge & other) const noexcept {
        return source_type == other.source_type &&
               target_type == other.target_type &&
               source_domain == other.source_domain &&
               target_domain == other.target_domain;
    }
};

// Upward reconstruction is deliberately one direct edge. Keeping the recipe
// explicit prevents schedule classification from being mistaken for backend
// authority, avoids compounding error through intermediate tiers, and leaves
// cross-domain reconstruction fail-closed.
struct vbr_upward_recipe {
    uint32_t version = VBR_UPWARD_RECIPE_VERSION;
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type target_type = GGML_TYPE_COUNT;
    std::array<vbr_upward_edge, 1> edges = {};
    size_t n_edges = 0;

    bool operator==(const vbr_upward_recipe & other) const noexcept {
        return version == other.version &&
               source_type == other.source_type &&
               target_type == other.target_type &&
               n_edges == other.n_edges &&
               n_edges <= edges.size() &&
               (n_edges == 0 || edges[0] == other.edges[0]);
    }
};

vbr_upward_recipe_status vbr_upward_resolve_recipe(
    ggml_type source_type,
    ggml_type target_type,
    vbr_upward_recipe & out) noexcept;

std::array<uint8_t, 32> vbr_upward_build_identity(
    const vbr_upward_recipe & recipe,
    int32_t meansub_model_id,
    const std::array<uint8_t, 32> & meansub_digest,
    const std::array<uint8_t, 32> & policy_digest,
    const std::array<uint8_t, 32> & tree_digest) noexcept;
