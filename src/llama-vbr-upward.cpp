#include "llama-vbr-upward.h"

#include "llama-sha256.h"
#include "llama-vbr-downward.h"

const char * vbr_upward_recipe_status_name(
        vbr_upward_recipe_status status) noexcept {
    switch (status) {
        case vbr_upward_recipe_status::resolved: return "resolved";
        case vbr_upward_recipe_status::equal_tier: return "equal_tier";
        case vbr_upward_recipe_status::cross_domain_unsupported:
            return "cross_domain_unsupported";
        case vbr_upward_recipe_status::tapped_domain_unsupported:
            return "tapped_domain_unsupported";
        case vbr_upward_recipe_status::unsupported_type:
            return "unsupported_type";
        case vbr_upward_recipe_status::invalid_argument:
            return "invalid_argument";
        case vbr_upward_recipe_status::_count: break;
    }
    return "invalid";
}

vbr_upward_recipe_status vbr_upward_resolve_recipe(
        ggml_type source_type,
        ggml_type target_type,
        vbr_upward_recipe & out) noexcept {
    out = {};
    out.source_type = source_type;
    out.target_type = target_type;
    if (source_type == GGML_TYPE_COUNT || target_type == GGML_TYPE_COUNT) {
        return vbr_upward_recipe_status::invalid_argument;
    }
    if (source_type == target_type) {
        return vbr_upward_recipe_status::equal_tier;
    }
    const auto source_domain = vbr_downward_tier_domain(source_type);
    const auto target_domain = vbr_downward_tier_domain(target_type);
    if (source_domain != target_domain) {
        return vbr_upward_recipe_status::cross_domain_unsupported;
    }
    if (source_domain == vbr_repr_domain::tapped) {
        vbr_downward_recipe canonical_order;
        const auto ordering = vbr_downward_resolve_recipe(
            source_type, target_type, GGML_TYPE_TURBO1_TCQ, true,
            canonical_order);
        if (ordering == vbr_downward_recipe_status::unsupported_type ||
            ordering == vbr_downward_recipe_status::invalid_argument) {
            return vbr_upward_recipe_status::unsupported_type;
        }
        if (ordering != vbr_downward_recipe_status::upward_forbidden) {
            return vbr_upward_recipe_status::tapped_domain_unsupported;
        }
    } else if (source_type != GGML_TYPE_TURBO8_0 ||
               target_type != GGML_TYPE_F16) {
        return vbr_upward_recipe_status::unsupported_type;
    }
    out.edges[0] = {
        source_type, target_type, source_domain, target_domain,
    };
    out.n_edges = 1;
    return vbr_upward_recipe_status::resolved;
}

std::array<uint8_t, 32> vbr_upward_build_identity(
        const vbr_upward_recipe & recipe,
        int32_t meansub_model_id,
        const std::array<uint8_t, 32> & meansub_digest,
        const std::array<uint8_t, 32> & policy_digest,
        const std::array<uint8_t, 32> & tree_digest) noexcept {
    try {
        if (meansub_model_id < 0 || recipe.version !=
                VBR_UPWARD_RECIPE_VERSION || recipe.n_edges != 1 ||
            recipe.n_edges > recipe.edges.size()) {
            return {};
        }
        vbr_upward_recipe resolved;
        if (vbr_upward_resolve_recipe(
                recipe.source_type, recipe.target_type, resolved) !=
                vbr_upward_recipe_status::resolved ||
            !(resolved == recipe)) {
            return {};
        }
        llama_sha256_writer writer;
        static constexpr char domain_label[] =
            "buun.vbr.upward/build-identity/v1";
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
        }
        writer.u32(uint32_t(meansub_model_id));
        writer.bytes(meansub_digest.data(), meansub_digest.size());
        writer.bytes(policy_digest.data(), policy_digest.size());
        writer.bytes(tree_digest.data(), tree_digest.size());
        return writer.finish();
    } catch (...) {
        return {};
    }
}
