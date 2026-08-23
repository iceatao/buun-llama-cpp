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
        vbr_downward_recipe tapped_membership;
        const auto tapped_status = vbr_downward_resolve_recipe(
            GGML_TYPE_TURBO4_0, source_type,
            GGML_TYPE_TURBO1_TCQ, true, tapped_membership);
        const bool canonical_tapped_source =
            tapped_status == vbr_downward_recipe_status::equal_tier ||
            tapped_status == vbr_downward_recipe_status::resolved;
        const bool tapped_to_full =
            canonical_tapped_source &&
            source_domain == vbr_repr_domain::tapped &&
            target_domain == vbr_repr_domain::full &&
            (target_type == GGML_TYPE_TURBO8_0 ||
             target_type == GGML_TYPE_F16);
        if (!tapped_to_full) {
            return vbr_upward_recipe_status::cross_domain_unsupported;
        }
        out.edges[0] = {
            source_type, target_type, source_domain, target_domain,
            vbr_upward_mean_action::add_baked_source_mean,
        };
        out.n_edges = 1;
        return vbr_upward_recipe_status::resolved;
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
        vbr_upward_mean_action::none,
    };
    out.n_edges = 1;
    return vbr_upward_recipe_status::resolved;
}

std::array<uint8_t, 32> vbr_upward_build_identity(
        const vbr_upward_recipe & recipe,
        const vbr_upward_representation_identity & source_identity,
    const vbr_upward_representation_identity & target_identity,
        const std::array<uint8_t, 32> & policy_digest,
    const std::array<uint8_t, 32> & tree_digest) noexcept {
    try {
        const auto digest_nonzero = [](const std::array<uint8_t, 32> & value) {
            for (uint8_t byte : value) {
                if (byte != 0) {
                    return true;
                }
            }
            return false;
        };
        const auto identity_complete = [&](
                const vbr_upward_representation_identity & identity) {
            return identity.codec_id != 0 && identity.codec_version != 0 &&
                   digest_nonzero(identity.codebook_digest) &&
                   digest_nonzero(identity.rotation_digest) &&
                   digest_nonzero(identity.meansub_digest) &&
                   digest_nonzero(identity.representation_reference_digest);
        };
        if (!identity_complete(source_identity) ||
            !identity_complete(target_identity) ||
            source_identity.meansub_model_id < 0 ||
            source_identity.meansub_layer < 0 ||
            target_identity.meansub_model_id < 0 ||
            target_identity.meansub_layer < 0 ||
            recipe.version !=
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
        const auto & edge = recipe.edges[0];
        if (edge.mean_action ==
                vbr_upward_mean_action::add_baked_source_mean &&
            (!source_identity.meansub_baked ||
             !target_identity.meansub_baked ||
             source_identity.meansub_model_id !=
                 target_identity.meansub_model_id ||
             source_identity.meansub_layer !=
                 target_identity.meansub_layer ||
             source_identity.meansub_digest !=
                 target_identity.meansub_digest)) {
            return {};
        }
        llama_sha256_writer writer;
        static constexpr char domain_label[] =
            "buun.vbr.upward/build-identity/v3";
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
            writer.u32(uint32_t(edge.mean_action));
        }
        const auto bind_identity = [&](
                const vbr_upward_representation_identity & identity) {
            writer.bytes(identity.codebook_digest.data(),
                         identity.codebook_digest.size());
            writer.bytes(identity.rotation_digest.data(),
                         identity.rotation_digest.size());
            writer.bytes(identity.meansub_digest.data(),
                         identity.meansub_digest.size());
            writer.u32(uint32_t(identity.meansub_model_id));
            writer.u32(uint32_t(identity.meansub_layer));
            writer.u32(identity.meansub_baked ? 1u : 0u);
            writer.u32(identity.codec_id);
            writer.u32(identity.codec_version);
            writer.bytes(identity.representation_reference_digest.data(),
                         identity.representation_reference_digest.size());
        };
        bind_identity(source_identity);
        bind_identity(target_identity);
        writer.bytes(policy_digest.data(), policy_digest.size());
        writer.bytes(tree_digest.data(), tree_digest.size());
        return writer.finish();
    } catch (...) {
        return {};
    }
}
