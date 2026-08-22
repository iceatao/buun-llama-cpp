#include "../src/llama-vbr-downward.h"
#include "../src/llama-kv-cache-iswa.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>

static const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);

static constexpr ggml_type tiers[] = {
    GGML_TYPE_F16, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0,
    GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ,
};

static const unsigned char VBR_TEST_BACKEND_IDENTITY = 0;
static const unsigned char VBR_TEST_BACKEND_IDENTITY_SECOND = 0;

struct fake_preflight_source {
    llama_memory_vbr_preflight_data result = {};
    std::vector<llama_memory_vbr_physical_growth> evidence;
    mutable uint32_t calls = 0;
    mutable uint32_t n_tokens_extra = 0;

    llama_memory_vbr_preflight_data vbr_retier_preflight(
            uint32_t requested,
            std::vector<llama_memory_vbr_physical_growth> * physical) const {
        calls++;
        n_tokens_extra = requested;
        assert(physical != nullptr);
        *physical = evidence;
        return result;
    }
};

static void test_iswa_physical_preflight_preserves_device_skew() {
    llama_memory_vbr_preflight_data first = {};
    llama_memory_vbr_preflight_data second = {};
    first.active = second.active = true;
    first.fits = second.fits = true;
    first.pools = second.pools = 2;
    const void * backend = &VBR_TEST_BACKEND_IDENTITY;
    const std::vector<llama_memory_vbr_physical_growth> first_rows {
        { backend, 0, 80, 0, 0, 0, 0, 100 },
        { backend, 1, 20, 0, 0, 0, 0, 100 },
    };
    const std::vector<llama_memory_vbr_physical_growth> second_rows {
        { backend, 0, 80, 0, 0, 0, 0, 100 },
        { backend, 1, 20, 0, 0, 0, 0, 100 },
    };
    std::vector<llama_memory_vbr_physical_growth> merged;
    const auto skewed = llama_memory_vbr_merge_preflight_children(
        first, first_rows, second, second_rows, &merged);
    assert(!skewed.fits);
    assert(skewed.physical_growth_needed == 200);
    assert(skewed.physical_growth_available == 200);
    assert(skewed.max_deficit == 60);
    assert(merged.size() == 2);

    const std::vector<llama_memory_vbr_physical_growth> balanced_second {
        { backend, 0, 20, 0, 0, 0, 0, 100 },
        { backend, 1, 80, 0, 0, 0, 0, 100 },
    };
    const auto balanced = llama_memory_vbr_merge_preflight_children(
        first, first_rows, second, balanced_second);
    assert(balanced.fits);
    assert(balanced.physical_growth_needed == 200);
    assert(balanced.physical_growth_available == 200);
    assert(balanced.max_deficit == 0);

    // Device ordinals are backend-local. Two backends' device 0 rows must stay
    // independent rather than sharing availability or combining requirements.
    const void * second_backend = &VBR_TEST_BACKEND_IDENTITY_SECOND;
    const std::vector<llama_memory_vbr_physical_growth> first_backend_row {
        { backend, 0, 80, 0, 0, 0, 0, 100 },
    };
    const std::vector<llama_memory_vbr_physical_growth> second_backend_row {
        { second_backend, 0, 80, 0, 0, 0, 0, 100 },
    };
    const auto separate_backends = llama_memory_vbr_merge_preflight_children(
        first, first_backend_row, second, second_backend_row, &merged);
    assert(separate_backends.fits);
    assert(separate_backends.physical_growth_needed == 160);
    assert(separate_backends.physical_growth_available == 200);
    assert(merged.size() == 2);
    assert(merged[0].backend != merged[1].backend);
    assert(merged[0].device == 0 && merged[1].device == 0);

    // Exercise the same collection helper used by the production iSWA override.
    // This pins both child calls, their output evidence, and the replay bound.
    fake_preflight_source first_source { first, first_rows };
    fake_preflight_source second_source { second, second_rows };
    std::vector<llama_memory_vbr_physical_growth> forwarded;
    const auto collected = llama_memory_vbr_preflight_children(
        first_source, second_source, 7, &forwarded);
    assert(!collected.fits);
    assert(first_source.calls == 1 && second_source.calls == 1);
    assert(first_source.n_tokens_extra == 7 && second_source.n_tokens_extra == 7);
    assert(forwarded.size() == 2);
}

static void test_iswa_physical_preflight_shared_scratch_and_overflow() {
    llama_memory_vbr_preflight_data first = {};
    llama_memory_vbr_preflight_data second = {};
    first.active = second.active = true;
    first.fits = second.fits = true;
    const void * backend = &VBR_TEST_BACKEND_IDENTITY;

    // KV mappings are child-owned and additive. K/V materialization scratch is
    // device-owned and each side is therefore reduced by max, not by sum.
    const std::vector<llama_memory_vbr_physical_growth> first_scratch {
        { backend, 0, 10, 100, 40, 20, 20, 150 },
    };
    const std::vector<llama_memory_vbr_physical_growth> second_scratch {
        { backend, 0, 15, 80, 60, 50, 10, 150 },
    };
    std::vector<llama_memory_vbr_physical_growth> merged;
    const auto shared = llama_memory_vbr_merge_preflight_children(
        first, first_scratch, second, second_scratch, &merged);
    assert(shared.fits);
    assert(shared.physical_growth_needed == 115);
    assert(shared.physical_growth_available == 150);
    assert(merged.size() == 1);
    assert(merged[0].kv_needed == 25);
    assert(merged[0].scratch_k_needed == 100);
    assert(merged[0].scratch_v_needed == 60);
    assert(merged[0].scratch_k_current == 50);
    assert(merged[0].scratch_v_current == 20);

    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    const std::vector<llama_memory_vbr_physical_growth> maximum_row {
        { backend, 0, maximum, 0, 0, 0, 0, maximum },
    };
    const std::vector<llama_memory_vbr_physical_growth> zero_row {
        { backend, 0, 0, 0, 0, 0, 0, maximum },
    };
    const auto exact_boundary = llama_memory_vbr_merge_preflight_children(
        first, maximum_row, second, zero_row);
    assert(exact_boundary.fits);
    assert(exact_boundary.physical_growth_needed == maximum);

    const std::vector<llama_memory_vbr_physical_growth> overflowing_row {
        { backend, 0, 1, 0, 0, 0, 0, maximum },
    };
    const auto overflow = llama_memory_vbr_merge_preflight_children(
        first, maximum_row, second, overflowing_row);
    assert(!overflow.fits);
    assert(overflow.physical_growth_needed == maximum);
    assert(overflow.max_deficit == std::numeric_limits<int64_t>::max());
}

static void test_all_recipes_and_rejections() {
    size_t pairs = 0;
    for (size_t source = 0; source < 6; ++source) {
        for (size_t target = source + 1; target < 6; ++target) {
            vbr_downward_recipe recipe;
            assert(vbr_downward_resolve_recipe(
                tiers[source], tiers[target], GGML_TYPE_TURBO1_TCQ, true, recipe) ==
                vbr_downward_recipe_status::resolved);
            assert(recipe.n_edges == target - source);
            assert(recipe.edges[0].source_type == tiers[source]);
            assert(recipe.edges[recipe.n_edges - 1].target_type == tiers[target]);
            for (size_t i = 0; i < recipe.n_edges; ++i) {
                assert(recipe.edges[i].source_type == tiers[source + i]);
                assert(recipe.edges[i].target_type == tiers[source + i + 1]);
            }
            pairs++;
        }
    }
    assert(pairs == 15);

    vbr_downward_recipe out;
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO1_TCQ, true, out) == vbr_downward_recipe_status::equal_tier);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO1_TCQ, true, out) == vbr_downward_recipe_status::upward_forbidden);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO1_TCQ, true, out) == vbr_downward_recipe_status::unsupported_type);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_F16, GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO3_TCQ, true, out) == vbr_downward_recipe_status::below_floor);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_F16, GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO1_TCQ, false, out) == vbr_downward_recipe_status::nonmovable);
}

static llama_vbr_policy::child policy_child(
        std::initializer_list<llama_vbr_policy::step> steps) {
    llama_vbr_policy::child child;
    child.terminal_progress = 100;
    child.steps = steps;
    return child;
}

static void test_policy_projection_tree_and_ordinary() {
    vbr_downward_policy_child ordinary;
    ordinary.initial_types = { GGML_TYPE_F16, GGML_TYPE_TURBO8_0 };
    ordinary.target_types = { GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0 };
    ordinary.policy = policy_child({
        { 0, 0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, 10 },
        { 1, 1, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0, 10 },
    });
    const auto one = vbr_downward_project_policy_prefix({ ordinary });
    assert(one.status == vbr_downward_policy_status::coherent);
    assert(one.prefix.size() == 2);
    assert(one.final_types[0] == ordinary.target_types);
    assert(one.final_cursors[0] == 2);

    vbr_downward_policy_child peer;
    peer.initial_types = { GGML_TYPE_TURBO4_0 };
    peer.target_types = { GGML_TYPE_TURBO3_TCQ };
    peer.policy = policy_child({
        { 0, 0, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_TCQ, 5 },
    });
    const auto tree = vbr_downward_project_policy_prefix({ ordinary, peer });
    assert(tree.status == vbr_downward_policy_status::coherent);
    assert(tree.prefix.size() == 3);
    assert(tree.final_types[0] == ordinary.target_types);
    assert(tree.final_types[1] == peer.target_types);
    assert(tree.final_cursors[0] == 2);
    assert(tree.final_cursors[1] == 1);
    assert(tree.tree_digest != one.tree_digest);

    // Cursor publication follows the absolute degrade-order index, not the
    // number of selected steps: skipped/non-applicable entries are normal in
    // straddled K/V schedules.
    vbr_downward_policy_child skipped;
    skipped.initial_types = { GGML_TYPE_F16 };
    skipped.target_types = { GGML_TYPE_TURBO8_0 };
    skipped.initial_cursor = 4;
    skipped.policy = policy_child({
        { 9, 0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, 10 },
    });
    const auto skipped_projection =
        vbr_downward_project_policy_prefix({ skipped });
    assert(skipped_projection.status ==
           vbr_downward_policy_status::coherent);
    assert(skipped_projection.final_cursors[0] == 10);

    peer.target_types = { GGML_TYPE_TURBO2_TCQ };
    const auto impossible = vbr_downward_project_policy_prefix({ ordinary, peer });
    assert(impossible.status == vbr_downward_policy_status::exhausted);

    vbr_downward_policy_child at_target;
    at_target.initial_types = { GGML_TYPE_TURBO4_0 };
    at_target.target_types = at_target.initial_types;
    at_target.policy = policy_child({
        { 0, 0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, -1 },
    });
    const auto already_coherent = vbr_downward_project_policy_prefix({ at_target });
    assert(already_coherent.status == vbr_downward_policy_status::coherent);
    assert(already_coherent.prefix.empty());
    assert(already_coherent.final_cursors[0] == 0);

    vbr_downward_policy_child bad_source;
    bad_source.initial_types = { GGML_TYPE_F16 };
    bad_source.target_types = { GGML_TYPE_TURBO8_0 };
    bad_source.policy = policy_child({
        { 0, 0, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO8_0, 1 },
    });
    assert(vbr_downward_project_policy_prefix({ bad_source }).status ==
        vbr_downward_policy_status::incoherent);

    vbr_downward_policy_child invalid = bad_source;
    invalid.policy.steps[0].type_a = GGML_TYPE_F16;
    invalid.policy.steps[0].logical_gain = -1;
    assert(vbr_downward_project_policy_prefix({ invalid }).status ==
        vbr_downward_policy_status::invalid);

    vbr_downward_policy_child overflow = bad_source;
    overflow.policy.steps[0].type_a = GGML_TYPE_F16;
    overflow.policy.initial_progress = std::numeric_limits<int64_t>::max();
    overflow.policy.steps[0].logical_gain = 1;
    assert(vbr_downward_project_policy_prefix({ overflow }).status ==
        vbr_downward_policy_status::overflow);
}

static bool live_capture(
        void *, ggml_type source, const std::vector<uint8_t> & bytes,
        std::vector<uint8_t> & stash) {
    stash.resize(std::min<size_t>(8, bytes.size()) + 1);
    stash[0] = uint8_t(source);
    std::copy_n(bytes.begin(), stash.size() - 1, stash.begin() + 1);
    return true;
}

static bool live_edge(
        void *, const vbr_downward_edge & edge, const std::vector<uint8_t> & source,
        const std::vector<uint8_t> * stash, std::vector<uint8_t> & target) {
    target.resize(source.size());
    const uint8_t salt = uint8_t(uint32_t(edge.source_type)*3u + uint32_t(edge.target_type)*5u);
    for (size_t i = 0; i < source.size(); ++i) {
        const uint8_t tap = stash && !stash->empty() ? (*stash)[i % stash->size()] : 0;
        target[i] = uint8_t((uint32_t(source[i])*17u + salt + tap) & 0xffu);
    }
    return true;
}

// Independent chaining/stash-policy oracle: the checking power is the explicit
// edge loop + capture-at-boundary rule; the per-edge byte math is shared with
// the fixture transforms so the two cannot drift apart.
static std::vector<uint8_t> explicit_live_oracle(
        const vbr_downward_recipe & recipe, const std::vector<uint8_t> & source,
        std::vector<uint8_t> * stash) {
    std::vector<uint8_t> current = source;
    for (size_t edge_index = 0; edge_index < recipe.n_edges; ++edge_index) {
        const auto & edge = recipe.edges[edge_index];
        if (edge.capture_stash_before && stash->empty()) {
            assert(live_capture(nullptr, edge.source_type, current, *stash));
        }
        std::vector<uint8_t> next;
        assert(live_edge(nullptr, edge, current,
            stash->empty() ? nullptr : stash, next));
        current = std::move(next);
    }
    return current;
}

static void test_edge_oracles_and_stash_boundaries() {
    const std::vector<uint8_t> source = { 1, 3, 5, 7, 9, 11, 13, 15, 17 };
    const vbr_downward_transform_iface iface { nullptr, live_capture, live_edge };
    for (size_t i = 0; i + 1 < 6; ++i) {
        vbr_downward_recipe recipe;
        assert(vbr_downward_resolve_recipe(tiers[i], tiers[i + 1],
            GGML_TYPE_TURBO1_TCQ, true, recipe) == vbr_downward_recipe_status::resolved);
        std::vector<uint8_t> oracle_stash;
        const auto oracle = explicit_live_oracle(recipe, source, &oracle_stash);
        const auto actual = vbr_downward_execute_recipe(recipe, source, nullptr, iface);
        assert(actual.status == vbr_downward_transform_status::transformed);
        assert(actual.bytes == oracle);
        assert(actual.stash == oracle_stash);
        assert(actual.stash_regenerated == (i >= 2));
    }

    vbr_downward_recipe boundary;
    assert(vbr_downward_resolve_recipe(GGML_TYPE_F16, GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO1_TCQ, true, boundary) == vbr_downward_recipe_status::resolved);
    const auto regenerated = vbr_downward_execute_recipe(boundary, source, nullptr, iface);
    assert(regenerated.status == vbr_downward_transform_status::transformed);
    assert(regenerated.stash_regenerated);
    assert(regenerated.stash[0] == uint8_t(GGML_TYPE_TURBO4_0));
}

static void test_straddled_kv_independent_chains() {
    const std::vector<uint8_t> k = { 2, 4, 6, 8 };
    const std::vector<uint8_t> v = { 1, 4, 9, 16 };
    const vbr_downward_transform_iface iface { nullptr, live_capture, live_edge };
    vbr_downward_recipe kr;
    vbr_downward_recipe vr;
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO1_TCQ, true, kr) == vbr_downward_recipe_status::resolved);
    assert(vbr_downward_resolve_recipe(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO1_TCQ,
        GGML_TYPE_TURBO1_TCQ, true, vr) == vbr_downward_recipe_status::resolved);
    assert(kr.n_edges == 2);
    assert(vr.n_edges == 2);
    assert(kr.source_type != vr.source_type && kr.target_type != vr.target_type);
    const auto ko = vbr_downward_execute_recipe(kr, k, nullptr, iface);
    const auto vo = vbr_downward_execute_recipe(vr, v, nullptr, iface);
    assert(ko.status == vbr_downward_transform_status::transformed);
    assert(vo.status == vbr_downward_transform_status::transformed);
    assert(ko.bytes != vo.bytes);
}

struct fake_workspace {
    size_t current = 0;
    size_t endpoint = 4096;
    bool fail_once = false;
    size_t reserves = 0;
};

static bool workspace_memory(
        ggml_backend_t backend, int, int64_t, int64_t, int64_t,
        size_t * now, size_t * endpoint) {
    auto * state = reinterpret_cast<fake_workspace *>(backend);
    *now = state->current;
    *endpoint = state->endpoint;
    return true;
}

static bool workspace_reserve(ggml_backend_t backend, int64_t, int64_t, int64_t) {
    auto * state = reinterpret_cast<fake_workspace *>(backend);
    state->reserves++;
    if (state->fail_once) {
        state->fail_once = false;
        state->current = state->endpoint / 2;
        return false;
    }
    state->current = state->endpoint;
    return true;
}

struct fake_stash {
    uint64_t current = 0;
    uint64_t endpoint = 8192;
    bool fail = false;
};

static bool stash_memory(void * context, uint64_t & now, uint64_t & endpoint) {
    auto * state = static_cast<fake_stash *>(context);
    now = state->current;
    endpoint = state->endpoint;
    return true;
}

static bool stash_reserve(void * context) {
    auto * state = static_cast<fake_stash *>(context);
    if (state->fail) {
        state->current = state->endpoint / 2;
        return false;
    }
    state->current = state->endpoint;
    return true;
}

static void configure_resource_ledger(llama_cache_acct_ledger & ledger) {
    const llama_cache_acct_completeness_requirement requirement = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    assert(ledger.configure_required_producers(&requirement, 1));
    for (size_t i = 0; i < size_t(llama_cache_acct_category::_count); ++i) {
        const auto category = static_cast<llama_cache_acct_category>(i);
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, HOST, measure, 0);
        }
    }
    assert(ledger.certify_complete(HOST, llama_cache_acct_producer::host_cache));
}

static void test_projection_reserve_retry_and_stashless() {
    llama_cache_acct_ledger ledger;
    configure_resource_ledger(ledger);
    const uint64_t baseline_ops = ledger.snapshot().live_ops;
    fake_workspace workspace;
    fake_stash stash;
    ggml_vbr_backend_iface iface = {};
    iface.kv_transcode_workspace_memory = workspace_memory;
    iface.kv_transcode_workspace_reserve = workspace_reserve;
    const int workspace_owner = 1;
    const int stash_owner = 2;
    vbr_downward_workspace_endpoint w;
    w.owner = &workspace_owner;
    w.iface = &iface;
    w.backend = reinterpret_cast<ggml_backend_t>(&workspace);
    w.device = 0;
    w.domain = HOST;
    w.requests.push_back({ 128, 256, 32 });
    vbr_downward_stash_endpoint s;
    s.owner = &stash_owner;
    s.unit_ids = { 77 };
    s.domain = HOST;
    s.context = &stash;
    s.memory = stash_memory;
    s.reserve = stash_reserve;

    {
        vbr_downward_resource_receipts receipts(ledger);
        workspace.fail_once = true;
        auto first = receipts.reserve_resources({}, { w }, { s });
        assert(first.status == vbr_downward_reserve_status::workspace_reserve_failed);
        assert(first.workspace_growth == 4096);
        assert(first.stash_growth == 8192);
        const uint64_t charged_ops = ledger.snapshot().live_ops;
        assert(charged_ops == baseline_ops + 2);

        stash.fail = true;
        auto retry = receipts.reserve_resources({}, { w }, { s });
        assert(retry.status == vbr_downward_reserve_status::reserved_stashless);
        assert(retry.stashless_units == std::vector<uint64_t>({ 77 }));
        assert(retry.workspace_growth == 0);
        assert(retry.stash_growth == 0);
        assert(ledger.snapshot().live_ops == charged_ops);
    }
    assert(ledger.snapshot().live_ops == baseline_ops);
}

int main() {
    test_iswa_physical_preflight_preserves_device_skew();
    test_iswa_physical_preflight_shared_scratch_and_overflow();
    test_all_recipes_and_rejections();
    test_policy_projection_tree_and_ordinary();
    test_edge_oracles_and_stash_boundaries();
    test_straddled_kv_independent_chains();
    test_projection_reserve_retry_and_stashless();
    std::puts("test-vbr-downward: OK");
    return 0;
}
