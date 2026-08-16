#include "server-cache-authority.h"
#include "server-cache-destruction-quote.h"
#include "server-cache-plan-authority.h"
#include "server-context.h"
#include "server-task.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <list>
#include <numeric>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(expr) do {                                                        \
    if (!(expr)) {                                                              \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n",                   \
                     __FILE__, __LINE__, #expr);                                \
        failures++;                                                             \
    }                                                                           \
} while (0)

void configure_host_accounting(
        server_cache_authority & authority,
        bool with_sidecar = false) {
    const auto host = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const llama_cache_acct_completeness_requirement required[] = {
        { host, llama_cache_acct_producer::host_cache },
        { host, llama_cache_acct_producer::retention_sidecar },
    };
    const size_t n_required = with_sidecar ? std::size(required) : 1;
    CHECK(authority.ledger.configure_required_producers(
        required, n_required));
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload,
            llama_cache_acct_category::artifact_descriptor_metadata }) {
        if (!with_sidecar && category ==
                llama_cache_acct_category::artifact_descriptor_metadata) {
            continue;
        }
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            authority.ledger.gauge_set(category, host, measure, 0);
        }
    }
    CHECK(authority.ledger.certify_complete(
        host, llama_cache_acct_producer::host_cache));
    if (with_sidecar) {
        CHECK(authority.ledger.certify_complete(
            host, llama_cache_acct_producer::retention_sidecar));
        authority.retention.configure(
            &authority.ledger, host, &authority.leases);
    }
}

std::list<server_prompt_cache_state> make_entry(
        const char * identity,
        size_t bytes) {
    std::list<server_prompt_cache_state> entry;
    entry.emplace_back();
    entry.front().adapter_config_key = identity;
    entry.front().data.main.resize(bytes);
    return entry;
}

std::list<server_prompt_cache_state> make_prompt_entry(
        const char * identity,
        std::initializer_list<llama_token> tokens) {
    auto entry = make_entry(identity, 1);
    entry.front().prompt.tokens = server_tokens(
        llama_tokens(tokens), false);
    return entry;
}

llama_cache_acct_artifact_id publish_host_retention(
        server_cache_authority & authority,
        server_prompt_cache::iterator state) {
    common_chat_msg_spans spans;
    for (int32_t i = 0; i < state->prompt.n_tokens(); ++i) {
        spans.add(COMMON_CHAT_ROLE_USER, i, 1);
    }
    const auto key =
        server_retention_instance_key::for_host_entry(&*state);
    CHECK(authority.retention.publish(
        key, common_retention_pool::attention, spans, true,
        state->prompt.n_tokens(), 1, true));
    const auto artifact = authority.retention.artifact_id(key);
    CHECK(artifact.v != 0);
    return artifact;
}

std::list<server_prompt_cache_state> make_redundant_entry() {
    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    entry.front().data.main.assign(16, 7);
    entry.front().data.drft.assign(4, 8);
    entry.front().prompt.checkpoints.emplace_back();
    auto & checkpoint = entry.front().prompt.checkpoints.back();
    checkpoint.n_tokens = 2;
    checkpoint.pos_min = 0;
    checkpoint.pos_max = 1;
    checkpoint.data_tgt.assign(8, 9);
    checkpoint.data_dft.assign(3, 10);
    checkpoint.accel.ring.assign(5, 11);
    checkpoint.accel.spec.assign(2, 12);
    return entry;
}

constexpr const char * HOST_TRADE_TEST_PROFILE =
    "qwen35-2b-q4-k---medium/nvidia-geforce-rtx-3090-ngl99/b512/kf16-vf16";

class available_host_fallback final : public server_cache_lease_fallback_provider {
public:
    server_cache_durable_fallback_proof acquire(
            const server_cache_lease_subject &,
            const server_cache_lease_identity &) noexcept override {
        return server_cache_durable_fallback_proof_for_test(
            server_cache_lease_fallback_state::available, owner);
    }

private:
    std::shared_ptr<void> owner = std::make_shared<int>(1);
};

struct control_vbr_fixture {
    server_cache_lease_identity identity;
    server_cache_lease_frontier frontier;
    std::shared_ptr<void> owner = std::make_shared<int>(1);
};

server_cache_control_status resolve_control_vbr_fixture(
        void * context,
        const server_cache_control_selector &,
        server_cache_lease_subject & subject,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier,
        server_cache_durable_fallback_proof & pin) noexcept {
    auto * fixture = static_cast<control_vbr_fixture *>(context);
    subject = {
        { 0xe11a }, common_retention_artifact_kind::host_entry, -1,
    };
    identity = fixture->identity;
    frontier = fixture->frontier;
    pin = server_cache_durable_fallback_proof_for_test(
        server_cache_lease_fallback_state::available, fixture->owner);
    return server_cache_control_status::ok;
}

struct control_host_refresh_fixture {
    server_prompt_cache * cache = nullptr;
    const std::string * execution_identity = nullptr;
    const server_prompt * live_prompt = nullptr;
    int32_t live_slot = -1;
    std::string live_adapter_identity;
};

bool refresh_control_host_fixture(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier) noexcept {
    auto * fixture = static_cast<control_host_refresh_fixture *>(context);
    if (selector.kind == server_cache_control_subject_kind::live_prefix) {
        if (!fixture->live_prompt ||
            !(selector.retention_key ==
                server_retention_instance_key::for_slot(fixture->live_slot)) ||
            !server_cache_lease_build_identity(
                *fixture->execution_identity,
                fixture->live_adapter_identity,
                fixture->live_prompt->tokens,
                fixture->live_prompt->n_tokens(), identity)) {
            return false;
        }
        frontier = {
            fixture->live_prompt->sequence_epoch,
            uint64_t(fixture->live_prompt->n_tokens()),
            fixture->live_prompt->n_tokens(),
        };
        return frontier.valid();
    }
    const auto * wanted = reinterpret_cast<const server_prompt_cache_state *>(
        selector.retention_key.instance);
    const auto found = std::find_if(
        fixture->cache->states.begin(), fixture->cache->states.end(),
        [&](const auto & value) { return &value == wanted; });
    if (found == fixture->cache->states.end() ||
        !server_cache_lease_build_identity(
            *fixture->execution_identity, found->adapter_config_key,
            found->prompt.tokens, found->prompt.n_tokens(), identity)) {
        return false;
    }
    frontier = {
        found->prompt.sequence_epoch,
        uint64_t(found->prompt.n_tokens()),
        found->prompt.n_tokens(),
    };
    return frontier.valid();
}

void configure_host_trade(
        server_cache_authority & authority,
        server_prompt_cache & cache,
        const std::string & execution_identity,
        server_cache_lease_table * leases = nullptr) {
    configure_host_accounting(authority, true);
    authority.calibration_profile = HOST_TRADE_TEST_PROFILE;
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &authority.retention;
    cache.lease_obs = leases ? leases : &authority.leases;
    cache.lease_execution_identity = &execution_identity;
}

server_prompt_cache::iterator install_host_trade_entry(
        server_prompt_cache & cache,
        server_cache_authority & authority,
        const char * unique_adapter,
        size_t bytes) {
    static llama_token next_token = 100;
    const llama_token first = next_token;
    next_token += 3;
    auto entry = make_prompt_entry(
        unique_adapter, { first, first + 1, first + 2 });
    entry.front().data.main.assign(bytes, uint8_t(next_token));
    CHECK(cache.publish(std::move(entry)));
    auto installed = std::prev(cache.states.end());
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER, 0, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 1, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 2, 1);
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_host_entry(&*installed),
        common_retention_pool::attention,
        spans,
        true,
        3,
        1,
        true));
    return installed;
}

void make_host_trade_pair(
        server_prompt_cache::iterator victim,
        server_prompt_cache::iterator recovery,
        const char * adapter,
        llama_token token,
        int32_t source_id,
        bool main_family = false) {
    victim->adapter_config_key = adapter;
    recovery->adapter_config_key = adapter;
    victim->prompt.tokens = server_tokens(
        llama_tokens { token, token + 1, token + 2 }, false);
    recovery->prompt.tokens = server_tokens(
        llama_tokens { token, token + 1, token + 2 }, false);
    victim->prompt.sequence_epoch = uint64_t(token);
    recovery->prompt.sequence_epoch = uint64_t(token);
    victim->data.main = recovery->data.main;
    victim->cache_plan_source_id = source_id;
    recovery->cache_plan_source_id = source_id + 100;
    victim->main_family = main_family;
    // Keep the proof source outside the victim candidate set while still
    // allowing D-A's short-lived pin to nest over it.
    recovery->recovery_pins = 1;
    CHECK(server_prompt_cache::exactly_redundant(*victim, *recovery));
}

server_cache_lease_id grant_host_lease(
        server_prompt_cache & cache,
        server_cache_lease_table & leases,
        server_prompt_cache::iterator victim,
        server_cache_lease_class cls) {
    const auto artifact = cache.retention_obs->artifact_id(
        server_retention_instance_key::for_host_entry(&*victim));
    server_cache_lease_identity identity;
    CHECK(server_cache_lease_build_identity(
        *cache.lease_execution_identity,
        victim->adapter_config_key,
        victim->prompt.tokens,
        victim->prompt.n_tokens(),
        identity));
    const server_cache_lease_subject subject {
        artifact,
        common_retention_artifact_kind::host_entry,
        -1,
    };
    const auto scope = server_cache_lease_scope::from(
        leases.new_context_scope());
    return cls == server_cache_lease_class::hard
        ? leases.grant_hard(subject, scope, identity,
              server_cache_lease_table::IMPLICIT_SOFT_TTL_NS)
        : leases.grant_soft(subject, scope, identity,
              server_cache_lease_table::IMPLICIT_SOFT_TTL_NS);
}

server_cache_lease_id grant_explicit_host_lease(
        server_prompt_cache & cache,
        server_cache_lease_table & leases,
        server_prompt_cache::iterator victim,
        uint64_t scope_id) {
    const auto artifact = cache.retention_obs->artifact_id(
        server_retention_instance_key::for_host_entry(&*victim));
    server_cache_lease_identity identity;
    CHECK(server_cache_lease_build_identity(
        *cache.lease_execution_identity, victim->adapter_config_key,
        victim->prompt.tokens, victim->prompt.n_tokens(), identity));
    return leases.grant_hard_owned(
        { artifact, common_retention_artifact_kind::host_entry, -1 },
        server_cache_lease_scope::from(
            server_cache_explicit_lease_scope_id { scope_id }),
        identity,
        server_cache_lease_owner_id { 1 },
        { 1, uint64_t(victim->prompt.n_tokens()), victim->prompt.n_tokens() },
        server_cache_lease_table::IMPLICIT_SOFT_TTL_NS);
}

void test_declared_family_round_trip_and_price() {
    const common_cache_family_binding declared_main {
        { 0xe11b }, common_cache_family_role::main,
    };
    CHECK(declared_main.declared());

    common_prompt_checkpoint checkpoint;
    checkpoint.n_tokens = 2;
    checkpoint.pos_min = 0;
    checkpoint.pos_max = 1;
    checkpoint.cache_family = declared_main;
    checkpoint.data_tgt.assign(4, 7);

    common_prompt_checkpoint copied = checkpoint;
    CHECK(copied.cache_family == declared_main);
    common_prompt_checkpoint assigned;
    assigned = checkpoint;
    CHECK(assigned.cache_family == declared_main);
    copied.clear();
    CHECK(!copied.cache_family.declared());

    server_prompt source;
    source.tokens = server_tokens(llama_tokens { 1, 2, 3 }, false);
    source.checkpoints.push_back(checkpoint);
    source.sequence_epoch = 9;
    const auto cloned = source.clone();
    CHECK(cloned.checkpoints.front().cache_family == declared_main);

    server_prompt_cache cache(0, 0);
    auto staged = cache.stage(source, 8, 0, "family-adapter");
    CHECK(staged.size() == 1);
    CHECK(staged.front().prompt.checkpoints.front().cache_family ==
          declared_main);
    server_prompt_cache_apply_family(
        staged.front(), declared_main, false);
    CHECK(staged.front().cache_family == declared_main);
    CHECK(staged.front().main_family);
    CHECK(cache.publish(std::move(staged)));

    server_prompt_cache_restore_delivery delivery;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), delivery));
    CHECK(delivery.cache_family == declared_main);
    const auto delivered_family = delivery.cache_family;
    server_prompt restored_prompt;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(delivery), restored_prompt, 3);
    CHECK(cache.states.empty());
    CHECK(delivered_family == declared_main);
    CHECK(restored_prompt.checkpoints.front().cache_family == declared_main);

    const common_cache_family_binding undeclared;
    server_prompt_cache_state automatic;
    server_prompt_cache_apply_family(automatic, undeclared, true);
    CHECK(automatic.main_family);
    CHECK(!automatic.cache_family.declared());

    const common_cache_plan_calib calib {
        "e1-family-test", 1, 0.0, 1.0, 10.0,
    };
    uint32_t automatic_weight = 0;
    uint32_t declared_weight = 0;
    uint64_t automatic_price = 0;
    uint64_t declared_price = 0;
    CHECK(server_cache_host_retention_price_us(
        calib, 100, false,
        common_cache_family_main_family(undeclared, true),
        automatic_weight, automatic_price));
    CHECK(server_cache_host_retention_price_us(
        calib, 100, false,
        common_cache_family_main_family(declared_main, false),
        declared_weight, declared_price));
    CHECK(automatic_weight == SERVER_CACHE_HOST_MAIN_FAMILY_WEIGHT);
    CHECK(declared_weight == automatic_weight);
    CHECK(declared_price == automatic_price);
    CHECK(!common_cache_family_allows_additional_weight(declared_main));
    const common_cache_family_binding declared_branch {
        declared_main.family, common_cache_family_role::branch,
    };
    CHECK(!common_cache_family_main_family(declared_branch, true));
    CHECK(!common_cache_family_allows_additional_weight(declared_branch));

    // One lineage rule covers slot reuse, undeclared append, and an explicit
    // declaration branching from retained content. A tokenizer-global BOS or
    // shared system prefix is not by itself conversation continuity.
    CHECK(common_cache_family_follow_lineage(
              declared_main, undeclared, 0, 3) == undeclared);
    CHECK(common_cache_family_follow_lineage(
              declared_main, declared_branch, 0, 3) == declared_branch);
    CHECK(common_cache_family_follow_lineage(
              declared_main, undeclared, 3, 3) == declared_main);
    CHECK(common_cache_family_follow_lineage(
              declared_main, declared_branch, 3, 3) == declared_main);
    CHECK(common_cache_family_follow_lineage(
              declared_main, declared_branch, 1, 3) == declared_branch);

    server_task parent(SERVER_TASK_TYPE_COMPLETION);
    parent.cache_family_binding_token = { 17, 29 };
    parent.add_child(1, 2);
    CHECK(parent.child_tasks.size() == 1);
    CHECK(parent.child_tasks.front().cache_family_binding_token ==
          parent.cache_family_binding_token);

    server_cache_authority cache_authority;
    server_cache_control_config control_config;
    control_config.leases = &cache_authority.leases;
    control_config.retention = &cache_authority.retention;
    server_cache_control_authority control(control_config);
    server_cache_control_request holder_request;
    holder_request.ttl_ns = 1000000000000ULL;
    const auto holder = control.execute(
        server_cache_control_operation::holder_create, holder_request);
    CHECK(holder.status == server_cache_control_status::ok);
    server_cache_control_request family_request;
    family_request.holder = holder.holder;
    family_request.idempotency_key = 1;
    const auto family = control.execute(
        server_cache_control_operation::family_register, family_request);
    CHECK(family.status == server_cache_control_status::ok);
    server_cache_control_request binding_request;
    binding_request.holder = holder.holder;
    binding_request.family = family.family;
    binding_request.family_role = common_cache_family_role::main;
    binding_request.idempotency_key = 2;
    const auto binding = control.execute(
        server_cache_control_operation::family_bind, binding_request);
    CHECK(binding.status == server_cache_control_status::ok);
    auto branch_request = binding_request;
    branch_request.family_role = common_cache_family_role::branch;
    branch_request.idempotency_key = 3;
    const auto branch_binding = control.execute(
        server_cache_control_operation::family_bind, branch_request);
    CHECK(branch_binding.status == server_cache_control_status::ok);
    CHECK(!(branch_binding.family_binding == binding.family_binding));
    const auto slot_round_trip =
        server_cache_family_slot_round_trip_for_test(
            control, binding.family_binding, branch_binding.family_binding);
    CHECK(slot_round_trip.resolved);
    CHECK(slot_round_trip.second_resolved);
    CHECK(slot_round_trip.roles_distinct);
    CHECK(slot_round_trip.host_roles_distinct);
    CHECK(slot_round_trip.no_restore_resume);
    CHECK(slot_round_trip.binding_intact);
    CHECK(slot_round_trip.host_save_carries);
    CHECK(slot_round_trip.checkpoint_carries);
    std::puts(
        "E1_FAMILY two_slot_save PASS main=1 branch=1 distinct=1");
    std::puts(
        "E1_FAMILY actual_slot_resume PASS binding_intact=1 host=1 checkpoint=1");
    std::puts("E1_FAMILY round_trip PASS main_price_equal no_stack");
}

void test_checkpoint_lineage_ignores_retier_but_rejects_content_change() {
    common_prompt_checkpoint checkpoint;
    checkpoint.checkpoint_epoch = 11;
    checkpoint.checkpoint_epoch_swa = 13;

    llama_memory_vbr_state_data state = {};
    state.checkpoint_epoch = 11;
    state.checkpoint_epoch_swa = 13;
    CHECK(common_prompt_checkpoint_lineage_matches(checkpoint, state));

    // Retiering advances representation identity but preserves the attention-content lineage.
    state.representation_epoch = 7;
    state.representation_epoch_swa = 9;
    CHECK(common_prompt_checkpoint_lineage_matches(checkpoint, state));

    state.checkpoint_epoch++;
    CHECK(!common_prompt_checkpoint_lineage_matches(checkpoint, state));
    state.checkpoint_epoch--;
    state.checkpoint_epoch_swa++;
    CHECK(!common_prompt_checkpoint_lineage_matches(checkpoint, state));
}

void test_checkpoint_suffix_trim_rebases_only_preserved_prefixes() {
    llama_memory_vbr_state_data before = {};
    before.checkpoint_epoch = 11;
    before.checkpoint_epoch_swa = 13;
    llama_memory_vbr_state_data after = before;
    after.checkpoint_epoch++;
    after.checkpoint_epoch_swa++;

    std::list<common_prompt_checkpoint> checkpoints(3);
    auto it = checkpoints.begin();
    it->pos_max = 9;
    it->checkpoint_epoch = before.checkpoint_epoch;
    it->checkpoint_epoch_swa = before.checkpoint_epoch_swa;
    auto & preserved = *it++;
    it->pos_max = 10;
    it->checkpoint_epoch = before.checkpoint_epoch;
    it->checkpoint_epoch_swa = before.checkpoint_epoch_swa;
    auto & removed_boundary = *it++;
    it->pos_max = 8;
    it->checkpoint_epoch = before.checkpoint_epoch - 1;
    it->checkpoint_epoch_swa = before.checkpoint_epoch_swa;
    auto & stale_lineage = *it;

    CHECK(server_cache_checkpoint_rebase_preserved_suffix(
        checkpoints, before, after, 10) == 1);
    CHECK(common_prompt_checkpoint_lineage_matches(preserved, after));
    CHECK(!common_prompt_checkpoint_lineage_matches(removed_boundary, after));
    CHECK(!common_prompt_checkpoint_lineage_matches(stale_lineage, after));
}

bool host_source_present(
        const server_prompt_cache & cache,
        int32_t source_id) {
    return std::any_of(cache.states.begin(), cache.states.end(),
        [&](const auto & state) {
            return state.cache_plan_source_id == source_id;
        });
}

// Regression for F0b review MUST-1: lifecycle accounting may prove/transact publication, but the
// prompt cache's configured limit remains a FIFO rotation policy—not an admission ceiling. A full
// 1 MiB cache must accept a second 700 KiB entry and evict the oldest, rather than become fill-once.
void test_lifecycle_full_cache_rotates() {
    server_cache_authority authority;
    configure_host_accounting(authority);

    server_prompt_cache cache(/* limit_size_mib */ 1, /* limit_tokens */ 1024);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;

    CHECK(cache.publish(make_entry("oldest", 700 * 1024)));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().adapter_config_key == "oldest");

    CHECK(cache.publish(make_entry("newest", 700 * 1024)));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().adapter_config_key == "newest");
    CHECK(cache.size() == 700 * 1024);
    CHECK(authority.admission_commits == 2);
    CHECK(authority.admission_refusals == 0);
    CHECK(authority.destruction.prepared_release_commits == 1);
    CHECK(authority.destruction.prepared_release_fallbacks == 0);
    CHECK(authority.destruction.n_events == 1);
    CHECK(authority.destruction.events[0].execution ==
          server_cache_destruction_execution::prepared_release);
}

void test_lifecycle_restore_retains_immutable_source() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    const std::string execution = "restore-retained-hard-fallback";

    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &authority.retention;
    cache.lease_obs = &authority.leases;
    cache.lease_execution_identity = &execution;

    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    const common_cache_family_binding declared_branch {
        { 0xe11b51de }, common_cache_family_role::branch,
    };
    server_prompt_cache_apply_family(
        entry.front(), declared_branch, true);
    entry.front().prompt.sequence_epoch = 17;
    entry.front().data.main.assign(32, 7);
    entry.front().prompt.checkpoints.emplace_back();
    entry.front().prompt.checkpoints.back().n_tokens = 2;
    entry.front().prompt.checkpoints.back().data_tgt.assign(8, 9);
    entry.front().prompt.checkpoints.emplace_back();
    entry.front().prompt.checkpoints.back().n_tokens = 3;
    entry.front().prompt.checkpoints.back().data_tgt.assign(8, 10);
    CHECK(cache.publish(std::move(entry)));
    CHECK(cache.states.size() == 1);
    common_chat_msg_spans checkpoint_spans;
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_host_entry(&cache.states.front()),
        common_retention_pool::attention, checkpoint_spans,
        false, 3, 3, true));
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_checkpoint(
            -1, &cache.states.front().prompt.checkpoints.front()),
        common_retention_pool::attention, checkpoint_spans,
        false, 3, 2, true));
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_checkpoint(
            -1, &cache.states.front().prompt.checkpoints.back()),
        common_retention_pool::attention, checkpoint_spans,
        false, 3, 3, true));
    CHECK(authority.retention.artifact_id(
        server_retention_instance_key::for_checkpoint(
            -1, &cache.states.front().prompt.checkpoints.front())).v != 0);
    CHECK(authority.retention.artifact_id(
        server_retention_instance_key::for_checkpoint(
            -1, &cache.states.front().prompt.checkpoints.back())).v != 0);
    llama_cache_acct_artifact_id durable_artifact;
    std::vector<llama_cache_acct_op_id> durable_ops;
    server_cache_recovery_pin durable_pin;
    CHECK(cache.acquire_durable_recovery(
        cache.states.front().prompt.tokens, "same",
        durable_artifact, durable_ops, durable_pin));
    CHECK(durable_artifact.v != 0);
    CHECK(durable_ops.size() == 3);
    CHECK(durable_pin.binds_exact(durable_artifact, durable_ops));
    durable_pin = {};
    server_tokens missing;
    missing.insert(llama_tokens { 99 });
    CHECK(!cache.acquire_durable_recovery(
        missing, "same", durable_artifact, durable_ops, durable_pin));
    const auto live_ops_before = authority.ledger.snapshot().live_ops;
    const auto host_size_before = cache.states.front().size();
    const auto * source_checkpoint =
        &cache.states.front().prompt.checkpoints.front();

    server_prompt_cache_restore_delivery first;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), first));
    CHECK(first.retains_source);
    CHECK(first.cache_family == declared_branch);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == host_size_before);

    server_prompt live_first;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(first), live_first, 4, -1, 3);
    const auto launch_prepared = [&](int32_t slot_id) {
        const auto key = server_retention_instance_key::for_slot(slot_id);
        CHECK(authority.retention.prepared_for_launch(key));
        server_retention_lineage_ticket source_ticket;
        CHECK(authority.retention.consume_prepared_launch(
            key, source_ticket));
        CHECK(authority.retention.credit_reuse(source_ticket) !=
              common_retention_credit_result::unavailable);
        authority.retention.release_lineage_ticket(source_ticket);
        server_retention_lineage_ticket destination_ticket;
        CHECK(authority.retention.acquire_lineage_ticket(
            key, destination_ticket));
        CHECK(authority.retention.activate_lineage_ticket(
            destination_ticket));
        authority.retention.release_lineage_ticket(destination_ticket);
    };
    launch_prepared(4);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == host_size_before);
    CHECK(live_first.n_tokens() == 3);
    CHECK(live_first.checkpoints.size() == 2);
    CHECK(&live_first.checkpoints.front() != source_checkpoint);
    CHECK(live_first.checkpoints.front().n_tokens == 2);
    CHECK(cache.states.front().cache_family == declared_branch);
    const auto live_ops_after_first = authority.ledger.snapshot().live_ops;
    CHECK(live_ops_after_first > live_ops_before);
    server_retention_candidate restored;
    CHECK(authority.retention.candidate_for_instance(
        server_retention_instance_key::for_checkpoint(
            4, &live_first.checkpoints.front()), restored));
    CHECK(!restored.release_ops.empty());
    CHECK(authority.destruction.host_restores_retained == 1);
    CHECK(authority.destruction.host_restores_consumed == 0);

    // The HTTP selector is exact. Model the real completion shape: {1,2} is
    // the submitted request prefix and 3 is the deterministic sampled suffix
    // stored in both the retained host source and the resumed live slot.
    // Looking up only the submitted prefix fails before lease admission; the
    // complete state is the selector that reaches the proof/disjointness door.
    auto prefix_entry = make_prompt_entry("same", { 1, 2 });
    CHECK(!cache.acquire_durable_recovery(
        prefix_entry.front().prompt.tokens, "same",
        durable_artifact, durable_ops, durable_pin));
    CHECK(cache.acquire_durable_recovery(
        live_first.tokens, "same",
        durable_artifact, durable_ops, durable_pin));
    durable_pin = {};

    // Exact production shape: a non-consuming restore leaves the source host
    // node and creates a distinct live-slot sidecar artifact for the same
    // lineage/frontier. The hard lease must bind the retained physical copy,
    // not confuse shared lineage with shared storage.
    const auto host_key = server_retention_instance_key::for_host_entry(
        &cache.states.front());
    const auto live_key = server_retention_instance_key::for_slot(4);
    const auto host_artifact = authority.retention.artifact_id(host_key);
    const auto live_artifact = authority.retention.artifact_id(live_key);
    CHECK(host_artifact.v != 0);
    CHECK(live_artifact.v != 0);
    CHECK(host_artifact != live_artifact);
    common_retention_lineage_record host_lineage;
    common_retention_lineage_record live_lineage;
    CHECK(authority.retention.lineage_for_instance(
        host_key, host_lineage));
    CHECK(authority.retention.lineage_for_instance(
        live_key, live_lineage));
    CHECK(host_lineage.lineage_id == live_lineage.lineage_id);
    CHECK(host_lineage.reuse_hits == 1);
    CHECK(host_lineage.state ==
          common_retention_frequency_state::probation);

    // A divergent request still credits the immutable host source, but its
    // live destination must start on probation rather than inheriting the
    // source's accumulated value. The real load path selects this transition
    // when LCP is shorter than the restored host frontier.
    server_prompt_cache_restore_delivery divergent;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), divergent));
    server_prompt live_branch;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(divergent), live_branch,
        5, -1, 2, false);
    const auto divergent_checkpoint_key =
        server_retention_instance_key::for_checkpoint(
            5, &live_branch.checkpoints.front());
    server_retention_checkpoint_inventory divergent_checkpoint;
    CHECK(!authority.retention.checkpoint_inventory(
        divergent_checkpoint_key, divergent_checkpoint));
    common_retention_lineage_record branch_lineage;
    CHECK(!authority.retention.lineage_for_instance(
        server_retention_instance_key::for_slot(5), branch_lineage));
    CHECK(authority.retention.lineage_for_instance(
        host_key, host_lineage));
    CHECK(host_lineage.reuse_hits == 1);
    launch_prepared(5);
    CHECK(authority.retention.lineage_for_instance(
        server_retention_instance_key::for_slot(5), branch_lineage));
    CHECK(authority.retention.checkpoint_inventory(
        divergent_checkpoint_key, divergent_checkpoint));
    CHECK(divergent_checkpoint.release_owned);
    CHECK(branch_lineage.lineage_id != host_lineage.lineage_id);
    CHECK(branch_lineage.reuse_hits == 0);
    CHECK(branch_lineage.state ==
          common_retention_frequency_state::probation);
    CHECK(authority.retention.lineage_for_instance(
        host_key, host_lineage));
    CHECK(host_lineage.reuse_hits == 1);
    CHECK(host_lineage.state ==
          common_retention_frequency_state::probation);

    control_host_refresh_fixture refresh {
        &cache, &execution, &live_first, 4, "same",
    };
    server_cache_control_config control_config;
    control_config.leases = &authority.leases;
    control_config.retention = &authority.retention;
    control_config.refresh_context = &refresh;
    control_config.refresh_subject = refresh_control_host_fixture;
    control_config.host_proof_context = &cache;
    control_config.acquire_host_proof = [](void * context,
        const server_cache_control_selector & selector) noexcept {
        return server_prompt_cache_host_fallback_proof(
            *static_cast<server_prompt_cache *>(context), selector);
    };
    server_cache_control_authority control(control_config);
    server_cache_control_request holder_request;
    holder_request.ttl_ns = 1000000000ULL;
    const auto holder = control.execute(
        server_cache_control_operation::holder_create, holder_request);
    CHECK(holder.status == server_cache_control_status::ok);
    server_cache_control_request acquire;
    acquire.holder = holder.holder;
    acquire.requested_class = server_cache_lease_class::hard;
    acquire.ttl_ns = holder_request.ttl_ns;
    acquire.subject.kind = server_cache_control_subject_kind::live_prefix;
    acquire.subject.retention_key = live_key;
    acquire.fallback.kind = server_cache_control_subject_kind::host_snapshot;
    acquire.fallback.retention_key = host_key;
    const auto hard = control.execute(
        server_cache_control_operation::lease_acquire, acquire);
    CHECK(hard.status == server_cache_control_status::ok);
    CHECK(cache.states.front().recovery_pins == 1);

    // A decoded append preserves the live artifact identity. Only the
    // frontier advances. Drive the real release-time sidecar publication: it
    // mints a new immutable record, migrates the lease from the old physical
    // record, and leaves the range beyond its proof partially stale.
    live_first.tokens.insert(llama_tokens { 4 });
    server_cache_lease_identity append_identity;
    CHECK(server_cache_lease_build_identity(
        execution, "same", live_first.tokens,
        live_first.n_tokens(), append_identity));
    const server_cache_lease_frontier append_frontier {
        live_first.sequence_epoch,
        uint64_t(live_first.n_tokens()),
        live_first.n_tokens(),
    };
    CHECK(authority.retention.publish(
        live_key, common_retention_pool::attention, checkpoint_spans,
        false, uint64_t(live_first.n_tokens()),
        uint64_t(live_first.n_tokens()), true,
        &append_identity, &append_frontier));
    const auto appended_artifact = authority.retention.artifact_id(live_key);
    CHECK(appended_artifact.v != 0);
    CHECK(appended_artifact != live_artifact);
    CHECK(!server_cache_lease_is_hard(
        authority.leases.inspect(live_artifact, append_identity)));
    CHECK(server_cache_lease_is_hard(
        authority.leases.inspect(appended_artifact, append_identity)));
    server_cache_control_request inspect;
    inspect.holder = holder.holder;
    inspect.lease = hard.lease;
    const auto partial = control.execute(
        server_cache_control_operation::lease_inspect, inspect);
    CHECK(partial.status ==
          server_cache_control_status::partially_stale);
    CHECK(partial.proven_frontier.token_count == 3);
    CHECK(partial.lease_frontier.token_count == 4);
    // The migrated hard lease must still own its retained-host proof before
    // explicit release. The terminal zero check alone would not detect a pin
    // accidentally dropped during artifact replacement.
    CHECK(cache.states.front().recovery_pins == 1);

    // Adapter/media/execution changes are identity changes, not frontier
    // growth. The real lease-table rebound terminal must still fail closed.
    server_cache_lease_identity changed_identity;
    CHECK(server_cache_lease_build_identity(
        execution, "different-adapter", live_first.tokens,
        live_first.n_tokens(), changed_identity));
    CHECK(authority.leases.artifact_rebound(
        appended_artifact, changed_identity));
    CHECK(control.execute(
        server_cache_control_operation::lease_inspect,
        inspect).status == server_cache_control_status::subject_lost);

    server_cache_control_request lease_release;
    lease_release.holder = holder.holder;
    lease_release.lease = hard.lease;
    CHECK(control.execute(
        server_cache_control_operation::lease_release,
        lease_release).status == server_cache_control_status::ok);
    CHECK(cache.states.front().recovery_pins == 0);
    std::printf(
        "E1_TWO_COPIES restored_host_fallback PASS live=%" PRIu64
        " appended=%" PRIu64 " host=%" PRIu64
        " distinct=1 prefix_lookup=0 append=partially_stale"
        " adapter_rebind=subject_lost\n",
        live_artifact.v, appended_artifact.v, host_artifact.v);

    server_prompt_cache_restore_delivery second;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), second));
    CHECK(second.cache_family == declared_branch);
    CHECK(authority.retention.begin_competition_wave());
    server_prompt live_second;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(second), live_second, 5, -1, 3);
    launch_prepared(5);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == host_size_before);
    CHECK(live_second.n_tokens() == 3);
    CHECK(live_second.checkpoints.size() == 2);
    CHECK(authority.ledger.snapshot().live_ops > live_ops_after_first);
    CHECK(authority.retention.candidate_for_instance(
        server_retention_instance_key::for_checkpoint(
            5, &live_second.checkpoints.front()), restored));
    CHECK(!restored.release_ops.empty());
    CHECK(authority.retention.lineage_for_instance(
        host_key, host_lineage));
    CHECK(host_lineage.reuse_hits == 2);
    CHECK(host_lineage.state ==
          common_retention_frequency_state::promoted);

    // Restored members own independent operations and therefore participate
    // in the exact D-A4 release terminal instead of remaining permanently
    // fail-closed. Releasing the newer member leaves its restored survivor
    // and survivor operations intact.
    auto restored_victim = std::next(live_second.checkpoints.begin());
    const auto victim_key = server_retention_instance_key::for_checkpoint(
        5, &*restored_victim);
    server_retention_candidate victim_candidate;
    CHECK(authority.retention.candidate_for_instance(
        victim_key, victim_candidate));
    auto release = llama_cache_prepare_release_set(
        authority.ledger, victim_candidate.release_ops,
        authority.ledger.snapshot().serial);
    CHECK(release.ready());
    CHECK(release.commit() ==
          llama_cache_conditional_release_status::released);
    authority.retention.retire_after_committed_release(victim_key);
    live_second.checkpoints.erase(restored_victim);
    CHECK(live_second.checkpoints.size() == 1);
    CHECK(authority.retention.candidate_for_instance(
        server_retention_instance_key::for_checkpoint(
            5, &live_second.checkpoints.front()), restored));
    CHECK(!restored.release_ops.empty());
    CHECK(authority.destruction.host_restores_retained == 3);

    cache.destroy_entry(
        cache.states.begin(), server_cache_destruction_reason::host_capacity);
    CHECK(cache.states.empty());
    authority.retention.retire_slot(4);
    authority.retention.retire_slot(5);
    CHECK(authority.ledger.snapshot().live_ops == 0);
    CHECK(authority.destruction.prepared_release_commits == 1);
}

void test_implicit_soft_append_chain_is_bounded() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    const std::string execution = "implicit-soft-append-bound";
    const auto key = server_retention_instance_key::for_slot(19);
    const auto scope = authority.leases.new_context_scope();
    CHECK(scope.v != 0);
    common_chat_msg_spans spans;
    server_tokens tokens(llama_tokens { 31 }, false);
    const uint64_t sequence_epoch = 91;
    server_cache_lease_id first_lease;

    for (size_t turn = 0; turn < 32; ++turn) {
        const auto source_artifact = authority.retention.artifact_id(key);
        if (turn == 1) {
            const server_cache_lease_subject stale_marker {
                source_artifact,
                common_retention_artifact_kind::live_slot,
                19,
            };
            authority.leases.artifact_identity_unavailable(stale_marker);
            server_cache_lease_replay_result marked;
            CHECK(server_cache_lease_table::replay(
                authority.leases.event_snapshot(), marked));
            CHECK(std::any_of(
                marked.identity_unavailable.begin(),
                marked.identity_unavailable.end(),
                [&](const auto & value) {
                    return value.artifact == source_artifact;
                }));
        }

        server_cache_lease_identity identity;
        CHECK(server_cache_lease_build_identity(
            execution, "append-adapter", tokens, tokens.size(), identity));
        const server_cache_lease_frontier frontier {
            sequence_epoch, uint64_t(tokens.size()), int64_t(tokens.size()),
        };
        CHECK(authority.retention.publish(
            key, common_retention_pool::attention, spans, false,
            uint64_t(tokens.size()), uint64_t(tokens.size()), true,
            &identity, source_artifact.v != 0 ? &frontier : nullptr));
        const auto artifact = authority.retention.artifact_id(key);
        CHECK(artifact.v != 0);
        const auto lease = authority.leases.grant_soft(
            { artifact, common_retention_artifact_kind::live_slot, 19 },
            server_cache_lease_scope::from(scope), identity,
            server_cache_lease_table::IMPLICIT_SOFT_TTL_NS);
        CHECK(bool(lease));
        if (turn == 0) {
            first_lease = lease;
        } else {
            CHECK(lease == first_lease);
        }
        server_cache_lease_replay_result replayed;
        CHECK(server_cache_lease_table::replay(
            authority.leases.event_snapshot(), replayed));
        CHECK(replayed.active.size() == 1);
        if (source_artifact.v != 0) {
            CHECK(std::none_of(
                replayed.identity_unavailable.begin(),
                replayed.identity_unavailable.end(),
                [&](const auto & value) {
                    return value.artifact == source_artifact;
                }));
        }
        tokens.insert(llama_tokens { llama_token(32 + turn) });
    }

    authority.retention.retire(key);
    server_cache_lease_replay_result retired;
    CHECK(server_cache_lease_table::replay(
        authority.leases.event_snapshot(), retired));
    CHECK(retired.active.empty());
    CHECK(authority.ledger.snapshot().live_ops == 0);
    std::puts("E1_FAMILY implicit_soft_append_bound PASS turns=32 leases=1");
}

void test_durable_recovery_binds_exact_published_peer() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    server_prompt_cache cache(0, 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.retention_obs = &authority.retention;
    cache.destruction_obs = &authority.destruction;

    server_prompt_cache::iterator older;
    CHECK(cache.publish(
        make_prompt_entry("same", { 1, 2, 3 }), nullptr, -1, &older));
    const auto older_artifact = publish_host_retention(authority, older);
    llama_cache_acct_artifact_id pinned_artifact;
    std::vector<llama_cache_acct_op_id> pinned_ops;
    server_cache_recovery_pin older_pin;
    CHECK(cache.acquire_durable_recovery(
        older, pinned_artifact, pinned_ops, older_pin));
    CHECK(pinned_artifact == older_artifact);

    // The older token-identical peer is pinned, so publish's dedup pass must
    // retain it. The returned iterator is the newly published physical node,
    // and durable recovery must bind that node rather than find_state_exact's
    // first (older) peer.
    server_prompt_cache::iterator fresh;
    CHECK(cache.publish(
        make_prompt_entry("same", { 1, 2, 3 }), nullptr, -1, &fresh));
    CHECK(cache.states.size() == 2);
    CHECK(fresh != older);
    const auto fresh_artifact = publish_host_retention(authority, fresh);
    CHECK(fresh_artifact != older_artifact);

    llama_cache_acct_artifact_id recovery_artifact;
    std::vector<llama_cache_acct_op_id> recovery_ops;
    server_cache_recovery_pin fresh_pin;
    CHECK(cache.acquire_durable_recovery(
        fresh, recovery_artifact, recovery_ops, fresh_pin));
    CHECK(recovery_artifact == fresh_artifact);
    CHECK(recovery_artifact != older_artifact);
    CHECK(fresh->recovery_pins == 1);
    CHECK(older->recovery_pins == 1);
}

void test_unlaunched_disarm_releases_recovery_pin() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    server_prompt_cache cache(0, 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.retention_obs = &authority.retention;
    cache.destruction_obs = &authority.destruction;

    server_prompt_cache::iterator displaced_copy;
    CHECK(cache.publish(
        make_prompt_entry("same", { 1, 2, 3 }),
        nullptr, -1, &displaced_copy));
    (void) publish_host_retention(authority, displaced_copy);
    llama_cache_acct_artifact_id artifact;
    std::vector<llama_cache_acct_op_id> ops;
    server_cache_recovery_pin recovery_pin;
    CHECK(cache.acquire_durable_recovery(
        displaced_copy, artifact, ops, recovery_pin));
    CHECK(displaced_copy->recovery_pins == 1);

    server_cache_plan_execution execution;
    execution.kind = server_cache_plan_execution_kind::cold_replay;
    execution.target = 0;
    auto plan = std::make_unique<common_cache_plan_record>();
    server_cache_plan_disarm_unlaunched(
        execution, plan, recovery_pin);
    CHECK(!execution.authoritative());
    CHECK(!plan);
    CHECK(!recovery_pin.valid());
    CHECK(displaced_copy->recovery_pins == 0);

    // The former recovery source is ordinary priced inventory again: a later
    // superset publication may retire it rather than treating it as pinned.
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3, 4 })));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().prompt.n_tokens() == 4);
}

void test_displacement_save_order_preserves_prefix_recovery() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    server_prompt_cache cache(0, 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.retention_obs = &authority.retention;
    cache.destruction_obs = &authority.destruction;

    // The old ordering (legacy prefix first, then longer victim) demonstrates
    // why contains(legacy) must be rechecked: ordinary publish dedup removes
    // the just-saved shorter state.
    server_prompt_cache::iterator legacy_first;
    CHECK(cache.publish(
        make_prompt_entry("same", { 1, 2 }), nullptr, -1, &legacy_first));
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));
    server_tokens legacy_tokens(llama_tokens { 1, 2 }, false);
    CHECK(!cache.contains(legacy_tokens, "same"));

    // Reset the fixture, then model the certified ordering: publish the
    // victim, bind and pin that exact node, publish the legacy frontier, and
    // verify both recovery sources remain physically present.
    cache.clear_accounting();
    cache.states.clear();
    server_prompt_cache::iterator victim;
    CHECK(cache.publish(
        make_prompt_entry("same", { 1, 2, 3 }), nullptr, -1, &victim));
    const auto victim_artifact = publish_host_retention(authority, victim);
    llama_cache_acct_artifact_id recovery_artifact;
    std::vector<llama_cache_acct_op_id> recovery_ops;
    server_cache_recovery_pin victim_pin;
    CHECK(cache.acquire_durable_recovery(
        victim, recovery_artifact, recovery_ops, victim_pin));
    CHECK(recovery_artifact == victim_artifact);

    server_prompt_cache::iterator legacy;
    CHECK(cache.publish(
        make_prompt_entry("same", { 1, 2 }), nullptr, -1, &legacy));
    const auto legacy_artifact = publish_host_retention(authority, legacy);
    CHECK(legacy_artifact != victim_artifact);
    CHECK(cache.states.size() == 2);
    CHECK(cache.contains(legacy_tokens, "same"));
    server_tokens victim_tokens(llama_tokens { 1, 2, 3 }, false);
    CHECK(cache.contains(victim_tokens, "same"));
    CHECK(victim_pin.valid());
    CHECK(victim->recovery_pins == 1);
}

void test_lifecycle_off_restore_consumes() {
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    server_cache_destruction_observer observer;
    cache.destruction_obs = &observer;
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));

    server_prompt_cache_restore_delivery delivery;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), delivery));
    CHECK(!delivery.retains_source);
    server_prompt live;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(delivery), live, 0);
    CHECK(cache.states.empty());
    CHECK(live.n_tokens() == 3);
    CHECK(observer.host_restores_retained == 0);
    CHECK(observer.host_restores_consumed == 1);
    CHECK(observer.n_events == 1);
    CHECK(observer.events[0].request.reason ==
          server_cache_destruction_reason::host_consumed_restore);
    CHECK(observer.events[0].execution ==
          server_cache_destruction_execution::pass_through);
}

void test_lifecycle_restore_batch_timing() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    server_prompt_cache cache(0, 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &authority.retention;

    auto entry = make_entry("batch-restore", 1);
    llama_tokens prompt_tokens(4096);
    std::iota(prompt_tokens.begin(), prompt_tokens.end(), 1);
    entry.front().prompt.tokens = server_tokens(
        std::move(prompt_tokens), false);
    entry.front().data.main.assign(32, 7);
    for (int i = 0; i < 8; ++i) {
        entry.front().prompt.checkpoints.emplace_back();
        auto & checkpoint = entry.front().prompt.checkpoints.back();
        checkpoint.n_tokens = 4096;
        checkpoint.data_tgt.assign(64 * 1024, uint8_t(i + 1));
        checkpoint.accel.ring.assign(4 * 1024, uint8_t(i + 2));
    }
    CHECK(cache.publish(std::move(entry)));
    common_chat_msg_spans spans;
    for (size_t i = 0; i < 2048; ++i) {
        spans.add(i % 2 == 0 ? COMMON_CHAT_ROLE_USER
                             : COMMON_CHAT_ROLE_ASSISTANT,
                  i * 2, 2);
    }
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_host_entry(&cache.states.front()),
        common_retention_pool::attention, spans, true, 4096, 4096, true));
    for (const auto & checkpoint : cache.states.front().prompt.checkpoints) {
        CHECK(authority.retention.publish(
            server_retention_instance_key::for_checkpoint(-1, &checkpoint),
            common_retention_pool::attention, spans, true, 4096,
            checkpoint.n_tokens, true));
    }

    std::vector<uint64_t> prepare_samples;
    std::vector<uint64_t> commit_samples;
    prepare_samples.reserve(21);
    commit_samples.reserve(21);
    for (int trial = 0; trial < 21; ++trial) {
        server_prompt_cache_restore_delivery delivery;
        const auto prepare_begin = std::chrono::steady_clock::now();
        CHECK(cache.prepare_restore_delivery(cache.states.begin(), delivery));
        const auto prepare_end = std::chrono::steady_clock::now();
        server_prompt live;
        const auto commit_begin = std::chrono::steady_clock::now();
        cache.commit_restore_delivery(
            cache.states.begin(), std::move(delivery), live, 100 + trial);
        const auto commit_end = std::chrono::steady_clock::now();
        prepare_samples.push_back(uint64_t(std::chrono::duration_cast<
            std::chrono::nanoseconds>(prepare_end - prepare_begin).count()));
        commit_samples.push_back(uint64_t(std::chrono::duration_cast<
            std::chrono::nanoseconds>(commit_end - commit_begin).count()));
        CHECK(live.checkpoints.size() == 8);
        for (const auto & checkpoint : live.checkpoints) {
            server_retention_candidate candidate;
            CHECK(authority.retention.candidate_for_instance(
                server_retention_instance_key::for_checkpoint(
                    100 + trial, &checkpoint), candidate));
            CHECK(candidate.release_ops.size() == 2);
        }
        authority.retention.retire_slot(100 + trial);
    }
    std::sort(prepare_samples.begin(), prepare_samples.end());
    std::sort(commit_samples.begin(), commit_samples.end());
    std::fprintf(stderr,
        "CHECKPOINT_RESTORE_TIMING members=8 prepare_median_ns=%" PRIu64
        " commit_median_ns=%" PRIu64 "\n",
        prepare_samples[prepare_samples.size() / 2],
        commit_samples[commit_samples.size() / 2]);

    cache.destroy_entry(
        cache.states.begin(), server_cache_destruction_reason::host_capacity);
    CHECK(authority.ledger.snapshot().live_ops == 0);
}

void test_checkpoint_creation_churn_timing() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);

    llama_tokens token_ids(2000);
    std::iota(token_ids.begin(), token_ids.end(), 1);
    server_tokens tokens(token_ids, false);
    common_chat_msg_spans spans;
    for (size_t i = 0; i < 1000; ++i) {
        spans.add(i % 2 == 0 ? COMMON_CHAT_ROLE_USER
                             : COMMON_CHAT_ROLE_ASSISTANT,
                  i * 2, 2);
    }

    std::list<common_prompt_checkpoint> ring;
    const std::string execution_identity = "checkpoint-churn-execution";
    const std::string adapter_identity = "checkpoint-churn-adapter";
    const auto publish_member = [&](common_prompt_checkpoint & checkpoint,
                                    std::array<uint64_t, 4> * timings = nullptr) {
        server_cache_lease_identity identity;
        CHECK(server_cache_lease_build_identity(
            execution_identity, adapter_identity, tokens,
            checkpoint.n_tokens, identity));
        checkpoint.computation_frontier.version =
            common_computation_frontier::VERSION;
        checkpoint.computation_frontier.sequence_epoch = 1;
        checkpoint.computation_frontier.token_count = checkpoint.n_tokens;
        checkpoint.computation_frontier.next_position =
            llama_pos(checkpoint.n_tokens);
        checkpoint.computation_frontier.execution_identity =
            identity.execution_identity;
        checkpoint.computation_frontier.adapter_config_identity =
            identity.adapter_config_identity;
        checkpoint.computation_frontier.media_content_identity =
            identity.media_content_identity;
        checkpoint.data_tgt.assign(64 * 1024, uint8_t(checkpoint.n_tokens));
        checkpoint.accel.ring.assign(4 * 1024, 7);
        const auto key = server_retention_instance_key::for_checkpoint(
            7, &checkpoint);
        const auto publish_begin = std::chrono::steady_clock::now();
        CHECK(authority.retention.publish(
            key, common_retention_pool::attention, spans, false,
            2000, uint64_t(checkpoint.n_tokens), true, &identity));
        const auto publish_end = std::chrono::steady_clock::now();
        const auto artifact = authority.retention.artifact_id(key);
        std::vector<llama_cache_acct_op_id> ops;
        const auto admit_begin = std::chrono::steady_clock::now();
        CHECK(authority.admit_live_checkpoint(
            artifact, checkpoint.data_tgt.size(), checkpoint.accel.size(), ops));
        const auto admit_end = std::chrono::steady_clock::now();
        CHECK(authority.retention.attach_release_ops(key, std::move(ops)));
        const auto attach_end = std::chrono::steady_clock::now();
        const auto scope = authority.leases.new_context_scope();
        CHECK(authority.leases.grant_soft(
            { artifact, common_retention_artifact_kind::checkpoint, 7 },
            server_cache_lease_scope::from(scope), identity,
            server_cache_lease_table::IMPLICIT_SOFT_TTL_NS));
        const auto lease_end = std::chrono::steady_clock::now();
        if (timings) {
            (*timings)[0] = uint64_t(std::chrono::duration_cast<
                std::chrono::nanoseconds>(publish_end - publish_begin).count());
            (*timings)[1] = uint64_t(std::chrono::duration_cast<
                std::chrono::nanoseconds>(admit_end - admit_begin).count());
            (*timings)[2] = uint64_t(std::chrono::duration_cast<
                std::chrono::nanoseconds>(attach_end - admit_end).count());
            (*timings)[3] = uint64_t(std::chrono::duration_cast<
                std::chrono::nanoseconds>(lease_end - attach_end).count());
        }
    };
    for (int i = 0; i < 8; ++i) {
        ring.emplace_back();
        ring.back().n_tokens = 600 + i * 200;
        publish_member(ring.back());
    }

    const common_cache_plan_calib calib {
        "checkpoint-churn", 1, 10.0, 0.01, 100.0,
    };
    std::array<std::vector<uint64_t>, 10> samples;
    for (auto & values : samples) {
        values.reserve(31);
    }
    const auto elapsed = [](const auto & begin) {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
    };

    for (int creation = 0; creation < 31; ++creation) {
        std::vector<server_cache_checkpoint_trade_input> legacy_prices;
        legacy_prices.reserve(ring.size());
        const auto legacy_inventory_begin = std::chrono::steady_clock::now();
        uint32_t ordinal = 0;
        uint32_t previous = UINT32_MAX;
        int64_t previous_tokens = 0;
        for (const auto & checkpoint : ring) {
            server_retention_candidate candidate;
            server_cache_lease_identity identity;
            const auto key = server_retention_instance_key::for_checkpoint(
                7, &checkpoint);
            CHECK(authority.retention.candidate_for_instance(key, candidate));
            CHECK(server_cache_lease_build_identity(
                execution_identity, adapter_identity, tokens,
                checkpoint.n_tokens, identity));
            const auto lease = authority.leases.inspect(
                candidate.artifact_id, identity);
            if (previous != UINT32_MAX) {
                server_cache_checkpoint_trade_input price;
                price.ordinal = ordinal;
                price.recovery_ordinal = previous;
                price.artifact = candidate.artifact_id;
                price.stable_id = candidate.record.stamp.stable_id;
                price.payload_bytes = checkpoint.size();
                price.replay_tokens = uint64_t(
                    checkpoint.n_tokens - previous_tokens);
                price.identity_known = identity.valid();
                price.recovery_available = true;
                price.mandatory_anchor =
                    candidate.record.stamp.mandatory_anchor;
                price.hard_leased = server_cache_lease_is_hard(lease);
                price.weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
                legacy_prices.push_back(price);
            }
            previous = ordinal++;
            previous_tokens = checkpoint.n_tokens;
        }
        samples[0].push_back(elapsed(legacy_inventory_begin));
        const auto legacy_priced_begin = std::chrono::steady_clock::now();
        const auto legacy_plan = server_cache_plan_checkpoint_thinning(
            legacy_prices, &calib);
        samples[1].push_back(elapsed(legacy_priced_begin));

        std::vector<server_cache_checkpoint_floor_input> legacy_floor;
        legacy_floor.reserve(ring.size());
        const auto legacy_floor_begin = std::chrono::steady_clock::now();
        ordinal = 0;
        for (const auto & checkpoint : ring) {
            server_retention_candidate candidate;
            server_cache_lease_identity identity;
            const auto key = server_retention_instance_key::for_checkpoint(
                7, &checkpoint);
            CHECK(authority.retention.candidate_for_instance(key, candidate));
            CHECK(server_cache_lease_build_identity(
                execution_identity, adapter_identity, tokens,
                checkpoint.n_tokens, identity));
            server_cache_checkpoint_floor_input input;
            input.ordinal = ordinal++;
            input.recovery_pinned = authority.retention.recovery_pinned(key);
            if (server_cache_lease_is_hard(authority.leases.inspect(
                    candidate.artifact_id, identity))) {
                input.protection =
                    server_cache_checkpoint_protection::hard_lease;
            }
            legacy_floor.push_back(input);
        }
        const auto legacy_floor_plan =
            server_cache_plan_checkpoint_capacity_floor(legacy_floor);
        samples[2].push_back(elapsed(legacy_floor_begin));
        CHECK(legacy_floor_plan.selected);

        std::vector<server_cache_checkpoint_trade_input> cached_prices;
        cached_prices.reserve(ring.size());
        const auto cached_inventory_begin = std::chrono::steady_clock::now();
        ordinal = 0;
        previous = UINT32_MAX;
        previous_tokens = 0;
        for (const auto & checkpoint : ring) {
            server_retention_checkpoint_inventory candidate;
            CHECK(authority.retention.checkpoint_inventory(
                server_retention_instance_key::for_checkpoint(
                    7, &checkpoint), candidate));
            CHECK(candidate.identity_known && candidate.release_owned);
            if (previous != UINT32_MAX) {
                server_cache_checkpoint_trade_input price;
                price.ordinal = ordinal;
                price.recovery_ordinal = previous;
                price.artifact = candidate.artifact_id;
                price.stable_id = candidate.stable_id;
                price.payload_bytes = checkpoint.size();
                price.replay_tokens = uint64_t(
                    checkpoint.n_tokens - previous_tokens);
                price.identity_known = candidate.identity_known;
                price.recovery_available = true;
                price.mandatory_anchor = candidate.mandatory_anchor;
                price.hard_leased = server_cache_lease_is_hard(
                    candidate.lease);
                price.weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
                cached_prices.push_back(price);
            }
            previous = ordinal++;
            previous_tokens = checkpoint.n_tokens;
        }
        samples[3].push_back(elapsed(cached_inventory_begin));
        const auto cached_priced_begin = std::chrono::steady_clock::now();
        const auto cached_plan = server_cache_plan_checkpoint_thinning(
            cached_prices, &calib);
        samples[4].push_back(elapsed(cached_priced_begin));
        CHECK(cached_plan.selected == legacy_plan.selected);
        CHECK(cached_plan.reason == legacy_plan.reason);
        if (cached_plan.selected) {
            CHECK(cached_plan.ordinal == legacy_plan.ordinal);
            CHECK(cached_plan.recovery_ordinal ==
                  legacy_plan.recovery_ordinal);
        }

        std::vector<server_cache_checkpoint_floor_input> cached_floor;
        cached_floor.reserve(ring.size());
        const auto cached_floor_begin = std::chrono::steady_clock::now();
        ordinal = 0;
        for (const auto & checkpoint : ring) {
            server_retention_checkpoint_inventory candidate;
            CHECK(authority.retention.checkpoint_inventory(
                server_retention_instance_key::for_checkpoint(
                    7, &checkpoint), candidate));
            server_cache_checkpoint_floor_input input;
            input.ordinal = ordinal++;
            input.recovery_pinned = candidate.recovery_pinned;
            if (server_cache_lease_is_hard(candidate.lease)) {
                input.protection =
                    server_cache_checkpoint_protection::hard_lease;
            }
            cached_floor.push_back(input);
        }
        const auto cached_floor_plan =
            server_cache_plan_checkpoint_capacity_floor(cached_floor);
        samples[5].push_back(elapsed(cached_floor_begin));
        CHECK(cached_floor_plan.selected);

        const auto victim_key = server_retention_instance_key::for_checkpoint(
            7, &ring.front());
        authority.retention.retire(victim_key);
        ring.pop_front();
        ring.emplace_back();
        ring.back().n_tokens = 2000;
        std::array<uint64_t, 4> creation_timings = {};
        publish_member(ring.back(), &creation_timings);
        for (size_t i = 0; i < creation_timings.size(); ++i) {
            samples[6 + i].push_back(creation_timings[i]);
        }
    }

    for (auto & values : samples) {
        std::sort(values.begin(), values.end());
    }
    std::fprintf(stderr,
        "CHECKPOINT_CREATION_CHURN_TIMING members=8 tokens=2000 "
        "before_inventory_ns=%" PRIu64 " before_priced_ns=%" PRIu64
        " before_floor_ns=%" PRIu64 " after_inventory_ns=%" PRIu64
        " after_priced_ns=%" PRIu64 " after_floor_ns=%" PRIu64
        " publish_ns=%" PRIu64 " admit_ns=%" PRIu64
        " attach_ns=%" PRIu64 " lease_ns=%" PRIu64 "\n",
        samples[0][samples[0].size() / 2],
        samples[1][samples[1].size() / 2],
        samples[2][samples[2].size() / 2],
        samples[3][samples[3].size() / 2],
        samples[4][samples[4].size() / 2],
        samples[5][samples[5].size() / 2],
        samples[6][samples[6].size() / 2],
        samples[7][samples[7].size() / 2],
        samples[8][samples[8].size() / 2],
        samples[9][samples[9].size() / 2]);

    authority.retention.retire_slot(7);
    CHECK(authority.ledger.snapshot().live_ops == 0);
}

void test_checkpoint_bounded_publication_skip_predicate() {
    common_prompt_checkpoint recovery;
    recovery.n_tokens = 1900;
    recovery.computation_frontier.version =
        common_computation_frontier::VERSION;
    recovery.computation_frontier.sequence_epoch = 7;
    recovery.computation_frontier.token_count = recovery.n_tokens;
    recovery.computation_frontier.next_position = llama_pos(recovery.n_tokens);
    recovery.computation_frontier.execution_identity = "execution";
    recovery.computation_frontier.adapter_config_identity = "adapter";
    recovery.computation_frontier.media_content_identity = "media";
    recovery.checkpoint_epoch = 11;
    recovery.checkpoint_epoch_swa = 13;

    common_prompt_checkpoint incoming = recovery;
    incoming.n_tokens = 2000;
    incoming.computation_frontier.token_count = incoming.n_tokens;
    incoming.computation_frontier.next_position = llama_pos(incoming.n_tokens);
    CHECK(server_cache_checkpoint_bounded_replay(recovery, incoming, 100));
    CHECK(!server_cache_checkpoint_bounded_replay(recovery, incoming, 99));

    incoming.computation_frontier.sequence_epoch++;
    CHECK(!server_cache_checkpoint_bounded_replay(recovery, incoming, 100));
    incoming.computation_frontier.sequence_epoch--;
    incoming.checkpoint_epoch_swa++;
    CHECK(!server_cache_checkpoint_bounded_replay(recovery, incoming, 100));
    incoming.checkpoint_epoch_swa--;
    CHECK(!server_cache_checkpoint_bounded_replay(incoming, recovery, 100));
}

void test_consuming_rebind_mints_checkpoint_ownership() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.retention_obs = &authority.retention;
    cache.destruction_obs = &authority.destruction;

    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    entry.front().prompt.checkpoints.emplace_back();
    entry.front().prompt.checkpoints.back().n_tokens = 2;
    entry.front().prompt.checkpoints.back().data_tgt.assign(8, 9);
    CHECK(cache.publish(std::move(entry)));
    common_chat_msg_spans spans;
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_host_entry(&cache.states.front()),
        common_retention_pool::attention, spans, false, 3, 3, true));
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_checkpoint(
            -1, &cache.states.front().prompt.checkpoints.front()),
        common_retention_pool::attention, spans, false, 3, 2, true));

    // A default delivery deliberately drives the consuming/rebind arm while
    // the lifecycle substrate remains available. That arm must mint fresh
    // live ownership rather than inheriting the host aggregate operations.
    server_prompt_cache_restore_delivery delivery;
    server_prompt live;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(delivery), live, 7);
    CHECK(cache.states.empty());
    server_retention_candidate candidate;
    CHECK(authority.retention.candidate_for_instance(
        server_retention_instance_key::for_checkpoint(
            7, &live.checkpoints.front()), candidate));
    CHECK(!candidate.release_ops.empty());
    authority.retention.retire_slot(7);
    CHECK(authority.ledger.snapshot().live_ops == 0);
}

void test_lifecycle_release_prepare_failure_keeps_legacy_bound() {
    server_cache_authority authority;
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;

    // A pre-authority/unaccounted node has no releasable op union. D-A1 may
    // not certify it, but must retain the legacy explicit-eviction bound.
    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    cache.states.splice(cache.states.end(), entry);
    cache.destroy_entry(
        cache.states.begin(), server_cache_destruction_reason::host_capacity);
    CHECK(cache.states.empty());
    CHECK(authority.destruction.prepared_release_commits == 0);
    CHECK(authority.destruction.prepared_release_fallbacks == 1);
    CHECK(authority.destruction.events[0].execution ==
          server_cache_destruction_execution::pass_through);
}

void test_lifecycle_restore_clone_fault() {
    server_cache_authority authority;
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.publish_authority = &authority;
    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    cache.states.splice(cache.states.end(), entry);
    const auto source_size = cache.states.front().size();

    // The injected tag exercises the explicit fail-closed seam. Deliberately
    // does not attempt to make the allocator throw std::bad_alloc.
    server_prompt_cache_restore_delivery delivery;
    CHECK(!cache.prepare_restore_delivery(cache.states.begin(), delivery));
    CHECK(!delivery.retains_source);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == source_size);
    CHECK(cache.states.front().prompt.n_tokens() == 3);
}

void test_lifecycle_authority_without_debug_is_silent() {
    server_cache_authority authority;
    configure_host_accounting(authority);
    server_cache_plan_authority plan_authority(
        common_cache_plan_authority_level::lru);
    CHECK(plan_authority.configured_level ==
          common_cache_plan_authority_level::lru);
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    CHECK(!cache.debug_observability);
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));

    server_prompt_cache_restore_delivery delivery;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), delivery));
    server_prompt live;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(delivery), live, 0, 7);
    CHECK(cache.states.size() == 1);
    CHECK(cache.debug_lifecycle_emissions == 0);

    // Positive control: the same retained restore emits exactly once when the
    // explicit debug view is enabled, proving the zero above is a real gate.
    cache.debug_observability = true;
    server_prompt_cache_restore_delivery debug_delivery;
    CHECK(cache.prepare_restore_delivery(
        cache.states.begin(), debug_delivery));
    server_prompt debug_live;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(debug_delivery), debug_live, 1, 7);
    CHECK(cache.debug_lifecycle_emissions == 1);
}

void test_authority_source_ids_survive_save_dedup() {
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    CHECK(cache.publish(make_prompt_entry("same", { 1 })));
    CHECK(cache.publish(make_prompt_entry("same", { 9 })));
    CHECK(cache.states.size() == 2);

    cache.cache_plan_begin_inventory();
    auto old = cache.states.begin();
    auto survivor = std::next(old);
    int32_t old_source = -1;
    int32_t survivor_source = -1;
    CHECK(cache.cache_plan_get_source_id(*old, old_source));
    CHECK(cache.cache_plan_get_source_id(*survivor, survivor_source));
    CHECK(old_source == 0);
    CHECK(survivor_source == 1);

    // Publishing the larger {1,2} prompt removes {1}. The surviving {9}
    // node keeps source 1, while the new node gets 2 even if the allocator
    // recycles the erased node's address.
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2 })));
    CHECK(cache.states.size() == 2);
    CHECK(cache.cache_plan_get_source_id(
        cache.states.front(), old_source));
    CHECK(old_source == survivor_source);
    CHECK(cache.cache_plan_get_source_id(
        cache.states.back(), old_source));
    CHECK(old_source == 2);
}

void test_exact_redundant_host_eviction() {
    server_cache_authority authority;
    const std::string execution_identity = "test-execution";
    configure_host_accounting(authority, true);

    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &authority.retention;
    cache.lease_obs = &authority.leases;
    cache.lease_execution_identity = &execution_identity;

    server_prompt source;
    source.tokens = server_tokens(llama_tokens { 1, 2, 3 }, false);
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER, 0, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 1, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 2, 1);
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_slot(0),
        common_retention_pool::attention,
        spans,
        true,
        3,
        1,
        true));

    auto first = make_redundant_entry();
    CHECK(cache.publish(std::move(first), &source, 0));
    CHECK(cache.states.size() == 1);
    const auto live_ops_before = authority.ledger.snapshot().live_ops;

    auto duplicate = make_redundant_entry();
    CHECK(server_prompt_cache::exactly_redundant(
        cache.states.front(), duplicate.front()));
    CHECK(cache.publish(std::move(duplicate), &source, 0));

    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().recovery_pins == 0);
    CHECK(authority.ledger.snapshot().live_ops == live_ops_before);
    CHECK(authority.destruction.redundant_host_certified == 1);
    CHECK(authority.destruction.redundant_host_executed == 1);
    CHECK(authority.destruction.redundant_host_refused == 0);
    CHECK(authority.destruction.redundant_host_release_bytes == 38);
    CHECK(authority.destruction.events[0].execution ==
          server_cache_destruction_execution::redundant_host_eviction);
    CHECK(authority.destruction_counters.has_receipt);
    CHECK(authority.destruction_counters.last_receipt.state ==
          common_cache_plan_destruction_state::executed);
    CHECK(authority.destruction_counters.last_receipt.displaced_fate ==
          common_cache_plan_displaced_fate::exact_duplicate);
    CHECK(authority.destruction_counters.last_receipt.recovery_citation ==
          common_cache_plan_recovery_citation::resolved);
    const auto & recovery_receipt =
        authority.destruction_counters.last_receipt;
    CHECK(recovery_receipt.recovery_source_artifact_id.v != 0);
    CHECK(recovery_receipt.recovery_source_artifact_id.v !=
          recovery_receipt.selected_attention.front().v);
    CHECK(recovery_receipt.recovery_source_manifest_digest.valid());
    const auto survivor_ops = cache.states.front().release_ops();
    const std::vector<llama_cache_acct_op_id> survivor_op_vector(
        survivor_ops.begin(), survivor_ops.end());
    CHECK(recovery_receipt.recovery_source_manifest_digest ==
          server_cache_destruction_recovery_source_digest(
              recovery_receipt.recovery_source_artifact_id,
              survivor_op_vector));
    CHECK(authority.destruction_counters.quoted
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_class::host_artifact_drop)] == 1);
    CHECK(authority.destruction_counters.certified
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_class::host_artifact_drop)] == 1);
    CHECK(authority.destruction_counters.executed
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_class::host_artifact_drop)] == 1);
    CHECK(authority.destruction_counters.lease_verdict
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_lease_verdict::unleased)] == 1);
    CHECK(authority.destruction_counters.recovery_outcome
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_displaced_fate::exact_duplicate)] == 1);
    // Lifecycle + authority without --cache-debug must not emit maintenance
    // evidence, even though the certified execution and process counters run.
    CHECK(cache.debug_destruction_emissions == 0);

    // Positive control for the same seam: explicit debug emits quoted,
    // certified, and executed receipts exactly once each.
    cache.debug_observability = true;
    auto debug_duplicate = make_redundant_entry();
    CHECK(cache.publish(std::move(debug_duplicate), &source, 0));
    CHECK(cache.debug_destruction_emissions == 3);
}

void test_redundancy_payload_mismatch_and_missing_catalog() {
    auto victim = make_prompt_entry("same", { 1, 2, 3 });
    victim.front().data.main.assign(4, 1);
    victim.front().prompt.checkpoints.emplace_back();
    victim.front().prompt.checkpoints.back().n_tokens = 2;
    victim.front().prompt.checkpoints.back().data_tgt.assign(2, 3);
    auto survivor = make_prompt_entry("same", { 1, 2, 3 });
    survivor.front().data.main.assign(4, 1);
    survivor.front().prompt.checkpoints.emplace_back();
    survivor.front().prompt.checkpoints.back().n_tokens = 2;
    survivor.front().prompt.checkpoints.back().data_tgt.assign(2, 3);
    survivor.front().prompt.tokens = server_tokens(
        llama_tokens { 1, 2, 3, 4 }, false);
    // Coverage superset is accepted only because all three physical payload
    // planes are still byte-identical.
    CHECK(server_prompt_cache::exactly_redundant(
        victim.front(), survivor.front()));
    survivor.front().prompt.checkpoints.back().data_tgt[1] = 4;
    CHECK(!server_prompt_cache::exactly_redundant(
        victim.front(), survivor.front()));

    server_cache_authority authority;
    configure_host_accounting(authority);
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));
    CHECK(cache.states.size() == 1);
    CHECK(authority.destruction.redundant_host_executed == 0);
    CHECK(authority.destruction.redundant_host_refused == 1);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
    CHECK(authority.destruction.prepared_release_commits == 1);
}

void test_host_trade_soft_lease_weight_flips_victim() {
    server_cache_authority authority;
    const std::string execution = "trade-soft";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto a = install_host_trade_entry(cache, authority, "a-v", 64);
    auto ar = install_host_trade_entry(cache, authority, "a-r", 64);
    auto b = install_host_trade_entry(cache, authority, "b-v", 64);
    auto br = install_host_trade_entry(cache, authority, "b-r", 64);
    make_host_trade_pair(a, ar, "pair-a", 10, 10);
    make_host_trade_pair(b, br, "pair-b", 20, 20);
    CHECK(grant_host_lease(
        cache, authority.leases, a, server_cache_lease_class::soft));

    cache.limit_size = cache.size() - b->size() + 1;
    cache.update();
    CHECK(host_source_present(cache, 10));
    CHECK(!host_source_present(cache, 20));
    CHECK(authority.destruction.host_trade_attempted == 1);
    CHECK(authority.destruction.host_trade_executed == 1);
    CHECK(authority.destruction.host_trade_soft_lease_evictions == 0);
    CHECK(authority.destruction_counters.last_receipt.state ==
          common_cache_plan_destruction_state::executed);
    CHECK(authority.destruction_counters.last_receipt.lease_verdict ==
          common_cache_plan_destruction_lease_verdict::unleased);

    // Soft protection is a price, never a veto: once it is the only
    // certifiable victim, the same lease must still permit eviction.
    cache.limit_size = cache.size() - a->size() + 1;
    cache.update();
    CHECK(!host_source_present(cache, 10));
    CHECK(authority.destruction.host_trade_executed == 2);
    CHECK(authority.destruction.host_trade_soft_lease_evictions == 1);
    CHECK(cache.debug_destruction_emissions == 0);
}

void test_host_trade_main_family_weight_flips_victim() {
    server_cache_authority authority;
    const std::string execution = "trade-main";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto main = install_host_trade_entry(cache, authority, "m-v", 64);
    auto main_r = install_host_trade_entry(cache, authority, "m-r", 64);
    auto child = install_host_trade_entry(cache, authority, "c-v", 64);
    auto child_r = install_host_trade_entry(cache, authority, "c-r", 64);
    make_host_trade_pair(main, main_r, "pair-main", 30, 30, true);
    make_host_trade_pair(child, child_r, "pair-child", 40, 40, false);
    const common_cache_family_binding declared_main {
        { 0xe11b30 }, common_cache_family_role::main,
    };
    const common_cache_family_binding declared_branch {
        declared_main.family, common_cache_family_role::branch,
    };
    main->cache_family = declared_main;
    main_r->cache_family = declared_main;
    child->cache_family = declared_branch;
    child_r->cache_family = declared_branch;
    size_t callback_calls = 0;
    authority.host_retention_weight_context = &callback_calls;
    authority.host_retention_weight = [](
            void * context,
            const server_prompt_cache_state &,
            uint32_t & weight) noexcept {
        ++*static_cast<size_t *>(context);
        weight = 9000;
        return true;
    };

    cache.limit_size = cache.size() - child->size() + 1;
    cache.update();
    CHECK(host_source_present(cache, 30));
    CHECK(!host_source_present(cache, 40));
    CHECK(authority.destruction.host_trade_attempted == 1);
    CHECK(authority.destruction.host_trade_main_family_evictions == 0);
    CHECK(callback_calls == 0);

    // The automatic family signal is likewise a finite pricing weight.
    cache.limit_size = cache.size() - main->size() + 1;
    cache.update();
    CHECK(!host_source_present(cache, 30));
    CHECK(authority.destruction.host_trade_executed == 2);
    CHECK(authority.destruction.host_trade_main_family_evictions == 1);
    CHECK(callback_calls == 0);
}

void test_host_trade_zero_destruction_tie_break() {
    server_cache_authority authority;
    const std::string execution = "trade-tie";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);
    cache.debug_observability = true;

    auto destructive = install_host_trade_entry(cache, authority, "d-v", 64);
    destructive->cache_plan_source_id = 1;
    auto duplicate = install_host_trade_entry(cache, authority, "z-v", 64);
    auto duplicate_r = install_host_trade_entry(cache, authority, "z-r", 64);
    make_host_trade_pair(
        duplicate, duplicate_r, "pair-zero", 50, 2, false);

    cache.limit_size = cache.size() - duplicate->size() + 1;
    cache.update();
    CHECK(host_source_present(cache, 1));
    CHECK(!host_source_present(cache, 2));
    CHECK(authority.destruction.host_trade_attempted == 1);
    CHECK(authority.destruction.host_trade_refused == 0);
    CHECK(authority.destruction.host_trade_zero_destruction_ties == 1);
    CHECK(cache.debug_recovery_pin_exclusions == 1);
    CHECK(cache.debug_host_pressure_floor_outcomes == 1);
    CHECK(cache.debug_destruction_emissions == 5);
}

void test_host_trade_all_refuse_falls_back_to_legacy() {
    server_cache_authority authority;
    const std::string execution = "trade-fallback";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto oldest = install_host_trade_entry(cache, authority, "old", 64);
    auto newer = install_host_trade_entry(cache, authority, "new", 64);
    oldest->cache_plan_source_id = 1;
    newer->cache_plan_source_id = 2;
    cache.limit_tokens = 3;
    cache.update();
    CHECK(!host_source_present(cache, 1));
    CHECK(host_source_present(cache, 2));
    CHECK(authority.destruction.host_trade_attempted == 2);
    CHECK(authority.destruction.host_trade_refused == 2);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 1);
    CHECK(authority.destruction.prepared_release_commits == 1);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::recovery_unavailable);
    const auto * event = authority.destruction.event_for_sequence(
        authority.destruction.n_events);
    CHECK(event != nullptr);
    CHECK(event->execution !=
          server_cache_destruction_execution::priced_host_eviction);
    CHECK(cache.debug_destruction_emissions == 0);
}

void test_host_trade_hard_lease_veto() {
    server_cache_authority authority;
    available_host_fallback fallback;
    server_cache_lease_table hard_leases(nullptr, &fallback);
    const std::string execution = "trade-hard";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution, &hard_leases);

    auto hard = install_host_trade_entry(cache, authority, "h-v", 64);
    auto open = install_host_trade_entry(cache, authority, "o-v", 64);
    hard->cache_plan_source_id = 1;
    open->cache_plan_source_id = 2;
    CHECK(grant_host_lease(
        cache, hard_leases, hard, server_cache_lease_class::hard));

    // Neither victim has durable recovery evidence, so the ranked ladder
    // refuses. The legacy floor must still honor the hard veto and evict the
    // next-oldest known-nonhard entry.
    cache.limit_tokens = 3;
    cache.update();
    CHECK(host_source_present(cache, 1));
    CHECK(!host_source_present(cache, 2));
    CHECK(authority.destruction.host_trade_attempted == 2);
    CHECK(authority.destruction.host_trade_hard_lease_vetoes == 1);
    CHECK(authority.destruction.host_trade_refused == 1);
    CHECK(authority.destruction.host_trade_executed == 0);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 1);
    CHECK(authority.destruction_counters.refused
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_reason::
                  hard_lease_blocked)] == 1);
}

void test_host_trade_all_hard_skips_publication() {
    server_cache_authority authority;
    available_host_fallback fallback;
    server_cache_lease_table hard_leases(nullptr, &fallback);
    const std::string execution = "trade-all-hard";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution, &hard_leases);
    cache.debug_observability = true;

    auto first = install_host_trade_entry(cache, authority, "hard-a", 64);
    auto second = install_host_trade_entry(cache, authority, "hard-b", 64);
    first->cache_plan_source_id = 11;
    second->cache_plan_source_id = 12;
    CHECK(grant_explicit_host_lease(cache, hard_leases, first, 101));
    CHECK(grant_explicit_host_lease(cache, hard_leases, second, 102));

    const auto live_ops_before = authority.ledger.snapshot().live_ops;
    cache.limit_tokens = cache.n_tokens();
    CHECK(!cache.publish(make_prompt_entry("incoming", { 90, 91, 92 })));
    CHECK(cache.states.size() == 2);
    CHECK(host_source_present(cache, 11));
    CHECK(host_source_present(cache, 12));
    CHECK(authority.destruction.host_trade_hard_lease_vetoes == 2);
    CHECK(authority.destruction.host_trade_refused == 0);
    CHECK(authority.destruction.host_trade_publication_skips == 1);
    CHECK(authority.ledger.snapshot().live_ops == live_ops_before);
    CHECK(authority.destruction_counters.last_receipt.state ==
          common_cache_plan_destruction_state::refused);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::hard_lease_blocked);
    CHECK(cache.debug_destruction_emissions == 3);
}

void test_host_trade_floor_skips_recovery_pin() {
    server_cache_authority authority;
    const std::string execution = "trade-pinned-floor";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto pinned = install_host_trade_entry(cache, authority, "pinned", 64);
    auto open = install_host_trade_entry(cache, authority, "open", 64);
    pinned->cache_plan_source_id = 21;
    open->cache_plan_source_id = 22;
    pinned->prompt.sequence_epoch = 1;

    server_cache_lease_identity identity;
    CHECK(server_cache_lease_build_identity(
        execution, pinned->adapter_config_key, pinned->prompt.tokens,
        pinned->prompt.n_tokens(), identity));
    const server_cache_lease_frontier frontier {
        pinned->prompt.sequence_epoch,
        uint64_t(pinned->prompt.n_tokens()),
        pinned->prompt.n_tokens(),
    };
    control_vbr_fixture vbr { identity, frontier };
    control_host_refresh_fixture refresh;
    refresh.cache = &cache;
    refresh.execution_identity = &execution;
    server_cache_control_config config;
    config.leases = &authority.leases;
    config.retention = &authority.retention;
    config.refresh_context = &refresh;
    config.refresh_subject = refresh_control_host_fixture;
    config.resolve_vbr_context = &vbr;
    config.resolve_vbr = resolve_control_vbr_fixture;
    config.host_proof_context = &cache;
    config.acquire_host_proof = [](void * context,
        const server_cache_control_selector & selector) noexcept {
        return server_prompt_cache_host_fallback_proof(
            *static_cast<server_prompt_cache *>(context), selector);
    };
    server_cache_control_authority control(config);
    server_cache_control_request holder_request;
    holder_request.ttl_ns = 1000000000000ULL;
    const auto holder = control.execute(
        server_cache_control_operation::holder_create, holder_request);
    CHECK(holder.status == server_cache_control_status::ok);
    server_cache_control_request acquire;
    acquire.holder = holder.holder;
    acquire.requested_class = server_cache_lease_class::hard;
    acquire.ttl_ns = holder_request.ttl_ns;
    acquire.subject.kind = server_cache_control_subject_kind::vbr_reference;
    acquire.subject.reference = "subject";
    acquire.subject.tenant_key = "tenant";
    acquire.fallback.kind = server_cache_control_subject_kind::host_snapshot;
    acquire.fallback.retention_key =
        server_retention_instance_key::for_host_entry(&*pinned);
    acquire.fallback.identity = identity;
    acquire.fallback.frontier = frontier;
    const auto granted = control.execute(
        server_cache_control_operation::lease_acquire, acquire);
    CHECK(granted.status == server_cache_control_status::ok);
    CHECK(pinned->recovery_pins == 1);
    const auto pinned_artifact = authority.retention.artifact_id(
        server_retention_instance_key::for_host_entry(&*pinned));
    CHECK(pinned_artifact.v != 0);
    cache.debug_observability = true;

    cache.limit_tokens = 3;
    cache.update();
    CHECK(host_source_present(cache, 21));
    CHECK(!host_source_present(cache, 22));
    CHECK(authority.destruction.host_trade_refused == 1);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 1);
    CHECK(cache.debug_recovery_pin_exclusions == 1);
    CHECK(cache.debug_host_pressure_floor_outcomes == 1);
    CHECK(cache.debug_last_recovery_pin_excluded == pinned_artifact);
    CHECK(cache.debug_destruction_emissions == 3);
    std::printf(
        "E1_TWO_COPIES floor_vs_pinned_fallback PASS pinned=%d open=%d pins=%u\n",
        host_source_present(cache, 21) ? 1 : 0,
        host_source_present(cache, 22) ? 1 : 0,
        pinned->recovery_pins);

    server_cache_control_request release;
    release.holder = holder.holder;
    release.lease = granted.lease;
    CHECK(control.execute(
        server_cache_control_operation::lease_release,
        release).status == server_cache_control_status::ok);
    CHECK(pinned->recovery_pins == 0);
}

void test_cache_control_shutdown_drains_host_pin() {
    server_cache_authority authority;
    const std::string execution = "control-shutdown";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);
    auto fallback = install_host_trade_entry(
        cache, authority, "shutdown-fallback", 64);
    fallback->prompt.sequence_epoch = 1;

    server_cache_lease_identity identity;
    CHECK(server_cache_lease_build_identity(
        execution, fallback->adapter_config_key, fallback->prompt.tokens,
        fallback->prompt.n_tokens(), identity));
    const server_cache_lease_frontier frontier {
        fallback->prompt.sequence_epoch,
        uint64_t(fallback->prompt.n_tokens()),
        fallback->prompt.n_tokens(),
    };
    control_vbr_fixture vbr { identity, frontier };
    control_host_refresh_fixture refresh;
    refresh.cache = &cache;
    refresh.execution_identity = &execution;
    server_cache_control_config config;
    config.leases = &authority.leases;
    config.retention = &authority.retention;
    config.refresh_context = &refresh;
    config.refresh_subject = refresh_control_host_fixture;
    config.resolve_vbr_context = &vbr;
    config.resolve_vbr = resolve_control_vbr_fixture;
    config.host_proof_context = &cache;
    config.acquire_host_proof = [](void * context,
        const server_cache_control_selector & selector) noexcept {
        return server_prompt_cache_host_fallback_proof(
            *static_cast<server_prompt_cache *>(context), selector);
    };
    auto control = std::make_unique<server_cache_control_authority>(config);
    server_cache_control_request holder_request;
    holder_request.ttl_ns = 1000000000000ULL;
    const auto holder = control->execute(
        server_cache_control_operation::holder_create, holder_request);
    server_cache_control_request acquire;
    acquire.holder = holder.holder;
    acquire.requested_class = server_cache_lease_class::hard;
    acquire.ttl_ns = holder_request.ttl_ns;
    acquire.subject.kind = server_cache_control_subject_kind::vbr_reference;
    acquire.subject.reference = "subject";
    acquire.subject.tenant_key = "tenant";
    acquire.fallback.kind = server_cache_control_subject_kind::host_snapshot;
    acquire.fallback.retention_key =
        server_retention_instance_key::for_host_entry(&*fallback);
    acquire.fallback.identity = identity;
    acquire.fallback.frontier = frontier;
    CHECK(control->execute(
        server_cache_control_operation::lease_acquire,
        acquire).status == server_cache_control_status::ok);
    CHECK(fallback->recovery_pins == 1);

    // Mirrors server_context_impl::destroy(): proofs close before the prompt
    // cache list nodes they call back into are released.
    control.reset();
    CHECK(fallback->recovery_pins == 0);
    cache.states.clear();
    std::puts("E1_SHUTDOWN live_hard_lease PASS pins=0");
}

void test_host_trade_partial_substrate_is_typed() {
    server_cache_authority authority;
    const std::string execution = "trade-partial-substrate";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);
    auto first = install_host_trade_entry(cache, authority, "first", 64);
    auto second = install_host_trade_entry(cache, authority, "second", 64);
    first->cache_plan_source_id = 31;
    second->cache_plan_source_id = 32;

    cache.lease_obs = nullptr;
    cache.limit_tokens = 3;
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(!host_source_present(cache, 31));
    CHECK(host_source_present(cache, 32));
    CHECK(cache.host_trade_substrate_warned);
    CHECK(authority.destruction.host_trade_substrate_unavailable == 1);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::lease_unavailable);
}

server_cache_checkpoint_trade_input checkpoint_trade(
        uint32_t ordinal,
        uint64_t replay_tokens,
        uint64_t stable_id = 1) {
    server_cache_checkpoint_trade_input out;
    out.ordinal = ordinal;
    out.recovery_ordinal = ordinal == 0 ? 99 : ordinal - 1;
    out.artifact = { uint64_t(ordinal) + 1 };
    out.stable_id = stable_id;
    out.payload_bytes = 4096;
    out.replay_tokens = replay_tokens;
    out.identity_known = true;
    out.recovery_available = true;
    return out;
}

void test_checkpoint_thinning_policy() {
    const common_cache_plan_calib calib {
        "checkpoint-test", 1, 10.0, 0.01, 100.0,
    };
    auto cheap = checkpoint_trade(1, 4, 8);
    auto costly = checkpoint_trade(2, 20, 9);
    auto plan = server_cache_plan_checkpoint_thinning(
        { costly, cheap }, &calib);
    CHECK(plan.selected);
    CHECK(plan.ordinal == cheap.ordinal);
    auto tie_high = checkpoint_trade(7, 4, 70);
    auto tie_low = checkpoint_trade(8, 4, 60);
    plan = server_cache_plan_checkpoint_thinning(
        { tie_high, tie_low }, &calib);
    CHECK(plan.selected && plan.ordinal == tie_low.ordinal);
    const auto permuted = server_cache_plan_checkpoint_thinning(
        { tie_low, tie_high }, &calib);
    CHECK(permuted.selected && permuted.ordinal == plan.ordinal);

    plan = server_cache_plan_checkpoint_thinning({ cheap }, nullptr);
    CHECK(!plan.selected);
    CHECK(plan.reason ==
          common_cache_plan_destruction_reason::profile_unfitted);

    // Soft protection is a price multiplier, never a veto. It can make the
    // next member the lower-cost destruction while both remain eligible.
    cheap.weight_milli = SERVER_CACHE_HOST_SOFT_LEASE_WEIGHT;
    costly.replay_tokens = 8;
    plan = server_cache_plan_checkpoint_thinning(
        { cheap, costly }, &calib);
    CHECK(plan.selected);
    CHECK(plan.ordinal == costly.ordinal);

    // The member the recovery seam would select never joins the optimum.
    cheap.weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
    cheap.seam_heuristic_protected = true;
    costly.replay_tokens = 20;
    plan = server_cache_plan_checkpoint_thinning(
        { cheap, costly }, &calib);
    CHECK(plan.selected);
    CHECK(plan.ordinal == costly.ordinal);

    // With no replay source, thinning refuses and leaves the ring intact at
    // the caller. Hard/mandatory members are equally non-selectable.
    cheap.seam_heuristic_protected = false;
    cheap.recovery_available = false;
    costly.recovery_available = false;
    plan = server_cache_plan_checkpoint_thinning(
        { cheap, costly }, &calib);
    CHECK(!plan.selected);
    CHECK(plan.reason ==
          common_cache_plan_destruction_reason::recovery_unavailable);

    cheap.recovery_available = true;
    cheap.hard_leased = true;
    plan = server_cache_plan_checkpoint_thinning({ cheap }, &calib);
    CHECK(!plan.selected);
    CHECK(plan.reason ==
          common_cache_plan_destruction_reason::hard_lease_blocked);
    CHECK(plan.protection ==
          server_cache_checkpoint_protection::hard_lease);

    cheap.hard_leased = false;
    cheap.seam_heuristic_protected = true;
    plan = server_cache_plan_checkpoint_thinning({ cheap }, &calib);
    CHECK(!plan.selected);
    CHECK(plan.reason ==
          common_cache_plan_destruction_reason::mandatory_anchor);
    CHECK(plan.protection ==
          server_cache_checkpoint_protection::seam_heuristic);

    cheap.seam_heuristic_protected = false;
    cheap.mandatory_anchor = true;
    plan = server_cache_plan_checkpoint_thinning({ cheap }, &calib);
    CHECK(!plan.selected);
    CHECK(plan.protection ==
          server_cache_checkpoint_protection::mandatory_anchor);

    auto seam = checkpoint_trade(9, 4, 90);
    seam.seam_heuristic_protected = true;
    auto hard = checkpoint_trade(10, 4, 100);
    hard.hard_leased = true;
    plan = server_cache_plan_checkpoint_thinning({ hard, seam }, &calib);
    const auto protected_permuted =
        server_cache_plan_checkpoint_thinning({ seam, hard }, &calib);
    CHECK(!plan.selected && !protected_permuted.selected);
    CHECK(plan.protection ==
          server_cache_checkpoint_protection::seam_heuristic);
    CHECK(protected_permuted.protection == plan.protection);

    auto selected_before_protected = checkpoint_trade(11, 1, 110);
    plan = server_cache_plan_checkpoint_thinning(
        { selected_before_protected, seam }, &calib);
    CHECK(plan.selected);
    CHECK(plan.ordinal == selected_before_protected.ordinal);
    CHECK(plan.reason == common_cache_plan_destruction_reason::none);
    CHECK(plan.protection == server_cache_checkpoint_protection::none);
}

void test_checkpoint_thin_lane_skips_pinned_member() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    authority.calibration_profile = HOST_TRADE_TEST_PROFILE;
    std::list<common_prompt_checkpoint> ring;
    ring.emplace_back();
    ring.emplace_back();
    ring.front().n_tokens = 100;
    ring.back().n_tokens = 150;
    const std::string execution = "thin-pin-execution";
    const std::string adapter = "thin-pin-adapter";
    llama_tokens token_ids(200);
    std::iota(token_ids.begin(), token_ids.end(), 1);
    server_tokens tokens(token_ids, false);
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER, 0, 200);
    for (auto & checkpoint : ring) {
        server_cache_lease_identity identity;
        CHECK(server_cache_lease_build_identity(
            execution, adapter, tokens, checkpoint.n_tokens, identity));
        checkpoint.computation_frontier.version =
            common_computation_frontier::VERSION;
        checkpoint.computation_frontier.sequence_epoch = 1;
        checkpoint.computation_frontier.token_count = checkpoint.n_tokens;
        checkpoint.computation_frontier.next_position = checkpoint.n_tokens;
        checkpoint.computation_frontier.execution_identity =
            identity.execution_identity;
        checkpoint.computation_frontier.adapter_config_identity =
            identity.adapter_config_identity;
        checkpoint.computation_frontier.media_content_identity =
            identity.media_content_identity;
        checkpoint.data_tgt.assign(32, uint8_t(checkpoint.n_tokens));
        const auto key = server_retention_instance_key::for_checkpoint(
            17, &checkpoint);
        CHECK(authority.retention.publish(
            key, common_retention_pool::attention, spans, false,
            200, uint64_t(checkpoint.n_tokens), true, &identity));
        std::vector<llama_cache_acct_op_id> ops;
        CHECK(authority.admit_live_checkpoint(
            authority.retention.artifact_id(key),
            checkpoint.data_tgt.size(), 0, ops));
        CHECK(authority.retention.attach_release_ops(key, std::move(ops)));
    }
    const auto pinned_key = server_retention_instance_key::for_checkpoint(
        17, &ring.back());
    auto pin = authority.retention.acquire_recovery_pin(pinned_key);
    CHECK(pin.valid());

    server_cache_checkpoint_attempt_latch attempts;
    const common_prompt_checkpoint * seam = nullptr;
    common_cache_plan_destruction_reason thin_reason =
        common_cache_plan_destruction_reason::none;
    common_cache_plan_destruction_reason floor_reason =
        common_cache_plan_destruction_reason::none;
    server_cache_checkpoint_authority_context context {
        17,
        ring,
        &authority,
        &authority.retention,
        &authority.destruction,
        &authority.leases,
        attempts,
        seam,
        thin_reason,
        floor_reason,
        false,
        {},
        false,
        nullptr,
        [](void *,
           server_cache_checkpoint_authority_context::checkpoint_iterator first,
           server_cache_checkpoint_authority_context::checkpoint_iterator) {
            return first;
        },
    };
    CHECK(!server_cache_checkpoint_thin_priced(
        context, -99, 100, nullptr, false));
    CHECK(ring.size() == 2);
    CHECK(thin_reason ==
          common_cache_plan_destruction_reason::mandatory_anchor);
    std::printf(
        "E1_TWO_COPIES thin_lane_pinned_member PASS members=%zu pinned=%d\n",
        ring.size(), authority.retention.recovery_pinned(pinned_key) ? 1 : 0);
    pin = {};
    authority.retention.retire_slot(17);
}

void test_checkpoint_capacity_floor() {
    server_cache_checkpoint_floor_input unprotected;
    unprotected.ordinal = 2;
    server_cache_checkpoint_floor_input heuristic;
    heuristic.ordinal = 3;
    heuristic.protection =
        server_cache_checkpoint_protection::seam_heuristic;
    server_cache_checkpoint_floor_input mandatory;
    mandatory.ordinal = 0;
    mandatory.protection =
        server_cache_checkpoint_protection::mandatory_anchor;
    server_cache_checkpoint_floor_input hard;
    hard.ordinal = 1;
    hard.protection = server_cache_checkpoint_protection::hard_lease;
    server_cache_checkpoint_floor_input pinned;
    pinned.ordinal = 4;
    pinned.recovery_pinned = true;

    auto plan = server_cache_plan_checkpoint_capacity_floor(
        { pinned, mandatory, hard, unprotected, heuristic });
    CHECK(plan.selected && plan.ordinal == unprotected.ordinal);

    plan = server_cache_plan_checkpoint_capacity_floor(
        { mandatory, hard, heuristic });
    CHECK(plan.selected && plan.ordinal == heuristic.ordinal);

    heuristic.recovery_pinned = true;
    plan = server_cache_plan_checkpoint_capacity_floor(
        { mandatory, hard, heuristic });
    CHECK(!plan.selected);
    CHECK(plan.reason ==
          common_cache_plan_destruction_reason::hard_lease_blocked);
}

void test_checkpoint_attempt_latch_rearms_on_ring_change() {
    server_cache_checkpoint_attempt_latch latch;
    uint64_t full_computations = 0;
    uint64_t receipts = 0;

    // Repeated publication attempts against one protected ring generation
    // perform and report the expensive optional-thinning pass exactly once.
    for (int i = 0; i < 8; ++i) {
        if (latch.begin(
                server_cache_checkpoint_attempt_lane::optional_thinning)) {
            full_computations++;
            if (latch.refusal_changed(
                    common_cache_plan_destruction_reason::mandatory_anchor)) {
                receipts++;
            }
        }
    }
    CHECK(full_computations == 1);
    CHECK(receipts == 1);

    // Capacity pricing and its protected-member floor are independent lanes,
    // but each is likewise single-shot for the same membership generation.
    CHECK(latch.begin(
        server_cache_checkpoint_attempt_lane::capacity_thinning));
    CHECK(!latch.begin(
        server_cache_checkpoint_attempt_lane::capacity_thinning));
    CHECK(latch.begin(server_cache_checkpoint_attempt_lane::capacity_floor));
    CHECK(!latch.begin(
        server_cache_checkpoint_attempt_lane::capacity_floor));

    // A committed member erase/publication is the only re-arm: computation
    // and evidence both become observable again for the new ring.
    latch.ring_changed();
    if (latch.begin(
            server_cache_checkpoint_attempt_lane::optional_thinning)) {
        full_computations++;
        if (latch.refusal_changed(
                common_cache_plan_destruction_reason::mandatory_anchor)) {
            receipts++;
        }
    }
    CHECK(full_computations == 2);
    CHECK(receipts == 2);
    CHECK(latch.begin(
        server_cache_checkpoint_attempt_lane::capacity_thinning));
    CHECK(latch.begin(server_cache_checkpoint_attempt_lane::capacity_floor));
}

void test_checkpoint_effect_matrix_consistency() {
    server_cache_authority authority;
    common_cache_plan_destruction_receipt receipt;
    receipt.state = common_cache_plan_destruction_state::executed;
    receipt.reason = common_cache_plan_destruction_reason::none;
    receipt.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::checkpoint_member_drop);
    receipt.actual_accounting_serial = 1;
    authority.observe_host_destruction(receipt, true);
    authority.destruction.note_checkpoint_thin_executed(0, 64);
    CHECK(authority.destruction_counters.executed
        [size_t(common_cache_plan_selection::none)]
        [size_t(common_cache_plan_destruction_class::checkpoint_drop)] == 1);
    CHECK(authority.destruction.checkpoint_thin_executed == 1);
}

void test_live_checkpoint_payload_ownership() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    const auto * checkpoint =
        reinterpret_cast<const common_prompt_checkpoint *>(uintptr_t(0x1234));
    const auto live =
        server_retention_instance_key::for_checkpoint(3, checkpoint);
    common_chat_msg_spans spans;
    CHECK(authority.retention.publish(
        live, common_retention_pool::recurrent, spans,
        false, 16, 8, true));
    const auto artifact = authority.retention.artifact_id(live);
    std::vector<llama_cache_acct_op_id> ops;
    CHECK(authority.admit_live_checkpoint(artifact, 64, 16, ops));
    CHECK(ops.size() == 2);
    CHECK(authority.retention.attach_release_ops(live, ops));
    server_retention_candidate candidate;
    CHECK(authority.retention.candidate_for_instance(live, candidate));
    CHECK(candidate.release_ops == ops);
    const auto provenance_op = candidate.provenance_op;
    CHECK(provenance_op);

    // Host copies remain aggregate-owned: clone the retention record but not
    // the live member's independently releasable operation set.
    const auto host =
        server_retention_instance_key::for_checkpoint(-1, checkpoint);
    CHECK(authority.retention.clone(live, host));
    CHECK(authority.retention.candidate_for_instance(host, candidate));
    CHECK(candidate.release_ops.empty());
    authority.retention.retire(host);
    auto committed_ops = ops;
    committed_ops.push_back(provenance_op);
    auto prepared = llama_cache_prepare_release_set(
        authority.ledger, committed_ops,
        authority.ledger.snapshot().serial);
    CHECK(prepared.ready());
    CHECK(prepared.commit() ==
          llama_cache_conditional_release_status::released);
    CHECK(!authority.retention.retire_slot_after_committed_release(
        3, { { artifact.v + 2 } }, {}));
    CHECK(authority.retention.candidate_for_instance(live, candidate));
    CHECK(authority.retention.retire_slot_after_committed_release(
        3, {}, { artifact }));
    CHECK(!authority.retention.candidate_for_instance(live, candidate));
    CHECK(authority.ledger.snapshot().live_ops == 0);
}

void test_live_checkpoint_batch_admission() {
    server_cache_authority authority;
    configure_host_accounting(authority, true);
    std::vector<uint64_t> sequential_samples;
    std::vector<uint64_t> batch_samples;
    sequential_samples.reserve(21);
    batch_samples.reserve(21);
    for (uint64_t trial = 0; trial < 21; ++trial) {
        std::vector<llama_cache_acct_op_id> all_ops;
        const auto begin = std::chrono::steady_clock::now();
        for (uint64_t member = 0; member < 8; ++member) {
            std::vector<llama_cache_acct_op_id> ops;
            CHECK(authority.admit_live_checkpoint(
                { 1000 + trial * 8 + member }, 64 * 1024, 4 * 1024, ops));
            all_ops.insert(all_ops.end(), ops.begin(), ops.end());
        }
        const auto end = std::chrono::steady_clock::now();
        sequential_samples.push_back(uint64_t(std::chrono::duration_cast<
            std::chrono::nanoseconds>(end - begin).count()));
        for (const auto op : all_ops) {
            CHECK(authority.ledger.release(op));
        }

        std::vector<server_cache_live_checkpoint_admission> batch(8);
        for (uint64_t member = 0; member < batch.size(); ++member) {
            batch[member].artifact = { 2000 + trial * 8 + member };
            batch[member].checkpoint_bytes = 64 * 1024;
            batch[member].accelerator_bytes = 4 * 1024;
        }
        const uint64_t commits_before = authority.admission_commits;
        const auto batch_begin = std::chrono::steady_clock::now();
        CHECK(authority.admit_live_checkpoints(batch));
        const auto batch_end = std::chrono::steady_clock::now();
        CHECK(authority.admission_commits ==
              commits_before + batch.size());
        batch_samples.push_back(uint64_t(std::chrono::duration_cast<
            std::chrono::nanoseconds>(batch_end - batch_begin).count()));
        for (const auto & member : batch) {
            CHECK(member.committed.size() == 2);
            for (const auto op : member.committed) {
                CHECK(authority.ledger.release(op));
            }
        }
    }
    std::sort(sequential_samples.begin(), sequential_samples.end());
    std::sort(batch_samples.begin(), batch_samples.end());
    std::fprintf(stderr,
        "CHECKPOINT_ADMIT_TIMING members=8 sequential_median_ns=%" PRIu64
        " batch_median_ns=%" PRIu64 "\n",
        sequential_samples[sequential_samples.size() / 2],
        batch_samples[batch_samples.size() / 2]);

    // One invalid member refuses the whole transaction. No sibling receives
    // an operation, and the ledger remains at its pre-batch baseline.
    const auto before = authority.ledger.snapshot();
    std::vector<server_cache_live_checkpoint_admission> invalid(2);
    invalid[0].artifact = { 9001 };
    invalid[0].checkpoint_bytes = 64;
    invalid[1].artifact = {};
    invalid[1].checkpoint_bytes = 64;
    CHECK(!authority.admit_live_checkpoints(invalid));
    CHECK(invalid[0].committed.empty());
    CHECK(invalid[1].committed.empty());
    CHECK(authority.ledger.snapshot().live_ops == before.live_ops);

    server_cache_authority unavailable;
    std::vector<server_cache_live_checkpoint_admission> refused(2);
    refused[0].artifact = { 9101 };
    refused[0].checkpoint_bytes = 64;
    refused[1].artifact = { 9102 };
    refused[1].checkpoint_bytes = 64;
    CHECK(!unavailable.admit_live_checkpoints(refused));
    CHECK(refused[0].committed.empty());
    CHECK(refused[1].committed.empty());
    CHECK(unavailable.ledger.snapshot().live_ops == 0);
}

} // namespace

int main(int argc, char ** argv) {
    llama_backend_init();
    if (argc == 2 && std::string(argv[1]) == "--clone-fault") {
        test_lifecycle_restore_clone_fault();
        llama_backend_free();
        if (failures == 0) {
            std::puts("test-server-prompt-cache: CLONE_FAULT_PASS");
        }
        return failures == 0 ? 0 : 1;
    }
    test_lifecycle_full_cache_rotates();
    test_declared_family_round_trip_and_price();
    test_checkpoint_lineage_ignores_retier_but_rejects_content_change();
    test_checkpoint_suffix_trim_rebases_only_preserved_prefixes();
    test_lifecycle_restore_retains_immutable_source();
    test_implicit_soft_append_chain_is_bounded();
    test_durable_recovery_binds_exact_published_peer();
    test_unlaunched_disarm_releases_recovery_pin();
    test_displacement_save_order_preserves_prefix_recovery();
    test_lifecycle_off_restore_consumes();
    test_lifecycle_restore_batch_timing();
    test_checkpoint_creation_churn_timing();
    test_checkpoint_bounded_publication_skip_predicate();
    test_consuming_rebind_mints_checkpoint_ownership();
    test_lifecycle_release_prepare_failure_keeps_legacy_bound();
    test_lifecycle_authority_without_debug_is_silent();
    test_authority_source_ids_survive_save_dedup();
    test_exact_redundant_host_eviction();
    test_redundancy_payload_mismatch_and_missing_catalog();
    test_host_trade_soft_lease_weight_flips_victim();
    test_host_trade_main_family_weight_flips_victim();
    test_host_trade_zero_destruction_tie_break();
    test_host_trade_all_refuse_falls_back_to_legacy();
    test_host_trade_hard_lease_veto();
    test_host_trade_all_hard_skips_publication();
    test_host_trade_floor_skips_recovery_pin();
    test_cache_control_shutdown_drains_host_pin();
    test_host_trade_partial_substrate_is_typed();
    test_checkpoint_thinning_policy();
    test_checkpoint_thin_lane_skips_pinned_member();
    test_checkpoint_capacity_floor();
    test_checkpoint_attempt_latch_rearms_on_ring_change();
    test_checkpoint_effect_matrix_consistency();
    test_live_checkpoint_payload_ownership();
    test_live_checkpoint_batch_admission();
    llama_backend_free();

    if (failures != 0) {
        std::fprintf(stderr, "test-server-prompt-cache: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("test-server-prompt-cache: PASS");
    return 0;
}
