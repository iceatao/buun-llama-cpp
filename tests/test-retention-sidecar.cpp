#include "common-retention-sidecar.h"
#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"
#include "server-retention-sidecar.h"
#include "llama-cache-authority.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <list>
#include <string>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

class retention_test_clock final : public server_cache_lease_clock {
public:
    uint64_t now_ns() noexcept override {
        return now++;
    }

private:
    uint64_t now = 1;
};

static std::string to_hex(const std::vector<uint8_t> & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

static common_chat_msg_spans make_spans() {
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER,      0,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT, 4,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     10,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT,14,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     20,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT,24,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     30,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT,34,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     40,  4);
    return spans;
}

static void test_turn_table_and_geometry() {
    common_retention_turn_table turns;
    CHECK(common_retention_build_turn_table(make_spans(), true, 44, turns));
    CHECK(turns.valid());
    CHECK(turns.boundaries.size() == 5);
    CHECK(turns.boundaries.front().token_pos == 0);
    CHECK(turns.boundaries.front().token_end == 4);

    common_retention_stamp stamp;
    stamp.stable_id = 1;
    stamp.lineage_id = 1;
    stamp.recency_ordinal = 1;
    stamp.coverage_tokens = 30;
    CHECK(common_retention_score(turns, 30, stamp));
    CHECK(stamp.state == common_retention_score_state::known);
    CHECK(stamp.mapped_turn_ordinal == 3);
    CHECK(stamp.anchor_rank == 3);
    CHECK(!stamp.mandatory_anchor);

    stamp.coverage_tokens = 40;
    CHECK(common_retention_score(turns, 40, stamp));
    CHECK(stamp.mapped_turn_ordinal == 4);
    CHECK(stamp.mandatory_anchor);

    common_retention_turn_table unavailable;
    common_chat_msg_spans missing;
    CHECK(common_retention_build_turn_table(missing, false, 44, unavailable));
    CHECK(unavailable.valid());
    CHECK(unavailable.source == common_retention_source_state::unavailable);

    auto malformed = make_spans();
    malformed.spans[2].pos = 3;
    CHECK(!common_retention_build_turn_table(malformed, true, 44, unavailable));
    malformed = make_spans();
    malformed.spans[2].role = COMMON_CHAT_ROLE_UNKNOWN;
    CHECK(!common_retention_build_turn_table(malformed, true, 44, unavailable));

    // Degenerate geometry is closed and deterministic: head-only and newest are
    // mandatory, while n=2 has exactly one optional geometric anchor.
    for (size_t n_user : { size_t(1), size_t(2), size_t(3) }) {
        common_chat_msg_spans small;
        for (size_t i = 0; i < n_user; ++i) {
            small.add(COMMON_CHAT_ROLE_USER, i*10, 2);
        }
        common_retention_turn_table table;
        CHECK(common_retention_build_turn_table(
            small, true, n_user*10, table));
        common_retention_stamp cur;
        cur.stable_id = 1;
        cur.lineage_id = 1;
        cur.recency_ordinal = 1;
        cur.coverage_tokens = n_user > 1 ? (n_user - 1)*10 : 0;
        CHECK(common_retention_score(table, cur.coverage_tokens, cur));
        CHECK(cur.mandatory_anchor);
        if (n_user == 3) {
            cur.coverage_tokens = 10;
            CHECK(common_retention_score(table, 10, cur));
            CHECK(!cur.mandatory_anchor);
            CHECK(cur.anchor_rank == 2);
        }
    }
}

static common_retention_sidecar_snapshot make_snapshot() {
    common_retention_sidecar_snapshot snapshot;
    common_retention_artifact_record record;
    record.kind = common_retention_artifact_kind::checkpoint;
    common_retention_turn_table turns;
    CHECK(common_retention_build_turn_table(make_spans(), true, 44, turns));
    record.turns = std::make_shared<const common_retention_turn_table>(
        std::move(turns));
    record.stamp.pool = common_retention_pool::recurrent;
    record.stamp.stable_id = 7;
    record.stamp.lineage_id = 5;
    record.stamp.recency_ordinal = 9;
    record.stamp.coverage_tokens = 30;
    CHECK(common_retention_score(*record.turns, 30, record.stamp));
    snapshot.stable_high_water[1] = 7;
    snapshot.recency_high_water[1] = 9;
    snapshot.lineage_high_water[1] = 5;
    common_retention_lineage_record lineage;
    lineage.pool = common_retention_pool::recurrent;
    lineage.lineage_id = 5;
    lineage.admission_epoch = snapshot.competition_epoch;
    lineage.frequency_epoch = snapshot.competition_epoch;
    snapshot.lineages.push_back(lineage);
    snapshot.artifacts.push_back(record);
    return snapshot;
}

static void test_codec() {
    const auto snapshot = make_snapshot();
    CHECK(snapshot.valid());

    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    CHECK(common_retention_sidecar_encode(snapshot, a));
    CHECK(common_retention_sidecar_encode(snapshot, b));
    CHECK(a == b);
    uint64_t arithmetic_size = 0;
    CHECK(common_retention_sidecar_artifact_encoded_size(
        snapshot.artifacts.front(), arithmetic_size));
    CHECK(arithmetic_size == a.size());

    common_retention_sidecar_snapshot decoded;
    CHECK(common_retention_sidecar_decode(a.data(), a.size(), decoded));
    CHECK(decoded.valid());
    CHECK(decoded.artifacts.size() == 1);
    CHECK(decoded.artifacts[0].stamp.stable_id == 7);
    CHECK(decoded.artifacts[0].stamp.lineage_id == 5);
    CHECK(decoded.lineages.size() == 1);
    CHECK(decoded.artifacts[0].turns->boundaries.size() == 5);

    auto corrupt = a;
    corrupt.back() ^= 1;
    CHECK(!common_retention_sidecar_decode(corrupt.data(), corrupt.size(), decoded));
    CHECK(decoded.version == 0 && decoded.artifacts.empty());
    auto bad_version = a;
    bad_version[4] = 3;
    CHECK(!common_retention_sidecar_decode(
        bad_version.data(), bad_version.size(), decoded));
    auto bad_length = a;
    bad_length[8] ^= 1;
    CHECK(!common_retention_sidecar_decode(
        bad_length.data(), bad_length.size(), decoded));
    CHECK(!common_retention_sidecar_decode(a.data(), a.size() - 1, decoded));
    auto trailing = a;
    trailing.push_back(0);
    CHECK(!common_retention_sidecar_decode(
        trailing.data(), trailing.size(), decoded));
    auto too_many = snapshot;
    too_many.artifacts.resize(8193, snapshot.artifacts.front());
    CHECK(!common_retention_sidecar_encode(too_many, corrupt));
    auto missing_lineage = snapshot;
    missing_lineage.lineages.clear();
    CHECK(!common_retention_sidecar_encode(missing_lineage, corrupt));
    auto duplicate_lineage = snapshot;
    duplicate_lineage.lineages.push_back(
        duplicate_lineage.lineages.front());
    CHECK(!common_retention_sidecar_encode(duplicate_lineage, corrupt));
    auto future_currency = snapshot;
    future_currency.lineages.front().frequency_epoch =
        future_currency.competition_epoch + 1;
    CHECK(!common_retention_sidecar_encode(future_currency, corrupt));
    auto impossible_credit = snapshot;
    impossible_credit.lineages.front().reuse_hits = 1;
    CHECK(!common_retention_sidecar_encode(impossible_credit, corrupt));

    // Golden locks the complete canonical envelope, not just a decoded field.
    static const char * golden =
        "523353440200000064010000000000008b18cdc15e55bc638717e6b96f42096d521ff4883443e418eaba3906eff33f0d0000000000000000090000000000000000000000000000000700000000000000000000000000000005000000000000000100000000000000010000000101050000000000000000000000000000000000000000000000010000000000000001000000000000000000000000000000e803000001000000020000010000010000002c00000000000000070000000000000005000000000000000900000000000000030000000000000003000000000000001e000000000000000500000000000000000000000000000000000000040000000000000001000000000000000a000000000000000e0000000000000002000000000000001400000000000000180000000000000003000000000000001e000000000000002200000000000000040000000000000028000000000000002c00000000000000";
    CHECK(to_hex(a) == golden);
}

static void test_store_and_allocator_import() {
    common_retention_allocator source;
    common_retention_stamp first;
    CHECK(source.issue(common_retention_pool::recurrent, first));
    CHECK(first.stable_id == 1 && first.recency_ordinal == 1);

    auto exported = make_snapshot();
    exported.stable_high_water[1] = 17;
    exported.recency_high_water[1] = 29;
    common_retention_allocator resumed;
    common_retention_allocator resumed_again;
    CHECK(resumed.import_snapshot(exported));
    CHECK(resumed_again.import_snapshot(exported));
    common_retention_stamp next;
    common_retention_stamp next_again;
    CHECK(resumed.issue(common_retention_pool::recurrent, next));
    CHECK(resumed_again.issue(common_retention_pool::recurrent, next_again));
    CHECK(next.stable_id == 18);
    CHECK(next.recency_ordinal == 30);
    CHECK(next.stable_id == next_again.stable_id);
    CHECK(next.recency_ordinal == next_again.recency_ordinal);
    CHECK(resumed.stable_high_water(common_retention_pool::recurrent) == 18);
    CHECK(resumed.recency_high_water(common_retention_pool::recurrent) == 30);
    common_retention_stamp attention;
    CHECK(resumed.issue(common_retention_pool::attention, attention));
    CHECK(attention.stable_id == 1 && attention.recency_ordinal == 1);
}

static void test_frequency_and_lineage_transitions() {
    common_retention_frequency_config config;
    common_retention_lineage_record lineage;
    lineage.lineage_id = 1;
    lineage.admission_epoch = 1;
    lineage.frequency_epoch = 1;
    CHECK(lineage.valid(1));
    CHECK(common_retention_frequency_credit(lineage, 1, config) ==
          common_retention_credit_result::credited);
    CHECK(lineage.reuse_hits == 1);
    CHECK(lineage.state == common_retention_frequency_state::probation);
    CHECK(common_retention_frequency_credit(lineage, 1, config) ==
          common_retention_credit_result::coalesced);
    CHECK(lineage.reuse_hits == 1);
    CHECK(common_retention_frequency_credit(lineage, 2, config) ==
          common_retention_credit_result::credited);
    CHECK(lineage.reuse_hits == 2);
    CHECK(lineage.state == common_retention_frequency_state::promoted);
    const auto hot_q = lineage.frequency_q;
    CHECK(common_retention_frequency_normalize(lineage, 18, config));
    CHECK(lineage.frequency_q == hot_q/4);

    // Inspection is not reuse and must not slide the decay window. A policy
    // that quotes every competition epoch reaches the same boundary as one
    // lazy jump.
    common_retention_lineage_record inspected = lineage;
    inspected.frequency_q = config.credit_q;
    const uint64_t start_epoch = inspected.frequency_epoch;
    for (uint64_t epoch = start_epoch + 1;
         epoch <= start_epoch + config.decay_interval_epochs; ++epoch) {
        CHECK(common_retention_frequency_normalize(
            inspected, epoch, config));
    }
    CHECK(inspected.frequency_q == config.credit_q/2);
    CHECK(inspected.frequency_epoch ==
          start_epoch + config.decay_interval_epochs);

    server_retention_sidecar_store store;
    store.configure(nullptr, {}, nullptr);
    const auto live = server_retention_instance_key::for_slot(1);
    const auto alias = server_retention_instance_key::for_slot(2);
    CHECK(store.publish(
        live, common_retention_pool::attention, make_spans(), true,
        44, 30, true));
    common_retention_lineage_record live_lineage;
    CHECK(store.lineage_for_instance(live, live_lineage));
    CHECK(store.competition_epoch_value() == 2);
    CHECK(live_lineage.state ==
          common_retention_frequency_state::probation);
    CHECK(store.set_lineage_prior(live, 2000));
    CHECK(store.clone(live, alias));
    common_retention_lineage_record alias_lineage;
    CHECK(store.lineage_for_instance(alias, alias_lineage));
    CHECK(alias_lineage.lineage_id == live_lineage.lineage_id);
    CHECK(alias_lineage.prior_milli == 2000);

    CHECK(store.credit_reuse(live) ==
          common_retention_credit_result::credited);
    CHECK(store.credit_reuse(alias) ==
          common_retention_credit_result::coalesced);
    CHECK(store.begin_competition_wave());
    CHECK(store.credit_reuse(alias) ==
          common_retention_credit_result::credited);
    CHECK(store.lineage_for_instance(live, live_lineage));
    CHECK(store.lineage_for_instance(alias, alias_lineage));
    CHECK(live_lineage.reuse_hits == 2);
    CHECK(live_lineage.state ==
          common_retention_frequency_state::promoted);
    CHECK(alias_lineage.reuse_hits == live_lineage.reuse_hits);

    const auto branch = server_retention_instance_key::for_slot(9);
    CHECK(store.branch(live, branch));
    common_retention_lineage_record branch_lineage;
    CHECK(store.lineage_for_instance(branch, branch_lineage));
    CHECK(branch_lineage.lineage_id != live_lineage.lineage_id);
    CHECK(branch_lineage.reuse_hits == 0);
    CHECK(branch_lineage.prior_milli == 1000);
    CHECK(branch_lineage.state ==
          common_retention_frequency_state::probation);

    // Same-key append supersession preserves the one lineage ledger.
    CHECK(store.publish(
        live, common_retention_pool::attention, make_spans(), true,
        44, 40, true));
    CHECK(store.lineage_for_instance(live, live_lineage));
    CHECK(live_lineage.lineage_id == alias_lineage.lineage_id);
    CHECK(live_lineage.reuse_hits == 2);

    // A distinct admitted lineage advances competition but never inherits
    // source credit.
    const auto unrelated = server_retention_instance_key::for_slot(3);
    CHECK(store.publish(
        unrelated, common_retention_pool::attention, make_spans(), true,
        44, 30, true));
    common_retention_lineage_record unrelated_lineage;
    CHECK(store.lineage_for_instance(unrelated, unrelated_lineage));
    CHECK(unrelated_lineage.lineage_id != live_lineage.lineage_id);
    CHECK(unrelated_lineage.reuse_hits == 0);
    CHECK(unrelated_lineage.state ==
          common_retention_frequency_state::probation);
}

static void test_scan_flood_and_phase_change_simulator() {
    server_retention_sidecar_store store;
    store.configure(nullptr, {}, nullptr);
    const auto main = server_retention_instance_key::for_slot(0);
    CHECK(store.publish(
        main, common_retention_pool::attention, make_spans(), true,
        44, 40, true));

    // Credit is ordered before branch admission. Hundreds of branches create
    // independent probationary lineages and cannot copy the source ledger.
    for (int32_t i = 1; i <= 256; ++i) {
        CHECK(store.credit_reuse(main) !=
              common_retention_credit_result::unavailable);
        const auto branch = server_retention_instance_key::for_slot(i);
        CHECK(store.publish(
            branch, common_retention_pool::attention, make_spans(), true,
            44, 30, true));
        common_retention_lineage_record branch_lineage;
        CHECK(store.lineage_for_instance(branch, branch_lineage));
        CHECK(branch_lineage.reuse_hits == 0);
        CHECK(branch_lineage.state ==
              common_retention_frequency_state::probation);
    }
    common_retention_lineage_record learned_main;
    CHECK(store.lineage_for_instance(main, learned_main));
    CHECK(learned_main.state ==
          common_retention_frequency_state::promoted);
    CHECK(learned_main.reuse_hits == 256);
    const auto learned_q = learned_main.frequency_q;

    // A complete phase change admits fresh lineages without reusing the old
    // main. Lazy normalization must age it without a timer or full scan.
    for (int32_t i = 257; i <= 384; ++i) {
        const auto fresh = server_retention_instance_key::for_slot(i);
        CHECK(store.publish(
            fresh, common_retention_pool::attention, make_spans(), true,
            44, 30, true));
    }
    common_retention_frequency_config config;
    CHECK(common_retention_frequency_normalize(
        learned_main, store.competition_epoch_value(), config));
    CHECK(learned_main.frequency_q < learned_q);

    // Exact appends supersede the prior physical record. A long-running chat
    // remains one lineage and one ranked catalog member instead of growing
    // one candidate per turn.
    server_retention_sidecar_store appends;
    appends.configure(nullptr, {}, nullptr);
    const auto append_key = server_retention_instance_key::for_slot(700);
    for (uint64_t i = 1; i <= 1000; ++i) {
        CHECK(appends.publish(
            append_key, common_retention_pool::attention,
            make_spans(), true, 44, 30 + i%11, true));
    }
    const auto append_snapshot = appends.snapshot();
    CHECK(append_snapshot.valid());
    CHECK(append_snapshot.lineages.size() == 1);
    CHECK(append_snapshot.artifacts.size() == 1);
    CHECK(appends.candidate_snapshot().size() == 1);
}

static void test_shared_geometry_and_scheduler_lineage_receipts() {
    common_chat_msg_spans large;
    for (size_t i = 0; i + 1 <
            COMMON_RETENTION_MAX_TURN_BOUNDARIES; ++i) {
        large.add(COMMON_CHAT_ROLE_USER, int32_t(i), 1);
    }
    common_chat_msg_spans too_large = large;
    too_large.add(COMMON_CHAT_ROLE_USER,
                  int32_t(COMMON_RETENTION_MAX_TURN_BOUNDARIES - 1), 1);
    too_large.add(COMMON_CHAT_ROLE_USER,
                  int32_t(COMMON_RETENTION_MAX_TURN_BOUNDARIES), 1);
    common_retention_turn_table rejected;
    CHECK(!common_retention_build_turn_table(
        too_large, true, COMMON_RETENTION_MAX_TURN_BOUNDARIES + 1, rejected));
    CHECK(rejected.boundaries.empty());

    // Capacity, rather than merely vector size, is the resident metadata
    // charge. Non-user spans must not reserve a hidden 8K-boundary vector;
    // sparse user geometry reserves exactly the derived table.
    common_chat_msg_spans no_users;
    common_chat_msg_spans one_user;
    for (size_t i = 0; i + 1 <
            COMMON_RETENTION_MAX_TURN_BOUNDARIES; ++i) {
        no_users.add(COMMON_CHAT_ROLE_ASSISTANT, int32_t(i), 1);
        one_user.add(
            i == COMMON_RETENTION_MAX_TURN_BOUNDARIES/2
                ? COMMON_CHAT_ROLE_USER : COMMON_CHAT_ROLE_ASSISTANT,
            int32_t(i), 1);
    }
    common_retention_turn_table no_user_table;
    common_retention_turn_table one_user_table;
    CHECK(common_retention_build_turn_table(
        no_users, true, COMMON_RETENTION_MAX_TURN_BOUNDARIES,
        no_user_table));
    CHECK(no_user_table.boundaries.empty());
    CHECK(no_user_table.boundaries.capacity() == 0);
    CHECK(common_retention_build_turn_table(
        one_user, true, COMMON_RETENTION_MAX_TURN_BOUNDARIES,
        one_user_table));
    CHECK(one_user_table.boundaries.size() == 2);
    CHECK(one_user_table.boundaries.capacity() ==
          one_user_table.boundaries.size());

    // Raw chat-span cardinality is not the geometry cap. Large tool/assistant
    // histories with zero or one sparse user turn remain representable; only
    // the derived user-boundary table is bounded.
    common_chat_msg_spans many_non_user;
    common_chat_msg_spans many_sparse_user;
    constexpr size_t MANY_SPANS =
        COMMON_RETENTION_MAX_TURN_BOUNDARIES + 1024;
    for (size_t i = 0; i < MANY_SPANS; ++i) {
        many_non_user.add(COMMON_CHAT_ROLE_TOOL, int32_t(i), 1);
        many_sparse_user.add(
            i == MANY_SPANS/2
                ? COMMON_CHAT_ROLE_USER : COMMON_CHAT_ROLE_TOOL,
            int32_t(i), 1);
    }
    common_retention_turn_table many_non_user_table;
    common_retention_turn_table many_sparse_user_table;
    CHECK(common_retention_build_turn_table(
        many_non_user, true, MANY_SPANS, many_non_user_table));
    CHECK(many_non_user_table.boundaries.empty());
    CHECK(many_non_user_table.boundaries.capacity() == 0);
    CHECK(common_retention_build_turn_table(
        many_sparse_user, true, MANY_SPANS, many_sparse_user_table));
    CHECK(many_sparse_user_table.boundaries.size() == 2);
    CHECK(many_sparse_user_table.boundaries.capacity() == 2);

    server_retention_sidecar_store store;
    store.configure(nullptr, {}, nullptr);
    std::list<common_prompt_checkpoint> checkpoints(32);
    auto first = checkpoints.begin();
    const auto first_key =
        server_retention_instance_key::for_checkpoint(4, &*first);
    CHECK(store.publish(
        first_key, common_retention_pool::attention, large, true,
        COMMON_RETENTION_MAX_TURN_BOUNDARIES - 1,
        COMMON_RETENTION_MAX_TURN_BOUNDARIES - 1, true));
    server_retention_lineage_ticket destination;
    CHECK(store.acquire_lineage_ticket(first_key, destination));
    server_retention_candidate first_candidate;
    CHECK(store.candidate_for_instance(first_key, first_candidate));

    for (auto it = std::next(first); it != checkpoints.end(); ++it) {
        CHECK(store.publish(
            server_retention_instance_key::for_checkpoint(4, &*it),
            common_retention_pool::attention, large, true,
            COMMON_RETENTION_MAX_TURN_BOUNDARIES - 1,
            COMMON_RETENTION_MAX_TURN_BOUNDARIES - 1, true,
            nullptr, nullptr, nullptr, &destination));
    }
    const auto live = server_retention_instance_key::for_slot(4);
    CHECK(store.publish(
        live, common_retention_pool::attention, large, true,
        COMMON_RETENTION_MAX_TURN_BOUNDARIES - 1,
        COMMON_RETENTION_MAX_TURN_BOUNDARIES - 1, true,
        nullptr, nullptr, nullptr, &destination));
    server_retention_candidate live_candidate;
    CHECK(store.candidate_for_instance(live, live_candidate));
    CHECK(live_candidate.record.turns.get() ==
          first_candidate.record.turns.get());
    CHECK(store.competition_epoch_value() == 2);
    CHECK(store.snapshot().lineages.size() == 1);
    CHECK(store.live_bytes() < 512*1024);
    store.release_lineage_ticket(destination);

    // A destructive divergent trim may retire the physical source while its
    // scheduler ticket survives to the success boundary and receives credit.
    server_retention_sidecar_store divergent;
    divergent.configure(nullptr, {}, nullptr);
    const auto source = server_retention_instance_key::for_slot(8);
    CHECK(divergent.publish(
        source, common_retention_pool::attention, make_spans(), true,
        44, 30, true));
    server_retention_lineage_ticket reused;
    CHECK(divergent.acquire_lineage_ticket(source, reused));
    divergent.retire(source);
    CHECK(divergent.credit_reuse(reused) ==
          common_retention_credit_result::credited);
    const auto child = server_retention_instance_key::for_slot(9);
    CHECK(divergent.publish(
        child, common_retention_pool::attention, make_spans(), true,
        44, 32, true));
    const auto after = divergent.snapshot();
    CHECK(after.lineages.size() == 2);
    const auto source_lineage = std::find_if(
        after.lineages.begin(), after.lineages.end(),
        [&](const auto & value) {
            return value.lineage_id == reused.lineage_id;
        });
    CHECK(source_lineage != after.lineages.end());
    CHECK(source_lineage->reuse_hits == 1);
    divergent.release_lineage_ticket(reused);

    // A host restore publishes its prepared live branch once. Launch consumes
    // the receipt and the final same-key publication preserves that lineage.
    server_retention_sidecar_store restored;
    restored.configure(nullptr, {}, nullptr);
    const auto * host_state = reinterpret_cast<const server_prompt_cache_state *>(
        uintptr_t(0x1234));
    const auto host =
        server_retention_instance_key::for_host_entry(host_state);
    const auto restored_live = server_retention_instance_key::for_slot(10);
    CHECK(restored.publish(
        host, common_retention_pool::attention, make_spans(), true,
        44, 30, true));
    CHECK(restored.branch(host, restored_live, nullptr, true));
    const uint64_t source_epoch = restored.competition_epoch_value();
    common_retention_lineage_record branch_before;
    CHECK(!restored.lineage_for_instance(restored_live, branch_before));
    CHECK(restored.prepare_for_launch(host, restored_live));
    CHECK(restored.prepared_for_launch(restored_live));
    server_retention_lineage_ticket restored_source;
    CHECK(restored.consume_prepared_launch(
        restored_live, restored_source));
    CHECK(!restored.prepared_for_launch(restored_live));
    CHECK(restored.credit_reuse(restored_source) ==
          common_retention_credit_result::credited);
    restored.release_lineage_ticket(restored_source);
    server_retention_lineage_ticket restored_destination;
    CHECK(restored.acquire_lineage_ticket(
        restored_live, restored_destination));
    CHECK(restored.activate_lineage_ticket(restored_destination));
    restored.release_lineage_ticket(restored_destination);
    CHECK(restored.lineage_for_instance(restored_live, branch_before));
    CHECK(restored.competition_epoch_value() == source_epoch + 1);
    CHECK(restored.publish(
        restored_live, common_retention_pool::attention,
        make_spans(), true, 44, 34, true));
    common_retention_lineage_record branch_after;
    CHECK(restored.lineage_for_instance(restored_live, branch_after));
    CHECK(branch_after.lineage_id == branch_before.lineage_id);
    CHECK(restored.competition_epoch_value() == source_epoch + 1);

    // A prepared branch which never launches carries no credit and leaves no
    // provisional destination/checkpoint aliases behind for a later task.
    const auto abandoned = server_retention_instance_key::for_slot(11);
    CHECK(restored.branch(host, abandoned, nullptr, true));
    CHECK(restored.prepare_for_launch(host, abandoned));
    restored.abandon_prepared_launch(abandoned);
    CHECK(restored.artifact_id(abandoned).v == 0);
    CHECK(!restored.prepared_for_launch(abandoned));
}

static void test_shadow_value_comparator() {
    common_retention_frequency_config config;
    common_retention_lineage_record probation;
    probation.lineage_id = 1;
    probation.admission_epoch = 1;
    probation.frequency_epoch = 1;
    common_retention_lineage_record promoted = probation;
    promoted.lineage_id = 2;
    promoted.state = common_retention_frequency_state::promoted;
    promoted.reuse_hits = 2;
    promoted.frequency_q = 2*COMMON_RETENTION_FREQUENCY_ONE;
    promoted.last_credit_epoch = 1;

    common_retention_shadow_value a;
    common_retention_shadow_value b;
    CHECK(common_retention_shadow_quote(
        probation, 1, 100, 100, 1, config, a));
    CHECK(common_retention_shadow_quote(
        promoted, 1, 100, 100, 2, config, b));
    CHECK(common_retention_shadow_compare(a, b) < 0);

    // Probation is not a blind eviction stratum: a new but very expensive
    // prefix can outrank a frequently reused trivial prefix. The bounded prior
    // gives cold starts a real opportunity without becoming a pin.
    common_retention_shadow_value expensive_new;
    common_retention_shadow_value cheap_hot;
    CHECK(common_retention_shadow_quote(
        probation, 1, 1000, 100, 3, config, expensive_new));
    CHECK(common_retention_shadow_quote(
        promoted, 1, 1, 100, 4, config, cheap_hot));
    CHECK(common_retention_shadow_compare(cheap_hot, expensive_new) < 0);

    // A cold-start prior is bounded help, not an immortal lease. It decays
    // against the same competition currency even before the first reuse.
    common_retention_shadow_value aged;
    CHECK(common_retention_shadow_quote(
        probation, 65, 100, 100, 1, config, aged));
    CHECK(aged.lost_value_q < a.lost_value_q);

    // Compare exact lost-value density and then stable recency/lineage
    // currency. No floating point participates.
    auto same = probation;
    same.lineage_id = 3;
    common_retention_shadow_value roomy;
    common_retention_shadow_value compact;
    CHECK(common_retention_shadow_quote(
        probation, 1, 100, 100, 4, config, roomy));
    CHECK(common_retention_shadow_quote(
        same, 1, 100, 50, 5, config, compact));
    CHECK(common_retention_shadow_compare(roomy, compact) < 0);

    common_retention_shadow_value max_a;
    common_retention_shadow_value max_b;
    max_a.state = max_b.state =
        common_retention_shadow_value_state::known;
    max_a.frequency_state = max_b.frequency_state =
        common_retention_frequency_state::promoted;
    max_a.lineage_id = 1;
    max_b.lineage_id = 2;
    max_a.lost_value_q = UINT64_MAX - 1;
    max_b.lost_value_q = UINT64_MAX;
    max_a.marginal_resource = max_b.marginal_resource = UINT64_MAX;
    max_a.recency_ordinal = 1;
    max_b.recency_ordinal = 2;
    CHECK(common_retention_shadow_compare(max_a, max_b) < 0);

    CHECK(!common_retention_shadow_quote(
        probation, 1, UINT64_MAX, 1, 1, config, a));
    CHECK(!common_retention_shadow_quote(
        probation, 1, 1, 0, 1, config, a));
}

static void test_observer_store_accounting() {
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    llama_cache_acct_ledger ledger;
    const llama_cache_acct_completeness_requirement requirement = {
        domain, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated,
            llama_cache_acct_measure::reserved }) {
        ledger.gauge_set(
            llama_cache_acct_category::artifact_descriptor_metadata,
            domain, measure, 0);
        ledger.gauge_set(
            llama_cache_acct_category::checkpoint_state_payload,
            domain, measure, 0);
    }
    CHECK(ledger.certify_complete(
        domain, llama_cache_acct_producer::retention_sidecar));

    retention_test_clock clock;
    server_cache_lease_table leases(&clock);
    server_retention_sidecar_store store;
    store.configure(&ledger, domain, &leases);
    const server_cache_lease_identity lease_identity = {
        "execution", "adapter", "media",
    };
    const auto live = server_retention_instance_key::for_slot(3);
    CHECK(store.publish(
        live, common_retention_pool::recurrent, make_spans(), true,
        44, 30, true, &lease_identity));
    CHECK(store.artifact_id(live).v != 0);
    const auto live_artifact = store.artifact_id(live);
    const auto live_payload_op = server_cache_acct_charge_shadow(
        ledger,
        llama_cache_acct_category::checkpoint_state_payload,
        domain,
        llama_cache_acct_producer::retention_sidecar,
        { llama_cache_acct_attr_kind::artifact, -1, live_artifact },
        32, 32);
    CHECK(live_payload_op);
    CHECK(store.attach_release_ops(live, { live_payload_op }));
    const auto lease = leases.grant_soft(
        {
            live_artifact,
            common_retention_artifact_kind::live_slot,
            3,
        },
        server_cache_lease_scope::from(
            server_cache_context_scope_id { 1 }),
        lease_identity,
        100);
    CHECK(lease);
    CHECK(store.live_bytes() > 0);
    CHECK(store.unavailable() == 0);
    const auto before_clone = store.live_bytes();

    const auto * checkpoint_ptr =
        reinterpret_cast<const common_prompt_checkpoint *>(uintptr_t(99));
    const auto host_checkpoint =
        server_retention_instance_key::for_checkpoint(-1, checkpoint_ptr);
    CHECK(store.clone(live, host_checkpoint));
    CHECK(store.live_bytes() > before_clone);
    const auto host_id = store.artifact_id(host_checkpoint);
    const auto rebound =
        server_retention_instance_key::for_checkpoint(7, checkpoint_ptr);
    CHECK(store.rebind(host_checkpoint, rebound));
    CHECK(store.artifact_id(rebound) == host_id);
    CHECK(store.artifact_id(host_checkpoint).v == 0);
    server_retention_candidate aggregate_clone;
    CHECK(store.candidate_for_instance(rebound, aggregate_clone));
    CHECK(aggregate_clone.release_ops.empty());
    const auto payload_op = server_cache_acct_charge_shadow(
        ledger,
        llama_cache_acct_category::checkpoint_state_payload,
        domain,
        llama_cache_acct_producer::retention_sidecar,
        { llama_cache_acct_attr_kind::artifact, -1, host_id },
        64, 64);
    CHECK(payload_op);
    CHECK(store.attach_release_ops(rebound, { payload_op }));
    server_retention_checkpoint_inventory inventory;
    CHECK(store.checkpoint_inventory(rebound, inventory));
    CHECK(inventory.artifact_id == host_id);
    CHECK(inventory.identity_known);
    CHECK(inventory.release_owned);
    CHECK(!inventory.recovery_pinned);
    {
        auto pin = store.acquire_recovery_pin(rebound);
        CHECK(pin.valid());
        CHECK(pin.binds_exact(host_id, { payload_op }));
        CHECK(store.checkpoint_inventory(rebound, inventory));
        CHECK(inventory.recovery_pinned);
    }
    CHECK(store.checkpoint_inventory(rebound, inventory));
    CHECK(!inventory.recovery_pinned);

    std::vector<uint8_t> exported;
    CHECK(store.export_bytes(exported));
    common_retention_sidecar_snapshot decoded;
    CHECK(common_retention_sidecar_decode(
        exported.data(), exported.size(), decoded));
    CHECK(decoded.artifacts.size() == 2);
    CHECK(decoded.lineages.size() == 1);
    CHECK(decoded.artifacts[0].stamp.lineage_id ==
          decoded.artifacts[1].stamp.lineage_id);

    const auto candidates = store.candidate_snapshot();
    CHECK(candidates.size() == 2);
    for (const auto & candidate : candidates) {
        CHECK(candidate.artifact_id.v != 0);
        CHECK(candidate.record.valid());
        CHECK(candidate.provenance_op);
        if (candidate.artifact_id == live_artifact) {
            CHECK(candidate.release_ops ==
                  std::vector<llama_cache_acct_op_id> { live_payload_op });
        } else {
            CHECK(candidate.release_ops ==
                  std::vector<llama_cache_acct_op_id> { payload_op });
        }
        CHECK(candidate.avail ==
              server_retention_candidate_availability::available);
        CHECK(store.artifact_id(candidate.instance_key) ==
              candidate.artifact_id);
    }

    server_retention_candidate victim_candidate;
    CHECK(store.candidate_for_instance(rebound, victim_candidate));
    server_cache_destruction_artifact victim;
    victim.candidate.artifact_id = victim_candidate.artifact_id;
    victim.candidate.record = victim_candidate.record;
    victim.candidate.availability = victim_candidate.avail;
    victim.candidate.release_ops = victim_candidate.release_ops;
    victim.candidate.identity_known = true;
    victim.candidate.lease = {
        server_cache_lease_eval_state::known,
        server_cache_lease_class::none,
        server_cache_lease_eligibility::eligible,
    };
    victim.kind = common_retention_artifact_kind::checkpoint;
    victim.pool = victim_candidate.record.stamp.pool;
    const auto preview = [&](const auto & cited, uint64_t serial, auto & out) {
        return ledger.preview_release_set(cited, serial, out);
    };
    const auto project = [](const auto & released, auto & out) {
        out.clear();
        for (const auto & row : released.rows) {
            common_cache_plan_yield_domain domain_row;
            domain_row.domain = row.domain;
            domain_row.current_resident_bytes =
                llama_cache_acct_value::measured(row.resident_allocated);
            domain_row.fit_before_bytes =
                domain_row.current_resident_bytes;
            domain_row.projected_release_bytes =
                llama_cache_acct_value::measured(row.resident_allocated);
            domain_row.projected_reserve_bytes =
                llama_cache_acct_value::measured(0);
            domain_row.projected_after_bytes =
                llama_cache_acct_value::measured(0);
            out.push_back(domain_row);
        }
        return !out.empty();
    };
    auto quote = server_cache_destruction_quote_single_artifact(
        victim, 0, ledger.snapshot().serial, 1, preview, project);
    CHECK(quote.receipt.state ==
          common_cache_plan_destruction_state::quoted);
    auto same_member_pin = store.acquire_recovery_pin(rebound);
    CHECK(same_member_pin.valid());
    auto disjoint_refusal = server_cache_prepare_release_set(
        quote, { victim }, ledger, ledger.snapshot().serial,
        project, std::move(same_member_pin));
    CHECK(disjoint_refusal.status ==
          server_cache_prepare_release_status::recovery_unavailable);
    same_member_pin = {};
    auto recovery_pin = store.acquire_recovery_pin(live);
    CHECK(recovery_pin.valid());
    auto prepared = server_cache_prepare_release_set(
        quote, { victim }, ledger, ledger.snapshot().serial,
        project, std::move(recovery_pin));
    CHECK(prepared.status == server_cache_prepare_release_status::prepared);
    server_cache_recovery_pin retained_pin;
    CHECK(prepared.capability.commit(retained_pin) ==
          common_cache_plan_destruction_reason::none);
    store.retire_after_committed_release(rebound);
    retained_pin = {};

    // A latent legacy drop racing a recovery pin must fail soft, not abort or
    // invalidate the pin callback. Retirement is deferred until the pin
    // closes, at which point both descriptor and payload ops are discharged.
    const auto * pinned_ptr =
        reinterpret_cast<const common_prompt_checkpoint *>(uintptr_t(100));
    const auto pinned_key =
        server_retention_instance_key::for_checkpoint(9, pinned_ptr);
    CHECK(store.clone(live, pinned_key));
    const auto pinned_artifact = store.artifact_id(pinned_key);
    const auto pinned_op = server_cache_acct_charge_shadow(
        ledger,
        llama_cache_acct_category::checkpoint_state_payload,
        domain,
        llama_cache_acct_producer::retention_sidecar,
        { llama_cache_acct_attr_kind::artifact, -1, pinned_artifact },
        16, 16);
    CHECK(pinned_op);
    CHECK(store.attach_release_ops(pinned_key, { pinned_op }));
    auto latent_pin = store.acquire_recovery_pin(pinned_key);
    CHECK(latent_pin.valid());
    const auto live_ops_before_pinned_drop = ledger.snapshot().live_ops;
    store.retire(pinned_key);
    CHECK(store.artifact_id(pinned_key).v == 0);
    CHECK(store.unavailable() == 1);
    CHECK(ledger.snapshot().live_ops == live_ops_before_pinned_drop);
    latent_pin = {};
    CHECK(ledger.snapshot().live_ops + 2 == live_ops_before_pinned_drop);

    store.retire(live);
    CHECK(leases.evaluate(live_artifact, lease_identity).cls ==
          server_cache_lease_class::none);
    CHECK(store.live_bytes() == 0);
    CHECK(ledger.snapshot().allocations.empty());
    CHECK(store.publish_ok() == 3);
    CHECK(store.unavailable() == 1);
}

int main() {
    test_turn_table_and_geometry();
    test_codec();
    test_store_and_allocator_import();
    test_frequency_and_lineage_transitions();
    test_scan_flood_and_phase_change_simulator();
    test_shared_geometry_and_scheduler_lineage_receipts();
    test_shadow_value_comparator();
    test_observer_store_accounting();
    if (failures != 0) {
        fprintf(stderr, "%d retention-sidecar test(s) failed\n", failures);
        return 1;
    }
    printf("retention sidecar: PASS\n");
    return 0;
}
