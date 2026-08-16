#include "server-retention-sidecar.h"
#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace {
constexpr size_t MAX_CATALOG_ARTIFACTS = SERVER_RETENTION_MAX_CANDIDATES;
constexpr uint64_t MAX_TURN_TABLE_BYTES = 16ull*1024*1024;
constexpr uint64_t TURN_BOUNDARY_BYTES =
    sizeof(common_retention_turn_boundary);

bool checked_add(uint64_t & dst, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - dst) {
        return false;
    }
    dst += value;
    return true;
}
}

void server_cache_acct_mark_shadow_unavailable(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer) noexcept {
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated }) {
        ledger.mark_unavailable(category, domain, measure);
    }
    ledger.mark_producer_unavailable(domain, producer);
}

llama_cache_acct_op_id server_cache_acct_charge_shadow(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer,
        const llama_cache_acct_attribution & attribution,
        uint64_t logical_bytes,
        uint64_t resident_bytes) noexcept {
    const auto op = ledger.reserve(
        category, domain, attribution, logical_bytes, resident_bytes);
    const auto artifact =
        attribution.kind == llama_cache_acct_attr_kind::artifact
            ? attribution.artifact
            : llama_cache_acct_artifact_id{};
    if (!op ||
        !ledger.stage(op, ledger.new_alloc(), resident_bytes, artifact) ||
        !ledger.commit(op, logical_bytes)) {
        if (op) {
            (void) ledger.abort(op);
        }
        server_cache_acct_mark_shadow_unavailable(
            ledger, category, domain, producer);
        return {};
    }
    return op;
}

size_t server_retention_instance_key_hash::operator()(
        const server_retention_instance_key & key) const noexcept {
    size_t result = std::hash<uintptr_t>{}(key.instance);
    result ^= std::hash<int32_t>{}(key.owner_slot) +
        0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<uint8_t>{}(uint8_t(key.kind)) +
        0x9e3779b9 + (result << 6) + (result >> 2);
    return result;
}

server_retention_sidecar_store::~server_retention_sidecar_store() {
    while (!associations.empty()) {
        retire_association(associations.begin());
    }
}

void server_retention_sidecar_store::configure(
        llama_cache_acct_ledger * ledger_in,
        const llama_cache_acct_resource_domain & domain_in,
        server_cache_lease_table * leases_in) noexcept {
    ledger = ledger_in;
    domain = domain_in;
    leases = leases_in;
}

llama_cache_acct_artifact_id
server_retention_sidecar_store::qualified_artifact_id(
        common_retention_pool pool, uint64_t stable_id) noexcept {
    if (stable_id == 0 ||
        stable_id > COMMON_RETENTION_MAX_POOL_COUNTER ||
        pool >= common_retention_pool::_count) {
        return {};
    }
    return { (stable_id << 1) | uint64_t(pool) };
}

uint64_t server_retention_sidecar_store::qualified_lineage_id(
        common_retention_pool pool, uint64_t lineage_id) noexcept {
    if (lineage_id == 0 ||
        lineage_id > COMMON_RETENTION_MAX_POOL_COUNTER ||
        pool >= common_retention_pool::_count) {
        return 0;
    }
    return (lineage_id << 1) | uint64_t(pool);
}

bool server_retention_sidecar_store::retain_lineage(
        common_retention_pool pool, uint64_t lineage_id) noexcept {
    const auto key = qualified_lineage_id(pool, lineage_id);
    const auto found = lineages.find(key);
    if (key == 0 || found == lineages.end() || found->second.refs == UINT32_MAX) {
        return false;
    }
    found->second.refs++;
    return true;
}

void server_retention_sidecar_store::release_lineage(
        common_retention_pool pool, uint64_t lineage_id) noexcept {
    const auto key = qualified_lineage_id(pool, lineage_id);
    const auto found = lineages.find(key);
    if (key == 0 || found == lineages.end() || found->second.refs == 0) {
        mark_unavailable();
        return;
    }
    found->second.refs--;
    if (found->second.refs == 0) {
        lineages.erase(found);
    }
}

bool server_retention_sidecar_store::retain_turn_table(
        const std::shared_ptr<const common_retention_turn_table> & table,
        llama_cache_acct_artifact_id attribution) noexcept {
    if (!table || !table->valid() ||
        table->boundaries.capacity() >
            (MAX_TURN_TABLE_BYTES - sizeof(*table))/TURN_BOUNDARY_BYTES) {
        return false;
    }
    const uint64_t bytes = sizeof(*table) +
        uint64_t(table->boundaries.capacity())*TURN_BOUNDARY_BYTES;
    const auto found = turn_tables.find(table.get());
    if (found != turn_tables.end()) {
        if (found->second.refs == UINT32_MAX) {
            return false;
        }
        found->second.refs++;
        return true;
    }
    if (bytes > MAX_TURN_TABLE_BYTES - turn_table_bytes_live) {
        return false;
    }
    try {
        turn_table_entry entry;
        entry.table = table;
        entry.bytes = bytes;
        entry.refs = 1;
        if (ledger) {
            const llama_cache_acct_attribution owner {
                llama_cache_acct_attr_kind::artifact, -1, attribution,
            };
            entry.accounting_op = server_cache_acct_charge_shadow(
                *ledger,
                llama_cache_acct_category::artifact_descriptor_metadata,
                domain,
                llama_cache_acct_producer::retention_sidecar,
                owner,
                bytes,
                bytes);
            if (!entry.accounting_op) {
                return false;
            }
        }
        const auto table_op = entry.accounting_op;
        try {
            if (!turn_tables.emplace(table.get(), std::move(entry)).second) {
                if (ledger && table_op) {
                    (void) ledger->release(table_op);
                }
                return false;
            }
        } catch (...) {
            if (ledger && table_op) {
                (void) ledger->release(table_op);
            }
            return false;
        }
        turn_table_bytes_live += bytes;
        if (!checked_add(bytes_live, bytes)) {
            bytes_live = std::numeric_limits<uint64_t>::max();
            mark_unavailable();
        }
        return true;
    } catch (...) {
        return false;
    }
}

void server_retention_sidecar_store::release_turn_table(
        const std::shared_ptr<const common_retention_turn_table> & table) noexcept {
    if (!table) {
        mark_unavailable();
        return;
    }
    const auto found = turn_tables.find(table.get());
    if (found == turn_tables.end() || found->second.refs == 0) {
        mark_unavailable();
        return;
    }
    if (--found->second.refs != 0) {
        return;
    }
    if (ledger && found->second.accounting_op &&
        !ledger->release(found->second.accounting_op)) {
        mark_unavailable();
    }
    const uint64_t bytes = found->second.bytes;
    turn_table_bytes_live = bytes <= turn_table_bytes_live
        ? turn_table_bytes_live - bytes : 0;
    bytes_live = bytes <= bytes_live ? bytes_live - bytes : 0;
    turn_tables.erase(found);
}

bool server_retention_sidecar_store::create_lineage(
        common_retention_pool pool, uint64_t & lineage_id) noexcept {
    lineage_id = 0;
    if (competition_epoch == UINT64_MAX) {
        return false;
    }
    common_retention_lineage_record record;
    if (!allocator.issue_lineage(pool, competition_epoch + 1, record)) {
        return false;
    }
    const auto key = qualified_lineage_id(pool, record.lineage_id);
    if (key == 0) {
        return false;
    }
    try {
        if (!lineages.emplace(key, lineage_entry { record, 0 }).second) {
            return false;
        }
    } catch (...) {
        return false;
    }
    lineage_id = record.lineage_id;
    return true;
}

bool server_retention_sidecar_store::advance_competition_epoch() noexcept {
    if (competition_epoch == UINT64_MAX) {
        return false;
    }
    competition_epoch++;
    return true;
}

bool server_retention_sidecar_store::activate_lineage_ticket(
        const server_retention_lineage_ticket & ticket) noexcept {
    if (!ticket.valid() || competition_epoch == UINT64_MAX) {
        return false;
    }
    const auto found = lineages.find(
        qualified_lineage_id(ticket.pool, ticket.lineage_id));
    if (found == lineages.end()) {
        return false;
    }
    if (found->second.admitted) {
        return true;
    }
    found->second.record.admission_epoch = competition_epoch + 1;
    found->second.record.frequency_epoch = competition_epoch + 1;
    if (!advance_competition_epoch()) {
        return false;
    }
    found->second.admitted = true;
    return true;
}

void server_retention_sidecar_store::mark_unavailable() noexcept {
    n_unavailable++;
    if (!ledger) {
        return;
    }
    server_cache_acct_mark_shadow_unavailable(
        *ledger,
        llama_cache_acct_category::artifact_descriptor_metadata,
        domain,
        llama_cache_acct_producer::retention_sidecar);
}

bool server_retention_sidecar_store::install(
        const server_retention_instance_key & key,
        common_retention_artifact_record && record,
        const server_cache_lease_identity * checkpoint_identity,
        const server_cache_lease_frontier * replacement_frontier) noexcept {
    try {
        if (catalog.size() >= MAX_CATALOG_ARTIFACTS) {
            mark_unavailable();
            return false;
        }
        uint64_t bytes = 0;
        if (!common_retention_sidecar_artifact_encoded_size(record, bytes)) {
            mark_unavailable();
            return false;
        }
        const uint64_t boundary_bytes =
            uint64_t(record.turns->boundaries.size())*TURN_BOUNDARY_BYTES;
        GGML_ASSERT(bytes >= boundary_bytes);
        bytes -= boundary_bytes;

        const auto artifact = qualified_artifact_id(
            record.stamp.pool, record.stamp.stable_id);
        if (artifact.v == 0 || catalog.find(artifact.v) != catalog.end()) {
            mark_unavailable();
            return false;
        }
        auto old = associations.find(key);
        const auto old_artifact = old == associations.end()
            ? llama_cache_acct_artifact_id{} : old->second;

        catalog_entry entry;
        entry.record = std::move(record);
        if (checkpoint_identity && checkpoint_identity->valid()) {
            entry.checkpoint_identity = *checkpoint_identity;
            entry.checkpoint_identity_known = true;
        }
        entry.encoded_size = bytes;
        auto inserted = catalog.emplace(artifact.v, std::move(entry));
        if (!inserted.second) {
            mark_unavailable();
            return false;
        }
        if (!retain_lineage(
                inserted.first->second.record.stamp.pool,
                inserted.first->second.record.stamp.lineage_id)) {
            catalog.erase(inserted.first);
            mark_unavailable();
            return false;
        }
        if (!retain_turn_table(inserted.first->second.record.turns, artifact)) {
            release_lineage(
                inserted.first->second.record.stamp.pool,
                inserted.first->second.record.stamp.lineage_id);
            catalog.erase(inserted.first);
            mark_unavailable();
            return false;
        }
        inserted.first->second.owner = this;
        inserted.first->second.artifact = artifact;

        auto & installed = inserted.first->second;
        if (ledger) {
            const llama_cache_acct_attribution attribution {
                llama_cache_acct_attr_kind::artifact, -1, artifact,
            };
            installed.accounting_op = server_cache_acct_charge_shadow(
                *ledger,
                llama_cache_acct_category::artifact_descriptor_metadata,
                domain,
                llama_cache_acct_producer::retention_sidecar,
                attribution,
                bytes,
                bytes);
            if (!installed.accounting_op) {
                release_turn_table(installed.record.turns);
                release_lineage(
                    installed.record.stamp.pool,
                    installed.record.stamp.lineage_id);
                catalog.erase(inserted.first);
                return false;
            }
        }
        if (!checked_add(bytes_live, bytes)) {
            bytes_live = std::numeric_limits<uint64_t>::max();
            mark_unavailable();
        }

        // Publish all new backing/accounting state before changing the one
        // association. A same-key append replacement can then migrate leases
        // without exposing a missing-artifact interval on the scheduler
        // thread. New-key insertion is the only throwing step after charge;
        // roll its complete catalog entry back on failure.
        if (old != associations.end()) {
            old->second = artifact;
        } else {
            try {
                if (!associations.emplace(key, artifact).second) {
                    retire_catalog_entry(inserted.first);
                    mark_unavailable();
                    return false;
                }
            } catch (...) {
                retire_catalog_entry(inserted.first);
                throw;
            }
        }

        if (old_artifact.v != 0) {
            bool continued = false;
            if (leases && checkpoint_identity && replacement_frontier) {
                continued = leases->artifact_replaced(
                    { old_artifact, key.kind, key.owner_slot },
                    { artifact, key.kind, key.owner_slot },
                    *checkpoint_identity, *replacement_frontier);
            }
            if (leases && !continued) {
                leases->artifact_retired(old_artifact);
            }
            const auto old_entry = catalog.find(old_artifact.v);
            if (old_entry == catalog.end()) {
                mark_unavailable();
            } else if (old_entry->second.recovery_pins != 0) {
                old_entry->second.retire_pending = true;
            } else {
                retire_catalog_entry(old_entry);
            }
        }
        n_publish_ok++;
        return true;
    } catch (...) {
        mark_unavailable();
        return false;
    }
}

bool server_retention_sidecar_store::publish(
        const server_retention_instance_key & key,
        common_retention_pool pool,
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t turn_token_count,
        uint64_t coverage_tokens,
        bool coverage_valid,
        const server_cache_lease_identity * checkpoint_identity,
        const server_cache_lease_frontier * replacement_frontier,
        const server_retention_instance_key * lineage_source,
        const server_retention_lineage_ticket * lineage_ticket,
        bool defer_lineage_admission) noexcept {
    common_retention_artifact_record record;
    record.kind = key.kind;
    uint64_t lineage_id = 0;
    bool created_lineage = false;
    const common_retention_artifact_record * geometry_source = nullptr;
    const auto existing = associations.find(key);
    // The scheduler retires a live association before any trim/branch/rebind.
    // Therefore a surviving same-key association is exact append continuity
    // even when the optional lease-identity client is disabled.
    if (existing != associations.end()) {
        const auto item = catalog.find(existing->second.v);
        if (item != catalog.end() && item->second.record.stamp.pool == pool) {
            lineage_id = item->second.record.stamp.lineage_id;
            geometry_source = &item->second.record;
        }
    }
    if (lineage_id == 0 && lineage_ticket && lineage_ticket->valid() &&
        lineage_ticket->pool == pool &&
        lineages.find(qualified_lineage_id(
            pool, lineage_ticket->lineage_id)) != lineages.end()) {
        lineage_id = lineage_ticket->lineage_id;
        record.turns = lineage_ticket->turns;
    }
    if (lineage_id == 0 && lineage_source) {
        const auto source = associations.find(*lineage_source);
        if (source != associations.end()) {
            const auto item = catalog.find(source->second.v);
            if (item != catalog.end() && item->second.record.stamp.pool == pool) {
                lineage_id = item->second.record.stamp.lineage_id;
                geometry_source = &item->second.record;
            }
        }
    }
    if (lineage_id == 0) {
        if (!create_lineage(pool, lineage_id)) {
            retire(key);
            mark_unavailable();
            return false;
        }
        created_lineage = true;
    }
    const auto discard_unreferenced_lineage = [&]() noexcept {
        if (!created_lineage) {
            return;
        }
        const auto id = qualified_lineage_id(pool, lineage_id);
        const auto item = lineages.find(id);
        if (item != lineages.end() && item->second.refs == 0) {
            lineages.erase(item);
        }
    };
    if (!allocator.issue(pool, record.stamp)) {
        discard_unreferenced_lineage();
        retire(key);
        mark_unavailable();
        return false;
    }
    record.stamp.lineage_id = lineage_id;
    record.stamp.coverage_tokens = coverage_tokens;
    if (record.turns && record.turns->token_count == turn_token_count) {
        // A retained lineage ticket already carries the immutable request
        // geometry shared by its checkpoints and final live publication.
    } else if (geometry_source && geometry_source->turns &&
        geometry_source->turns->token_count == turn_token_count) {
        record.turns = geometry_source->turns;
    } else {
        common_retention_turn_table turns;
        if (!common_retention_build_turn_table(
                spans, source_known, turn_token_count, turns)) {
            turns = {};
        }
        try {
            record.turns =
                std::make_shared<const common_retention_turn_table>(
                    std::move(turns));
        } catch (...) {
            discard_unreferenced_lineage();
            retire(key);
            mark_unavailable();
            return false;
        }
    }
    if (!coverage_valid ||
        !common_retention_score(*record.turns, coverage_tokens, record.stamp)) {
        record.stamp.state = common_retention_score_state::unavailable;
        record.stamp.mandatory_anchor = false;
        record.stamp.mapped_turn_ordinal = 0;
        record.stamp.anchor_rank = 0;
    }
    if (key.kind != common_retention_artifact_kind::checkpoint) {
        record.stamp.mandatory_anchor = false;
    }
    const auto activation_turns = record.turns;
    if (!install(
            key, std::move(record), checkpoint_identity,
            replacement_frontier)) {
        discard_unreferenced_lineage();
        retire(key);
        return false;
    }
    if (!defer_lineage_admission) {
        server_retention_lineage_ticket activation {
            pool, lineage_id, activation_turns,
        };
        if (!activate_lineage_ticket(activation)) {
            retire(key);
            mark_unavailable();
            return false;
        }
    }
    return true;
}

bool server_retention_sidecar_store::clone(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept {
    try {
        const auto assoc = associations.find(source);
        if (assoc == associations.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        const auto item = catalog.find(assoc->second.v);
        if (item == catalog.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        const auto source_lineage = lineages.find(qualified_lineage_id(
            item->second.record.stamp.pool,
            item->second.record.stamp.lineage_id));
        if (source_lineage == lineages.end() ||
            !source_lineage->second.admitted) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        auto record = item->second.record;
        server_cache_lease_identity checkpoint_identity;
        const server_cache_lease_identity * checkpoint_identity_ptr = nullptr;
        if (item->second.checkpoint_identity_known) {
            checkpoint_identity = item->second.checkpoint_identity;
            checkpoint_identity_ptr = &checkpoint_identity;
        }
        record.kind = destination.kind;
        if (!allocator.issue(record.stamp.pool, record.stamp)) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        if (!install(
                destination, std::move(record), checkpoint_identity_ptr,
                nullptr)) {
            retire(destination);
            return false;
        }
        return true;
    } catch (...) {
        retire(destination);
        mark_unavailable();
        return false;
    }
}

bool server_retention_sidecar_store::branch(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination,
        const server_retention_instance_key *
            destination_lineage_source,
        bool defer_lineage_admission) noexcept {
    common_retention_pool pool = common_retention_pool::attention;
    uint64_t lineage_id = 0;
    const auto discard_unreferenced = [&]() noexcept {
        const auto key = qualified_lineage_id(pool, lineage_id);
        const auto found = lineages.find(key);
        if (key != 0 && found != lineages.end() && found->second.refs == 0) {
            lineages.erase(found);
        }
    };
    try {
        const auto assoc = associations.find(source);
        if (assoc == associations.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        const auto item = catalog.find(assoc->second.v);
        if (item == catalog.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        const auto source_lineage = lineages.find(qualified_lineage_id(
            item->second.record.stamp.pool,
            item->second.record.stamp.lineage_id));
        if (source_lineage == lineages.end() ||
            !source_lineage->second.admitted) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        pool = item->second.record.stamp.pool;
        bool created_lineage = false;
        if (destination_lineage_source) {
            const auto lineage_assoc =
                associations.find(*destination_lineage_source);
            const auto lineage_item = lineage_assoc == associations.end()
                ? catalog.end() : catalog.find(lineage_assoc->second.v);
            if (lineage_item == catalog.end() ||
                lineage_item->second.record.stamp.pool != pool) {
                retire(destination);
                mark_unavailable();
                return false;
            }
            lineage_id = lineage_item->second.record.stamp.lineage_id;
        } else {
            if (!create_lineage(pool, lineage_id)) {
                retire(destination);
                mark_unavailable();
                return false;
            }
            created_lineage = true;
        }
        auto record = item->second.record;
        record.kind = destination.kind;
        if (!allocator.issue(pool, record.stamp)) {
            discard_unreferenced();
            retire(destination);
            mark_unavailable();
            return false;
        }
        record.stamp.lineage_id = lineage_id;
        const auto * checkpoint_identity =
            item->second.checkpoint_identity_known
                ? &item->second.checkpoint_identity : nullptr;
        const auto activation_turns = record.turns;
        if (!install(
                destination, std::move(record), checkpoint_identity,
                nullptr)) {
            discard_unreferenced();
            retire(destination);
            return false;
        }
        if (created_lineage && !defer_lineage_admission) {
            const server_retention_lineage_ticket activation {
                pool, lineage_id, activation_turns,
            };
            if (!activate_lineage_ticket(activation)) {
                retire(destination);
                mark_unavailable();
                return false;
            }
        }
        return true;
    } catch (...) {
        retire(destination);
        discard_unreferenced();
        mark_unavailable();
        return false;
    }
}

bool server_retention_sidecar_store::rebind(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept {
    try {
        const auto src = associations.find(source);
        if (src == associations.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        const auto item = catalog.find(src->second.v);
        if (item == catalog.end() ||
            item->second.record.kind != destination.kind) {
            retire(destination);
            retire(source);
            mark_unavailable();
            return false;
        }
        const auto artifact = src->second;
        auto old = associations.find(destination);
        if (old != associations.end() && old != src) {
            retire_association(old);
        }
        if (!associations.emplace(destination, artifact).second) {
            retire(source);
            mark_unavailable();
            return false;
        }
        associations.erase(source);
        return true;
    } catch (...) {
        retire(source);
        mark_unavailable();
        return false;
    }
}

bool server_retention_sidecar_store::prepare_for_launch(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept {
    const auto association = associations.find(destination);
    const auto item = association == associations.end()
        ? catalog.end() : catalog.find(association->second.v);
    if (item == catalog.end() ||
        destination.kind != common_retention_artifact_kind::live_slot ||
        item->second.prepared_source.valid()) {
        return false;
    }
    return acquire_lineage_ticket(source, item->second.prepared_source);
}

bool server_retention_sidecar_store::prepared_for_launch(
        const server_retention_instance_key & destination) const noexcept {
    const auto association = associations.find(destination);
    const auto item = association == associations.end()
        ? catalog.end() : catalog.find(association->second.v);
    return item != catalog.end() && item->second.prepared_source.valid();
}

bool server_retention_sidecar_store::consume_prepared_launch(
        const server_retention_instance_key & destination,
        server_retention_lineage_ticket & source) noexcept {
    source = {};
    const auto association = associations.find(destination);
    const auto item = association == associations.end()
        ? catalog.end() : catalog.find(association->second.v);
    if (item == catalog.end() || !item->second.prepared_source.valid()) {
        return false;
    }
    source = std::move(item->second.prepared_source);
    item->second.prepared_source = {};
    return true;
}

void server_retention_sidecar_store::abandon_prepared_launch(
        const server_retention_instance_key & destination) noexcept {
    const auto association = associations.find(destination);
    const auto item = association == associations.end()
        ? catalog.end() : catalog.find(association->second.v);
    if (item != catalog.end()) {
        const auto pool = item->second.record.stamp.pool;
        const auto lineage_id = item->second.record.stamp.lineage_id;
        const auto lineage = lineages.find(
            qualified_lineage_id(pool, lineage_id));
        if (lineage != lineages.end() && !lineage->second.admitted) {
            // A restore which never launches must not leave a provisional
            // branch (or any checkpoint aliases sharing it) in the catalog.
            // Erase in place: abandonment is a failure/defer terminal and
            // must not depend on allocating a victim list.
            for (auto it = associations.begin(); it != associations.end();) {
                const auto candidate = catalog.find(it->second.v);
                if (candidate == catalog.end() ||
                    candidate->second.record.stamp.pool != pool ||
                    candidate->second.record.stamp.lineage_id != lineage_id) {
                    ++it;
                    continue;
                }
                auto victim = it++;
                retire_association(victim);
            }
            return;
        }
    }
    server_retention_lineage_ticket source;
    if (consume_prepared_launch(destination, source)) {
        release_lineage_ticket(source);
    }
}

common_retention_credit_result server_retention_sidecar_store::credit_reuse(
        const server_retention_instance_key & source) noexcept {
    const auto association = associations.find(source);
    if (association == associations.end()) {
        return common_retention_credit_result::unavailable;
    }
    const auto artifact = catalog.find(association->second.v);
    if (artifact == catalog.end()) {
        mark_unavailable();
        return common_retention_credit_result::unavailable;
    }
    const auto key = qualified_lineage_id(
        artifact->second.record.stamp.pool,
        artifact->second.record.stamp.lineage_id);
    const auto lineage = lineages.find(key);
    if (key == 0 || lineage == lineages.end() ||
        !lineage->second.admitted) {
        mark_unavailable();
        return common_retention_credit_result::unavailable;
    }
    return common_retention_frequency_credit(
        lineage->second.record, competition_epoch, frequency_config);
}

bool server_retention_sidecar_store::acquire_lineage_ticket(
        const server_retention_instance_key & source,
        server_retention_lineage_ticket & out) noexcept {
    out = {};
    const auto association = associations.find(source);
    const auto artifact = association == associations.end()
        ? catalog.end() : catalog.find(association->second.v);
    if (artifact == catalog.end()) {
        return false;
    }
    const auto pool = artifact->second.record.stamp.pool;
    const auto lineage_id = artifact->second.record.stamp.lineage_id;
    if (!retain_lineage(pool, lineage_id)) {
        mark_unavailable();
        return false;
    }
    if (!retain_turn_table(
            artifact->second.record.turns, association->second)) {
        release_lineage(pool, lineage_id);
        mark_unavailable();
        return false;
    }
    out = { pool, lineage_id, artifact->second.record.turns };
    return true;
}

void server_retention_sidecar_store::release_lineage_ticket(
        server_retention_lineage_ticket & ticket) noexcept {
    if (!ticket.valid()) {
        return;
    }
    const auto released = ticket;
    ticket = {};
    release_turn_table(released.turns);
    release_lineage(released.pool, released.lineage_id);
}

common_retention_credit_result server_retention_sidecar_store::credit_reuse(
        const server_retention_lineage_ticket & source) noexcept {
    if (!source.valid()) {
        return common_retention_credit_result::unavailable;
    }
    const auto key = qualified_lineage_id(source.pool, source.lineage_id);
    const auto lineage = lineages.find(key);
    if (key == 0 || lineage == lineages.end() ||
        !lineage->second.admitted) {
        mark_unavailable();
        return common_retention_credit_result::unavailable;
    }
    return common_retention_frequency_credit(
        lineage->second.record, competition_epoch, frequency_config);
}

bool server_retention_sidecar_store::set_lineage_prior(
        const server_retention_instance_key & source,
        uint32_t prior_milli) noexcept {
    if (prior_milli == 0 ||
        prior_milli > COMMON_RETENTION_MAX_PRIOR_MILLI) {
        return false;
    }
    const auto association = associations.find(source);
    const auto artifact = association == associations.end()
        ? catalog.end() : catalog.find(association->second.v);
    if (artifact == catalog.end()) {
        return false;
    }
    const auto key = qualified_lineage_id(
        artifact->second.record.stamp.pool,
        artifact->second.record.stamp.lineage_id);
    const auto lineage = lineages.find(key);
    if (key == 0 || lineage == lineages.end() ||
        !lineage->second.admitted) {
        mark_unavailable();
        return false;
    }
    lineage->second.record.prior_milli = prior_milli;
    return true;
}

bool server_retention_sidecar_store::begin_competition_wave() noexcept {
    return advance_competition_epoch();
}

bool server_retention_sidecar_store::lineage_for_instance(
        const server_retention_instance_key & key,
        common_retention_lineage_record & out) const noexcept {
    out = {};
    const auto association = associations.find(key);
    if (association == associations.end()) {
        return false;
    }
    const auto artifact = catalog.find(association->second.v);
    if (artifact == catalog.end()) {
        return false;
    }
    const auto lineage_key = qualified_lineage_id(
        artifact->second.record.stamp.pool,
        artifact->second.record.stamp.lineage_id);
    const auto lineage = lineages.find(lineage_key);
    if (lineage_key == 0 || lineage == lineages.end() ||
        !lineage->second.admitted) {
        return false;
    }
    out = lineage->second.record;
    return out.valid(competition_epoch);
}

server_retention_value_snapshot_result
server_retention_sidecar_store::value_snapshots(
        void * context,
        server_retention_value_snapshot_visitor visitor) const noexcept {
    server_retention_value_snapshot_result result;
    if (!visitor ||
        associations.size() > SERVER_RETENTION_MAX_CANDIDATES) {
        result.status = server_retention_value_snapshot_status::overflow;
        return result;
    }
    for (const auto & association : associations) {
        const auto artifact = catalog.find(association.second.v);
        if (artifact == catalog.end() || !artifact->second.record.valid()) {
            return result;
        }
        const auto & record = artifact->second.record;
        if (record.kind == common_retention_artifact_kind::checkpoint) {
            continue;
        }
        if (record.stamp.state != common_retention_score_state::known) {
            return result;
        }
        const auto lineage_key = qualified_lineage_id(
            record.stamp.pool, record.stamp.lineage_id);
        const auto lineage = lineages.find(lineage_key);
        if (lineage_key == 0 || lineage == lineages.end()) {
            return result;
        }
        if (!lineage->second.admitted) {
            continue;
        }
        if (!lineage->second.record.valid(competition_epoch)) {
            return result;
        }
        const server_retention_value_snapshot value {
            association.second,
            record.kind,
            record.stamp,
            lineage->second.record,
        };
        if (!visitor(context, value)) {
            result = {};
            result.status = server_retention_value_snapshot_status::overflow;
            return result;
        }
        result.size++;
    }
    result.status = server_retention_value_snapshot_status::complete;
    return result;
}

void server_retention_sidecar_store::retire_association(
        association_map::iterator it) noexcept {
    const auto artifact = it->second;
    const auto entry = catalog.find(artifact.v);
    if (entry == catalog.end()) {
        associations.erase(it);
        if (leases) {
            leases->artifact_retired(artifact);
        }
        mark_unavailable();
        return;
    }
    // The policy inventory excludes pinned entries. If latent catalog drift
    // reaches this legacy terminal anyway, fail soft: surface unavailable,
    // detach the stale association, and defer catalog/accounting retirement
    // until the final pin closes. The pin callback owns that terminal.
    if (entry->second.recovery_pins != 0) {
        entry->second.retire_pending = true;
        associations.erase(it);
        if (leases) {
            leases->artifact_retired(artifact);
        }
        mark_unavailable();
        return;
    }
    associations.erase(it);
    if (leases) {
        leases->artifact_retired(artifact);
    }
    retire_catalog_entry(entry);
}

void server_retention_sidecar_store::retire_catalog_entry(
        catalog_map::iterator entry) noexcept {
    if (ledger && !entry->second.release_ops.empty()) {
        auto release = llama_cache_prepare_release_set(
            *ledger, entry->second.release_ops,
            ledger->snapshot().serial);
        if (!release.ready() || release.commit() !=
                llama_cache_conditional_release_status::released) {
            // A failed conditional commit never releases a member. The
            // legacy drop has already won, so discharge each still-live op
            // and mark the catalog unavailable rather than aborting.
            mark_unavailable();
            for (const auto op : entry->second.release_ops) {
                (void) ledger->release(op);
            }
        }
    }
    if (entry->second.accounting_op && ledger) {
        if (!ledger->release(entry->second.accounting_op)) {
            mark_unavailable();
        }
    }
    const uint64_t bytes = entry->second.encoded_size;
    const auto turns = entry->second.record.turns;
    auto prepared_source = std::move(entry->second.prepared_source);
    const auto lineage_pool = entry->second.record.stamp.pool;
    const auto lineage_id = entry->second.record.stamp.lineage_id;
    bytes_live = bytes <= bytes_live ? bytes_live - bytes : 0;
    catalog.erase(entry);
    release_turn_table(turns);
    release_lineage(lineage_pool, lineage_id);
    release_lineage_ticket(prepared_source);
}

bool server_retention_sidecar_store::recovery_pinned(
        const server_retention_instance_key & key) const noexcept {
    const auto association = associations.find(key);
    if (association == associations.end()) {
        return false;
    }
    const auto item = catalog.find(association->second.v);
    return item != catalog.end() && item->second.recovery_pins != 0;
}

void server_retention_sidecar_store::retire(
        const server_retention_instance_key & key) noexcept {
    const auto it = associations.find(key);
    if (it != associations.end()) {
        retire_association(it);
    }
}

void server_retention_sidecar_store::retire_slot(int32_t owner_slot) noexcept {
    for (auto it = associations.begin(); it != associations.end();) {
        if (it->first.owner_slot == owner_slot) {
            auto victim = it++;
            retire_association(victim);
        } else {
            ++it;
        }
    }
}

llama_cache_acct_artifact_id server_retention_sidecar_store::artifact_id(
        const server_retention_instance_key & key) const noexcept {
    const auto it = associations.find(key);
    return it == associations.end() ? llama_cache_acct_artifact_id{} : it->second;
}

bool server_retention_sidecar_store::candidate_for_instance(
        const server_retention_instance_key & key,
        server_retention_candidate & out) const noexcept {
    out = {};
    try {
        const auto association = associations.find(key);
        if (association == associations.end()) {
            return false;
        }
        const auto item = catalog.find(association->second.v);
        if (item == catalog.end()) {
            return false;
        }
        out.artifact_id = association->second;
        out.instance_key = key;
        out.record = item->second.record;
        const auto lineage_key = qualified_lineage_id(
            out.record.stamp.pool, out.record.stamp.lineage_id);
        const auto lineage = lineages.find(lineage_key);
        if (lineage_key == 0 || lineage == lineages.end() ||
            !lineage->second.admitted) {
            return false;
        }
        out.lineage = lineage->second.record;
        out.provenance_op = item->second.accounting_op;
        out.release_ops = item->second.release_ops;
        out.avail = server_retention_candidate_availability::available;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_retention_sidecar_store::checkpoint_admission_artifact(
        const server_retention_instance_key & key,
        llama_cache_acct_artifact_id & artifact) const noexcept {
    artifact = {};
    if (key.kind != common_retention_artifact_kind::checkpoint) {
        return false;
    }
    const auto association = associations.find(key);
    if (association == associations.end()) {
        return false;
    }
    const auto item = catalog.find(association->second.v);
    if (item == catalog.end() || !item->second.release_ops.empty()) {
        return false;
    }
    // This is an internal publication handle, not policy inventory.
    // Provisional restore checkpoints acquire balanced ledger ownership here;
    // candidate/inventory/snapshot doors remain closed until their lineage is
    // admitted at successful launch. Abandon retires both payload ops and the
    // provisional catalog atomically.
    artifact = association->second;
    return artifact.v != 0;
}

bool server_retention_sidecar_store::checkpoint_inventory(
        const server_retention_instance_key & key,
        server_retention_checkpoint_inventory & out) const noexcept {
    out = {};
    if (key.kind != common_retention_artifact_kind::checkpoint) {
        return false;
    }
    const auto association = associations.find(key);
    if (association == associations.end()) {
        return false;
    }
    const auto item = catalog.find(association->second.v);
    if (item == catalog.end() ||
        item->second.record.kind != common_retention_artifact_kind::checkpoint) {
        return false;
    }
    const auto lineage = lineages.find(qualified_lineage_id(
        item->second.record.stamp.pool,
        item->second.record.stamp.lineage_id));
    if (lineage == lineages.end() || !lineage->second.admitted) {
        return false;
    }
    out.artifact_id = association->second;
    out.pool = item->second.record.stamp.pool;
    out.stable_id = item->second.record.stamp.stable_id;
    out.mandatory_anchor = item->second.record.stamp.mandatory_anchor;
    out.release_owned = !item->second.release_ops.empty();
    out.recovery_pinned = item->second.recovery_pins != 0;
    out.identity_known = leases &&
        item->second.checkpoint_identity_known &&
        item->second.checkpoint_identity.valid();
    if (out.identity_known) {
        out.lease = leases->inspect(
            out.artifact_id, item->second.checkpoint_identity);
    }
    return out.artifact_id.v != 0;
}

bool server_retention_sidecar_store::attach_release_ops(
        const server_retention_instance_key & key,
        std::vector<llama_cache_acct_op_id> ops) noexcept {
    const auto release_supplied = [&]() noexcept {
        if (!ledger) {
            return;
        }
        for (const auto op : ops) {
            if (op && !ledger->release(op)) {
                mark_unavailable();
            }
        }
    };
    try {
        if (ops.empty() || std::any_of(ops.begin(), ops.end(),
                [](const auto op) { return !op; })) {
            release_supplied();
            return false;
        }
        std::sort(ops.begin(), ops.end());
        ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
        const auto association = associations.find(key);
        if (association == associations.end()) {
            release_supplied();
            return false;
        }
        const auto item = catalog.find(association->second.v);
        if (item == catalog.end() || !item->second.release_ops.empty()) {
            release_supplied();
            return false;
        }
        item->second.release_ops = std::move(ops);
        return true;
    } catch (...) {
        release_supplied();
        mark_unavailable();
        return false;
    }
}

void server_retention_sidecar_store::release_recovery_pin(
        void * context) noexcept {
    auto * entry = static_cast<catalog_entry *>(context);
    if (!entry || entry->recovery_pins == 0) {
        return;
    }
    entry->recovery_pins--;
    if (entry->recovery_pins == 0 && entry->retire_pending &&
        entry->owner && entry->artifact.v != 0) {
        auto * owner = entry->owner;
        const auto found = owner->catalog.find(entry->artifact.v);
        if (found != owner->catalog.end() && &found->second == entry) {
            owner->retire_catalog_entry(found);
        } else {
            owner->mark_unavailable();
        }
    }
}

server_cache_recovery_pin
server_retention_sidecar_store::acquire_recovery_pin(
        const server_retention_instance_key & key) noexcept {
    try {
        const auto association = associations.find(key);
        if (association == associations.end()) {
            return {};
        }
        const auto item = catalog.find(association->second.v);
        if (item == catalog.end() || item->second.release_ops.empty() ||
            item->second.recovery_pins == UINT32_MAX) {
            return {};
        }
        item->second.recovery_pins++;
        auto pin = server_cache_recovery_pin::acquire(
            &item->second,
            release_recovery_pin,
            { association->second },
            item->second.release_ops);
        if (!pin.valid()) {
            item->second.recovery_pins--;
        }
        return pin;
    } catch (...) {
        mark_unavailable();
        return {};
    }
}

void server_retention_sidecar_store::retire_after_committed_release(
        const server_retention_instance_key & key) noexcept {
    const auto association = associations.find(key);
    if (association == associations.end()) {
        mark_unavailable();
        return;
    }
    const auto item = catalog.find(association->second.v);
    if (item == catalog.end() || item->second.recovery_pins != 0 ||
        item->second.release_ops.empty()) {
        mark_unavailable();
        return;
    }
    item->second.release_ops.clear();
    retire_association(association);
}

bool server_retention_sidecar_store::retire_slot_after_committed_release(
        int32_t owner_slot,
        const std::vector<llama_cache_acct_artifact_id> & selected_attention,
        const std::vector<llama_cache_acct_artifact_id> & selected_recurrent) noexcept {
    const auto selected = [&](llama_cache_acct_artifact_id artifact) {
        return std::find(selected_attention.begin(), selected_attention.end(), artifact) !=
                   selected_attention.end() ||
               std::find(selected_recurrent.begin(), selected_recurrent.end(), artifact) !=
                   selected_recurrent.end();
    };
    bool found = false;
    for (const auto & association : associations) {
        if (association.first.owner_slot != owner_slot) {
            continue;
        }
        found = true;
        const auto item = catalog.find(association.second.v);
        if (item == catalog.end() || item->second.recovery_pins != 0 ||
            !selected(association.second)) {
            mark_unavailable();
            return false;
        }
    }
    if (!found) {
        mark_unavailable();
        return false;
    }
    for (auto it = associations.begin(); it != associations.end();) {
        if (it->first.owner_slot != owner_slot) {
            ++it;
            continue;
        }
        auto victim = it++;
        const auto item = catalog.find(victim->second.v);
        GGML_ASSERT(item != catalog.end());
        GGML_ASSERT(item->second.recovery_pins == 0);
        item->second.release_ops.clear();
        item->second.accounting_op = {};
        retire_association(victim);
    }
    return true;
}

std::vector<server_retention_candidate>
server_retention_sidecar_store::candidate_snapshot() const noexcept {
    try {
        std::vector<server_retention_candidate> out;
        out.reserve(associations.size());
        for (const auto & [key, artifact] : associations) {
            server_retention_candidate candidate;
            candidate.artifact_id = artifact;
            candidate.instance_key = key;
            const auto item = catalog.find(artifact.v);
            if (item != catalog.end()) {
                candidate.record = item->second.record;
                const auto lineage_key = qualified_lineage_id(
                    candidate.record.stamp.pool,
                    candidate.record.stamp.lineage_id);
                const auto lineage = lineages.find(lineage_key);
                if (lineage != lineages.end() &&
                    !lineage->second.admitted) {
                    continue;
                }
                if (lineage == lineages.end()) {
                    candidate.avail = server_retention_candidate_availability::
                        backing_missing_or_stale;
                    out.push_back(std::move(candidate));
                    continue;
                }
                candidate.lineage = lineage->second.record;
                candidate.provenance_op = item->second.accounting_op;
                candidate.release_ops = item->second.release_ops;
                candidate.avail =
                    server_retention_candidate_availability::available;
            }
            out.push_back(std::move(candidate));
        }
        std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) {
            if (a.record.stamp.pool != b.record.stamp.pool) {
                return a.record.stamp.pool < b.record.stamp.pool;
            }
            if (a.record.stamp.stable_id != b.record.stamp.stable_id) {
                return a.record.stamp.stable_id < b.record.stamp.stable_id;
            }
            return a.artifact_id.v < b.artifact_id.v;
        });
        return out;
    } catch (...) {
        server_retention_candidate failed;
        failed.avail =
            server_retention_candidate_availability::backing_missing_or_stale;
        try {
            return { std::move(failed) };
        } catch (...) {
            return {};
        }
    }
}

common_retention_sidecar_snapshot
server_retention_sidecar_store::snapshot() const noexcept {
    try {
        common_retention_sidecar_snapshot out;
        for (const auto pool : {
                common_retention_pool::attention,
                common_retention_pool::recurrent }) {
            const size_t i = size_t(pool);
            out.recency_high_water[i] = allocator.recency_high_water(pool);
            out.stable_high_water[i] = allocator.stable_high_water(pool);
            out.lineage_high_water[i] = allocator.lineage_high_water(pool);
        }
        out.competition_epoch = competition_epoch;
        out.lineages.reserve(lineages.size());
        for (const auto & item : lineages) {
            if (item.second.admitted) {
                out.lineages.push_back(item.second.record);
            }
        }
        std::sort(out.lineages.begin(), out.lineages.end(),
            [](const auto & a, const auto & b) {
                if (a.pool != b.pool) {
                    return a.pool < b.pool;
                }
                return a.lineage_id < b.lineage_id;
            });
        out.artifacts.reserve(catalog.size());
        for (const auto & item : catalog) {
            const auto lineage = lineages.find(qualified_lineage_id(
                item.second.record.stamp.pool,
                item.second.record.stamp.lineage_id));
            if (lineage != lineages.end() && lineage->second.admitted) {
                out.artifacts.push_back(item.second.record);
            }
        }
        std::sort(out.artifacts.begin(), out.artifacts.end(),
            [](const auto & a, const auto & b) {
                if (a.stamp.pool != b.stamp.pool) {
                    return a.stamp.pool < b.stamp.pool;
                }
                return a.stamp.stable_id < b.stamp.stable_id;
            });
        return out;
    } catch (...) {
        common_retention_sidecar_snapshot invalid;
        invalid.version = 0;
        return invalid;
    }
}

bool server_retention_sidecar_store::import_snapshot(
        const common_retention_sidecar_snapshot & imported) noexcept {
    if (!allocator.import_snapshot(imported)) {
        mark_unavailable();
        return false;
    }
    competition_epoch = std::max(
        competition_epoch, imported.competition_epoch);
    return true;
}

bool server_retention_sidecar_store::export_bytes(
        std::vector<uint8_t> & out) const noexcept {
    return common_retention_sidecar_encode(snapshot(), out);
}
