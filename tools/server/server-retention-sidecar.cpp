#include "server-retention-sidecar.h"
#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace {
constexpr size_t MAX_CATALOG_ARTIFACTS = SERVER_RETENTION_MAX_CANDIDATES;
constexpr uint64_t MAX_TURN_TABLE_BYTES = 16ull*1024*1024;
constexpr uint64_t MAX_PREFIX_INDEX_TOKEN_BYTES = 16ull*1024*1024;
constexpr uint64_t MAX_PREFIX_INDEX_EDGE_BYTES = 16ull*1024*1024;
constexpr size_t MAX_PREFIX_INDEX_NODES = 2*SERVER_RETENTION_MAX_CANDIDATES + 1;
constexpr size_t MAX_PREFIX_INDEX_LINEAGE_REFS = 65536;
constexpr uint64_t MAX_PREFIX_TRACKING_SCOPE_BYTES = 2ull*1024*1024;
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

struct server_retention_prefix_index::impl {
    struct node;

    struct edge {
        std::vector<llama_token> label;
        std::unique_ptr<node> child;
    };

    struct node {
        uint32_t total_refs = 0;
        uint32_t terminal_refs = 0;
        uint64_t terminal_head = 0;
        std::unordered_map<uint64_t, uint32_t> lineage_refs;
        std::unordered_map<llama_token, edge> edges;
    };

    struct token_block {
        std::vector<llama_token> values;
    };

    struct artifact_record {
        uint64_t artifact = 0;
        uint64_t lineage_id = 0;
        node * terminal_node = nullptr;
        uint64_t terminal_prev = 0;
        uint64_t terminal_next = 0;
        std::shared_ptr<const token_block> tokens;
    };

    node root;
    std::unordered_map<uint64_t, artifact_record> artifacts;
    uint64_t artifact_token_bytes = 0;
    uint64_t edge_token_bytes = 0;
    size_t nodes = 1;
    size_t lineage_ref_entries = 0;
    bool healthy = true;

    void poison() noexcept {
        healthy = false;
        artifacts.clear();
        root = {};
        artifact_token_bytes = 0;
        edge_token_bytes = 0;
        nodes = 1;
        lineage_ref_entries = 0;
    }

    bool add_ref(node & cur, uint64_t lineage_id) {
        if (cur.total_refs == UINT32_MAX) {
            return false;
        }
        auto found = cur.lineage_refs.find(lineage_id);
        if (found == cur.lineage_refs.end()) {
            if (lineage_ref_entries == MAX_PREFIX_INDEX_LINEAGE_REFS) {
                return false;
            }
            found = cur.lineage_refs.emplace(lineage_id, 0).first;
            lineage_ref_entries++;
        }
        if (found->second == UINT32_MAX) {
            return false;
        }
        cur.total_refs++;
        found->second++;
        return true;
    }

    bool insert(
            uint64_t artifact,
            uint64_t lineage_id,
            const std::vector<llama_token> & tokens,
            node * & terminal_node,
            uint64_t & terminal_prev,
            uint64_t & terminal_next) {
        node * cur = &root;
        size_t pos = 0;
        if (!add_ref(*cur, lineage_id)) {
            return false;
        }

        while (pos < tokens.size()) {
            auto found = cur->edges.find(tokens[pos]);
            if (found == cur->edges.end()) {
                if (nodes == MAX_PREFIX_INDEX_NODES) {
                    return false;
                }
                edge added;
                added.label.assign(tokens.begin() + pos, tokens.end());
                added.child = std::make_unique<node>();
                if (!add_ref(*added.child, lineage_id)) {
                    return false;
                }
                added.child->terminal_refs = 1;
                added.child->terminal_head = artifact;
                terminal_node = added.child.get();
                terminal_prev = 0;
                terminal_next = 0;
                const uint64_t added_bytes =
                    uint64_t(added.label.capacity())*sizeof(llama_token);
                if (added_bytes > MAX_PREFIX_INDEX_EDGE_BYTES - edge_token_bytes) {
                    return false;
                }
                cur->edges.emplace(tokens[pos], std::move(added));
                edge_token_bytes += added_bytes;
                nodes++;
                return true;
            }

            edge & current = found->second;
            size_t shared = 0;
            while (shared < current.label.size() &&
                   pos + shared < tokens.size() &&
                   current.label[shared] == tokens[pos + shared]) {
                shared++;
            }
            if (shared == current.label.size()) {
                pos += shared;
                cur = current.child.get();
                if (!cur || !add_ref(*cur, lineage_id)) {
                    return false;
                }
                continue;
            }
            if (shared == 0 || nodes + 1 > MAX_PREFIX_INDEX_NODES) {
                return false;
            }

            auto middle = std::make_unique<node>();
            middle->total_refs = current.child->total_refs;
            middle->lineage_refs = current.child->lineage_refs;
            if (lineage_ref_entries >
                MAX_PREFIX_INDEX_LINEAGE_REFS - middle->lineage_refs.size()) {
                return false;
            }
            lineage_ref_entries += middle->lineage_refs.size();
            if (!add_ref(*middle, lineage_id)) {
                return false;
            }

            edge old_suffix;
            old_suffix.label.assign(
                current.label.begin() + shared, current.label.end());
            std::vector<llama_token> prefix(
                current.label.begin(), current.label.begin() + shared);

            const bool ends_at_split = pos + shared == tokens.size();
            edge new_suffix;
            if (!ends_at_split) {
                if (nodes + 2 > MAX_PREFIX_INDEX_NODES) {
                    return false;
                }
                new_suffix.label.assign(
                    tokens.begin() + pos + shared, tokens.end());
                new_suffix.child = std::make_unique<node>();
                if (!add_ref(*new_suffix.child, lineage_id)) {
                    return false;
                }
                new_suffix.child->terminal_refs = 1;
                new_suffix.child->terminal_head = artifact;
                terminal_node = new_suffix.child.get();
                terminal_prev = 0;
                terminal_next = 0;
            } else {
                middle->terminal_refs = 1;
                middle->terminal_head = artifact;
                terminal_node = middle.get();
                terminal_prev = 0;
                terminal_next = 0;
            }

            middle->edges.reserve(ends_at_split ? 1 : 2);
            const llama_token old_first = old_suffix.label.front();
            middle->edges.emplace(old_first, std::move(old_suffix));
            if (!ends_at_split) {
                const llama_token new_first = new_suffix.label.front();
                middle->edges.emplace(new_first, std::move(new_suffix));
            }

            uint64_t replacement_bytes =
                uint64_t(prefix.capacity())*sizeof(llama_token);
            for (const auto & item : middle->edges) {
                const uint64_t bytes =
                    uint64_t(item.second.label.capacity())*sizeof(llama_token);
                if (bytes > UINT64_MAX - replacement_bytes) {
                    return false;
                }
                replacement_bytes += bytes;
            }
            const uint64_t old_bytes =
                uint64_t(current.label.capacity())*sizeof(llama_token);
            if (replacement_bytes < old_bytes ||
                replacement_bytes - old_bytes >
                    MAX_PREFIX_INDEX_EDGE_BYTES - edge_token_bytes) {
                return false;
            }

            edge_token_bytes += replacement_bytes - old_bytes;
            old_suffix = {};
            auto old_child = std::move(current.child);
            auto old_edge = middle->edges.find(old_first);
            old_edge->second.child = std::move(old_child);
            current.label = std::move(prefix);
            current.child = std::move(middle);
            nodes += ends_at_split ? 1 : 2;
            return true;
        }

        if (cur->terminal_refs == UINT32_MAX) {
            return false;
        }
        terminal_node = cur;
        terminal_prev = 0;
        terminal_next = cur->terminal_head;
        if (terminal_next != 0) {
            const auto old_head = artifacts.find(terminal_next);
            if (old_head == artifacts.end() ||
                old_head->second.terminal_node != cur ||
                old_head->second.terminal_prev != 0) {
                return false;
            }
            old_head->second.terminal_prev = artifact;
        }
        cur->terminal_head = artifact;
        cur->terminal_refs++;
        return true;
    }

    static uint32_t lineage_refs_at(const node & cur, uint64_t lineage_id) {
        const auto found = cur.lineage_refs.find(lineage_id);
        return found == cur.lineage_refs.end() ? 0 : found->second;
    }

    bool remove_ref(node & cur, uint64_t lineage_id) noexcept {
        const auto found = cur.lineage_refs.find(lineage_id);
        if (cur.total_refs == 0 || found == cur.lineage_refs.end() ||
            found->second == 0) {
            return false;
        }
        cur.total_refs--;
        found->second--;
        if (found->second == 0) {
            cur.lineage_refs.erase(found);
            lineage_ref_entries--;
        }
        return true;
    }

    bool retire_artifact(const artifact_record & record) noexcept {
        struct path_entry {
            node * parent;
            llama_token edge_key;
            node * child;
        };
        if (!record.tokens) {
            return false;
        }
        const auto & tokens = record.tokens->values;
        std::vector<path_entry> path;
        try {
            path.reserve(std::min(tokens.size(), nodes));
        } catch (...) {
            return false;
        }
        node * cur = &root;
        size_t pos = 0;
        if (!remove_ref(*cur, record.lineage_id)) {
            return false;
        }
        while (pos < tokens.size()) {
            auto found = cur->edges.find(tokens[pos]);
            if (found == cur->edges.end() || !found->second.child ||
                found->second.label.size() > tokens.size() - pos ||
                !std::equal(
                    found->second.label.begin(), found->second.label.end(),
                    tokens.begin() + pos)) {
                return false;
            }
            node * child = found->second.child.get();
            path.push_back({ cur, found->first, child });
            if (!remove_ref(*child, record.lineage_id)) {
                return false;
            }
            pos += found->second.label.size();
            cur = child;
        }
        if (cur->terminal_refs == 0 || record.artifact == 0 ||
            record.terminal_node != cur) {
            return false;
        }
        if (record.terminal_prev == 0) {
            if (cur->terminal_head != record.artifact) {
                return false;
            }
            cur->terminal_head = record.terminal_next;
        } else {
            const auto previous = artifacts.find(record.terminal_prev);
            if (previous == artifacts.end() ||
                previous->second.terminal_node != cur ||
                previous->second.terminal_next != record.artifact) {
                return false;
            }
            previous->second.terminal_next = record.terminal_next;
        }
        if (record.terminal_next != 0) {
            const auto next = artifacts.find(record.terminal_next);
            if (next == artifacts.end() ||
                next->second.terminal_node != cur ||
                next->second.terminal_prev != record.artifact) {
                return false;
            }
            next->second.terminal_prev = record.terminal_prev;
        }
        cur->terminal_refs--;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            auto edge_it = it->parent->edges.find(it->edge_key);
            if (edge_it == it->parent->edges.end() ||
                edge_it->second.child.get() != it->child) {
                return false;
            }
            if (it->child->total_refs == 0) {
                if (!it->child->edges.empty() || !it->child->lineage_refs.empty() ||
                    it->child->terminal_refs != 0 ||
                    it->child->terminal_head != 0) {
                    return false;
                }
                edge_token_bytes -=
                    uint64_t(edge_it->second.label.capacity())*sizeof(llama_token);
                it->parent->edges.erase(edge_it);
                nodes--;
                continue;
            }

            // A retired one-shot branch can leave its split point as a
            // nonterminal unary node. Recompress it immediately so historical
            // branch churn cannot consume the live-cardinality node bound.
            while (edge_it->second.child &&
                   edge_it->second.child->terminal_refs == 0 &&
                   edge_it->second.child->terminal_head == 0 &&
                   edge_it->second.child->edges.size() == 1) {
                auto * unary = edge_it->second.child.get();
                auto only = unary->edges.begin();
                if (!only->second.child ||
                    lineage_ref_entries < unary->lineage_refs.size()) {
                    return false;
                }
                std::vector<llama_token> merged;
                try {
                    if (only->second.label.size() >
                        std::numeric_limits<size_t>::max() -
                            edge_it->second.label.size()) {
                        return false;
                    }
                    merged.reserve(
                        edge_it->second.label.size() + only->second.label.size());
                    merged.insert(
                        merged.end(), edge_it->second.label.begin(),
                        edge_it->second.label.end());
                    merged.insert(
                        merged.end(), only->second.label.begin(),
                        only->second.label.end());
                } catch (...) {
                    return false;
                }
                const uint64_t old_bytes =
                    (uint64_t(edge_it->second.label.capacity()) +
                     uint64_t(only->second.label.capacity()))*sizeof(llama_token);
                const uint64_t new_bytes =
                    uint64_t(merged.capacity())*sizeof(llama_token);
                if (old_bytes > edge_token_bytes ||
                    new_bytes >
                        MAX_PREFIX_INDEX_EDGE_BYTES - (edge_token_bytes - old_bytes)) {
                    return false;
                }
                auto grandchild = std::move(only->second.child);
                lineage_ref_entries -= unary->lineage_refs.size();
                edge_token_bytes = edge_token_bytes - old_bytes + new_bytes;
                edge_it->second.label = std::move(merged);
                edge_it->second.child = std::move(grandchild);
                nodes--;
            }
        }
        return true;
    }
};

server_retention_prefix_index::server_retention_prefix_index() noexcept {
    try {
        pimpl = std::make_unique<impl>();
    } catch (...) {
    }
}

server_retention_prefix_index::~server_retention_prefix_index() = default;

bool server_retention_prefix_index::publish(
        llama_cache_acct_artifact_id artifact,
        uint64_t lineage_id,
        const std::vector<llama_token> & tokens) noexcept {
    if (!pimpl || !pimpl->healthy) {
        return false;
    }
    if (artifact.v == 0 || lineage_id == 0 || tokens.empty() ||
        pimpl->artifacts.count(artifact.v) != 0) {
        return false;
    }
    if (pimpl->artifacts.size() == SERVER_RETENTION_MAX_CANDIDATES) {
        pimpl->poison();
        return false;
    }
    if (tokens.size() > MAX_PREFIX_INDEX_TOKEN_BYTES/sizeof(llama_token) ||
        uint64_t(tokens.size())*sizeof(llama_token) >
            MAX_PREFIX_INDEX_TOKEN_BYTES - pimpl->artifact_token_bytes) {
        pimpl->poison();
        return false;
    }
    try {
        impl::artifact_record record;
        record.artifact = artifact.v;
        record.lineage_id = lineage_id;
        auto block = std::make_shared<impl::token_block>();
        block->values = tokens;
        record.tokens = std::move(block);
        const uint64_t bytes =
            uint64_t(record.tokens->values.capacity())*sizeof(llama_token);
        if (bytes > MAX_PREFIX_INDEX_TOKEN_BYTES - pimpl->artifact_token_bytes ||
            !pimpl->insert(
                artifact.v, lineage_id, record.tokens->values,
                record.terminal_node, record.terminal_prev,
                record.terminal_next)) {
            pimpl->poison();
            return false;
        }
        pimpl->artifact_token_bytes += bytes;
        pimpl->artifacts.emplace(artifact.v, std::move(record));
        return true;
    } catch (...) {
        pimpl->poison();
        return false;
    }
}

bool server_retention_prefix_index::clone(
        llama_cache_acct_artifact_id source,
        llama_cache_acct_artifact_id destination,
        uint64_t lineage_id) noexcept {
    if (!pimpl || !pimpl->healthy || source.v == 0 || destination.v == 0 ||
        lineage_id == 0 || source == destination ||
        pimpl->artifacts.count(destination.v) != 0 ||
        pimpl->artifacts.size() == SERVER_RETENTION_MAX_CANDIDATES) {
        return false;
    }
    const auto found = pimpl->artifacts.find(source.v);
    if (found == pimpl->artifacts.end() || !found->second.tokens) {
        return false;
    }
    try {
        impl::artifact_record record;
        record.artifact = destination.v;
        record.lineage_id = lineage_id;
        record.tokens = found->second.tokens;
        if (!pimpl->insert(
                destination.v, lineage_id, record.tokens->values,
                record.terminal_node, record.terminal_prev,
                record.terminal_next) ||
            !pimpl->artifacts.emplace(
                destination.v, std::move(record)).second) {
            pimpl->poison();
            return false;
        }
        return true;
    } catch (...) {
        pimpl->poison();
        return false;
    }
}

bool server_retention_prefix_index::visit_prefixes(
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_visitor visitor) const noexcept {
    if (!pimpl || !pimpl->healthy || tokens.empty() || !visitor) {
        return false;
    }
    const impl::node * cur = &pimpl->root;
    size_t pos = 0;
    while (pos < tokens.size()) {
        const auto found = cur->edges.find(tokens[pos]);
        if (found == cur->edges.end() || !found->second.child ||
            found->second.label.size() > tokens.size() - pos ||
            !std::equal(
                found->second.label.begin(), found->second.label.end(),
                tokens.begin() + pos)) {
            break;
        }
        pos += found->second.label.size();
        cur = found->second.child.get();
        uint64_t artifact = cur->terminal_head;
        uint64_t previous = 0;
        uint32_t visited = 0;
        while (artifact != 0) {
            const auto record = pimpl->artifacts.find(artifact);
            if (record == pimpl->artifacts.end() ||
                !record->second.tokens ||
                record->second.tokens->values.size() != pos ||
                record->second.terminal_node != cur ||
                record->second.terminal_prev != previous ||
                ++visited > cur->terminal_refs) {
                return false;
            }
            if (!visitor(
                    context, llama_cache_acct_artifact_id { artifact }, pos)) {
                return true;
            }
            previous = artifact;
            artifact = record->second.terminal_next;
        }
        if (visited != cur->terminal_refs) {
            return false;
        }
    }
    return true;
}

bool server_retention_prefix_index::visit_longest_common_prefix(
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_visitor visitor) const noexcept {
    if (!pimpl || !pimpl->healthy || tokens.empty() || !visitor) {
        return false;
    }

    const impl::node * cur = &pimpl->root;
    size_t path_tokens = 0;
    size_t common_tokens = 0;
    while (path_tokens < tokens.size()) {
        const auto found = cur->edges.find(tokens[path_tokens]);
        if (found == cur->edges.end()) {
            common_tokens = path_tokens;
            break;
        }
        const auto & edge = found->second;
        if (!edge.child || edge.label.empty() ||
            edge.label.front() != found->first) {
            return false;
        }
        size_t shared = 0;
        while (shared < edge.label.size() &&
               path_tokens + shared < tokens.size() &&
               edge.label[shared] == tokens[path_tokens + shared]) {
            shared++;
        }
        if (shared != edge.label.size()) {
            common_tokens = path_tokens + shared;
            cur = edge.child.get();
            path_tokens += edge.label.size();
            break;
        }
        path_tokens += shared;
        common_tokens = path_tokens;
        cur = edge.child.get();
    }
    if (common_tokens == 0) {
        return true;
    }

    struct pending_node {
        const impl::node * value;
        size_t path_tokens;
    };
    try {
        std::vector<pending_node> pending;
        std::vector<uint64_t> artifacts;
        pending.reserve(std::min<size_t>(
            cur->total_refs, SERVER_RETENTION_MAX_CANDIDATES));
        artifacts.reserve(std::min<size_t>(
            cur->total_refs, SERVER_RETENTION_MAX_CANDIDATES));
        pending.push_back({ cur, path_tokens });

        while (!pending.empty()) {
            const auto current = pending.back();
            pending.pop_back();
            if (!current.value || current.value->total_refs == 0 ||
                current.value->total_refs > SERVER_RETENTION_MAX_CANDIDATES ||
                (current.value->terminal_refs == 0) !=
                    (current.value->terminal_head == 0)) {
                return false;
            }

            uint64_t artifact = current.value->terminal_head;
            uint64_t previous = 0;
            uint32_t visited = 0;
            while (artifact != 0) {
                const auto record = pimpl->artifacts.find(artifact);
                if (record == pimpl->artifacts.end() ||
                    !record->second.tokens ||
                    record->second.tokens->values.size() !=
                        current.path_tokens ||
                    record->second.terminal_node != current.value ||
                    record->second.terminal_prev != previous ||
                    ++visited > current.value->terminal_refs ||
                    artifacts.size() == SERVER_RETENTION_MAX_CANDIDATES) {
                    return false;
                }
                artifacts.push_back(artifact);
                previous = artifact;
                artifact = record->second.terminal_next;
            }
            if (visited != current.value->terminal_refs) {
                return false;
            }

            for (const auto & item : current.value->edges) {
                const auto & edge = item.second;
                if (!edge.child || edge.label.empty() ||
                    edge.label.front() != item.first ||
                    edge.label.size() >
                        std::numeric_limits<size_t>::max() -
                            current.path_tokens ||
                    edge.child->total_refs == 0 ||
                    edge.child->total_refs >
                        SERVER_RETENTION_MAX_CANDIDATES) {
                    return false;
                }
                pending.push_back({
                    edge.child.get(), current.path_tokens + edge.label.size(),
                });
            }
        }
        if (artifacts.size() != cur->total_refs) {
            return false;
        }
        std::sort(artifacts.begin(), artifacts.end());
        if (std::adjacent_find(artifacts.begin(), artifacts.end()) !=
            artifacts.end()) {
            return false;
        }
        for (const uint64_t artifact : artifacts) {
            if (!visitor(
                    context, llama_cache_acct_artifact_id { artifact },
                    common_tokens)) {
                break;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

void server_retention_prefix_index::retire(
        llama_cache_acct_artifact_id artifact) noexcept {
    if (!pimpl || !pimpl->healthy || artifact.v == 0) {
        return;
    }
    const auto found = pimpl->artifacts.find(artifact.v);
    if (found == pimpl->artifacts.end()) {
        return;
    }
    if (!found->second.tokens) {
        pimpl->poison();
        return;
    }
    const bool last_token_owner = found->second.tokens.use_count() == 1;
    const uint64_t bytes = uint64_t(
        found->second.tokens->values.capacity())*sizeof(llama_token);
    if (!pimpl->retire_artifact(found->second) ||
        (last_token_owner && bytes > pimpl->artifact_token_bytes)) {
        pimpl->poison();
        return;
    }
    pimpl->artifacts.erase(found);
    if (last_token_owner) {
        pimpl->artifact_token_bytes -= bytes;
    }
}

bool server_retention_prefix_index::external_shared_coverage(
        llama_cache_acct_artifact_id artifact,
        uint64_t & coverage_tokens,
        llama_cache_acct_artifact_id excluded,
        uint64_t * compared_tokens) const noexcept {
    coverage_tokens = 0;
    if (!pimpl || !pimpl->healthy || artifact.v == 0) {
        return false;
    }
    const auto found = pimpl->artifacts.find(artifact.v);
    if (found == pimpl->artifacts.end()) {
        return false;
    }
    const auto & record = found->second;
    const impl::artifact_record * excluded_record = nullptr;
    if (excluded.v) {
        const auto excluded_found = pimpl->artifacts.find(excluded.v);
        if (excluded_found == pimpl->artifacts.end() ||
            !excluded_found->second.tokens) {
            return false;
        }
        excluded_record = &excluded_found->second;
    }
    const impl::node * cur = &pimpl->root;
    size_t pos = 0;
    if (!record.tokens) {
        return false;
    }
    const auto & tokens = record.tokens->values;
    bool excluded_still_matches = excluded_record &&
        excluded_record->lineage_id != record.lineage_id;
    while (pos < tokens.size()) {
        const size_t edge_begin = pos;
        const auto edge = cur->edges.find(tokens[pos]);
        if (edge == cur->edges.end() || !edge->second.child ||
            edge->second.label.size() > tokens.size() - pos ||
            !std::equal(
                edge->second.label.begin(), edge->second.label.end(),
                tokens.begin() + pos)) {
            return false;
        }
        pos += edge->second.label.size();
        cur = edge->second.child.get();
        if (excluded_still_matches) {
            const uint64_t compared = uint64_t(pos - edge_begin);
            if (compared_tokens && compared >
                    UINT64_MAX - *compared_tokens) {
                return false;
            }
            excluded_still_matches =
                excluded_record->tokens->values.size() >= pos &&
                std::equal(
                    tokens.begin() + edge_begin, tokens.begin() + pos,
                    excluded_record->tokens->values.begin() + edge_begin);
            if (compared_tokens) {
                *compared_tokens += compared;
            }
        }
        const uint32_t same_lineage_refs =
            impl::lineage_refs_at(*cur, record.lineage_id);
        if (same_lineage_refs > cur->total_refs) {
            return false;
        }
        uint32_t external_refs = cur->total_refs - same_lineage_refs;
        if (excluded_still_matches) {
            if (external_refs == 0) {
                return false;
            }
            external_refs--;
        }
        if (external_refs != 0) {
            coverage_tokens = pos;
        } else {
            break;
        }
    }
    return true;
}

bool server_retention_prefix_index::available() const noexcept {
    return pimpl && pimpl->healthy;
}

size_t server_retention_prefix_index::size() const noexcept {
    return available() ? pimpl->artifacts.size() : 0;
}

size_t server_retention_prefix_index::node_count() const noexcept {
    return available() ? pimpl->nodes : 0;
}

uint64_t server_retention_prefix_index::token_bytes() const noexcept {
    return available() ? pimpl->artifact_token_bytes + pimpl->edge_token_bytes : 0;
}

uint64_t server_retention_prefix_index::source_token_bytes() const noexcept {
    return available() ? pimpl->artifact_token_bytes : 0;
}

struct server_retention_sidecar_store::prefix_tracking {
    struct scope_entry {
        std::string exact_scope;
        server_retention_prefix_index index;
        uint32_t refs = 0;
    };

    struct artifact_entry {
        scope_entry * scope = nullptr;
        server_retention_instance_key instance;
    };

    std::unordered_map<std::string, std::unique_ptr<scope_entry>> scopes;
    std::unordered_map<uint64_t, artifact_entry> artifacts;
    uint64_t source_token_bytes = 0;
    uint64_t scope_bytes = 0;
    bool healthy = true;

    void poison() noexcept {
        healthy = false;
        artifacts.clear();
        scopes.clear();
        source_token_bytes = 0;
        scope_bytes = 0;
    }

    bool publish(
            llama_cache_acct_artifact_id artifact,
            common_retention_pool pool,
            uint64_t lineage_id,
            const server_retention_instance_key & instance,
            const std::string & exact_scope,
            const std::vector<llama_token> & tokens) noexcept {
        if (!healthy) {
            return false;
        }
        if (artifact.v == 0 || lineage_id == 0 ||
            pool >= common_retention_pool::_count || exact_scope.empty() ||
            tokens.empty() || artifacts.count(artifact.v) != 0) {
            poison();
            return false;
        }
        if (exact_scope.size() >
            MAX_PREFIX_TRACKING_SCOPE_BYTES/2 - 1) {
            poison();
            return false;
        }
        if (artifacts.size() == SERVER_RETENTION_MAX_CANDIDATES ||
            tokens.size() > MAX_PREFIX_INDEX_TOKEN_BYTES/sizeof(llama_token)) {
            poison();
            return false;
        }
        const uint64_t token_bytes =
            uint64_t(tokens.size())*sizeof(llama_token);
        if (token_bytes > MAX_PREFIX_INDEX_TOKEN_BYTES - source_token_bytes) {
            poison();
            return false;
        }
        try {
            std::string scope_key;
            scope_key.reserve(1 + exact_scope.size());
            scope_key.push_back(char(pool));
            scope_key.append(exact_scope);
            auto scope = scopes.find(scope_key);
            if (scope == scopes.end()) {
                if (exact_scope.size() + 1 >
                    (MAX_PREFIX_TRACKING_SCOPE_BYTES - scope_bytes)/2) {
                    poison();
                    return false;
                }
                auto value = std::make_unique<scope_entry>();
                value->exact_scope = scope_key;
                auto inserted = scopes.emplace(scope_key, std::move(value));
                if (!inserted.second) {
                    poison();
                    return false;
                }
                scope = inserted.first;
                const uint64_t added_scope_bytes =
                    uint64_t(scope->first.capacity()) +
                    uint64_t(scope->second->exact_scope.capacity());
                if (added_scope_bytes >
                    MAX_PREFIX_TRACKING_SCOPE_BYTES - scope_bytes) {
                    poison();
                    return false;
                }
                scope_bytes += added_scope_bytes;
            }
            auto * owner = scope->second.get();
            const uint64_t source_bytes_before =
                owner ? owner->index.source_token_bytes() : 0;
            if (!owner || owner->refs == UINT32_MAX ||
                !owner->index.publish(artifact, lineage_id, tokens)) {
                poison();
                return false;
            }
            const uint64_t source_bytes_after =
                owner->index.source_token_bytes();
            if (source_bytes_after < source_bytes_before ||
                source_bytes_after - source_bytes_before >
                    MAX_PREFIX_INDEX_TOKEN_BYTES - source_token_bytes) {
                poison();
                return false;
            }
            const auto inserted_artifact = artifacts.emplace(
                artifact.v, artifact_entry { owner, instance });
            if (!inserted_artifact.second) {
                poison();
                return false;
            }
            owner->refs++;
            source_token_bytes += source_bytes_after - source_bytes_before;
            return true;
        } catch (...) {
            poison();
            return false;
        }
    }

    bool clone(
            llama_cache_acct_artifact_id source,
            llama_cache_acct_artifact_id destination,
            uint64_t lineage_id,
            const server_retention_instance_key & instance) noexcept {
        if (!healthy || source.v == 0 || destination.v == 0 ||
            lineage_id == 0 || artifacts.size() ==
                SERVER_RETENTION_MAX_CANDIDATES ||
            artifacts.count(destination.v) != 0) {
            return false;
        }
        const auto found = artifacts.find(source.v);
        if (found == artifacts.end() || !found->second.scope ||
            found->second.scope->refs == UINT32_MAX) {
            return false;
        }
        auto * owner = found->second.scope;
        const uint64_t source_bytes_before =
            owner->index.source_token_bytes();
        try {
            if (!owner->index.clone(source, destination, lineage_id) ||
                owner->index.source_token_bytes() != source_bytes_before ||
                !artifacts.emplace(
                    destination.v, artifact_entry { owner, instance }).second) {
                poison();
                return false;
            }
            owner->refs++;
            return true;
        } catch (...) {
            poison();
            return false;
        }
    }

    void retire(llama_cache_acct_artifact_id artifact) noexcept {
        if (!healthy || artifact.v == 0) {
            return;
        }
        const auto found = artifacts.find(artifact.v);
        if (found == artifacts.end()) {
            poison();
            return;
        }
        auto * owner = found->second.scope;
        if (!owner || owner->refs == 0) {
            poison();
            return;
        }
        const uint64_t source_bytes_before =
            owner->index.source_token_bytes();
        const size_t indexed_before = owner->index.size();
        owner->index.retire(artifact);
        const uint64_t source_bytes_after =
            owner->index.source_token_bytes();
        if (!owner->index.available() || indexed_before == 0 ||
            owner->index.size() + 1 != indexed_before ||
            source_bytes_after > source_bytes_before ||
            source_bytes_before - source_bytes_after > source_token_bytes) {
            poison();
            return;
        }
        source_token_bytes -= source_bytes_before - source_bytes_after;
        artifacts.erase(found);
        owner->refs--;
        if (owner->refs == 0) {
            const auto scope = scopes.find(owner->exact_scope);
            if (scope == scopes.end() || scope->second.get() != owner) {
                poison();
                return;
            }
            const uint64_t removed_scope_bytes =
                uint64_t(scope->first.capacity()) +
                uint64_t(owner->exact_scope.capacity());
            if (removed_scope_bytes > scope_bytes) {
                poison();
                return;
            }
            scope_bytes -= removed_scope_bytes;
            scopes.erase(scope);
        }
    }

    bool coverage(
            llama_cache_acct_artifact_id artifact,
            uint64_t & out,
            llama_cache_acct_artifact_id excluded = {}) const noexcept {
        out = 0;
        if (!healthy || artifact.v == 0) {
            return false;
        }
        const auto found = artifacts.find(artifact.v);
        if (found == artifacts.end() || !found->second.scope) {
            return false;
        }
        // A scope with one artifact cannot contain cross-lineage coverage.
        // Avoid walking its private radix tree on every pressure projection;
        // the index is still the authority as soon as the scope is shared.
        const auto excluded_found = excluded.v
            ? artifacts.find(excluded.v) : artifacts.end();
        const bool excludes_same_scope = excluded_found != artifacts.end() &&
            excluded_found->second.scope == found->second.scope;
        if (excluded.v && excluded_found == artifacts.end()) {
            return false;
        }
        if (found->second.scope->refs == 1 ||
            (found->second.scope->refs == 2 && excludes_same_scope &&
             excluded != artifact)) {
            return true;
        }
        return found->second.scope->index.external_shared_coverage(
            artifact, out, excludes_same_scope ? excluded :
                llama_cache_acct_artifact_id {});
    }

    bool visit_impl(
            common_retention_pool pool,
            const std::string & exact_scope,
            const std::vector<llama_token> & tokens,
            void * context,
            server_retention_sidecar_store::prefix_instance_visitor visitor,
            bool longest_common_prefix)
            const noexcept {
        if (!healthy || pool >= common_retention_pool::_count ||
            exact_scope.empty() || tokens.empty() || !visitor) {
            return false;
        }
        try {
            std::string scope_key;
            scope_key.reserve(1 + exact_scope.size());
            scope_key.push_back(char(pool));
            scope_key.append(exact_scope);
            const auto scope = scopes.find(scope_key);
            if (scope == scopes.end() || !scope->second) {
                return true;
            }
            struct bridge {
                const prefix_tracking * owner;
                void * context;
                server_retention_sidecar_store::prefix_instance_visitor visitor;
                bool valid = true;
            } state { this, context, visitor, true };
            const auto bridge_visitor =
                [](void * opaque, llama_cache_acct_artifact_id artifact,
                   uint64_t prefix) noexcept {
                    auto & bridge_state = *static_cast<bridge *>(opaque);
                    const auto found =
                        bridge_state.owner->artifacts.find(artifact.v);
                    if (found == bridge_state.owner->artifacts.end()) {
                        bridge_state.valid = false;
                        return false;
                    }
                    return bridge_state.visitor(
                        bridge_state.context,
                        found->second.instance, prefix);
                };
            const bool walked = longest_common_prefix
                ? scope->second->index.visit_longest_common_prefix(
                    tokens, &state, bridge_visitor)
                : scope->second->index.visit_prefixes(
                    tokens, &state, bridge_visitor);
            return walked && state.valid;
        } catch (...) {
            return false;
        }
    }

    bool visit(
            common_retention_pool pool,
            const std::string & exact_scope,
            const std::vector<llama_token> & tokens,
            void * context,
            server_retention_sidecar_store::prefix_instance_visitor visitor)
            const noexcept {
        return visit_impl(
            pool, exact_scope, tokens, context, visitor, false);
    }

    bool visit_longest_common_prefix(
            common_retention_pool pool,
            const std::string & exact_scope,
            const std::vector<llama_token> & tokens,
            void * context,
            server_retention_sidecar_store::prefix_instance_visitor visitor)
            const noexcept {
        return visit_impl(
            pool, exact_scope, tokens, context, visitor, true);
    }
};

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

server_retention_sidecar_store::server_retention_sidecar_store() = default;

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

bool server_retention_sidecar_store::enable_prefix_tracking(
        bool force_failure_for_test) noexcept {
    prefix_tracking_requested = true;
    if (prefixes) {
        return prefixes->healthy;
    }
    if (force_failure_for_test) {
        return false;
    }
    try {
        prefixes = std::make_unique<prefix_tracking>();
    } catch (...) {
    }
    return prefixes && prefixes->healthy;
}

bool server_retention_sidecar_store::publish_prefix(
        const server_retention_instance_key & key,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens) noexcept {
    if (!prefixes) {
        return !prefix_tracking_requested;
    }
    const auto association = associations.find(key);
    if (association == associations.end()) {
        prefixes->poison();
        return false;
    }
    const auto artifact = catalog.find(association->second.v);
    if (artifact == catalog.end() ||
        artifact->second.record.kind ==
            common_retention_artifact_kind::checkpoint ||
        artifact->second.record.stamp.coverage_tokens != tokens.size()) {
        prefixes->poison();
        return false;
    }
    if (artifact->second.prefix_indexed) {
        return false;
    }
    if (!prefixes->publish(
        association->second,
        artifact->second.record.stamp.pool,
        artifact->second.record.stamp.lineage_id,
        key,
        exact_scope,
        tokens)) {
        return false;
    }
    artifact->second.prefix_indexed = true;
    return true;
}

bool server_retention_sidecar_store::clone_prefix(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept {
    if (!prefixes) {
        return !prefix_tracking_requested;
    }
    const auto source_assoc = associations.find(source);
    const auto destination_assoc = associations.find(destination);
    if (source_assoc == associations.end() ||
        destination_assoc == associations.end()) {
        prefixes->poison();
        return false;
    }
    const auto source_artifact = catalog.find(source_assoc->second.v);
    auto destination_artifact = catalog.find(destination_assoc->second.v);
    if (source_artifact == catalog.end() ||
        destination_artifact == catalog.end() ||
        !source_artifact->second.prefix_indexed ||
        destination_artifact->second.prefix_indexed ||
        source_artifact->second.record.stamp.pool !=
            destination_artifact->second.record.stamp.pool ||
        source_artifact->second.record.stamp.coverage_tokens !=
            destination_artifact->second.record.stamp.coverage_tokens ||
        !prefixes->clone(
            source_assoc->second, destination_assoc->second,
            destination_artifact->second.record.stamp.lineage_id,
            destination)) {
        return false;
    }
    destination_artifact->second.prefix_indexed = true;
    return true;
}

bool server_retention_sidecar_store::visit_prefix_instances(
        common_retention_pool pool,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_instance_visitor visitor) const noexcept {
    return prefixes && prefixes->visit(
        pool, exact_scope, tokens, context, visitor);
}

bool server_retention_sidecar_store::visit_longest_common_prefix_instances(
        common_retention_pool pool,
        const std::string & exact_scope,
        const std::vector<llama_token> & tokens,
        void * context,
        prefix_instance_visitor visitor) const noexcept {
    return prefixes && prefixes->visit_longest_common_prefix(
        pool, exact_scope, tokens, context, visitor);
}

bool server_retention_sidecar_store::prefix_tracking_available() const noexcept {
    return !prefix_tracking_requested || (prefixes && prefixes->healthy);
}

bool server_retention_sidecar_store::prefix_tracking_enabled() const noexcept {
    return prefix_tracking_requested;
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
                // Replacement detaches the old artifact just as surely as an
                // explicit association retirement. A recovery pin preserves
                // its payload, not its authority to contribute shared-prefix
                // coverage after the key has moved to a new artifact.
                if (prefixes && old_entry->second.prefix_indexed) {
                    prefixes->retire(old_artifact);
                    old_entry->second.prefix_indexed = false;
                }
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
        const auto * item = find_clone_source(source);
        if (!item) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        auto record = item->record;
        server_cache_lease_identity checkpoint_identity;
        const server_cache_lease_identity * checkpoint_identity_ptr = nullptr;
        if (item->checkpoint_identity_known) {
            checkpoint_identity = item->checkpoint_identity;
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

const server_retention_sidecar_store::catalog_entry *
server_retention_sidecar_store::find_clone_source(
        const server_retention_instance_key & source) const noexcept {
    const auto assoc = associations.find(source);
    if (assoc == associations.end()) {
        return nullptr;
    }
    const auto item = catalog.find(assoc->second.v);
    if (item == catalog.end()) {
        return nullptr;
    }
    const auto source_lineage = lineages.find(qualified_lineage_id(
        item->second.record.stamp.pool,
        item->second.record.stamp.lineage_id));
    if (source_lineage == lineages.end() ||
        !source_lineage->second.admitted) {
        return nullptr;
    }
    return &item->second;
}

bool server_retention_sidecar_store::clone_source_available(
        const server_retention_instance_key & source) const noexcept {
    return find_clone_source(source) != nullptr;
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
        server_retention_value_snapshot_visitor visitor,
        llama_cache_acct_artifact_id excluded) const noexcept {
    server_retention_value_snapshot_result result;
    if (!prefix_tracking_available()) {
        return result;
    }
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
        server_retention_value_snapshot value {
            association.second,
            association.first,
            record.kind,
            record.stamp,
            lineage->second.record,
            0,
        };
        if (prefixes && association.second != excluded &&
            !prefixes->coverage(
                association.second,
                value.external_shared_coverage_tokens, excluded)) {
            return result;
        }
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
        if (prefixes && entry->second.prefix_indexed) {
            prefixes->retire(artifact);
            entry->second.prefix_indexed = false;
        }
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
    if (prefixes && entry->second.prefix_indexed) {
        prefixes->retire(entry->second.artifact);
        entry->second.prefix_indexed = false;
    }
    if (ledger && !entry->second.release_ops.empty()) {
        auto release = llama_cache_prepare_release_set(
            *ledger, entry->second.release_ops,
            ledger->serial());
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
