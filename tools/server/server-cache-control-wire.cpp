#include "server-cache-control.h"

#include "../../src/llama-sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <limits>

using json = nlohmann::ordered_json;

namespace {

template<typename Enum, size_t N>
struct vocab_table {
    std::array<std::pair<std::string_view, Enum>, N> values;

    const char * name(Enum value, const char * fallback) const noexcept {
        for (const auto & item : values) {
            if (item.second == value) {
                return item.first.data();
            }
        }
        return fallback;
    }

    bool parse(std::string_view name, Enum & out) const noexcept {
        for (const auto & item : values) {
            if (item.first == name) {
                out = item.second;
                return true;
            }
        }
        return false;
    }
};

constexpr vocab_table<server_cache_lease_class, 3> LEASE_CLASSES { {{
    { "none", server_cache_lease_class::none },
    { "soft", server_cache_lease_class::soft },
    { "hard", server_cache_lease_class::hard },
}} };

constexpr vocab_table<server_cache_control_subject_kind, 4> SUBJECT_KINDS { {{
    { "live_prefix", server_cache_control_subject_kind::live_prefix },
    { "host_snapshot", server_cache_control_subject_kind::host_snapshot },
    { "vbr_reference", server_cache_control_subject_kind::vbr_reference },
    { "live_checkpoint", server_cache_control_subject_kind::live_checkpoint },
}} };

constexpr vocab_table<common_cache_family_role, 3> FAMILY_ROLES { {{
    { "main", common_cache_family_role::main },
    { "branch", common_cache_family_role::branch },
    { "background", common_cache_family_role::background },
}} };

constexpr vocab_table<server_cache_control_event_kind, 5> EVENT_KINDS { {{
    { "grant", server_cache_control_event_kind::grant },
    { "refuse", server_cache_control_event_kind::refuse },
    { "renew", server_cache_control_event_kind::renew },
    { "expire", server_cache_control_event_kind::expire },
    { "release", server_cache_control_event_kind::release },
}} };

struct operation_field {
    server_cache_control_operation operation;
    std::string_view field;
};

constexpr operation_field OPERATION_FIELDS[] = {
    { server_cache_control_operation::holder_create, "ttl_ms" },
    { server_cache_control_operation::holder_create, "idempotency_key" },
    { server_cache_control_operation::holder_reattach, "holder_recovery" },
    { server_cache_control_operation::holder_reattach, "ttl_ms" },
    { server_cache_control_operation::holder_close, "holder" },
    { server_cache_control_operation::family_register, "holder" },
    { server_cache_control_operation::family_register, "label" },
    { server_cache_control_operation::family_register, "idempotency_key" },
    { server_cache_control_operation::family_bind, "holder" },
    { server_cache_control_operation::family_bind, "family" },
    { server_cache_control_operation::family_bind, "role" },
    { server_cache_control_operation::family_bind, "idempotency_key" },
    { server_cache_control_operation::lease_acquire, "holder" },
    { server_cache_control_operation::lease_acquire, "class" },
    { server_cache_control_operation::lease_acquire, "ttl_ms" },
    { server_cache_control_operation::lease_acquire, "floor" },
    { server_cache_control_operation::lease_acquire, "family_binding" },
    { server_cache_control_operation::lease_acquire, "subject" },
    { server_cache_control_operation::lease_acquire, "fallback" },
    { server_cache_control_operation::lease_acquire, "allow_soft_fallback" },
    { server_cache_control_operation::lease_acquire, "idempotency_key" },
    { server_cache_control_operation::lease_inspect, "holder" },
    { server_cache_control_operation::lease_inspect, "lease" },
    { server_cache_control_operation::lease_renew, "holder" },
    { server_cache_control_operation::lease_renew, "lease" },
    { server_cache_control_operation::lease_renew, "ttl_ms" },
    { server_cache_control_operation::lease_renew, "fallback" },
    { server_cache_control_operation::lease_release, "holder" },
    { server_cache_control_operation::lease_release, "lease" },
    { server_cache_control_operation::events, "holder" },
    { server_cache_control_operation::events, "after_ordinal" },
    { server_cache_control_operation::events, "limit" },
};

struct selector_field {
    server_cache_control_subject_kind kind;
    std::string_view field;
};

constexpr selector_field SELECTOR_FIELDS[] = {
    { server_cache_control_subject_kind::live_prefix, "kind" },
    { server_cache_control_subject_kind::live_prefix, "slot_id" },
    { server_cache_control_subject_kind::host_snapshot, "kind" },
    { server_cache_control_subject_kind::host_snapshot, "prompt" },
    { server_cache_control_subject_kind::host_snapshot, "lora" },
    { server_cache_control_subject_kind::host_snapshot, "message_delimiters" },
    { server_cache_control_subject_kind::vbr_reference, "kind" },
    { server_cache_control_subject_kind::vbr_reference, "reference" },
    { server_cache_control_subject_kind::live_checkpoint, "kind" },
};

char handle_tag(server_cache_control_handle_kind kind) noexcept {
    switch (kind) {
        case server_cache_control_handle_kind::holder: return 'h';
        case server_cache_control_handle_kind::holder_recovery: return 'r';
        case server_cache_control_handle_kind::lease: return 'l';
        case server_cache_control_handle_kind::family: return 'f';
        case server_cache_control_handle_kind::family_binding: return 'b';
    }
    return '\0';
}

int hex_value(char value) noexcept {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

const char * protection_name(
        server_cache_control_protection_state value) noexcept {
    switch (value) {
        case server_cache_control_protection_state::current: return "current";
        case server_cache_control_protection_state::partially_stale: return "partially_stale";
        case server_cache_control_protection_state::subject_lost: return "subject_lost";
        case server_cache_control_protection_state::orphaned: return "orphaned";
        case server_cache_control_protection_state::released: return "released";
        case server_cache_control_protection_state::_count: break;
    }
    return "released";
}

const char * fallback_kind_name(
        server_cache_control_subject_kind value) noexcept {
    switch (value) {
        case server_cache_control_subject_kind::host_snapshot: return "retained_host";
        case server_cache_control_subject_kind::vbr_reference: return "sealed_artifact";
        default: return "unavailable";
    }
}

bool parse_handle(
        const json & body,
        const char * field,
        server_cache_control_handle_kind kind,
        server_cache_control_token & out) {
    if (!body.contains(field)) {
        return true;
    }
    return body[field].is_string() && server_cache_control_decode_handle(
        kind, body[field].get<std::string>(), out);
}

bool parse_ttl(const json & body, server_cache_control_request & out) {
    if (!body.contains("ttl_ms")) {
        return true;
    }
    if (!body["ttl_ms"].is_number_unsigned()) {
        return false;
    }
    const uint64_t milliseconds = body["ttl_ms"].get<uint64_t>();
    static constexpr uint64_t MAX_TTL_MS = 24ULL * 60 * 60 * 1000;
    if (milliseconds == 0 ||
        milliseconds > MAX_TTL_MS ||
        milliseconds > std::numeric_limits<uint64_t>::max() / 1000000ULL) {
        return false;
    }
    out.ttl_ns = milliseconds * 1000000ULL;
    return true;
}

json frontier_json(const server_cache_lease_frontier & frontier) {
    return {
        { "token_count", frontier.token_count },
        { "next_position", frontier.next_position },
    };
}

json payload_kind_json(server_cache_control_subject_kind kind) {
    switch (kind) {
        case server_cache_control_subject_kind::host_snapshot:
            return "fixed_state";
        case server_cache_control_subject_kind::vbr_reference:
            return "vbr_artifact";
        case server_cache_control_subject_kind::live_prefix:
        case server_cache_control_subject_kind::live_checkpoint:
        case server_cache_control_subject_kind::_count:
            return nullptr;
    }
    return nullptr;
}

json host_payload_scope_json() {
    return json::array({ "fixed_state", "vbr_artifact" });
}

json fallback_json(const server_cache_control_result & result) {
    if (result.fallback_kind >= server_cache_control_subject_kind::_count) {
        return nullptr;
    }
    return {
        { "state", "resolved" },
        { "kind", fallback_kind_name(result.fallback_kind) },
        { "payload_kind", payload_kind_json(result.fallback_kind) },
    };
}

} // namespace

std::string server_cache_control_encode_handle(
        server_cache_control_handle_kind kind,
        server_cache_control_token token) {
    if (!token) {
        return {};
    }
    static constexpr char HEX[] = "0123456789abcdef";
    const char tag = handle_tag(kind);
    if (tag == '\0') {
        return {};
    }
    std::string out = { 'e', '1', tag, '_' };
    out.reserve(36);
    for (int shift = 60; shift >= 0; shift -= 4) {
        out.push_back(HEX[(token.high >> shift) & 0xf]);
    }
    for (int shift = 60; shift >= 0; shift -= 4) {
        out.push_back(HEX[(token.low >> shift) & 0xf]);
    }
    return out;
}

bool server_cache_control_decode_handle(
        server_cache_control_handle_kind kind,
        const std::string & text,
        server_cache_control_token & out) noexcept {
    out = {};
    const char tag = handle_tag(kind);
    if (tag == '\0' || text.size() != 36 || text[0] != 'e' ||
        text[1] != '1' || text[2] != tag || text[3] != '_') {
        return false;
    }
    uint64_t high = 0;
    uint64_t low = 0;
    for (size_t i = 4; i < text.size(); ++i) {
        const int nibble = hex_value(text[i]);
        if (nibble < 0) {
            return false;
        }
        auto & word = i < 20 ? high : low;
        word = (word << 4) | uint64_t(nibble);
    }
    out = { high, low };
    return bool(out);
}

uint64_t server_cache_control_idempotency_digest(
        const std::string & text) noexcept {
    if (text.empty() || text.size() > 128) {
        return 0;
    }
    try {
        llama_sha256_writer writer;
        static constexpr char domain_label[] = "buun.cache-control-idempotency/v1";
        writer.string(domain_label, sizeof(domain_label) - 1);
        writer.string(text.data(), text.size());
        const auto digest = writer.finish();
        uint64_t out = 0;
        for (size_t i = 0; i < sizeof(out); ++i) {
            out = (out << 8) | digest[i];
        }
        return out == 0 ? 1 : out;
    } catch (...) {
        return 0;
    }
}

bool server_cache_control_request_field_allowed(
        server_cache_control_operation operation,
        std::string_view field) noexcept {
    return std::any_of(
        std::begin(OPERATION_FIELDS), std::end(OPERATION_FIELDS),
        [operation, field](const operation_field & item) {
            return item.operation == operation && item.field == field;
        });
}

bool server_cache_control_selector_field_allowed(
        server_cache_control_subject_kind kind,
        std::string_view field) noexcept {
    return std::any_of(
        std::begin(SELECTOR_FIELDS), std::end(SELECTOR_FIELDS),
        [kind, field](const selector_field & item) {
            return item.kind == kind && item.field == field;
        });
}

const char * server_cache_control_subject_kind_name(
        server_cache_control_subject_kind kind) noexcept {
    return SUBJECT_KINDS.name(kind, "unavailable");
}

bool server_cache_control_parse_subject_kind(
        std::string_view name,
        server_cache_control_subject_kind & out) noexcept {
    return SUBJECT_KINDS.parse(name, out);
}

const char * server_cache_control_lease_class_name(
        server_cache_lease_class value) noexcept {
    return LEASE_CLASSES.name(value, "none");
}

bool server_cache_control_parse_lease_class(
        std::string_view name,
        server_cache_lease_class & out) noexcept {
    return LEASE_CLASSES.parse(name, out);
}

const char * server_cache_control_family_role_name(
        common_cache_family_role value) noexcept {
    return FAMILY_ROLES.name(value, "automatic");
}

bool server_cache_control_parse_family_role(
        std::string_view name,
        common_cache_family_role & out) noexcept {
    return FAMILY_ROLES.parse(name, out);
}

server_cache_control_status server_cache_control_prepare_request(
        server_cache_control_operation operation,
        const json & body,
        server_cache_control_request & out) noexcept {
    out = {};
    try {
        if (!body.is_object()) {
            return server_cache_control_status::invalid_request;
        }
        for (const auto & item : body.items()) {
            if (!server_cache_control_request_field_allowed(
                    operation, item.key())) {
                return server_cache_control_status::invalid_request;
            }
        }
        if (!parse_handle(body, "holder",
                          server_cache_control_handle_kind::holder,
                          out.holder) ||
            !parse_handle(body, "holder_recovery",
                          server_cache_control_handle_kind::holder_recovery,
                          out.recovery) ||
            !parse_handle(body, "lease",
                          server_cache_control_handle_kind::lease,
                          out.lease) ||
            !parse_handle(body, "family",
                          server_cache_control_handle_kind::family,
                          out.family) ||
            !parse_handle(body, "family_binding",
                          server_cache_control_handle_kind::family_binding,
                          out.family_binding) ||
            !parse_ttl(body, out)) {
            return server_cache_control_status::invalid_request;
        }
        if (body.contains("idempotency_key")) {
            if (!body["idempotency_key"].is_string() ||
                (out.idempotency_key = server_cache_control_idempotency_digest(
                     body["idempotency_key"].get<std::string>())) == 0) {
                return server_cache_control_status::invalid_request;
            }
        }
        if (body.contains("class") &&
            (!body["class"].is_string() ||
             !server_cache_control_parse_lease_class(
                 body["class"].get<std::string>(), out.requested_class))) {
            return server_cache_control_status::invalid_request;
        }
        if (body.contains("role") &&
            (!body["role"].is_string() ||
             !server_cache_control_parse_family_role(
                 body["role"].get<std::string>(), out.family_role))) {
            return server_cache_control_status::invalid_request;
        }
        if (body.contains("label")) {
            if (!body["label"].is_string()) {
                return server_cache_control_status::invalid_request;
            }
            out.family_label = body["label"].get<std::string>();
            if (out.family_label.empty() || out.family_label.size() > 128) {
                return server_cache_control_status::invalid_request;
            }
        }
        if (body.contains("floor") &&
            (!body["floor"].is_string() || body["floor"] != "t4")) {
            return server_cache_control_status::invalid_request;
        }
        if (body.contains("allow_soft_fallback")) {
            if (!body["allow_soft_fallback"].is_boolean()) {
                return server_cache_control_status::invalid_request;
            }
            out.allow_soft_fallback = body["allow_soft_fallback"].get<bool>();
        }
        if (body.contains("after_ordinal")) {
            if (!body["after_ordinal"].is_number_unsigned()) {
                return server_cache_control_status::invalid_request;
            }
            out.after_ordinal = body["after_ordinal"].get<uint64_t>();
        }
        if (body.contains("limit")) {
            if (!body["limit"].is_number_unsigned()) {
                return server_cache_control_status::invalid_request;
            }
            const uint64_t limit = body["limit"].get<uint64_t>();
            if (limit == 0 || limit > SERVER_CACHE_LEASE_EVENT_RING ||
                limit > std::numeric_limits<uint32_t>::max()) {
                return server_cache_control_status::invalid_request;
            }
            out.event_limit = uint32_t(limit);
        }
        bool common_valid = false;
        switch (operation) {
            case server_cache_control_operation::holder_create:
                common_valid = out.ttl_ns != 0;
                break;
            case server_cache_control_operation::holder_reattach:
                common_valid = out.ttl_ns != 0 && bool(out.recovery);
                break;
            case server_cache_control_operation::holder_close:
                common_valid = bool(out.holder);
                break;
            case server_cache_control_operation::family_register:
                common_valid = bool(out.holder);
                break;
            case server_cache_control_operation::family_bind:
                common_valid = bool(out.holder) && bool(out.family) &&
                    out.family_role < common_cache_family_role::_count;
                break;
            case server_cache_control_operation::lease_acquire:
                common_valid = bool(out.holder) && out.ttl_ns != 0 &&
                    body.contains("class") && body.contains("subject") &&
                    out.requested_class > server_cache_lease_class::none &&
                    out.requested_class < server_cache_lease_class::_count &&
                    (!out.allow_soft_fallback ||
                     out.requested_class == server_cache_lease_class::hard);
                break;
            case server_cache_control_operation::lease_inspect:
            case server_cache_control_operation::lease_release:
                common_valid = bool(out.holder) && bool(out.lease);
                break;
            case server_cache_control_operation::lease_renew:
                common_valid = bool(out.holder) && bool(out.lease) &&
                    out.ttl_ns != 0;
                break;
            case server_cache_control_operation::events:
                common_valid = bool(out.holder) && out.event_limit > 0 &&
                    out.event_limit <= SERVER_CACHE_LEASE_EVENT_RING;
                break;
            case server_cache_control_operation::_count:
                break;
        }
        if (!common_valid) {
            out = {};
            return server_cache_control_status::invalid_request;
        }
        return server_cache_control_status::ok;
    } catch (...) {
        out = {};
        return server_cache_control_status::invalid_request;
    }
}

json server_cache_control_json(
        server_cache_control_operation operation,
        const server_cache_control_result & result) {
    json body = json::object();
    const bool idempotent_terminal =
        operation == server_cache_control_operation::holder_close ||
        operation == server_cache_control_operation::lease_release;
    if (result.status != server_cache_control_status::ok &&
        !(idempotent_terminal &&
          result.status == server_cache_control_status::already_released)) {
        return {
            { "object", "cache_control" },
            { "schema_version", 1 },
            { "status", server_cache_control_status_name(result.status) },
            { "result", std::move(body) },
        };
    }
    const auto expires_ms = result.expires_at_ns == 0
        ? json(nullptr) : json(result.expires_at_ns / 1000000);
    switch (operation) {
        case server_cache_control_operation::holder_create:
            body = {
                { "holder", server_cache_control_encode_handle(
                    server_cache_control_handle_kind::holder, result.holder) },
                { "holder_recovery", server_cache_control_encode_handle(
                    server_cache_control_handle_kind::holder_recovery,
                    result.holder_recovery) },
                { "expires_at_ms", expires_ms },
                { "max_leases", result.max_leases },
                { "hard_expiry_policy", "explicit_only" },
            };
            break;
        case server_cache_control_operation::holder_reattach: {
            json leases = json::array();
            for (const auto & lease : result.orphaned_leases) {
                leases.push_back({
                    { "lease", server_cache_control_encode_handle(
                        server_cache_control_handle_kind::lease, lease.lease) },
                    { "subject_kind", server_cache_control_subject_kind_name(
                        lease.subject_kind) },
                    { "payload_kind", payload_kind_json(lease.subject_kind) },
                    { "proven_frontier", frontier_json(lease.proven_frontier) },
                });
            }
            json families = json::array();
            for (const auto & family : result.families) {
                families.push_back({
                    { "family", server_cache_control_encode_handle(
                        server_cache_control_handle_kind::family, family.family) },
                    { "label", family.label },
                    { "payload_scope", host_payload_scope_json() },
                });
            }
            body = {
                { "holder", server_cache_control_encode_handle(
                    server_cache_control_handle_kind::holder, result.holder) },
                { "expires_at_ms", expires_ms },
                { "orphaned_leases", std::move(leases) },
                { "families", std::move(families) },
            };
        } break;
        case server_cache_control_operation::holder_close:
            body = { { "closed", result.status == server_cache_control_status::ok } };
            break;
        case server_cache_control_operation::family_register:
            body = {
                { "family", server_cache_control_encode_handle(
                    server_cache_control_handle_kind::family, result.family) },
                { "label", result.families.empty()
                    ? json(nullptr) : json(result.families.front().label) },
                { "payload_scope", host_payload_scope_json() },
            };
            break;
        case server_cache_control_operation::family_bind:
            body = {
                { "family_binding", server_cache_control_encode_handle(
                    server_cache_control_handle_kind::family_binding,
                    result.family_binding) },
                { "payload_scope", host_payload_scope_json() },
            };
            break;
        case server_cache_control_operation::lease_acquire:
            body = {
                { "lease", server_cache_control_encode_handle(
                    server_cache_control_handle_kind::lease, result.lease) },
                { "granted_class", server_cache_control_lease_class_name(
                    result.granted_class) },
                { "effective_floor", result.granted_class ==
                    server_cache_lease_class::hard ? json("t4") : json(nullptr) },
                { "expires_at_ms", expires_ms },
                { "fallback", fallback_json(result) },
                { "protected_bytes", result.protected_bytes_known
                    ? json(result.protected_bytes) : json(nullptr) },
                { "fallback_pinned_bytes", result.fallback_pinned_bytes_known
                    ? json(result.fallback_pinned_bytes) : json(nullptr) },
                { "shared_fallback", result.shared_fallback },
                { "payload_kind", payload_kind_json(result.subject_kind) },
            };
            break;
        case server_cache_control_operation::lease_inspect:
        case server_cache_control_operation::lease_renew:
            body = {
                { "granted_class", server_cache_control_lease_class_name(
                    result.granted_class) },
                { "expires_at_ms", expires_ms },
                { "protection_state", protection_name(result.protection) },
                { "lease_frontier", frontier_json(result.lease_frontier) },
                { "proven_frontier", frontier_json(result.proven_frontier) },
                { "family_role", result.cache_family.declared()
                    ? server_cache_control_family_role_name(result.cache_family.role)
                    : "automatic" },
                { "family_label", result.family_label.empty()
                    ? json(nullptr) : json(result.family_label) },
                { "fallback", fallback_json(result) },
                { "protected_bytes", result.protected_bytes_known
                    ? json(result.protected_bytes) : json(nullptr) },
                { "fallback_pinned_bytes", result.fallback_pinned_bytes_known
                    ? json(result.fallback_pinned_bytes) : json(nullptr) },
                { "shared_fallback", result.shared_fallback },
                { "payload_kind", payload_kind_json(result.subject_kind) },
            };
            break;
        case server_cache_control_operation::lease_release:
            body = { { "released", result.status == server_cache_control_status::ok ||
                result.status == server_cache_control_status::already_released } };
            break;
        case server_cache_control_operation::events: {
            json events = json::array();
            for (const auto & event : result.events) {
                events.push_back({
                    { "ordinal", event.ordinal },
                    { "timestamp_ms", event.timestamp_ms },
                    { "kind", EVENT_KINDS.name(event.kind, "refuse") },
                    { "status", server_cache_control_status_name(event.status) },
                    { "class", server_cache_control_lease_class_name(event.cls) },
                    { "subject_kind", event.subject_kind <
                        server_cache_control_subject_kind::_count
                            ? json(server_cache_control_subject_kind_name(
                                  event.subject_kind)) : json(nullptr) },
                    { "payload_kind", payload_kind_json(event.subject_kind) },
                    { "family_role", event.family_role < common_cache_family_role::_count
                        ? json(server_cache_control_family_role_name(event.family_role))
                        : json("automatic") },
                    { "lease", event.lease ? json(server_cache_control_encode_handle(
                        server_cache_control_handle_kind::lease, event.lease))
                        : json(nullptr) },
                });
            }
            body = { { "events", std::move(events) },
                     { "overflowed", result.events_overflowed } };
        } break;
        case server_cache_control_operation::_count:
            break;
    }
    return {
        { "object", "cache_control" },
        { "schema_version", 1 },
        { "status", server_cache_control_status_name(result.status) },
        { "result", std::move(body) },
    };
}
