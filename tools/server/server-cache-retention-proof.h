#pragma once

#include "server-cache-lease.h"

class server_cache_recovery_pin;

// Production adapter. Only the scheduler-owned resolver may call it;
// contract scans ban other production call sites.
server_cache_durable_fallback_proof
server_cache_retention_fallback_proof(
    server_cache_recovery_pin && pin) noexcept;

// Private compatibility/test door.
server_cache_durable_fallback_proof
server_cache_retention_fallback_proof_for_test(
    server_cache_recovery_pin && pin) noexcept;
