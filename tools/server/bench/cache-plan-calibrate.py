#!/usr/bin/env python3
# cache-plan-calibrate.py — cache-plan calibration sweep and fit. Drives a --cache-debug server
# with controlled probes, reads the measured actuals back out of its CACHE_PLAN records,
# and fits the common_cache_plan_calib coefficients:
#   replay_us_per_token — least-squares over COLD records (ttft vs n_replayed_tokens)
#   restore_us_per_byte, workspace_setup_us — least-squares over RESTORED host-cache
#       records (ttft minus fitted replay share vs payload bytes)
# Emits a C++ initializer snippet for common/common-cache-plan-estimate.cpp's checked-in
# table (data reviewed like code) plus fit diagnostics. The profile id is read from the
# records themselves; the server composes it and this script never invents one.
#
# Usage: point at a FRESH server log + run the sweep, e.g.
#   cache-plan-calibrate.py --server http://localhost:8241 --log server.log

import argparse
import json
import math
import random
import re
import sys
import urllib.request

from cache_plan_common import UnsupportedSchemaError, iter_cache_plan_records

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
         "omicron pi rho sigma tau upsilon").split()


def post_completion(base, prompt, n_predict, timeout):
    req = urllib.request.Request(
        base + "/completion",
        data=json.dumps({"prompt": prompt, "n_predict": n_predict, "cache_prompt": True}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def least_squares(xs, ys):
    # y = a*x + b
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else 0.0
    return a, my - a * mx


def fit_restore(xs, ys):
    """Restore/workspace fit with the unidentifiable-slope rule: when payloads are
    (near-)constant the slope is forced to exactly zero and workspace becomes the mean
    residual. Returns (slope, workspace, flat_branch_taken)."""
    a, b = least_squares(xs, ys)
    if max(xs) - min(xs) < 0.01 * max(xs):
        return 0.0, sum(ys) / len(ys), True
    return a, b, False


def self_test():
    fails = 0
    def check(cond, what):
        nonlocal fails
        if not cond:
            print(f"FAIL: {what}", file=sys.stderr)
            fails += 1

    # exact-constant payloads (the 27B hybrid case): slope 0, workspace = mean
    xs = [156894364] * 5
    ys = [100.0, 110.0, 90.0, 105.0, 95.0]
    a, b, flat = fit_restore(xs, ys)
    check(flat and a == 0.0 and abs(b - 100.0) < 1e-9, "exact-constant -> 0 / mean")

    # near-constant with a garbage POSITIVE raw slope (old bug kept it + a negative intercept)
    xs = [1000000, 1001000, 1002000, 1003000, 1004000]
    ys = [500.0, 100.0, 900.0, 200.0, 800.0]
    raw_a, raw_b = least_squares(xs, ys)
    a, b, flat = fit_restore(xs, ys)
    check(raw_a > 0 and raw_b < 0, "near-constant raw fit is the pathological case")
    check(flat and a == 0.0 and abs(b - 500.0) < 1e-9, "near-constant+ -> 0 / mean")

    # near-constant with a garbage NEGATIVE raw slope
    ys = [800.0, 200.0, 900.0, 100.0, 500.0]
    raw_a, _ = least_squares(xs, ys)
    a, b, flat = fit_restore(xs, ys)
    check(raw_a < 0, "negative-slope cell is negative before the branch")
    check(flat and a == 0.0 and abs(b - 500.0) < 1e-9, "near-constant- -> 0 / mean")

    # genuinely identifiable slope: the branch must NOT fire
    xs = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    ys = [1100.0, 10100.0, 50100.0, 100100.0]   # ~1e-3 us/byte + 100 us
    a, b, flat = fit_restore(xs, ys)
    check(not flat and abs(a - 0.001) < 1e-6 and abs(b - 100.0) < 1.0,
          "identifiable slope preserved")

    print("all cache-plan-calibrate self-tests passed" if not fails else f"{fails} failure(s)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description="cache-plan calibration sweep + fit")
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--log", required=True, help="the server's log file (--cache-debug on)")
    ap.add_argument("--lengths", default="64,128,256,512,1024",
                    help="cold prompt lengths (tokens, approximate) to sweep")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--fit-only", action="store_true", help="skip the sweep, fit the existing log")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if not args.fit_only:
        # cold-replay sweep: unique prompts (no reuse possible) at several lengths
        for n_tok in (int(x) for x in args.lengths.split(",")):
            for r in range(args.repeats):
                salt = f"probe {n_tok} {r} {rng.random():.9f} "
                prompt = salt + " ".join(rng.choice(WORDS) for _ in range(n_tok))
                post_completion(args.server, prompt, args.n_predict, args.timeout)
        print("sweep complete; fitting from log", file=sys.stderr)

    cold, restored = [], []
    profiles = set()
    try:
        records = list(iter_cache_plan_records([args.log]))
    except UnsupportedSchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for rec in records:
        if rec.get("calibration_profile"):
            profiles.add(rec["calibration_profile"])
        ttft = rec.get("ttft_us")
        if not isinstance(ttft, (int, float)):
            continue
        if rec.get("outcome") == "cold":
            nre = rec.get("n_replayed_tokens")
            if isinstance(nre, int) and nre > 0:
                cold.append((nre, ttft))
        elif rec.get("outcome") == "restored":
            # the SHIPPED PLAN decides what was restored: a bare host entry, or a composed
            # host→checkpoint chain (chosen names the terminal provider, not the plan) —
            # total restored bytes = the plan's components' payloads
            cid = rec.get("shipped_plan_candidate", rec.get("chosen_candidate"))
            by_id = {c.get("id"): c for c in rec.get("candidates", [])}
            plan = by_id.get(cid)
            nre  = rec.get("n_replayed_tokens")
            if plan is None or not isinstance(nre, int):
                continue
            if plan.get("is_chain"):
                comps = [by_id.get(i) for i in plan.get("components", [])]
                if any(c is None or not isinstance(c.get("payload_bytes"), int) for c in comps):
                    continue
                if any(c.get("provider") == "host_cache_entry" for c in comps):
                    restored.append((sum(c["payload_bytes"] for c in comps), nre, ttft))
            elif plan.get("provider") in ("host_cache_entry", "live_context_checkpoint") and \
                    isinstance(plan.get("payload_bytes"), int):
                # bare host restores AND bare checkpoint restores both exercise the same
                # state-install path — under dynamic VBR (host cache disabled) checkpoint
                # restores are the ONLY restore-fit source
                restored.append((plan["payload_bytes"], nre, ttft))

    # one fit = one profile: mixing hardware/model regimes in a single log would fit
    # garbage coefficients under whichever profile string came last.
    if len(profiles) > 1:
        print(f"error: log mixes {len(profiles)} calibration profiles: {sorted(profiles)}; "
              "fit each regime from its own log", file=sys.stderr)
        return 2
    profile = next(iter(profiles), None)

    if len(cold) < 4:
        print(f"not enough cold records to fit ({len(cold)}); need a longer sweep", file=sys.stderr)
        return 1

    replay_us_per_token, cold_intercept = least_squares([c[0] for c in cold], [c[1] for c in cold])
    print(f"# cold records: {len(cold)}, replay fit: {replay_us_per_token:.3f} us/token "
          f"(intercept {cold_intercept:.0f} us)")
    if not (replay_us_per_token > 0 and math.isfinite(replay_us_per_token)):
        print(f"error: invalid replay fit ({replay_us_per_token}); refusing to emit an entry",
              file=sys.stderr)
        return 2

    restore_us_per_byte = workspace_setup_us = None
    if len(restored) >= 4:
        xs = [r[0] for r in restored]
        ys = [r[2] - r[1] * replay_us_per_token for r in restored]  # ttft minus replay share
        restore_us_per_byte, workspace_setup_us, flat = fit_restore(xs, ys)
        if flat:
            # payloads (near-)constant — common on hybrid models whose checkpoint blob is
            # dominated by the fixed-size recurrent state: the slope is UNIDENTIFIABLE (see
            # fit_restore + --self-test)
            print(f"# note: restored payloads are (near-)constant ({min(xs)}..{max(xs)} B); "
                  "slope forced to 0, workspace = mean residual")
        print(f"# restored records: {len(restored)}, restore fit: {restore_us_per_byte:.6f} us/byte, "
              f"workspace {workspace_setup_us:.0f} us")
        # the estimator contract accepts nonnegative finite coefficients; refuse only
        # negative/non-finite (a genuinely flat restore cost is valid data)
        if not (restore_us_per_byte >= 0 and math.isfinite(restore_us_per_byte)
                and workspace_setup_us >= 0 and math.isfinite(workspace_setup_us)):
            print(f"error: invalid restore fit ({restore_us_per_byte}, {workspace_setup_us}); "
                  "refusing to emit an entry", file=sys.stderr)
            return 2
    else:
        print(f"# restored records: {len(restored)} — too few; restore/workspace NOT fitted "
              "(drive host-cache restores and re-run with --fit-only)")

    if profile and restore_us_per_byte is not None:
        print("\n// fitted entry for common/common-cache-plan-estimate.cpp (bump the version")
        print("// on ANY coefficient change; review like code):")
        print("static const common_cache_plan_calib CALIB_" +
              re.sub(r"[^a-z0-9]", "_", profile).upper() + " = {")
        print(f"    \"{profile}\", 1,")
        print(f"    {replay_us_per_token:.3f}, {restore_us_per_byte:.6f}, "
              f"{max(workspace_setup_us, 0.0):.1f},")
        print("};")
    return 0


if __name__ == "__main__":
    sys.exit(main())
