"""Project tokens-per-request, KV residency and turns from the harness window.

The whole projection rests on one measured fact: the harness's trim loop
(`tool_agent.py:1686`) drops the oldest message block until its own
`_estimate_tokens` (chars/3 over the JSON payload) fits
`_context_budget_tokens = LOCAL_ANALYZER_CONTEXT_WINDOW - 512 - 512`, so the
request is always pushed right up against that budget. Measured fill on the
baseline run is 29,404 of 31,744 estimated tokens = 92.6%.

Therefore the *real* prompt length is set by the window and nothing else, and
scales with it -- minus the system prompt, which is fixed and cannot be trimmed
(`_trim_messages_for_context` always keeps `messages[0]`).

    E(W)     = fill * (W - 1024)                    estimated payload
    E_var(W) = E(W) - E_sys                         trimmable part
    R(W)     = R_sys + rho * E_var(W)               real prompt tokens
    rho      = R_var_baseline / E_var_baseline      real-per-estimated

Residency and turns then follow the identities established in
`experiments/stage7_turn_latency.md`:

    residency     = KV_POOL_TOKENS / (R + generation)
    turns_per_game scales with aggregate generation throughput, which scales
    with residency while the pool -- not `max_num_seqs` -- is the binding
    constraint (Running p50 = 3, cap 8, never reached).

Usage:
    python scripts/project_context_window.py experiments/stage7_context_budget_data.json
"""

from __future__ import annotations

import argparse
import json

# ---- measured from the 10.69 baseline run (see stage7_turn_latency.md) ------
KV_POOL_TOKENS = 105_202          # vllm-openai-server.log, scheduler capacity line
MEAN_GENERATION_TOKENS = 1_433    # vllm-metrics-final.prom
BASELINE_REAL_PROMPT = 20_126     # vllm-metrics-final.prom, 26,948,103 / 1,339
BASELINE_WINDOW = 32_768
RESERVE = 512 + 512               # reply reserve + safety margin, tool_agent.py:944-948
BASELINE_ACTIONS = 3_633
BASELINE_SCORE = 10.689


def project(window, fill, e_sys, rho, r_sys):
    budget = window - RESERVE
    e_total = fill * budget
    e_var = max(0.0, e_total - e_sys)
    r = r_sys + rho * e_var
    per_request = r + MEAN_GENERATION_TOKENS
    residency = KV_POOL_TOKENS / per_request
    return budget, e_total, r, per_request, residency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_json")
    ap.add_argument("--fill", type=float, default=0.926,
                    help="measured estimated-payload / budget fill factor")
    ap.add_argument("--e-sys", type=float, default=None,
                    help="harness-estimated tokens for the untrimmable system message + tool schema")
    args = ap.parse_args()

    d = json.load(open(args.data_json, encoding="utf-8"))
    r_sys = d["means"]["system"]

    # The system message is untrimmable, as is the tool schema. Its harness
    # estimate is chars/3; back it out of the real token count using the
    # measured chars-per-token of that specific text if not given.
    if args.e_sys is None:
        # system real tokens * (chars/token for prose ~3.85) / 3, + ~1450 chars
        # of tool schema JSON. Both measured in analyze_context_budget.py.
        e_sys = r_sys * 3.85 / 3 + 1450 / 3
    else:
        e_sys = args.e_sys

    baseline_budget = BASELINE_WINDOW - RESERVE
    e_total_base = args.fill * baseline_budget
    e_var_base = e_total_base - e_sys
    r_var_base = BASELINE_REAL_PROMPT - r_sys
    rho = r_var_base / e_var_base

    print("calibration (from the 10.69 baseline run)")
    print(f"  real prompt              {BASELINE_REAL_PROMPT:>9,.0f}")
    print(f"  of which system (fixed)  {r_sys:>9,.0f}")
    print(f"  estimated payload        {e_total_base:>9,.0f}  ({100*args.fill:.1f}% of {baseline_budget:,})")
    print(f"  of which system estimate {e_sys:>9,.0f}")
    print(f"  rho (real per estimated) {rho:>9.4f}")
    print()

    hdr = (f"{'window':>8}{'budget':>9}{'real prompt':>13}{'tok/req':>10}"
           f"{'residency':>11}{'x turns':>9}{'turns/game':>12}{'ret. turns':>12}")
    print(hdr)
    print("-" * len(hdr))

    _, _, _, base_per_req, base_res = project(BASELINE_WINDOW, args.fill, e_sys, rho, r_sys)
    base_turns = 52.4  # measured, stage7_turn_latency.md section 5
    base_retained = d["means"].get("_n_user_turns", 8.6)

    rows = []
    for w in (32768, 28672, 24576, 20480, 16384, 12288, 8192):
        budget, e_total, r, per_req, res = project(w, args.fill, e_sys, rho, r_sys)
        mult = res / base_res
        # retained turns scale with the trimmable budget
        ret = 8.6 * (e_total - e_sys) / e_var_base
        flag = "  <- baseline" if w == BASELINE_WINDOW else ""
        print(f"{w:>8,}{budget:>9,}{r:>13,.0f}{per_req:>10,.0f}"
              f"{res:>11.2f}{mult:>9.2f}{base_turns*mult:>12.1f}{ret:>12.1f}{flag}")
        rows.append({"window": w, "real_prompt": r, "tokens_per_request": per_req,
                     "residency": res, "turn_multiplier": mult,
                     "turns_per_game": base_turns * mult, "retained_turns": ret})

    print()
    print("RHAE upside at the projected turn multipliers")
    print("(from experiments/stage7_turn_latency.md section 6.3; the")
    print(" '+1 only' column is the floor -- it credits no level the agent")
    print(" was not already working on)")
    table = {1.00: (10.689, 10.689), 1.25: (16.830, 14.595), 1.50: (24.578, 15.879),
             2.00: (37.690, 17.701), 2.55: (45.096, 17.701), 3.00: (51.031, 17.701)}
    print(f"{'x turns':>9}{'optimistic':>13}{'floor (+1)':>13}")
    for k in sorted(table):
        o, f = table[k]
        print(f"{k:>9.2f}{o:>13.2f}{f:>13.2f}")

    print()
    print("caveats")
    print("  * the fill factor may fall at smaller budgets: the loop drops whole")
    print("    message blocks, and a block is a larger fraction of a small budget,")
    print("    so real prompts could land further under the projection (which")
    print("    would make residency BETTER and history SHORTER than shown).")
    print("  * rho is assumed constant. It is not exactly: the image base64 that")
    print("    inflates the estimate scales with retained turns, as does the text,")
    print("    so the two move together, but not identically.")
    print("  * turns/game assumes generation throughput scales with residency,")
    print("    which holds only while the KV pool is the binding constraint.")
    print("    max_num_seqs is 8 and Running p50 is 3, so there is room to ~2.6x")
    print("    before the sequence cap starts to bind.")


if __name__ == "__main__":
    main()
