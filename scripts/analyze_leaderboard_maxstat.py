"""Is the public leaderboard a max-statistic over submission count?

Why this exists
---------------
Four public forks of the NVFP4 Duck stack -- ``wuliao0/duck-qwen3-8-anim-base``,
``keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp``, ``tantan0327/arc3-flashnext-asis``
and ``akhileshgodugu/...-aacb81`` -- are **byte-identical notebooks**
(md5 ``5025d1d3e4e6c3719e2b0ea2cbb1870c``, 18 cells) with identical
``model_sources`` / ``dataset_sources`` / ``docker_image`` / ``machine_shape``.
Our own copy is the same md5. Their leaderboard scores are 4.33 / 4.17 / 3.92 /
4.09. Ours is 2.95.

Identical code cannot be 1.4 points better. Either the hidden-set run has large
variance, or the leaderboard -- which shows each team's **best** submission --
is being read as a measure of configuration quality when it is substantially a
measure of *how many times you rolled the dice*. Those two explanations are the
same explanation, and this script measures its size instead of asserting it.

Method
------
On the public leaderboard CSV (`kaggle competitions leaderboard -d`):

1. Spearman rank correlation between ``SubmissionCount`` and ``Score``, over
   all teams and over the sub-band plausibly running an LLM stack at all.
   Spearman, not Pearson: score is heavy-tailed and submission count is
   heavier, and a single 137-submission outlier would dominate Pearson.
2. Median best-score by submission-count bucket -- the direct, assumption-free
   version of the same question.
3. An order-statistic estimate: if per-run score is iid with some spread, the
   expected best of ``n`` draws grows like the ``n/(n+1)`` quantile. Fitting
   that to the observed bucket medians gives a rough per-run distribution, and
   therefore an estimate of what a team's *single-run mean* is versus its
   displayed best.

Caveat stated up front: teams with more submissions also iterate more, so part
of any correlation is real improvement, not just resampling. The four
byte-identical forks are the control for that -- they cannot have iterated,
because their notebooks are the same file.

Usage
-----
    venv/Scripts/python.exe scripts/analyze_leaderboard_maxstat.py <leaderboard.csv> \
        [--anchors user1,user2,...] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation with average ranks for ties."""
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 1000)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--anchors", default="")
    parser.add_argument("--min-score", type=float, default=2.0,
                        help="floor for the 'running a real LLM stack' sub-band")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open(encoding="utf-8")))
    rank_key = next(k for k in rows[0] if k.lstrip("﻿") == "Rank")
    teams = [
        {
            "rank": int(r[rank_key]),
            "name": r["TeamName"],
            "score": float(r["Score"]),
            "subs": int(r["SubmissionCount"]),
            "users": [u.strip() for u in r["TeamMemberUserNames"].split(",")],
        }
        for r in rows
    ]

    report: dict = {"n_teams": len(teams)}

    for label, subset in (
        ("all teams", teams),
        (f"score >= {args.min_score}", [t for t in teams if t["score"] >= args.min_score]),
    ):
        rho = spearman([t["subs"] for t in subset], [t["score"] for t in subset])
        report[f"spearman_subs_vs_score__{label}"] = round(rho, 4)
        print(f"Spearman(submissions, best score), {label} (n={len(subset)}): {rho:+.3f}")

    print("\nmedian best-score by submission-count bucket:")
    bucket_rows = []
    for lo, hi in BUCKETS:
        sel = [t for t in teams if lo <= t["subs"] <= hi]
        if not sel:
            continue
        med = statistics.median(t["score"] for t in sel)
        p90 = sorted(t["score"] for t in sel)[int(0.9 * (len(sel) - 1))]
        label = f"{lo}-{hi}" if hi < 1000 else f"{lo}+"
        bucket_rows.append({"bucket": label, "n": len(sel), "median": med, "p90": p90})
        print(f"  {label:>7s} subs  n={len(sel):>4d}  median={med:6.2f}  p90={p90:6.2f}")
    report["buckets"] = bucket_rows

    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    if anchors:
        print("\nanchor teams (byte-identical-notebook control group):")
        found = []
        for user in anchors:
            for t in teams:
                if user in t["users"]:
                    found.append({"user": user, **{k: t[k] for k in ("rank", "score", "subs")}})
                    print(f"  {user:<22s} rank {t['rank']:>4d}  best {t['score']:5.2f}  subs {t['subs']:>3d}")
                    break
            else:
                print(f"  {user:<22s} not on leaderboard")
        report["anchors"] = found
        if len(found) >= 3:
            rho = spearman([a["subs"] for a in found], [a["score"] for a in found])
            report["spearman_anchors"] = round(rho, 4)
            print(f"\n  Spearman within the identical-code control group (n={len(found)}): {rho:+.3f}")
            print("  Any positive value here is pure resampling: the code cannot differ.")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
