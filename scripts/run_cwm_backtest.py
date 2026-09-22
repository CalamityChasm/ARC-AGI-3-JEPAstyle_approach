"""Run the CodeWorldModel backtest against a recorded free run.

Two modes, and the first one costs nothing:

  --self-check   Extract, census determinism, and verify the positive-
                 control oracle. No LLM is contacted. Run this before
                 spending GPU time -- it proves the instrument works on
                 the exact data about to be measured.

  (default)      The real backtest: draft a WorldModel per segment via
                 the configured LLM and replay it.

The model lives behind `llm_engine.llm_client.make_client`, so pointing
this at the in-kernel vLLM server on Kaggle is a matter of environment
(`LLM_BACKEND=openai`, `CODER_LLM_BASE_URL=...`), not a code change.

Examples
--------
    python scripts/run_cwm_backtest.py --artifacts <run>/artifacts --self-check
    python scripts/run_cwm_backtest.py --artifacts <run>/artifacts \\
        --out results.json --max-segments 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc3_cwm.determinism import census, deterministic_only  # noqa: E402
from arc3_cwm.extract import extract_run  # noqa: E402
from arc3_cwm.harness import BacktestConfig, run_segment  # noqa: E402
from arc3_cwm.oracle import verify_oracle  # noqa: E402
from arc3_cwm.render import MAX_STEPS, measure_sizes  # noqa: E402
from arc3_cwm.report import build_report, per_game_table  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", required=True, type=Path,
                   help="a free run's artifacts/ directory (*_events.jsonl)")
    p.add_argument("--self-check", action="store_true",
                   help="validate the instrument only; contact no LLM")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help=f"steps per segment shown AND scored (default {MAX_STEPS})")
    p.add_argument("--min-transitions", type=int, default=4,
                   help="skip segments shorter than this (default 4)")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="draft/repair attempts per segment (default 3)")
    p.add_argument("--max-segments", type=int, default=None,
                   help="cap segments processed, for a cheap pilot")
    p.add_argument("--deterministic-only", action="store_true",
                   help="skip segments with a proven board/action contradiction")
    p.add_argument("--out", type=Path, default=None, help="write JSON results here")
    p.add_argument("--save-sources", type=Path, default=None,
                   help="directory to write each passing WorldModel source into")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    segments, stats = extract_run(args.artifacts, min_transitions=args.min_transitions)
    windowed = [s.window(args.max_steps) for s in segments]
    windowed = [s for s in windowed if len(s) >= args.min_transitions]

    print(f"extracted {len(windowed)} segments from "
          f"{len({s.game_id for s in windowed})} games")
    print(f"extraction stats: {json.dumps(stats.as_dict())}")

    det = census(windowed)
    print(det.summary())

    if args.deterministic_only:
        before = len(windowed)
        windowed = deterministic_only(windowed)
        print(f"deterministic-only: {before} -> {len(windowed)} segments")

    if args.max_segments is not None:
        windowed = windowed[: args.max_segments]
        print(f"capped to {len(windowed)} segments")

    if args.self_check:
        return _self_check(windowed, det)

    config = BacktestConfig(max_steps=args.max_steps, max_attempts=args.max_attempts)

    from llm_engine.llm_client import make_client

    client = make_client("coder")

    results = []
    for index, segment in enumerate(windowed, start=1):
        result = run_segment(client, segment, config)
        results.append(result)
        print(f"[{index}/{len(windowed)}] {segment.key:24s} n={len(segment):3d} "
              f"{result.outcome:16s} prefix={result.best_prefix:3d} "
              f"attempts={result.attempts} {result.elapsed_s:6.1f}s")
        if result.source and args.save_sources:
            args.save_sources.mkdir(parents=True, exist_ok=True)
            (args.save_sources / f"{segment.key.replace('/', '_')}.py").write_text(
                result.source, encoding="utf-8"
            )

    report = build_report(results)
    print()
    print(report.summary())
    print()
    print(per_game_table(results))

    if args.out:
        payload = report.as_dict()
        payload["determinism"] = det.as_dict()
        payload["extraction"] = stats.as_dict()
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return 0


def _self_check(windowed, det) -> int:
    """Prove the instrument works before any GPU time is spent."""
    print()
    print("SELF-CHECK (no LLM contacted)")
    print("=" * 62)

    passed = 0
    failures = []
    for segment in windowed:
        ok, err = verify_oracle(segment)
        if ok:
            passed += 1
        else:
            failures.append((segment.key, err))

    print(f"positive control: oracle replays {passed}/{len(windowed)} segments exactly")
    for key, err in failures:
        print(f"  cannot build a passing oracle for {key}: {str(err)[:100]}")

    sizes = [measure_sizes(s) for s in windowed]
    if sizes:
        total_compact = sum(s.compact_chars for s in sizes)
        total_engine = sum(s.engine_chars for s in sizes)
        largest = max(sizes, key=lambda s: s.compact_chars)
        print()
        print(f"prompt size: compact renderer {total_compact:,} chars total vs "
              f"{total_engine:,} for llm_engine's own ({total_engine / total_compact:.1f}x)")
        print(f"largest single prompt: {largest.compact_chars:,} chars "
              f"(~{largest.compact_chars // 4:,} tokens) on {largest.segment_key}")

    # An oracle failure that is NOT explained by a proven contradiction is
    # an instrument bug, and must not be waved through as a data property.
    unexplained = [k for k, _ in failures if k not in set(det.contradictory_segments)]
    print()
    if unexplained:
        print(f"WARNING: {len(unexplained)} oracle failure(s) not explained by a "
              f"board/action contradiction: {unexplained}")
        print("These are instrument limitations, not model results.")
    else:
        print("every oracle failure is explained by a proven contradiction in the data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
