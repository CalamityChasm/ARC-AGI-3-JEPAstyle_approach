"""What actually differs between two TAAF solver bundles, beyond the headline.

`stage7_components.md` §2.2 established that the AVO bundle is a strict superset
of the anim incumbent by checking which *files exist* (`animation.py` present,
`noop_guard.py` present, `inference/avo/` added). File presence is necessary and
not sufficient: the two bundles are 3.5 weeks of upstream commits apart
(`9158303` 2026-08-07 -> `74ff3df` 2026-09-01), so shared files can have changed
underneath, and a "one variable" claim has to survive that.

This script checks the sufficient version: it diffs the two source trees, the
two pickled solvers field by field, and the two deploy targets, and prints what
is left once `__pycache__` and non-Python files are set aside.

It found the thing the feature matrix could not: the AVO lineage gates the
`animation()` retrieval tool behind a new `animation_retrieval` flag that
defaults to **False**, and deletes the "suggest animation() when stuck" hint
system outright. The incumbent installs that tool unconditionally. So the swap
removes a capability at the same time as it adds one.

Usage:
    venv/Scripts/python.exe scripts/diff_solver_bundles.py <anim-dir> <avo-dir>

Each dir is an unpacked solver bundle (`src/`, `benchmark_initial.pkl`,
`deploy_target.pkl`). On Windows the pickles hold `PosixPath`, so this shims
`pathlib.PosixPath` to `PurePosixPath` for the read; nothing is executed.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import difflib
import sys
from pathlib import Path
from typing import Any

SKIP = ("__pycache__", ".pyc")


def _load(bundle: Path) -> tuple[Any, Any]:
    pathlib.PosixPath = pathlib.PurePosixPath  # type: ignore[misc]
    added = []
    for repo in sorted((bundle / "src").iterdir(), reverse=True):
        for cand in (repo / "src", repo):
            if cand.is_dir():
                sys.path.insert(0, str(cand))
                added.append(str(cand))
    for mod in [m for m in list(sys.modules) if m.split(".")[0] in ("inference", "taaf")]:
        del sys.modules[mod]
    try:
        bm = pickle.load((bundle / "benchmark_initial.pkl").open("rb"))
        target = pickle.load((bundle / "deploy_target.pkl").open("rb"))
    finally:
        for p in added:
            sys.path.remove(p)
    return bm, target


def _py_files(root: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(root)).replace("\\", "/"): p
        for p in root.rglob("*.py")
        if not any(s in str(p) for s in SKIP)
    }


def _tree_diff(a: Path, b: Path) -> list[tuple[str, int]]:
    """Pure-Python so it does not depend on a `diff` binary or on path flavour."""
    fa, fb = _py_files(a / "src"), _py_files(b / "src")
    rows: list[tuple[str, int]] = []
    for name in sorted(set(fa) | set(fb)):
        if name not in fa:
            rows.append((f"(only in other) {name}", len(fb[name].read_text(
                encoding="utf-8", errors="replace").splitlines())))
            continue
        if name not in fb:
            rows.append((f"(only in base) {name}", len(fa[name].read_text(
                encoding="utf-8", errors="replace").splitlines())))
            continue
        la = fa[name].read_text(encoding="utf-8", errors="replace").splitlines()
        lb = fb[name].read_text(encoding="utf-8", errors="replace").splitlines()
        if la == lb:
            continue
        n = sum(1 for ln in difflib.unified_diff(la, lb, n=0)
                if ln[:1] in "+-" and ln[:2] not in ("++", "--"))
        rows.append((name, n))
    return sorted(rows, key=lambda r: -r[1])


def _obj_diff(a: Any, b: Any) -> list[tuple[str, str, str]]:
    fa, fb = vars(a), vars(b)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        va, vb = repr(fa.get(k, "<absent>")), repr(fb.get(k, "<absent>"))
        if va != vb:
            rows.append((k, va[:56], vb[:56]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("other", type=Path)
    args = ap.parse_args()

    print(f"base  = {args.base}")
    print(f"other = {args.other}\n")
    for tag, path in (("base", args.base), ("other", args.other)):
        gs = (path / "git_status.txt")
        if gs.is_file():
            line = next((l for l in gs.read_text().splitlines() if "ARC3-Inference" in l), "")
            print(f"{tag:<6}{line.strip()}")

    print("\n=== source files that differ (.py only, __pycache__ excluded) ===")
    rows = _tree_diff(args.base, args.other)
    for name, n in rows:
        print(f"  {name:<56}{n if n else '':>6}{' changed lines' if n else ''}")
    if not rows:
        print("  (none)")

    base_bm, base_t = _load(args.base)
    other_bm, other_t = _load(args.other)

    for title, a, b in (
        ("pickled solver fields", base_bm.solver, other_bm.solver),
        ("deploy target fields", base_t, other_t),
    ):
        print(f"\n=== {title} ===")
        for k, va, vb in _obj_diff(a, b):
            print(f"  {k:<34}{va:>58}  ->  {vb}")

    print("\n=== benchmark-level fields (excluding solver) ===")
    for k, va, vb in _obj_diff(base_bm, other_bm):
        if k != "solver":
            print(f"  {k:<34}{va:>58}  ->  {vb}")


if __name__ == "__main__":
    main()
