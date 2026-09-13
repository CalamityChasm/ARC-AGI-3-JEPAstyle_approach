"""Locate the harness's own RHAE scorer inside the TAAF serving bundle zip.

CLAUDE.md states the RHAE formula but flags it as "high confidence, not
confirmed" (the Kaggle Evaluation page is a JS SPA that could not be read at
source). Every score projection in experiments/stage7_turn_latency.md depends on
that formula being right - in particular on whether the efficiency term's
denominator runs over SOLVED levels or over ALL levels, which flips the sign of
"what does one more slowly-won level do to the score".

The bundle ships the scorer that actually produced the run's `score.json`, so
the formula can be read from source rather than assumed. Members are read
straight out of the archive: extracting this zip on Windows half-fails, because
its vendored `sglang-rtxpro6000` tree has paths past the 260-char MAX_PATH limit
(see experiments/stage7_duck_nvfp4.md section 4).

Usage:
    python scripts/find_rhae_scorer.py <bundle.zip> [--dump <member>]
"""

from __future__ import annotations

import argparse
import re
import zipfile

HINTS = (
    "rhae",
    "human_action",
    "action_efficiency",
    "def score",
    "levels_completed",
    "1.15",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--dump", default=None, help="print this member verbatim")
    ap.add_argument("--grep", default=None, help="print lines matching this regex")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip_path)
    names = [n for n in zf.namelist() if n.endswith(".py")]
    print("python members in bundle: %d (of %d entries)" % (len(names), len(zf.namelist())))

    if args.dump:
        print(zf.read(args.dump).decode("utf-8", "replace"))
        return

    hits = []
    for name in names:
        try:
            text = zf.read(name).decode("utf-8", "replace")
        except Exception:
            continue
        low = text.lower()
        score = sum(low.count(h) for h in HINTS)
        if "1.15" in text and ("min(" in text or "min (" in text):
            score += 20
        if score:
            hits.append((score, name, text))
    hits.sort(key=lambda t: -t[0])

    print("\ntop candidates by RHAE-shaped content:")
    for score, name, _ in hits[:15]:
        print("  %5d  %s" % (score, name))

    if args.grep:
        pat = re.compile(args.grep)
        print("\nlines matching %r:" % args.grep)
        for _, name, text in hits[:15]:
            for i, line in enumerate(text.splitlines(), start=1):
                if pat.search(line):
                    print("  %s:%d: %s" % (name, i, line.strip()[:200]))


if __name__ == "__main__":
    main()
