"""Build the free (non-scored) vLLM throughput-vs-concurrency diagnostic kernel.

The diagnostic notebook is *generated*, not hand-maintained, so that its setup
path is provably byte-identical to the real submission notebook's: cells 1-5 of
`kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb`
(env setup, wheelhouse install, model/dataset mount config, input audit, and
the TAAF/vLLM boot) are copied verbatim. Only the trailing game-playing cells
are dropped and replaced with `benchmark_cell.py`.

This matters because the whole point of the diagnostic is to measure the *real*
server -- same model, same machine shape, same bundled vLLM launch arguments,
same attention backend. Any divergence in the setup path would make the numbers
unrepresentative of production.

Usage (from repo root):
    python scripts/_build_duck_concurrency_diag.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_NB = (
    REPO_ROOT
    / "kaggle_submission_duck"
    / "notebook"
    / "lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb"
)
DIAG_DIR = REPO_ROOT / "kaggle_submission_duck" / "notebook_concurrency_diag"
BENCH_CELL = DIAG_DIR / "benchmark_cell.py"
OUT_NB = DIAG_DIR / "arc3-duck-concurrency-benchmark.ipynb"

# Cells 1-5 of the production notebook: everything up to and including the
# TAAF/vLLM setup. Cell 0 is markdown (replaced), cells 6-10 are the game loop
# and diagnostics rendering (dropped -- this kernel plays no games).
SETUP_CELL_INDICES = [1, 2, 3, 4, 5]

HEADER_MD = """\
# stage7-duck-concurrency — vLLM aggregate throughput vs. concurrency

**Free diagnostic kernel. Plays no games. Consumes no submission quota.**

Cells 1–5 below are **byte-identical** to the production submission notebook
(`lb-9-arc3-duck-v12-with-qwen-3-8-27b`), so vLLM boots the real
`Qwen3.8-27B-FP8` on the real `NvidiaRtxPro6000` with the real bundled server
arguments and attention backend. The final cell replaces the Duck harness's
game loop with a fixed generation workload issued at several concurrency
levels.

**Question.** A competition rerun has a fixed ~9h wall-clock. Aggregate
throughput is a property of the server, so total tokens generated ≈
`throughput × wall_clock` *regardless* of concurrency — raising concurrency
28 → 37 gives each game more wall-clock but a proportionally thinner slice of
the GPU, and tokens-per-game come out about the same. **Concurrency is only a
win if aggregate throughput actually rises with more concurrent sequences.**
A real run showed KV-cache utilisation at only ~22%, which *suggests* headroom
— but that is an inference, not a measurement. This kernel measures it.

**Caveat on prompt length.** The analyzer runs a 32K rolling window at steady
state; this benchmark uses ~11–13K-token prompts (unique per request, so
prefix caching cannot collapse the prefill) as a compromise between realism and
prefill cost. Absolute KV-utilisation figures here are therefore *lower* than
production's; the *shape* of the throughput-vs-concurrency curve is the result
of interest.
"""


def main() -> None:
    prod = json.loads(PROD_NB.read_text(encoding="utf-8"))
    bench_src = BENCH_CELL.read_text(encoding="utf-8")

    cells: list[dict] = [
        {"cell_type": "markdown", "metadata": {}, "source": HEADER_MD.splitlines(keepends=True)}
    ]
    for idx in SETUP_CELL_INDICES:
        cell = prod["cells"][idx]
        if cell["cell_type"] != "code":
            raise RuntimeError(f"production cell {idx} is not code: {cell['cell_type']}")
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": list(cell["source"]),
            }
        )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": bench_src.splitlines(keepends=True),
        }
    )

    nb = {
        "cells": cells,
        "metadata": prod.get("metadata", {}),
        "nbformat": prod.get("nbformat", 4),
        "nbformat_minor": prod.get("nbformat_minor", 5),
    }
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(cells)} cells: 1 markdown + "
          f"{len(SETUP_CELL_INDICES)} verbatim setup + 1 benchmark)")


if __name__ == "__main__":
    main()
