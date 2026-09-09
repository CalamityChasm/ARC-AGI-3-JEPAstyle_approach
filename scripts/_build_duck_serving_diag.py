"""Build the free (non-scored) vLLM serving-configuration diagnostic kernel.

Sibling of `_build_duck_concurrency_diag.py`, and deliberately the same shape:
the diagnostic notebook is *generated*, not hand-maintained, so its setup path
is provably byte-identical to the real submission notebook's. Cells 1-5 of
`kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb`
(env setup, wheelhouse install, model/dataset mount config, input audit, and
the TAAF/vLLM boot) are copied verbatim; the trailing game-playing cells are
dropped and replaced with `serving_benchmark_cell.py`.

Why the setup path must stay identical: the benchmark recovers its BASELINE
vLLM argv from `/proc/<pid>/cmdline` of the server that this setup path starts.
If the setup diverged from production, the baseline would be a different server
and every "vs. baseline" number in the sweep would be measuring the wrong thing.

Usage (from repo root):
    python scripts/_build_duck_serving_diag.py
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
DIAG_DIR = REPO_ROOT / "kaggle_submission_duck" / "notebook_serving_diag"
BENCH_CELL = DIAG_DIR / "serving_benchmark_cell.py"
OUT_NB = DIAG_DIR / "arc3-duck-serving-benchmark.ipynb"

# Cells 1-5 of the production notebook: everything up to and including the
# TAAF/vLLM setup. Cell 0 is markdown (replaced), cells 6-10 are the game loop
# and diagnostics rendering (dropped -- this kernel plays no games).
SETUP_CELL_INDICES = [1, 2, 3, 4, 5]

HEADER_MD = """\
# stage7-duck-throughput — vLLM serving configuration vs. throughput

**Free diagnostic kernel. Plays no games. Consumes no submission quota.**

Cells 1–5 below are **byte-identical** to the production submission notebook
(`lb-9-arc3-duck-v12-with-qwen-3-8-27b`), so vLLM boots the real
`Qwen3.8-27B-FP8` on the real `NvidiaRtxPro6000` with the real bundled server
arguments. The final cell then sweeps **server configurations** at a fixed
client concurrency of 37 (the optimum measured in
`experiments/stage7_duck_concurrency.md`).

**Question.** A public fork claims a large gain from serving changes alone:
NVFP4 weights + 3-token NEXTN MTP speculative decoding + async scheduling +
prefix caching off + a small KV cache. Two things found before this kernel was
written change what needs measuring:

1. That fork's `TAAF_VLLM_*` environment variables are read only by **its own**
   source bundle. Ours contains zero references to them, so setting them on our
   stack is a silent no-op. This cell therefore mutates the **real vLLM argv**.
2. Our **existing** model already ships MTP weights
   (`text_config.mtp_num_hidden_layers = 1`, plus `mtp.safetensors` in the
   checkpoint). So speculative decoding — the recipe's biggest single lever —
   may be reachable with **no model swap at all**. That is the primary
   hypothesis here.

**Method.** Dump `vllm serve --help` so flag availability is verified rather
than assumed; recover the baseline argv from `/proc/<pid>/cmdline` of the
already-running server; measure the baseline; then restart the server once per
configuration and re-measure the identical workload. `mtp3` and `flags` are
each measured **alone** against the same baseline so the combined row can never
be mistaken for an attributable one.
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
