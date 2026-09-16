"""Build the free (non-scored) NVFP4 KV-capacity / sequence-slot diagnostic kernel.

Sibling of `_build_duck_serving_diag.py` and `_build_duck_concurrency_diag.py`,
and deliberately the same shape: the diagnostic notebook is *generated*, not
hand-maintained, so its setup path is provably byte-identical to the real NVFP4
notebook's. Cells 0, 4, 6, 8, 10 of
`kaggle_submission_duck_nvfp4/notebook/duck-qwen3-8-anim-base.ipynb` (hardware
probe, env + vLLM profile, ARC wheel install, mount resolution, and the bundled
`serving_setup.py` invocation that boots vLLM) are copied verbatim; the
benchmark-loading and game-playing cells are dropped and replaced with
`kv_benchmark_cell.py`.

Why the setup path must stay identical: the benchmark recovers its BASELINE
vLLM argv from the server that this setup path starts. If the setup diverged
from production, the baseline would be a different server and every "vs.
baseline" number in the sweep would be measuring the wrong thing.

Note the NVFP4 notebook's cell 4 is the one that exports the
`TAAF_VLLM_*` profile (`kv5-bf16-mtp3-c8-cg32`). It is copied UNCHANGED, so the
baseline this kernel measures is exactly the configuration the 2026-09-10
public-25 run used.

Usage (from repo root):
    python scripts/_build_duck_kv_diag.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_NB = (
    REPO_ROOT
    / "kaggle_submission_duck_nvfp4"
    / "notebook"
    / "duck-qwen3-8-anim-base.ipynb"
)
DIAG_DIR = REPO_ROOT / "kaggle_submission_duck_nvfp4" / "notebook_kv_diag"
BENCH_CELL = DIAG_DIR / "kv_benchmark_cell.py"
OUT_NB = DIAG_DIR / "arc3-duck-nvfp4-kv-benchmark.ipynb"

# Code cells of the NVFP4 notebook up to and including the vLLM boot.
#   0  hardware probe (added by this fork)
#   4  env + PUBLIC25_VLLM_PROFILE export
#   6  ARC runtime wheel install
#   8  mount resolution / TAAF_KAGGLE_INPUT_PATHS
#  10  source path setup + run setup_commands.json  <- boots vLLM
# 12+ load the pickled benchmark and play games: dropped, this kernel plays none.
SETUP_CELL_INDICES = [0, 4, 6, 8, 10]

HEADER_MD = """\
# stage7-turn-latency — what actually limits analyzer turns on the NVFP4 stack

**Free diagnostic kernel. Plays no games. Consumes no submission quota.**

The cells below are **byte-identical** to the NVFP4 submission notebook
(`arc3-duck-nvfp4-baseline`) up to and including the bundled `serving_setup.py`
call, so vLLM boots the real `Qwen3.8-Flash-Next-NVFP4` on the real
`NvidiaRtxPro6000` with the real `kv5-bf16-mtp3-c8-cg32` profile. The final cell
then sweeps server configurations and measures the one quantity that decides
anything: **analyzer turns per game**.

**Why.** The 2026-09-10 free public-25 run was 100% time-bound — all 25 games
hit the 7920 s wall, each getting only ~52 turns at a median 153 s per turn. Its
own `/metrics` scrape decomposes that turn as **queue 126.71 s (86.97%)** +
inference 18.61 s (prefill 1.60 s, decode 17.01 s). So turn latency is
queueing.

**But the obvious cause is already refuted.** `TAAF_VLLM_MAX_NUM_SEQS = 8`
against 28 game threads looks like 3.5× oversubscription — yet the server log
shows `Running` at **2 / 3 / 6** (min/p50/max) across all 792 scheduler
snapshots and **never once reaches 8**, while `Waiting` sits at 18–22. The real
limit is the **KV pool**: `GPU KV cache size: 105,202 tokens` against
21,608 tokens resident per request — **4.87 requests fit**, and 191 preemptions
were recorded.

**Method.** Recover the baseline argv from the bundle's own
`vllm-server-identity.json`; kill the bundle's vLLM watchdog so it cannot
relaunch the original configuration underneath a mutated one; then restart the
server once per configuration and re-measure an identical workload matched to
production's real shape (25 concurrent, ~20,175-token prompts, 1,433 output
tokens). `seqs40` is measured **alone** as a direct falsification test of the
sequence-slot hypothesis before any KV change is tried.
"""


def main() -> None:
    prod = json.loads(PROD_NB.read_text(encoding="utf-8"))
    bench_src = BENCH_CELL.read_text(encoding="utf-8")

    cells: list[dict] = [
        {"cell_type": "markdown", "metadata": {},
         "source": HEADER_MD.splitlines(keepends=True)}
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

    # Verify the copied cells really are byte-identical to production's.
    for out_idx, src_idx in enumerate(SETUP_CELL_INDICES, start=1):
        if "".join(cells[out_idx]["source"]) != "".join(prod["cells"][src_idx]["source"]):
            raise RuntimeError(f"setup cell {src_idx} was not copied verbatim")

    print(f"wrote {OUT_NB} ({len(cells)} cells: 1 markdown + "
          f"{len(SETUP_CELL_INDICES)} verbatim setup + 1 benchmark)")
    print("verified: all setup cells byte-identical to the production notebook")


if __name__ == "__main__":
    main()
