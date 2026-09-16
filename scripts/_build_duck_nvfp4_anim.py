"""Build our copy of the NVFP4-serving + anim-solver graft, from the public source.

What this is
------------
``experiments/stage7_model_search.md`` establishes that a **model** swap on the
NVFP4 stack is impossible: the serving bundle verifies the checkpoint's manifest
hash, repo, revision, file count, byte total, every one of 419 per-file SHA-256s,
config.json, the quant config, the vLLM runtime layers, the PLE patch's pre/post
hashes -- and its own hash. Nothing else can be served by it.

What that seal does **not** cover is the *solver*, which the notebook puts on
``sys.path`` from a separate, freely-chosen dataset mount. That is the one real
degree of freedom left on the winning stack, and `stage7_duck_nvfp4.md` already
flagged the relevant confound without resolving it:

> the bundle's ``git_status.txt`` pins a *different* solver revision from our own
> FP8 fork's bundle: ``ARC3-Inference aa69123`` ... where ours pins ``9158303``
> on ``feature/animation-awareness``. ... this is not literally the same solver
> snapshot our 2.57 ran. That is an uncontrolled difference ... and it should not
> be waved away.

``yocybercode/thui-animfast-b71-full25-r1`` (public, 10 votes, v1) is exactly
that experiment, already built and used by teams on the leaderboard: Keith
Tyser's NVFP4 serving stack, unmodified, with the **anim** solver
(``jakobbrggen/taaf-kaggle-source-anim-20260807-anim`` -- the same bundle our own
FP8 stack runs) grafted over the June duck solver, plus two analyzer knobs
(``LOCAL_ANALYZER_SEED=20260825``, ``LOCAL_ANALYZER_YIELD_SECONDS=180`` against
the NVFP4 bundle's persisted 60) and the anim bundle's own deploy target.

Attribution
-----------
Three upstreams, none of them ours, all credited in THIRD_PARTY_NOTICE.md:
Tufa Labs (the Duck solver), Jakob Brueggen (the anim branch bundle), Keith
Tyser / wuliao0 (the NVFP4 serving stack), and Thuitanium / Knowless Crew
(yocybercode, sahasawatt) for the graft itself. **No score quoted by any of them
is ours.** We reproduce it to measure it on our own account.

What this script changes, exhaustively
--------------------------------------
1. Every **code** cell is copied byte-for-byte and the script asserts it.
   The experiment is worthless if any of them drifts.
2. The two leading **markdown** cells -- which are that team's own provenance
   narrative, and would read as our claims under our kernel id -- are replaced
   with our attribution header.
3. One print-only probe cell is prepended, identical in purpose to the one in
   ``kaggle_submission_duck_nvfp4``: the upstream README warns that a manual
   copy must select the RTX PRO 6000 by hand, and getting that silently wrong
   is how a run is wasted.
4. ``kernel-metadata.json`` gets our id/title/privacy and
   ``competition_sources``; every mount, the docker image and ``machine_shape``
   are carried over untouched.

Usage
-----
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim.py <pulled-src-dir> \
        --out kaggle_submission_duck_nvfp4_anim/notebook
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

KERNEL_ID = "calamitychasm/arc3-duck-nvfp4-anim"
KERNEL_TITLE = "arc3-duck-nvfp4-anim"

# md5 of the source notebook this graft was built from. Asserted so a silent
# upstream re-push cannot change what we think we are measuring.
EXPECTED_SRC_MD5 = "87bd2660ab73a36a39249712a765859d"

PROBE_CELL = """\
# [calamitychasm] ADDED FOR THIS FORK -- diagnostic only, changes no behaviour.
# The upstream README warns that a manual copy must select the RTX PRO 6000 by hand.
# We push via the API with machine_shape=NvidiaRtxPro6000, which IS honoured (verified
# in experiments/stage7_duck_nvfp4.md), but a wrong card would waste the whole run, so
# print what we actually got before anything expensive happens.
import os, shutil, subprocess

print(subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
     "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip() or "nvidia-smi unavailable")
try:
    _mem_kb = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")))
    _ram = f"{_mem_kb / 1048576:.1f}"
except Exception:
    _ram = "?"
print(f"HW_PROBE host_ram_gib={_ram}  cpu_count={os.cpu_count()}  "
      f"free_disk_gib={shutil.disk_usage('/kaggle/working').free / 2**30:.1f}  "
      f"rerun={os.getenv('KAGGLE_IS_COMPETITION_RERUN')!r}")
"""

HEADER_MD = """\
# arc3-duck-nvfp4-anim - NVFP4 serving stack + the anim solver

**This is a reproduction of other people's work, run on our own account to
measure it. No score quoted by any upstream is ours.** Full credit and licence
text in `THIRD_PARTY_NOTICE.md`.

Four upstreams:

- **Solver** - the Tufa Labs Duck harness (Harold Bessis, Jeroen Cottaar, Isaiah
  Pressman, Andries Smit, Michal Tesnar, Stefano Viel), on Jakob Brueggen's
  `feature/animation-awareness` branch, mounted as
  `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` and executed unmodified.
- **Serving** - Keith Tyser's `duck-qwen3-8-flash-next-nvfp4-mtp`: the pinned
  `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint, offline vLLM runtime, NVFP4
  PLE patch, MTP-3 profile and server watchdog. Unmodified.
- **Weights** - RadixArk's NVFP4 quantisation of Qwen/Qwen3.8-Flash-Next
  (Qwen licence terms apply).
- **The graft itself** - Thuitanium / Knowless Crew
  (`yocybercode/thui-animfast-b71-full25-r1`, `sahasawatt/thui-animfast-v1`).
  Every code cell below is byte-identical to theirs; the build script asserts it.

## Why we are running it

Our own `experiments/stage7_duck_nvfp4.md` recorded, and never resolved, that the
NVFP4 serving bundle ships a *different* solver revision (`ARC3-Inference
aa69123`) from the anim bundle our FP8 stack ran (`9158303`,
`feature/animation-awareness`) - "an uncontrolled difference between the two
stacks ... it should not be waved away". This notebook is that controlled
comparison: same model, same serving profile, anim solver instead of the June
duck, plus the graft's two analyzer knobs (`LOCAL_ANALYZER_SEED=20260825`,
`LOCAL_ANALYZER_YIELD_SECONDS=180` against the bundle's persisted 60) and the
anim bundle's own deploy target.

Baseline to beat, measured on our account on the identical 25 public games:
**10.69 mean self-eval, 3,633 actions** (`calamitychasm/arc3-duck-nvfp4-baseline`
v2). Method and pitfalls - including that the log emits a *progressive* summary
block and only the last one is the result - in `experiments/stage7_model_search.md`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", type=Path, help="dir from `kaggle kernels pull -m`")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-md5-drift", action="store_true")
    args = parser.parse_args()

    src_nb = next(p for p in args.src_dir.glob("*.ipynb"))
    raw = src_nb.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != EXPECTED_SRC_MD5:
        msg = f"source notebook md5 {md5} != expected {EXPECTED_SRC_MD5}"
        if not args.allow_md5_drift:
            raise SystemExit(msg + " (upstream re-pushed; re-review the diff, then pass --allow-md5-drift)")
        print("WARNING: " + msg)

    nb = json.loads(raw.decode("utf-8"))
    src_cells = nb["cells"]
    code_before = [c for c in src_cells if c["cell_type"] == "code"]

    # Cells 0 and 1 are the graft author's own provenance markdown. Replace with
    # ours; assert they really are markdown so an upstream reshuffle cannot make
    # this silently delete code.
    if not (src_cells[0]["cell_type"] == "markdown" and src_cells[1]["cell_type"] == "markdown"):
        raise SystemExit("cells 0/1 are not both markdown; upstream layout changed, re-review")

    def cell(kind: str, text: str) -> dict:
        lines = text.splitlines(keepends=True)
        base = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            base["execution_count"] = None
            base["outputs"] = []
        return base

    out_cells = [cell("markdown", HEADER_MD), cell("code", PROBE_CELL)] + src_cells[2:]
    nb["cells"] = out_cells

    code_after = [c for c in out_cells if c["cell_type"] == "code"]
    if code_after[0]["source"] != cell("code", PROBE_CELL)["source"]:
        raise SystemExit("probe cell is not first")
    if len(code_after) - 1 != len(code_before):
        raise SystemExit(f"code cell count changed: {len(code_before)} -> {len(code_after) - 1}")
    for i, (a, b) in enumerate(zip(code_before, code_after[1:])):
        if "".join(a["source"]) != "".join(b["source"]):
            raise SystemExit(f"code cell {i} is not byte-identical to the source")
    compile("".join(code_after[0]["source"]), "<probe>", "exec")
    print(f"OK: {len(code_before)} code cells carried byte-identically; probe cell compiles")

    args.out.mkdir(parents=True, exist_ok=True)
    nb_path = args.out / f"{KERNEL_ID.split('/')[1]}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    src_meta = json.loads((args.src_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": src_meta["dataset_sources"],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": src_meta["model_sources"],
        "docker_image": src_meta["docker_image"],
        "machine_shape": src_meta["machine_shape"],
    }
    (args.out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {nb_path} ({len(out_cells)} cells)")
    print(f"      mounts: {meta['dataset_sources']} + {meta['model_sources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
