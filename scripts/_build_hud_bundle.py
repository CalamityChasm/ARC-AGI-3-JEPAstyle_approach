"""Build the HUD-annotation arm: our own copy of the anim solver bundle, with
four files patched, plus the kernel that mounts it.

Why a republished bundle at all
-------------------------------
Every previous arm swapped *which upstream dataset* the notebook mounts. This
one changes source inside the solver, so there is no upstream dataset to point
at: we have to publish the patched tree ourselves under ``calamitychasm`` and
mount that. The solver is MIT-licensed (``ARC3-Inference`` pyproject), so this
is permitted; ``THIRD_PARTY_NOTICE.md`` carries the attribution, and the MIT
notice for the detector rules (Rudakov, Shock & Cowley, arXiv:2512.24156) is
copied in beside it.

What changes, exhaustively
--------------------------
Four files under ``src/ARC3-Inference``, all via anchored replacements defined
in ``arc3_hud/splice.py``:

* ``inference/utils/segmentation.py`` -- the detector, and the two additive
  fields. **Gated off by default**: with ``ARC3_HUD_ANNOTATION`` unset the
  return value is equal to the unpatched bundle's, key for key.
* ``inference/agent/python_tool_sandbox.py`` -- one env passthrough. The sandbox
  is a subprocess with an explicit allowlist env and it is the *only* caller of
  ``segment_layer``; without this the gate could never reach the detector.
* ``inference/agent/prompts.py`` -- the field's documentation, written to be
  honest about the heuristic's failure modes. Empty strings when the gate is off.
* ``inference/agent/tool_agent.py`` -- the three duplicated per-turn
  descriptions, from the same two constants.

Every other file in the bundle is sha256-checked to be byte-identical, and the
four patched ones have their *pre-patch* sha256 pinned below, so a bundle that
moved underneath us fails the build instead of shipping a half-patched tree.

Usage
-----
    venv/Scripts/python.exe scripts/_build_hud_bundle.py --bundle <pristine-dir>
    venv/Scripts/python.exe scripts/_build_hud_bundle.py --bundle <dir> --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from arc3_hud import splice  # noqa: E402

DATASET_SLUG = "calamitychasm/taaf-anim-hud-v1"
DATASET_TITLE = "taaf-anim-hud-v1"
BENCHMARK_LABEL = "anim-20260807-anim"
SRC_ROOT = "src/ARC3-Inference"

#: sha256 (LF-normalised) of each patched file *before* the patch, in the
#: pristine `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` v1 bundle.
EXPECTED_PRE_PATCH = {
    "inference/utils/segmentation.py": "e9499bd859b4488455b481d3f0577a2c59feba9f78b92f20befc5d294ee09ee0",
    "inference/agent/python_tool_sandbox.py": "c7d47f14a556953fff3f889d34d1a8a42e1413104e2444cf5a7e33ee4445a08f",
    "inference/agent/prompts.py": "463f09b308450782408361ba37393b6b75f98b457dee1e562f5cf6f53495753f",
    "inference/agent/tool_agent.py": "856bf9b895d0ad8b959c8f828c7132b0e09eaa47f4c5cc6173785354090f8be7",
}

NOTICE = """\
# Third-party notice -- `calamitychasm/taaf-anim-hud-v1`

This dataset is **other people's work with a four-file patch of ours**,
republished on our own Kaggle account because the patch cannot be expressed as a
notebook change: it edits solver source that the notebook only mounts.

**No score reported by any upstream is ours, and none is quoted here as ours.**

## The bundle

The Tufa Labs ARC-AGI-3 Duck harness -- Harold Bessis, Jeroen Cottaar, Isaiah
Pressman, Andries Smit, Michal Tesnar, Stefano Viel -- on Jakob Brueggen's
`feature/animation-awareness` branch, distributed as the Kaggle dataset
`jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (v1). Licensed MIT
(`src/ARC3-Inference/pyproject.toml`). Copied verbatim except for the four files
listed in `HUD_PATCH.md`; `scripts/_build_hud_bundle.py` sha256-checks every
other file on every build.

## The HUD detection rules

Ported from Evgenii Rudakov, Ryan Shock and Nathan Cowley, *"Graph-Based
Exploration for ARC-AGI-3 Interactive Reasoning Tasks"* (AAAI 2026 Workshop on
AI for Scientific Research, arXiv:2512.24156), original code at
`github.com/dolphin-in-a-coma/arc-agi-3-just-explore`, MIT-licensed. Their
`identify_status_bars_with_rule` and its helpers are the source of the rules and
of all three thresholds (edge distance 3, aspect ratio 5, twin count 3). The
required copyright notice and licence text is reproduced verbatim in
`GRAPH_EXPLORER_THIRD_PARTY_LICENSE` beside this file, and this repo also
carries it at
`ARC-AGI-3-Agents/agents/templates/graph_explorer_THIRD_PARTY_LICENSE`.

## What is ours

`arc3_hud/hud_detect.py` (the port and its two documented deviations),
`arc3_hud/splice.py` (the four anchored edits), `scripts/_build_hud_bundle.py`,
the prompt wording, and the measurement in
`experiments/stage7_hud_perception.md`.
"""

PATCH_DOC = """\
# `taaf-anim-hud-v1` -- what is patched

Base: `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` v1, verbatim except
these four files under `src/ARC3-Inference/`.

| file | change |
|---|---|
| `inference/utils/segmentation.py` | adds `detect_hud_nodes` and, **only when `ARC3_HUD_ANNOTATION` is set**, a per-node `hud: bool` and a top-level `hud_node_ids: [...]` |
| `inference/agent/python_tool_sandbox.py` | passes `ARC3_HUD_ANNOTATION` through the sandbox's allowlist env (the sandbox subprocess is the only caller of `segment_layer`) |
| `inference/agent/prompts.py` | documents the two new fields, honestly, behind the same flag |
| `inference/agent/tool_agent.py` | the three duplicated per-turn descriptions, from the same two constants |

**Nothing is removed, masked, hidden or altered.** The annotation is additive
and advisory. With `ARC3_HUD_ANNOTATION` unset, `segment_layer`'s return value
is equal to the unpatched bundle's key for key, no node carries a `hud` field,
and every prompt string in the two agent modules is byte-identical.

The detector is a rule over shape and position only. It catches edge *bars*; it
does **not** catch HUD drawn as a block away from the frame edges, and it can
flag a genuine playable object that happens to be a long edge-hugging bar. That
is why the flag is advisory and why the prompt says so.
"""


def _read(path: Path) -> str:
    """Decode a source file with its line endings normalised to ``\\n``."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(path: Path, text: str, *, crlf: bool) -> None:
    out = text.replace("\n", "\r\n") if crlf else text
    path.write_bytes(out.encode("utf-8"))


def build(bundle: Path, out_dir: Path, check_only: bool) -> None:
    src_root = bundle / SRC_ROOT
    marker = bundle / "taaf-kaggle-bundle.json"
    if not marker.is_file():
        raise SystemExit(f"{bundle} has no taaf-kaggle-bundle.json -- not a TAAF bundle")
    label = json.loads(marker.read_text(encoding="utf-8")).get("benchmark_label")
    if label != BENCHMARK_LABEL:
        raise SystemExit(f"bundle label is {label!r}, expected {BENCHMARK_LABEL!r}")

    # --- pre-patch identity of the four files ------------------------------
    for rel, want in EXPECTED_PRE_PATCH.items():
        got = _sha(_read(src_root / rel))
        if got != want:
            raise SystemExit(
                f"{rel}: pre-patch sha256 {got} != pinned {want} -- the upstream bundle "
                "moved. Re-review the anchors before rebuilding this arm."
            )

    manifest: dict[str, dict[str, str]] = {}
    patched_text: dict[str, str] = {}
    for rel in splice.PATCHED_FILES:
        original = _read(src_root / rel)
        new = splice.patch_text(rel, original)
        compile(new, rel, "exec")  # never ship a bundle that cannot import
        if new == original:
            raise SystemExit(f"{rel}: patch was a no-op")
        patched_text[rel] = new
        manifest[rel] = {"before": _sha(original), "after": _sha(new)}

    if check_only:
        for rel, text in patched_text.items():
            live = out_dir / SRC_ROOT / rel
            if not live.is_file():
                raise SystemExit(f"{live} not built")
            if _read(live) != text:
                raise SystemExit(f"{live} is stale -- rebuild")
        print(f"{out_dir} is up to date ({len(patched_text)} patched files)")
        return

    # --- copy the bundle verbatim, then overwrite exactly four files -------
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # Never ship bytecode: importing the tree for a local smoke test leaves
    # __pycache__ behind, and a stale .pyc inside a Kaggle dataset is a silent
    # way to run source that is not the source in the diff.
    shutil.copytree(bundle, out_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    unchanged = 0
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel_bundle = path.relative_to(bundle).as_posix()
        rel_src = rel_bundle[len(SRC_ROOT) + 1 :] if rel_bundle.startswith(SRC_ROOT + "/") else None
        target = out_dir / path.relative_to(bundle)
        if rel_src in patched_text:
            crlf = b"\r\n" in path.read_bytes()
            _write(target, patched_text[rel_src], crlf=crlf)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(
            target.read_bytes()
        ).hexdigest():
            raise SystemExit(f"{rel_bundle} differs from the source bundle after copy")
        unchanged += 1

    (out_dir / "THIRD_PARTY_NOTICE.md").write_text(NOTICE, encoding="utf-8")
    (out_dir / "HUD_PATCH.md").write_text(PATCH_DOC, encoding="utf-8")
    shutil.copyfile(
        REPO / "ARC-AGI-3-Agents/agents/templates/graph_explorer_THIRD_PARTY_LICENSE",
        out_dir / "GRAPH_EXPLORER_THIRD_PARTY_LICENSE",
    )
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": DATASET_TITLE, "id": DATASET_SLUG, "licenses": [{"name": "other"}]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = REPO / "experiments" / "stage7_hud_perception_artifacts" / "bundle_patch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "base_dataset": "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
                "published_as": DATASET_SLUG,
                "benchmark_label": BENCHMARK_LABEL,
                "env_flag": splice.ENV_FLAG,
                "files_unchanged": unchanged,
                "files_patched": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"built {out_dir}")
    print(f"  base      {bundle} (label={label})")
    print(f"  patched   {len(patched_text)} files; {unchanged} files byte-identical")
    for rel, digests in manifest.items():
        print(f"            {rel}  {digests['before'][:12]} -> {digests['after'][:12]}")
    print(f"  dataset   {DATASET_SLUG}")
    print(f"  manifest  {manifest_path.relative_to(REPO)}")
    print("\nnext:  kaggle datasets create -p <out_dir> --dir-mode zip")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bundle", type=Path, required=True, help="pristine anim bundle dir")
    ap.add_argument("--out", type=Path, required=True, help="staging dir for the patched copy")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    build(args.bundle, args.out, args.check)


if __name__ == "__main__":
    main()
