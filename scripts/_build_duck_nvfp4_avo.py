"""Build the AVO arm: the anim graft with the solver bundle swapped for AVO.

What this is
------------
``experiments/stage7_components.md`` enumerated 551 Kaggle datasets, found 13
distinct TAAF solver lineages, and found that exactly **two** are strict
supersets of the configuration that currently scores 3.79. This builds the
first of them.

``jakobbrggen/taaf-kaggle-source`` v27 (``ARC3-Inference 74ff3df``, branch
``experiment/avo-v2``) keeps ``animation.py`` and ``noop_guard.py`` -- the +25%
win this project measured -- and adds ``inference/avo/`` (733 lines): durable
cross-level memory, a phased inspect/plan/implement/evaluate loop, and a
stagnation supervisor. The supervisor costs **zero extra model calls**: it
appends a paragraph to the prompt the turn already sends. That is why it
survives the clock arithmetic in ``stage7_components.md`` §5, which excludes
a verifier pass (+43% of the clock), self-consistency (+86%) and a prompt
ensemble (+69%).

We mount ``raist321/taaf-avo-v27-bundle``, a **frozen v1 pin** of that tree,
**not** the ``jakobbrggen`` slug, which is rolling and could change under a
rerun. The pin was re-verified byte-identical on 2026-09-19 (``diff -rq``
against a fresh download of the rolling slug: no differences except
raist321's own added ``README.md``).

Arms
----
``--phased-loop 0`` (arm 1, the default) sets ``ARC3_AVO_PHASED_LOOP=0``,
which the bundle ships precisely as its memory+supervisor-only ablation.
AVO's ``INSPECT`` phase opens with *"Do not act this turn unless the
situation is already unambiguous"* -- one turn in four instructed not to act,
on top of the 37.7% dead-turn rate ``stage7_depth.md`` §2 measured. Our own
dose-response (commit floor +48% actions / -10% score; no-thinking +203%
actions / 0.64) shows removing deliberation hurts, but does **not** establish
that adding more helps. Arm 1 holds that second variable off.

``--phased-loop 1`` (arm 2) is the same notebook with the flag on, as a
separate kernel id. One variable between them.

The two traps, and how they are handled
---------------------------------------
1. **``deploy_target.pkl`` carries ``max_runtime_s = 54000.0``** (15 h -- the
   AVO settings docstring's "15h Kaggle run"). The inherited cell 13 *asserts*
   32400.0, so inheriting it either fails the run outright or, if the assert
   were simply deleted, overruns Kaggle's 9 h hard cap. This script rewrites
   that block to **set** 32400.0, print what it overrode, and still fail loudly
   on a value that is neither of the two known ones.
2. **The phased loop is a second variable.** Arm 1 pins it off, and asserts
   ``AvoSettings.from_env().phased_loop is False`` inside the notebook rather
   than trusting that the env var was read.

What this script changes, exhaustively
--------------------------------------
Every one of the 18 inherited cells is sha256-recorded before the patch and
re-checked after, so only the cells named here can differ:

* cell 0 (markdown) -- replaced with our AVO attribution header.
* cell 7 -- the mount ref and the ``_find_bundle_dir`` label.
  ``ANIM_BUNDLE_DIR`` keeps its name on purpose: cell 9 (the graft's teeth --
  the ``sys.path`` substitution and the ``LOCAL_ANALYZER_*`` overrides)
  references it, and renaming would force an edit to a cell that must stay
  byte-identical. The name is not even wrong: the AVO bundle *is* the anim
  bundle plus AVO, and cell 7's inherited ``animation.py`` assert proves it.
* cell 11 -- the ``bm.label`` assert, plus a new ``avo_agent is True`` assert.
* cell 13 -- trap 1.
* two cells inserted after 13 -- the arm's env var and its verification.
* ``kernel-metadata.json`` -- ``dataset_sources[2]`` and the kernel id/title.

Cells 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 17 are asserted unchanged.
Notably that includes cell 9 and cell 15, so the serving profile, the analyzer
knobs and the run loop are identical to the incumbent's.

Attribution
-----------
Unchanged from the anim graft, plus one: the AVO subsystem is Tufa Labs'
reimplementation (by Jakob Brueggen, branch ``experiment/avo-v2``) of the
architecture NVIDIA described; NVIDIA released no code. Neither their 100.00
public-set figure nor any upstream's leaderboard score is ours.

Usage
-----
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_avo.py               # arm 1
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_avo.py --phased-loop 1
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_avo.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook"
SRC_NB = SRC_DIR / "arc3-duck-nvfp4-anim.ipynb"
SRC_META = SRC_DIR / "kernel-metadata.json"

ANIM_DATASET = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
AVO_DATASET = "raist321/taaf-avo-v27-bundle"

# The whole notebook, as inherited. A drift here means the base moved and every
# telemetry comparison against the incumbent's free run is off.
# sha256 of the file read in *text* mode, so it is line-ending independent (the
# working tree is CRLF on this box, the git blob is LF; both hash the same here).
EXPECTED_SRC_SHA256 = "4fdd395c6974240dbfa7d463e12ce1e10dafdf4000e9f1201b9733a4c79e8220"

ARMS = {
    0: {
        "dir": "kaggle_submission_duck_nvfp4_avo",
        "nb": "arc3-duck-nvfp4-avo.ipynb",
        "kernel": "calamitychasm/arc3-duck-nvfp4-avo",
        "note": "memory + supervisor + exploit deadline; phased loop OFF",
    },
    1: {
        "dir": "kaggle_submission_duck_nvfp4_avo_phased",
        "nb": "arc3-duck-nvfp4-avo-pl.ipynb",
        "kernel": "calamitychasm/arc3-duck-nvfp4-avo-pl",
        "note": "arm 1 plus the four-phase inspect/plan/implement/evaluate loop",
    },
}

# ---------------------------------------------------------------- cell patches

# (cell index, old, new). Each `old` must occur exactly once in that cell.
CELL_PATCHES: list[tuple[int, str, str]] = [
    # --- cell 7: the mount and the bundle label -----------------------------
    (
        7,
        f'"{ANIM_DATASET}"]',
        f'"{AVO_DATASET}"]',
    ),
    (
        7,
        'ANIM_BUNDLE_DIR = _find_bundle_dir("anim-20260807-anim")     '
        "# ours: the solver tree + its pickled benchmark / target",
        'ANIM_BUNDLE_DIR = _find_bundle_dir("avo-kaggle")     '
        "# [calamitychasm] ours: the AVO solver tree (anim + inference/avo) + its pickles.\n"
        "# Name kept so cell 9's graft teeth stay byte-identical. The inherited animation.py\n"
        "# assert two lines below is what proves this bundle is still the anim superset.",
    ),
    # --- cell 11: label, and the assert that is the point of the arm --------
    (
        11,
        '# thui-animfast: the anim bundle\'s target (32400 s)',
        "# [calamitychasm] the AVO bundle's target -- 54000 s, overridden to 32400 in cell 13",
    ),
    (
        11,
        "# thui-animfast: the unpickled solver must be the anim chassis, not the June duck.\n"
        'assert bm.label == "anim-20260807-anim", bm.label',
        "# thui-animfast: the unpickled solver must be the anim chassis, not the June duck.\n"
        "# [calamitychasm] AVO is that chassis plus inference/avo, so both halves are checked.\n"
        'assert bm.label == "avo-kaggle", bm.label',
    ),
    (
        11,
        'assert type(bm.solver).__module__ == "inference.framework.solver"',
        'assert getattr(bm.solver, "avo_agent", None) is True, vars(bm.solver)   '
        "# [calamitychasm] the point of this arm\n"
        'assert type(bm.solver).__module__ == "inference.framework.solver"',
    ),
    (
        11,
        'print(f"thui-animfast: bm.label={bm.label} solver={type(bm.solver).__name__} '
        'animation_awareness={bm.solver.animation_awareness} "',
        'print(f"thui-animfast: bm.label={bm.label} solver={type(bm.solver).__name__} '
        'animation_awareness={bm.solver.animation_awareness} avo_agent={bm.solver.avo_agent} "',
    ),
    # --- cell 13: TRAP 1 -- set the budget, do not assert it ----------------
    (
        13,
        "if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:\n"
        "    raise RuntimeError(\n"
        "        f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.'\n"
        "    )\n",
        "# [calamitychasm] AVO TRAP 1 -- SET the notebook budget, do not assert it.\n"
        "# The anim bundle's deploy_target.pkl carried 32400.0 and the inherited cell asserted it.\n"
        "# The AVO bundle's carries 54000.0 (15 h -- inference/avo/settings.py's \"15h Kaggle run\").\n"
        "# Kaggle's hard cap is 9 h, so asserting here fails the run and inheriting 54000 overruns\n"
        "# it. Set 32400 and print the override so it is visible in the log instead of silent.\n"
        "# Still fails loudly on a third value: that would mean the pinned bundle moved.\n"
        "_inherited_max_runtime_s = float(getattr(target, 'max_runtime_s', 0.0) or 0.0)\n"
        "if _inherited_max_runtime_s not in (32400.0, 54000.0):\n"
        "    raise RuntimeError(\n"
        "        f'Unexpected inherited notebook budget {target.max_runtime_s!r}; expected 32400 (anim) '\n"
        "        f'or 54000 (avo-v27). The pinned bundle moved -- re-review before running.'\n"
        "    )\n"
        "target.max_runtime_s = 32400.0\n"
        "print(\n"
        "    f'AVO_BUDGET_OVERRIDE inherited_s={_inherited_max_runtime_s} applied_s={target.max_runtime_s} '\n"
        "    f'per_game_s={bm.solver.max_runtime_s_per_game} '\n"
        "    f'exploit_deadline_s={bm.solver.max_runtime_s_per_game * 0.6}',\n"
        "    flush=True,\n"
        ")\n",
    ),
]

# ------------------------------------------------------------- inserted cells

ANCHOR_INDEX = 13
ANCHOR_MARKER = "PUBLIC25_SETTINGS"

ARM_HEADING = """\
## 6b. The AVO arm

The one behavioural variable in this notebook relative to the arm it will be
compared against. `ARC3_AVO_PHASED_LOOP` is the bundle's own ablation switch:
off gives memory + stagnation supervisor + exploit deadline against the same
base agent; on adds the four-phase inspect/plan/implement/evaluate directive.

The flag is *verified* here, not merely exported: `AvoSettings.from_env()` is
the exact call the solver makes per game, so reading it back is the same value
the run will use. It also resolves `inference.avo` off the mounted bundle, which
is the check that catches a stale `sys.path` before 9 hours are spent on it.
"""

ARM_CELL_TEMPLATE = '''\
# [calamitychasm] THE ONE VARIABLE OF THIS ARM.
# {note}
#
# Set after cell 9 on purpose: cell 9 asserts `inference` is not yet imported before it
# applies the graft's LOCAL_ANALYZER_* overrides, and nothing in inference/avo reads this
# flag at import time -- AvoSettings.from_env() is called per game inside the solver
# (_make_analyzer) and once at run start (_write_effective_flags), both after this cell.
os.environ["ARC3_AVO_PHASED_LOOP"] = "{flag}"

import inference.avo.settings as _avo_settings_mod
import inference.avo.prompts as _avo_prompts_mod

# The module must come from the mounted AVO bundle, not from anything else on sys.path.
for _m in (_avo_settings_mod, _avo_prompts_mod):
    assert str(Path(_m.__file__).resolve()).startswith(str(ANIM_BUNDLE_DIR.resolve())), (_m.__name__, _m.__file__)

# Read back through the solver's own call, not through os.environ.
_avo = _avo_settings_mod.AvoSettings.from_env()
assert _avo.phased_loop is {expect}, _avo
# Everything else is the bundle's shipped default: one variable, and this proves it.
assert (_avo.stagnation_turns, _avo.unrewarded_turns, _avo.escalation_interventions) == (3, 12, 3), _avo
assert _avo.exploit_deadline == 0.6, _avo
assert (_avo.max_facts, _avo.max_rules, _avo.max_failures) == (24, 16, 16), _avo

# INSPECT is the directive that tells the model not to act. Recorded either way so the
# log says which arm ran without needing the kernel metadata.
print("AVO_ARM " + json.dumps(_avo.as_dict(), sort_keys=True), flush=True)
print(f"AVO_ARM phased_loop={{_avo.phased_loop}} inspect_directive_active={{_avo.phased_loop}} "
      f"module={{Path(_avo_settings_mod.__file__).parent}}", flush=True)
'''

HEADER_MD = """\
# arc3-duck-nvfp4-avo{title_suffix} - NVFP4 serving stack + the AVO solver

**This is a reproduction of other people's work, run on our own account to
measure it. No score quoted by any upstream is ours.** Full credit and licence
text in `THIRD_PARTY_NOTICE.md`.

Identical to `calamitychasm/arc3-duck-nvfp4-anim` except that the **solver
bundle** is swapped. The serving stack, the model, the analyzer knobs
(`LOCAL_ANALYZER_SEED=20260825`, `LOCAL_ANALYZER_YIELD_SECONDS=180`),
`concurrency=28`, `max_runtime_s_per_game=7920` and the run loop are all
carried over byte-identically, and the build script asserts every one of the
18 inherited cells by sha256.

Upstreams:

- **Solver** - the Tufa Labs Duck harness, on Jakob Brueggen's
  `experiment/avo-v2` branch (`ARC3-Inference 74ff3df`), mounted as the frozen
  pin **`raist321/taaf-avo-v27-bundle`** rather than the rolling
  `jakobbrggen/taaf-kaggle-source` slug, and executed unmodified. Verified
  byte-identical to that slug's v27 on 2026-09-19.
- **Serving / weights** - Keith Tyser's `duck-qwen3-8-flash-next-nvfp4-mtp`
  appliance over RadixArk's NVFP4 quantisation of Qwen3.8-Flash-Next.
  Unmodified and sealed.
- **The NVFP4+anim graft this is patched onto** - Thuitanium / Knowless Crew
  (`yocybercode/thui-animfast-b71-full25-r1`).
- **AVO** - Tufa Labs' reimplementation, from the published description, of the
  architecture NVIDIA's AVO team described. NVIDIA released no code and no
  ablations, and their 100.00 public-set figure was obtained with a frontier
  model, not this one.

## What AVO adds

`inference/avo/`, 733 lines, subclassing `ToolAgent` rather than replacing it:

1. **Durable memory** - the base agent clears `_summarized_knowledge` whenever
   the runtime dir changes, i.e. at every level boundary, which is exactly
   where it is worth most. AVO persists it and reloads it per game.
2. **Stagnation supervisor** - two triggers (`barren_turns >= 3`, which yields
   to frame novelty, and `unrewarded_turns >= 12`, which ignores it), with a
   3-step escalation to a hard redirect. **Zero extra model calls**: it appends
   a paragraph to the prompt the turn already sends.
3. **Exploit deadline** - at 0.6 x 7,920 s = 4,752 s the arm stops building a
   world model and plays its best policy.
4. **Phased loop** - {phased_desc}

## Why the score of this free run cannot rank the arm

A single public-25 mean carries SE +/-2.46 (`experiments/stage7_noise_floor.md`).
This run is for **counted** telemetry -- dead-turn rate, actions, calls, levels
by index, supervisor firings -- and for catastrophe detection. The incumbent's
matched free run is **9.97 mean / 2,615 actions / 42 levels / 20-of-25 games
scoring**; the mean is reported only for comparability, not to rank anything.
"""

PHASED_DESC = {
    0: (
        "**disabled in this arm** (`ARC3_AVO_PHASED_LOOP=0`, the bundle's own\n"
        "   ablation switch). Its `INSPECT` phase opens *\"Do not act this turn unless\n"
        "   the situation is already unambiguous\"*, which would instruct one turn in four\n"
        "   not to act on top of a measured 37.7% dead-turn rate. That is a second\n"
        "   variable and it is held off here."
    ),
    1: (
        "**enabled in this arm** (`ARC3_AVO_PHASED_LOOP=1`, the bundle default).\n"
        "   INSPECT / PLAN / IMPLEMENT / EVALUATE rotate on turn index, not on model\n"
        "   self-report. This is the second variable, and it is the only difference\n"
        "   between this notebook and `arc3-duck-nvfp4-avo`."
    ),
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cell(kind: str, text: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build(phased_loop: int, check_only: bool = False) -> None:
    arm = ARMS[phased_loop]
    out_dir = REPO / arm["dir"] / "notebook"
    out_nb = out_dir / arm["nb"]
    out_meta = out_dir / "kernel-metadata.json"

    raw = SRC_NB.read_text(encoding="utf-8")
    got_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if got_sha != EXPECTED_SRC_SHA256:
        raise SystemExit(
            f"source notebook sha256 {got_sha} != expected {EXPECTED_SRC_SHA256}; "
            "the anim base moved -- re-review before building an arm against it"
        )

    nb = json.loads(raw)
    cells = nb["cells"]
    if len(cells) != 18:
        raise SystemExit(f"anim notebook has {len(cells)} cells, expected 18")
    inherited = [_digest("".join(c["source"])) for c in cells]

    anchor = "".join(cells[ANCHOR_INDEX]["source"])
    if cells[ANCHOR_INDEX]["cell_type"] != "code" or ANCHOR_MARKER not in anchor:
        raise SystemExit(f"cell {ANCHOR_INDEX} is not the customization hook")

    # --- apply the patches, each of which must match exactly once ------------
    patched: set[int] = set()
    for idx, old, new in CELL_PATCHES:
        src = "".join(cells[idx]["source"])
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"cell {idx}: expected exactly 1 occurrence of {old[:70]!r}, found {n} "
                "-- the base drifted, refusing to build a half-patched arm"
            )
        cells[idx]["source"] = src.replace(old, new).splitlines(keepends=True)
        patched.add(idx)

    # Nothing anim-specific may survive in a patched cell.
    for idx in sorted(patched):
        src = "".join(cells[idx]["source"])
        for stale in (ANIM_DATASET, '"anim-20260807-anim"'):
            if stale in src:
                raise SystemExit(f"cell {idx} still references {stale!r} after patching")

    # --- header ------------------------------------------------------------
    cells[0] = _cell(
        "markdown",
        HEADER_MD.format(
            title_suffix="-pl" if phased_loop else "",
            phased_desc=PHASED_DESC[phased_loop],
        ),
    )
    patched.add(0)

    # --- the arm's own cells, inserted after the customization hook ---------
    arm_code = ARM_CELL_TEMPLATE.format(
        note=arm["note"],
        flag=phased_loop,
        expect="True" if phased_loop else "False",
    )
    compile(arm_code, "<arm>", "exec")
    new_cells = (
        cells[: ANCHOR_INDEX + 1]
        + [_cell("markdown", ARM_HEADING), _cell("code", arm_code)]
        + cells[ANCHOR_INDEX + 1 :]
    )
    nb["cells"] = new_cells

    # --- post-condition: only the named cells differ ------------------------
    got = [_digest("".join(c["source"])) for c in new_cells]
    kept = got[: ANCHOR_INDEX + 1] + got[ANCHOR_INDEX + 3 :]
    changed = {i for i, (a, b) in enumerate(zip(inherited, kept)) if a != b}
    if changed != patched:
        raise SystemExit(
            f"cells changed: {sorted(changed)}, expected exactly {sorted(patched)} "
            "-- refusing to build a confounded arm"
        )
    if len(new_cells) != 20:
        raise SystemExit(f"built {len(new_cells)} cells, expected 20")

    # --- metadata -----------------------------------------------------------
    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    ds = list(meta["dataset_sources"])
    if ds[-1] != ANIM_DATASET:
        raise SystemExit(f"dataset_sources[-1] is {ds[-1]!r}, expected the anim bundle")
    ds[-1] = AVO_DATASET
    meta["dataset_sources"] = ds
    meta["id"] = arm["kernel"]
    meta["title"] = arm["kernel"].split("/", 1)[1]
    meta["code_file"] = out_nb.name
    for key, want in (
        ("enable_internet", False),
        ("enable_gpu", True),
        ("is_private", True),
        ("machine_shape", "NvidiaRtxPro6000"),
    ):
        if meta.get(key) != want:
            raise SystemExit(f"kernel metadata {key}={meta.get(key)!r}, expected {want!r}")
    if meta["competition_sources"] != ["arc-prize-2026-arc-agi-3"]:
        raise SystemExit("competition_sources drifted")

    nb_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"

    if check_only:
        for path, want in ((out_nb, nb_text), (out_meta, meta_text)):
            if not path.exists():
                raise SystemExit(f"{path} not built")
            if path.read_text(encoding="utf-8") != want:
                raise SystemExit(f"{path} is stale -- rebuild")
        print(f"arm {phased_loop} ({arm['kernel']}) is up to date")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb.write_text(nb_text, encoding="utf-8")
    out_meta.write_text(meta_text, encoding="utf-8")
    print(f"built {out_nb}")
    print(f"  base      {SRC_NB.name} sha256={got_sha[:16]} (18 cells, all asserted)")
    print(f"  patched   cells {sorted(patched)}; 14 inherited cells byte-identical")
    print(f"  inserted  1 markdown + 1 code cell after index {ANCHOR_INDEX} ({ANCHOR_MARKER})")
    print(f"  mount     {ANIM_DATASET}\n            -> {AVO_DATASET}")
    print(f"  arm       ARC3_AVO_PHASED_LOOP={phased_loop} ({arm['note']})")
    print(f"  kernel    {arm['kernel']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phased-loop", type=int, choices=(0, 1), default=0)
    ap.add_argument("--check", action="store_true", help="verify the built arm is current")
    ap.add_argument("--both", action="store_true", help="build/check both arms")
    args = ap.parse_args()
    for pl in ((0, 1) if args.both else (args.phased_loop,)):
        build(pl, check_only=args.check)


if __name__ == "__main__":
    main()
