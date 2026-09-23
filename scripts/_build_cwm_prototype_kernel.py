"""Build the free CodeWorldAgent prototype kernel.

The prior live test (2026-09-09) ran CodeWorldAgent with
`LLM_BACKEND=transformers` against **Qwen3-Coder-30B-A3B** and concluded
0/25 replay passes. Two things have changed since, and both need a fresh
run rather than an argument:

  1. The served model is now **Qwen3.8-Flash-Next-NVFP4** behind in-kernel
     vLLM -- a different model on a much faster path. The 2026-09-22
     backtest showed that model *can* write a replay-passing world model
     and that with reasoning disabled its candidates actually load and
     run, which the 30B coder's never did.
  2. The per-level transcript reset now exists. Before it, clearing one
     level made the replay gate permanently unsatisfiable for the rest of
     that game, so the prior test never measured a game past its first
     level fairly.

This kernel therefore = the anim notebook's serving cells (which boot
vLLM) + the existing diag driver, repointed at that server. It plays real
games and costs no submission quota.

Usage:
    python scripts/_build_cwm_prototype_kernel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ANIM_NOTEBOOK = (
    REPO_ROOT / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "arc3-duck-nvfp4-anim.ipynb"
)
DIAG_NOTEBOOK = (
    REPO_ROOT / "kaggle_submission_llm_world_engine" / "notebook_diag"
    / "arc3-codeworld-live-diag.ipynb"
)
#: Cells 0..9 of the anim notebook are the serving stack; cell 9 runs
#: setup_commands.json, which is what starts vLLM on 127.0.0.1:1234.
SETUP_THROUGH = 9

KERNEL_SLUG = "arc3-cwm-prototype"
OWNER = "calamitychasm"

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_ID = "Qwen/Qwen3.8-Flash-Next-NVFP4"

#: Env replacements applied to the diag cell's `run_env` dict. The diag
#: kernel built this for the transformers backend; every line here moves
#: it onto the served model instead.
ENV_PATCHES = {
    '"LLM_BACKEND": "transformers",': (
        f'"LLM_BACKEND": "openai",\n'
        f'    "CODER_LLM_BASE_URL": "{BASE_URL}",\n'
        f'    "CODER_LLM_MODEL": "{MODEL_ID}",\n'
        f'    "ACTION_LLM_BASE_URL": "{BASE_URL}",\n'
        f'    "ACTION_LLM_MODEL": "{MODEL_ID}",\n'
        # Measured: with reasoning ON this model spent the entire reply\n'
        # budget thinking and emitted no code at all (27/27 truncated).\n'
        f'    "LLM_ENABLE_THINKING": "0",'
    ),
    # More games than the 2-game diagnostic: vLLM is far faster than the
    # in-process transformers path this driver was written for.
    '"DIAG_N_GAMES": "2",': '"DIAG_N_GAMES": "12",',
    # Enough actions to actually reach a second level, which is the thing
    # the per-level reset changes.
    '"DIAG_MAX_ACTIONS": "40",': '"DIAG_MAX_ACTIONS": "120",',
    '"DIAG_CODER_BUDGET": "3",': '"DIAG_CODER_BUDGET": "8",',
    # Fit inside a free session with room to write evidence.
    '"DIAG_LLM_DEADLINE_MIN": "240",': '"DIAG_LLM_DEADLINE_MIN": "150",',
    '"DIAG_RUN_DEADLINE_MIN": "300",': '"DIAG_RUN_DEADLINE_MIN": "180",',
}

#: The diag cell hard-asserts a local coder model directory, which does
#: not exist (and is not wanted) on the served path.
ASSERT_PATCHES = {
    'assert CODER_MODEL_DIR, "could not locate coder model directory"':
        '# CODER_MODEL_DIR is unused on the served path (LLM_BACKEND=openai).\n'
        'print("CODER_MODEL_DIR (unused, served path):", CODER_MODEL_DIR)',
}


def build() -> dict:
    anim = json.loads(ANIM_NOTEBOOK.read_text(encoding="utf-8"))
    diag = json.loads(DIAG_NOTEBOOK.read_text(encoding="utf-8"))

    header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# arc3-cwm-prototype - CodeWorldAgent on the served model\n",
            "\n",
            "Free run, **no submission quota**. Setup cells are reused verbatim ",
            "from `arc3-duck-nvfp4-anim` (itself a reproduction of other people's ",
            "work -- see that notebook's `THIRD_PARTY_NOTICE.md`) purely to boot ",
            "the NVFP4 vLLM server; the games are then played by our own ",
            "`CodeWorldAgent`, not by the duck solver.\n",
            "\n",
            "Changes since the 2026-09-09 live test, which scored 0/25 replay ",
            "passes with Qwen3-Coder-30B over `transformers`:\n",
            "\n",
            "* the served model is **Qwen3.8-Flash-Next-NVFP4** behind vLLM;\n",
            "* reasoning is **disabled** (measured: with it on, 27/27 replies hit ",
            "the token cap and none contained a `class WorldModel`);\n",
            "* the transcript now **resets per level**, so clearing a level no ",
            "longer makes the replay gate permanently unsatisfiable.\n",
        ],
    }

    diag_cells = []
    for cell in diag["cells"]:
        source = "".join(cell["source"])
        if cell.get("cell_type") == "code":
            for old, new in {**ENV_PATCHES, **ASSERT_PATCHES}.items():
                if old in source:
                    source = source.replace(old, new)
        diag_cells.append({**cell, "source": source.splitlines(True)})

    notebook = dict(anim)
    notebook["cells"] = [header] + anim["cells"][1:SETUP_THROUGH + 1] + diag_cells
    return notebook


def main() -> int:
    notebook = build()

    out_dir = REPO_ROOT / "kaggle_submission_cwm_prototype" / "notebook"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{KERNEL_SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(notebook['cells'])} cells)")

    # Prove every patch landed. A silently-unapplied replacement would run
    # the whole kernel on the wrong backend and look like a model result.
    body = "".join("".join(c["source"]) for c in notebook["cells"])
    required = [
        ('LLM_BACKEND -> openai', '"LLM_BACKEND": "openai"'),
        ('coder base_url', f'"CODER_LLM_BASE_URL": "{BASE_URL}"'),
        ('served model', f'"CODER_LLM_MODEL": "{MODEL_ID}"'),
        ('thinking disabled', '"LLM_ENABLE_THINKING": "0"'),
        ('12 games', '"DIAG_N_GAMES": "12"'),
        ('120 actions', '"DIAG_MAX_ACTIONS": "120"'),
        ('vLLM boot present', 'setup_commands.json'),
        ('agent dataset mounted', 'llm-world-engine-agent-fixed'),
    ]
    forbidden = [
        ('no transformers backend', '"LLM_BACKEND": "transformers"'),
        ('no hard model-dir assert', 'assert CODER_MODEL_DIR'),
    ]
    failures = [name for name, probe in required if probe not in body]
    failures += [name for name, probe in forbidden if probe in body]
    if failures:
        print("REFUSING TO SHIP -- checks failed: " + ", ".join(failures))
        return 1
    for name, _ in required + forbidden:
        print(f"  OK  {name}")

    from _check_notebook_cell import check_notebook

    problems = check_notebook(path)
    if problems:
        print(f"REFUSING TO SHIP -- {len(problems)} use-before-definition problem(s):")
        for p in problems[:10]:
            print("  " + p)
        return 1
    print("use-before-definition check PASSED")

    meta = json.loads(
        (ANIM_NOTEBOOK.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    meta["id"] = f"{OWNER}/{KERNEL_SLUG}"
    meta["title"] = KERNEL_SLUG
    meta["code_file"] = f"{KERNEL_SLUG}.ipynb"
    sources = list(meta.get("dataset_sources", []))
    for extra in (f"{OWNER}/llm-world-engine-agent-fixed",):
        if extra not in sources:
            sources.append(extra)
    meta["dataset_sources"] = sources
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'kernel-metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
