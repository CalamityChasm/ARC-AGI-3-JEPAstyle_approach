"""Generate arc3-codeworld-live-diag.ipynb from diag_driver.py.

The driver lives as a real .py file so it is reviewable/diffable; this
script inlines it into the notebook so a change needs only a free
`kaggle kernels push`, never a dataset re-version.

Usage:  python build_diag_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = (HERE / "diag_driver.py").read_text(encoding="utf-8")

# The driver is embedded via a base64 blob rather than a heredoc so that
# nothing in it (quotes, backslashes, triple-quotes, ``` fences inside the
# prompt strings) can possibly break the notebook cell that writes it out.
import base64

DRIVER_B64 = base64.b64encode(DRIVER.encode("utf-8")).decode("ascii")

CELL_INSTALL = """\
!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv
"""

CELL_SETUP = '''\
# ---------------------------------------------------------------------------
# CodeWorldAgent LIVE DIAGNOSTIC -- runs UNCONDITIONALLY (no
# KAGGLE_IS_COMPETITION_RERUN gate). That gate is exactly why every previous
# free push validated nothing: it hides all the real setup + agent code.
#
# THIS KERNEL IS NEVER SUBMITTED FOR SCORING. It exists only to answer:
# can Qwen3-Coder-30B-A3B write a replay-passing WorldModel for a real game?
# ---------------------------------------------------------------------------
import base64
import glob
import os
import shutil
import subprocess
import sys
import time

_t0 = time.time()
def _el():
    return f"{time.time() - _t0:6.1f}s"


def find_model_dir(keyword):
    """Same resolution the real submission notebook uses."""
    candidates = []
    for path in glob.glob("/kaggle/input/**/config.json", recursive=True):
        if keyword.lower() in path.lower():
            candidates.append(os.path.dirname(path))
    if not candidates:
        return None
    return sorted(candidates)[-1]


CODER_MODEL_DIR = find_model_dir("qwen3-coder") or find_model_dir("qwen")
GEMMA_MODEL_DIR = find_model_dir("gemma-3-12b-it") or find_model_dir("gemma")
print(f"[{_el()}] CODER_MODEL_DIR = {CODER_MODEL_DIR}", flush=True)
print(f"[{_el()}] GEMMA_MODEL_DIR = {GEMMA_MODEL_DIR}  (resolved for parity; NOT loaded)", flush=True)
assert CODER_MODEL_DIR, "could not locate coder model directory"

print(f"[{_el()}] === copying competition harness + environment_files ===", flush=True)
_COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
if not os.path.exists("/kaggle/working/ARC-AGI-3-Agents"):
    shutil.copytree(
        f"{_COMP}/ARC-AGI-3-Agents",
        "/kaggle/working/ARC-AGI-3-Agents",
        ignore=shutil.ignore_patterns(".git"),
    )
    shutil.copytree(
        f"{_COMP}/environment_files",
        "/kaggle/working/ARC-AGI-3-Agents/environment_files",
    )
_n_envs = len(glob.glob("/kaggle/working/ARC-AGI-3-Agents/environment_files/**/metadata.json", recursive=True))
print(f"[{_el()}] environment_files: {_n_envs} metadata.json found", flush=True)

print(f"[{_el()}] === copying llm_engine + agent from the FIXED dataset ===", flush=True)
_DATASET = "/kaggle/input/datasets/calamitychasm/llm-world-engine-agent-fixed"
print(f"[{_el()}] dataset mount exists: {os.path.exists(_DATASET)}", flush=True)
if not os.path.exists(_DATASET):
    # Dump the real tree rather than guessing at the mount convention.
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count("/") <= 5:
            print("   ", root, dirs[:8], files[:8], flush=True)
    raise SystemExit("dataset mount path not found -- see tree above")

if os.path.exists("/kaggle/working/llm_engine"):
    shutil.rmtree("/kaggle/working/llm_engine")
shutil.copytree(f"{_DATASET}/llm_engine", "/kaggle/working/llm_engine")
shutil.copy(
    f"{_DATASET}/code_world_agent.py",
    "/kaggle/working/ARC-AGI-3-Agents/agents/templates/code_world_agent.py",
)

with open("/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py", "w") as f:
    f.write(
        "from typing import Type, cast\\n"
        "from dotenv import load_dotenv\\n"
        "from .agent import Agent, Playback\\n"
        "from .swarm import Swarm\\n"
        "from .templates.random_agent import Random\\n"
        "from .templates.code_world_agent import CodeWorldAgent\\n"
        "\\n"
        "load_dotenv()\\n"
        "\\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\\n"
        "    \\"random\\": Random,\\n"
        "    \\"codeworldagent\\": CodeWorldAgent,\\n"
        "}\\n"
    )

with open("/kaggle/working/ARC-AGI-3-Agents/.env", "w") as f:
    f.write(
        "ARC_API_KEY=offline-diag\\n"
        "OPERATION_MODE=offline\\n"
        "ENVIRONMENTS_DIR=/kaggle/working/ARC-AGI-3-Agents/environment_files\\n"
        "RECORDINGS_DIR=/kaggle/working/diag_recordings\\n"
    )

print(f"[{_el()}] === writing diag driver ===", flush=True)
DRIVER_B64 = "__DRIVER_B64__"
with open("/kaggle/working/diag_driver.py", "wb") as f:
    f.write(base64.b64decode(DRIVER_B64))

run_env = {
    **os.environ,
    "MPLBACKEND": "agg",
    "LLM_BACKEND": "transformers",
    "CODER_MODEL_DIR": CODER_MODEL_DIR,
    # The action head is budget-disabled in the driver; pointing it at the
    # coder dir means get_shared_transformers_client() hands back the same
    # already-loaded model instead of putting a second multi-GB model in
    # VRAM alongside the 30B coder.
    "ACTION_MODEL_DIR": CODER_MODEL_DIR,
    "GEMMA_MODEL_DIR_PARITY": GEMMA_MODEL_DIR or "",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "ARC_API_KEY": "offline-diag",
    "OPERATION_MODE": "offline",
    "ENVIRONMENTS_DIR": "/kaggle/working/ARC-AGI-3-Agents/environment_files",
    "RECORDINGS_DIR": "/kaggle/working/diag_recordings",
    "DIAG_N_GAMES": "2",
    "DIAG_MAX_ACTIONS": "40",
    "DIAG_CODER_BUDGET": "3",
    "DIAG_DRAFT_ATTEMPTS": "5",
    "DIAG_REPAIR_ATTEMPTS": "2",
    "DIAG_LLM_DEADLINE_MIN": "240",
    "DIAG_RUN_DEADLINE_MIN": "300",
    "PYTHONUNBUFFERED": "1",
}

print(f"[{_el()}] === running diag driver ===", flush=True)
result = subprocess.run(
    [sys.executable, "-u", "/kaggle/working/diag_driver.py"],
    cwd="/kaggle/working/ARC-AGI-3-Agents",
    env=run_env,
)
print(f"[{_el()}] === diag driver exited with code {result.returncode} ===", flush=True)
'''

CELL_TAIL = '''\
# Surface the machine-readable evidence in the notebook output too, so it
# survives even if the output file download is unavailable.
import json
import os

p = "/kaggle/working/diag_evidence.json"
if os.path.exists(p):
    ev = json.load(open(p))
    print("SUMMARY:", json.dumps(ev.get("summary"), indent=2))
    print("FIXED-CODE CHECKS:", json.dumps(ev.get("fixed_code_checks"), indent=2))
    print("ROUNDS:", json.dumps(ev.get("rounds"), indent=2))
    print("REPLAY RESULTS:", json.dumps(ev.get("replay_results"), indent=2))
    print("LOAD RESULTS:", json.dumps(
        [{k: v for k, v in r.items()} for r in ev.get("load_results", [])], indent=2))
    print("GAMES:", json.dumps(
        [{k: v for k, v in g.items() if k != "model_source"} for g in ev.get("games", [])],
        indent=2))
    print("ERRORS:", json.dumps(ev.get("errors"), indent=2))
else:
    print("NO EVIDENCE FILE at", p)

# Also drop the world_models/ revision trail into the output.
for root, dirs, files in os.walk("/kaggle/working/world_models"):
    for fn in files:
        print("revision artifact:", os.path.join(root, fn))
'''


def cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


nb = {
    "cells": [
        cell(CELL_INSTALL),
        cell(CELL_SETUP.replace("__DRIVER_B64__", DRIVER_B64)),
        cell(CELL_TAIL),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "arc3-codeworld-live-diag.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes, driver {len(DRIVER)} chars)")
