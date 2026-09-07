"""One-off script to generate kaggle_submission_graph_explorer_learned/
notebook/arc3-graph-explorer-learned-submission.ipynb -- combines
kaggle_submission_graph_explorer's setup pattern (GraphExplorerAgent copy,
minimal __init__.py) with kaggle_submission's torch-availability-check and
checkpoint-copying pattern (this agent, unlike the pure GraphExplorerAgent,
has a small torch model). Not part of the reproducible pipeline -- run
once, inspect/diff the output, discard."""

import json
from pathlib import Path

CELL2_SOURCE = '''\
import os
import shutil
import subprocess
import sys
import time

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    _t0 = time.time()

    def _el():
        return f"{time.time() - _t0:6.1f}s"

    # Write a placeholder submission before anything risky runs -- same
    # insurance pattern as every other submission notebook in this
    # project (rules.md: a submission is "auto-generated as long as the
    # agent acts on the games", so a total setup crash before that would
    # otherwise leave no submission file at all).
    import pandas as pd
    pd.DataFrame(
        data=[["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet("/kaggle/working/submission.parquet", index=False)
    print(f"[{_el()}] === step: wrote placeholder submission.parquet ===", flush=True)

    print(f"[{_el()}] === step: starting gateway wait in background ===", flush=True)
    # retry-max-time=600 matches the official reference notebook
    # (arcprize/ARC-AGI-3-Kaggle-Starter) -- the one value actually proven
    # against the real gateway sidecar's startup latency. A prior version
    # of this cell shrunk it to 90s, which main.py's own no-retry
    # /api/games check (confirmed by reading main.py directly) has no way
    # to recover from if the gateway is genuinely slower than that under
    # real competition load. Running it as a background subprocess (below)
    # keeps the latency win from overlapping with our own setup without
    # giving up the official's proven wait budget.
    gateway_proc = subprocess.Popen(
        ["curl", "--fail", "--retry", "999", "--retry-all-errors",
         "--retry-delay", "2", "--retry-max-time", "600",
         "http://gateway:8001/api/games"],
    )

    print(f"[{_el()}] === step: checking torch/numpy availability ===", flush=True)
    # This agent (unlike the pure GraphExplorerAgent) has a small torch
    # model (jepa/click_effect_model.py, a few thousand params) -- CPU-only
    # is plenty for it, so enable_gpu=false in kernel-metadata.json and no
    # CUDA setup is needed. Same "no fallback available if this triggers
    # for real, only a clearer error" reasoning as kaggle_submission's own
    # hypothesis-agent notebook -- internet is disabled during scored runs.
    try:
        import torch  # noqa: F401
        import numpy  # noqa: F401
        print(f"[{_el()}] torch {torch.__version__}, numpy {numpy.__version__} already available", flush=True)
    except ImportError as e:
        print(f"[{_el()}] torch/numpy not available ({e}) -- no internet during scored runs, "
              f"so there is no fallback; attempting pip install anyway in case this "
              f"is a manual test push with internet enabled", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "numpy"], check=True)
        import torch  # noqa: F401
        print(f"[{_el()}] installed torch {torch.__version__}", flush=True)

    print(f"[{_el()}] === step: copying competition harness repo (excluding .git) ===", flush=True)
    shutil.copytree(
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
        "/kaggle/working/ARC-AGI-3-Agents",
        ignore=shutil.ignore_patterns(".git"),
    )

    print(f"[{_el()}] === step: copying jepa package + checkpoint + agent files ===", flush=True)
    # A Kaggle dataset_sources attachment mounts at
    # /kaggle/input/datasets/<owner>/<slug>/, not /kaggle/input/<slug>/ --
    # see CLAUDE.md's "Kaggle competition submission" section for the full
    # story of how this cost a whole debugging session the first time.
    # graph_explorer_learned_agent.py resolves its own repo root the same
    # way graph_explorer_jepa_agent.py / hypothesis_agent.py do
    # (Path(__file__).resolve().parents[3]) -- jepa/ and
    # checkpoints_click_effect/ need to sit directly under /kaggle/working,
    # one level above ARC-AGI-3-Agents, matching this project's own local
    # layout.
    _dataset_root = "/kaggle/input/datasets/calamitychasm/graph-explorer-learned-agent"
    _templates_dir = "/kaggle/working/ARC-AGI-3-Agents/agents/templates"
    shutil.copytree(f"{_dataset_root}/jepa", "/kaggle/working/jepa")
    shutil.copytree(f"{_dataset_root}/checkpoints_click_effect", "/kaggle/working/checkpoints_click_effect")
    shutil.copy(f"{_dataset_root}/graph_explorer_core.py", f"{_templates_dir}/graph_explorer_core.py")
    shutil.copy(f"{_dataset_root}/graph_explorer_agent.py", f"{_templates_dir}/graph_explorer_agent.py")
    shutil.copy(f"{_dataset_root}/graph_explorer_learned_agent.py", f"{_templates_dir}/graph_explorer_learned_agent.py")

    print(f"[{_el()}] === step: verifying copied files exist ===", flush=True)
    for p in [
        "/kaggle/working/jepa/click_effect_model.py",
        "/kaggle/working/jepa/click_effect_adapter.py",
        "/kaggle/working/jepa/click_effect_features.py",
        "/kaggle/working/checkpoints_click_effect/click_effect_model.pt",
        f"{_templates_dir}/graph_explorer_core.py",
        f"{_templates_dir}/graph_explorer_agent.py",
        f"{_templates_dir}/graph_explorer_learned_agent.py",
    ]:
        assert os.path.exists(p), f"missing expected file: {p}"
    print(f"[{_el()}] all expected files present", flush=True)

    print(f"[{_el()}] === step: writing agents/__init__.py ===", flush=True)
    # Minimal __init__.py -- avoids eagerly importing every other template
    # (langgraph/smolagents/etc, unmet deps in this image) and registers
    # only what we need.
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write(
            "from typing import Type, cast\\n"
            "from dotenv import load_dotenv\\n"
            "from .agent import Agent, Playback\\n"
            "from .swarm import Swarm\\n"
            "from .templates.random_agent import Random\\n"
            "from .templates.graph_explorer_learned_agent import GraphExplorerLearnedAgent\\n"
            "\\n"
            "load_dotenv()\\n"
            "\\n"
            "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\\n"
            "    \\"random\\": Random,\\n"
            "    \\"graphexplorerlearnedagent\\": GraphExplorerLearnedAgent,\\n"
            "}\\n"
        )

    print(f"[{_el()}] === step: sanity-importing our agent before running main.py ===", flush=True)
    sys.path.insert(0, "/kaggle/working/ARC-AGI-3-Agents")
    sys.path.insert(0, "/kaggle/working")
    from agents.templates.graph_explorer_learned_agent import GraphExplorerLearnedAgent  # noqa: F401
    print(f"[{_el()}] agent import OK", flush=True)

    print(f"[{_el()}] === step: writing .env ===", flush=True)
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write(
            "SCHEME=http\\n"
            "HOST=gateway\\n"
            "PORT=8001\\n"
            "ARC_API_KEY=test-key-123\\n"
            "ARC_BASE_URL=http://gateway:8001/\\n"
            "OPERATION_MODE=online\\n"
            "ENVIRONMENTS_DIR=\\n"
            "RECORDINGS_DIR=/kaggle/working/server_recording\\n"
        )

    print(f"[{_el()}] === step: joining background gateway-wait ===", flush=True)
    gw_rc = gateway_proc.wait()
    print(f"[{_el()}] gateway curl exited with code {gw_rc}", flush=True)
    if gw_rc != 0:
        print(f"[{_el()}] WARNING: gateway did not respond within the retry "
              f"window -- proceeding to main.py anyway (it may still come up moments "
              f"later); if this run errors, this line is the first thing to check.",
              flush=True)

    print(f"[{_el()}] === step: running agent ===", flush=True)
    # GRAPH_EXPLORER_LEARNED_FORCE_CPU=1: runs this kernel under
    # enable_gpu:true (kernel-metadata.json) -- matching the ONE proven
    # torch+GPU-enabled infrastructure profile in this project
    # (hypothesis_agent.py) -- while forcing this agent's own model to
    # never touch the confirmed-broken P100 CUDA compute path (see
    # CLAUDE.md's own gotcha entry). torch+enable_gpu:false (this class's
    # original config) is the one combination never proven to work
    # anywhere in this project and failed 3x for unexplained reasons.
    result = subprocess.run(
        [sys.executable, "main.py", "--agent", "graphexplorerlearnedagent"],
        cwd="/kaggle/working/ARC-AGI-3-Agents",
        env={**os.environ, "MPLBACKEND": "agg", "GRAPH_EXPLORER_LEARNED_FORCE_CPU": "1"},
    )
    print(f"[{_el()}] === main.py exited with code {result.returncode} ===", flush=True)
'''

CELL3_SOURCE = '''\
# Non-rerun mode: produce a dummy submission
import pandas as pd

if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
'''

CELL1_SOURCE = "!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv"


def make_cell(cell_id, source):
    lines = source.split("\n")
    src_lines = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None,
        "outputs": [], "source": src_lines,
    }


notebook = {
    "cells": [
        make_cell("gel0001", CELL1_SOURCE),
        make_cell("gel0002", CELL2_SOURCE),
        make_cell("gel0003", CELL3_SOURCE),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "kaggle_submission_graph_explorer_learned" / "notebook" / "arc3-graph-explorer-learned-submission.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"wrote {out}")
