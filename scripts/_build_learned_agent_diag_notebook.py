"""One-off script to generate kaggle_submission_graph_explorer_learned/
notebook_diag/diag.ipynb -- an UNCONDITIONAL diagnostic (no
KAGGLE_IS_COMPETITION_RERUN gate) that exercises the real setup path
(dataset mount, file copies, torch model load, adapter construction, a
full choose_action() call) on Kaggle's actual environment, before
spending a real scored submission. Mirrors
kaggle_submission_graph_explorer/notebook_diag/diag.ipynb's own pattern
and rationale (a free test push only exercises the non-rerun dummy-
submission branch, never the real gated setup code)."""

import json
from pathlib import Path

CELL2_SOURCE = '''\
import os
import shutil
import sys
import traceback

print("=== UNCONDITIONAL DIAGNOSTIC (mirrors the real submission notebook's exact setup order) ===", flush=True)

import numpy
print(f"python {sys.version}", flush=True)
print(f"numpy {numpy.__version__}", flush=True)

print("\\n--- step: walk /kaggle/input to confirm mount layout ---", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
    if depth <= 3:
        print(root, flush=True)

print("\\n--- step: check torch availability ---", flush=True)
try:
    import torch
    print(f"torch {torch.__version__} available", flush=True)
except ImportError as e:
    print(f"TORCH IMPORT FAILED: {e}", flush=True)

print("\\n--- step: copy competition harness repo ---", flush=True)
shutil.copytree(
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
    "/kaggle/working/ARC-AGI-3-Agents",
    ignore=shutil.ignore_patterns(".git"),
)
print("harness copy OK", flush=True)

print("\\n--- step: copy jepa package + checkpoint + agent files from dataset ---", flush=True)
_dataset_root = "/kaggle/input/datasets/calamitychasm/graph-explorer-learned-agent"
_templates_dir = "/kaggle/working/ARC-AGI-3-Agents/agents/templates"
shutil.copytree(f"{_dataset_root}/jepa", "/kaggle/working/jepa")
shutil.copytree(f"{_dataset_root}/checkpoints_click_effect", "/kaggle/working/checkpoints_click_effect")
shutil.copy(f"{_dataset_root}/graph_explorer_core.py", f"{_templates_dir}/graph_explorer_core.py")
shutil.copy(f"{_dataset_root}/graph_explorer_agent.py", f"{_templates_dir}/graph_explorer_agent.py")
shutil.copy(f"{_dataset_root}/graph_explorer_learned_agent.py", f"{_templates_dir}/graph_explorer_learned_agent.py")
for p in [
    "/kaggle/working/jepa/click_effect_model.py",
    "/kaggle/working/jepa/click_effect_adapter.py",
    "/kaggle/working/jepa/click_effect_features.py",
    "/kaggle/working/jepa/device.py",
    "/kaggle/working/checkpoints_click_effect/click_effect_model.pt",
    f"{_templates_dir}/graph_explorer_core.py",
    f"{_templates_dir}/graph_explorer_agent.py",
    f"{_templates_dir}/graph_explorer_learned_agent.py",
]:
    assert os.path.exists(p), f"missing: {p}"
print("file copy OK", flush=True)

print("\\n--- step: write minimal agents/__init__.py (EXACTLY matching the real submission notebook) ---", flush=True)
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
print("wrote minimal __init__.py", flush=True)

print("\\n--- step: sanity-import the agent ---", flush=True)
sys.path.insert(0, "/kaggle/working/ARC-AGI-3-Agents")
sys.path.insert(0, "/kaggle/working")
from agents.templates.graph_explorer_learned_agent import GraphExplorerLearnedAgent
from agents.templates.graph_explorer_agent import FrameProcessor
import numpy as np
print("import OK", flush=True)

print("\\n--- step: directly test ClickEffectModel load + ClickEffectAdapter construction ---", flush=True)
try:
    from jepa.click_effect_model import ClickEffectModel
    from jepa.click_effect_adapter import ClickEffectAdapter
    from jepa.device import get_device

    # Mirror the real agent's own device-selection exactly (see
    # graph_explorer_learned_agent.py: _init_model) -- this diagnostic now
    # runs under enable_gpu:true (kernel-metadata.json) to match the one
    # proven torch+GPU infrastructure profile, while FORCE_CPU replicates
    # what the real submission notebook sets to avoid the confirmed-broken
    # P100 CUDA compute path.
    os.environ["GRAPH_EXPLORER_LEARNED_FORCE_CPU"] = "1"
    if os.getenv("GRAPH_EXPLORER_LEARNED_FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = get_device()
    print(f"device: {device}", flush=True)
    model = ClickEffectModel().to(device)
    model.load_state_dict(torch.load("/kaggle/working/checkpoints_click_effect/click_effect_model.pt", map_location=device))
    model.eval()
    print("ClickEffectModel loaded OK", flush=True)
    adapter = ClickEffectAdapter(model, device)
    print("ClickEffectAdapter constructed OK", flush=True)

    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[20:25, 20:25] = 9
    prob = adapter.predict(frame, 22, 22, np.array([9/15.0, 25/4096.0, 1.0, 5/64.0, 5/64.0], dtype=np.float32))
    print(f"adapter.predict OK: prob={prob}", flush=True)

    print("=== MODEL DIAGNOSTICS PASSED ===", flush=True)
except Exception:
    print("MODEL DIAGNOSTIC FAILED:", flush=True)
    traceback.print_exc()

print("\\n--- step: full end-to-end choose_action simulation using a fake Agent construction ---", flush=True)
try:
    from arcengine import FrameData, GameState, GameAction
    from agents.templates.graph_explorer_core import GraphExplorer
    from jepa.memory import TransitionGraph
    import time

    agent = GraphExplorerLearnedAgent.__new__(GraphExplorerLearnedAgent)
    agent.game_id = "diag-game"
    agent.frame_processor = FrameProcessor()
    agent.status_bar_mask = None
    agent.hashed_frame2action_results = {}
    agent.hashed_frame2transitions = {}
    agent.last_hashed_frame = None
    agent.last_action = None
    agent.arrow_control = True
    agent.favor_new_actions = False
    agent.favor_frontier_search = True
    agent.graph_explorer = GraphExplorer(verbose_level=0, n_groups=agent.N_GROUPS)
    agent.level_first_frame = None
    agent.failed = False
    agent.level_up = True
    agent.last_action_object = GameAction.RESET
    agent.time_start = time.time()
    agent.last_time = time.time()
    agent.last_transition_suspicious = False
    agent._prev_levels_completed = 0
    agent._last_levels_completed_delta = 0
    agent._consecutive_fallbacks = 0
    agent.transition_memory = TransitionGraph()
    agent._last_action_id = None
    agent._last_xy = None

    agent._model_ready = False
    agent._pending_click_obs = None
    agent._init_model()
    agent.graph_explorer.tie_break_fn = agent._learned_tie_break
    agent._model_ready = True
    print("agent model init OK", flush=True)

    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[0:3, :] = 5
    frame[20:25, 20:25] = 9
    frame[40:44, 10:20] = 12

    fake_frame = FrameData(
        game_id="diag-game",
        frame=[frame.tolist()],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 6],
    )
    action = agent.choose_action([fake_frame], fake_frame)
    print(f"choose_action (1st call, explore-mode tie_break exercised) OK: returned {action}", flush=True)

    # Second call, same frame -- exercises _record_pending_observation
    # (the adapter.observe path) against the pending click from the first call.
    action2 = agent.choose_action([fake_frame, fake_frame], fake_frame)
    print(f"choose_action (2nd call, adapter.observe exercised) OK: returned {action2}", flush=True)

    print("=== END-TO-END DIAGNOSTIC PASSED ===", flush=True)
except Exception:
    print("END-TO-END DIAGNOSTIC FAILED:", flush=True)
    traceback.print_exc()

print("\\n=== DIAGNOSTIC COMPLETE ===", flush=True)
'''

CELL1_SOURCE = "!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv"

CELL3_SOURCE = '''\
import pandas as pd
submission = pd.DataFrame(
    data=[['1_0', '1', True, 1]],
    columns=['row_id', 'game_id', 'end_of_game', 'score'])
submission.to_parquet('/kaggle/working/submission.parquet', index=False)
'''


def make_cell(cell_id, source):
    lines = source.split("\n")
    src_lines = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None,
        "outputs": [], "source": src_lines,
    }


notebook = {
    "cells": [
        make_cell("geld0001", CELL1_SOURCE),
        make_cell("geld0002", CELL2_SOURCE),
        make_cell("geld0003", CELL3_SOURCE),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "kaggle_submission_graph_explorer_learned" / "notebook_diag" / "diag.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"wrote {out}")
