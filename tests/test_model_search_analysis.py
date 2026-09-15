"""Tests for the stage7-model-search analysis tooling.

Two of these guard specific ways this project has previously produced a
confident, plausible-looking, entirely wrong table:

* ``test_kernel_log_takes_the_last_summary_block`` -- the harness emits a
  *progressive* summary roughly every 10 games' worth of progress. Reading the
  first block instead of the last inverted the NVFP4 conclusion completely
  ("4x worse" when it was 3.2x better); ``stage7_duck_nvfp4.md`` records that
  near-miss. A parser regression here would silently reintroduce it.
* ``test_preemptions_absent_is_unknown_not_zero`` -- preemption counts live only
  in the Prometheus scrape, never in the server log text. An earlier draft
  counted regex hits in the log and reported the NVFP4 run as having **0**
  preemptions when its own metrics file says **191**.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ms_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tradeoff = _load("analyze_model_tradeoff")
maxstat = _load("analyze_leaderboard_maxstat")
builder = _load("_build_duck_nvfp4_anim")


# --------------------------------------------------------------------------
# analyze_model_tradeoff
# --------------------------------------------------------------------------

SERVER_LOG = """\
(EngineCore pid=629) INFO gpu_model_runner.py Model loading took 28.95 GiB memory and 99.7 seconds
(EngineCore pid=629) INFO gpu_worker.py Available KV cache memory: 44.57 GiB
(EngineCore pid=629) INFO kv_cache_utils.py GPU KV cache size: 342,144 tokens
(EngineCore pid=629) INFO kv_cache_utils.py Maximum concurrency for 65,536 tokens per request: 18.06x
Engine 000: Running: 25 reqs, Waiting: 0 reqs, GPU KV cache usage: 63.7%, Prefix cache hit rate: 0.0%
Engine 000: Running: 24 reqs, Waiting: 2 reqs, GPU KV cache usage: 61.0%, Prefix cache hit rate: 0.0%
Engine 000: Running: 25 reqs, Waiting: 0 reqs, GPU KV cache usage: 62.0%, Prefix cache hit rate: 0.0%
"""

# The NVFP4 bundle takes the other allocator path: it pins a byte count and
# skips profiling, so it never prints "Available KV cache memory".
SERVER_LOG_PINNED = """\
(Worker pid=242) INFO model_runner.py Model loading took 81.8 GiB memory and 180.1 seconds
(EngineCore pid=224) INFO gpu_worker.py reserved 5.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config
(EngineCore pid=224) INFO kv_cache_utils.py GPU KV cache size: 105,202 tokens, Maximum concurrency for 32,768 tokens per request: 3.21x
Engine 000: Running: 3 reqs, Waiting: 22 reqs, GPU KV cache usage: 79.3%, Prefix cache hit rate: 0.0%
"""


def test_server_log_parses_profiled_allocator_path(tmp_path):
    log = tmp_path / "vllm-openai-server.log"
    log.write_text(SERVER_LOG, encoding="utf-8")
    out = tradeoff.parse_server_log(log)
    assert out["weights_gib"] == pytest.approx(28.95)
    assert out["kv_gib"] == pytest.approx(44.57)
    assert out["kv_tokens"] == 342_144
    assert out["max_model_len"] == 65_536
    assert out["running_p50"] == 25
    assert out["waiting_p50"] == 0
    assert out["snapshots"] == 3
    assert out["snapshots_queue_empty"] == 2


def test_server_log_parses_pinned_allocator_path(tmp_path):
    """The `reserved N GiB ... kv_cache_memory_bytes` spelling must also be read.

    The two stacks take different paths into the allocator; a parser that only
    knows one of them silently drops the KV size for the other, which is exactly
    the column the whole comparison turns on.
    """
    log = tmp_path / "vllm-openai-server.log"
    log.write_text(SERVER_LOG_PINNED, encoding="utf-8")
    out = tradeoff.parse_server_log(log)
    assert out["weights_gib"] == pytest.approx(81.8)
    assert out["kv_gib"] == pytest.approx(5.0)
    assert out["kv_tokens"] == 105_202


def test_kernel_log_takes_the_last_summary_block(tmp_path):
    """Progressive summaries: the LAST block is the result, never the first."""
    log = tmp_path / "run.log"
    log.write_text(
        "mean score:    0.80\ntotal actions: 236\ntotal tokens:  120000\n"
        "mean score:    6.71\ntotal actions: 2800\ntotal tokens:  1400000\n"
        "mean score:    10.69\ntotal actions: 3633\ntotal tokens:  1870896\n"
        "duration:  2h 12m 1s\n",
        encoding="utf-8",
    )
    out = tradeoff.parse_kernel_log(log)
    assert out["score"] == pytest.approx(10.69)
    assert out["actions"] == 3633
    assert out["gen_tokens"] == 1_870_896
    assert out["wallclock_s"] == 2 * 3600 + 12 * 60 + 1


def test_preemptions_absent_is_unknown_not_zero(tmp_path):
    assert tradeoff.parse_metrics(tmp_path / "nope.prom") == {}
    prom = tmp_path / "vllm-metrics-final.prom"
    prom.write_text(
        '# HELP vllm:num_preemptions_total Cumulative preemptions\n'
        'vllm:num_preemptions_total{engine="0",model_name="x"} 191.0\n',
        encoding="utf-8",
    )
    assert tradeoff.parse_metrics(prom) == {"preemptions": 191}


def test_turn_identity_closes_on_the_real_nvfp4_numbers(tmp_path):
    """End-to-end: the identity from stage7_turn_latency.md s5 must close.

    Uses the real baseline figures. If this drifts, either the parser broke or
    the identity's inputs were mixed up (e.g. prompt tokens counted as
    generated), both of which have bitten this project.
    """
    (tmp_path / "vllm-openai-server.log").write_text(SERVER_LOG_PINNED, encoding="utf-8")
    (tmp_path / "run.log").write_text(
        "mean score:    10.69\ntotal actions: 3633\ntotal tokens:  1870896\n"
        "duration:  2h 12m 1s\n",
        encoding="utf-8",
    )
    row = tradeoff.analyze("nvfp4", tmp_path)
    assert row["gen_tokens_per_action"] == pytest.approx(515.0, abs=0.5)
    assert row["actions_per_game"] == pytest.approx(145.3, abs=0.1)
    assert abs(row["turns_identity_residual_pct"]) < 0.1


# --------------------------------------------------------------------------
# analyze_leaderboard_maxstat
# --------------------------------------------------------------------------

def test_spearman_handles_ties_and_perfect_orders():
    assert maxstat.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert maxstat.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Ties must use average ranks, not input order.
    assert maxstat.spearman([1, 1, 2, 2], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_spearman_is_zero_for_a_constant_column():
    assert maxstat.spearman([1, 2, 3], [7, 7, 7]) == 0.0


# --------------------------------------------------------------------------
# _build_duck_nvfp4_anim
# --------------------------------------------------------------------------

def _fake_source(tmp_path: Path, code_bodies: list[str]) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# theirs\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## upstream notes\n"]},
    ]
    for body in code_bodies:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": [body]})
    (src / "thui.ipynb").write_text(
        json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4}),
        encoding="utf-8")
    (src / "kernel-metadata.json").write_text(json.dumps({
        "dataset_sources": ["a/b", "c/d"], "model_sources": ["e/f/PyTorch/g/1"],
        "docker_image": "gcr.io/x@sha256:deadbeef", "machine_shape": "NvidiaRtxPro6000",
    }), encoding="utf-8")
    return src


def test_builder_refuses_an_unexpected_source_md5(tmp_path, monkeypatch, capsys):
    src = _fake_source(tmp_path, ["print('a')\n"])
    monkeypatch.setattr(sys, "argv",
                        ["b", str(src), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc:
        builder.main()
    assert "md5" in str(exc.value)


def test_builder_carries_code_cells_byte_identically(tmp_path, monkeypatch):
    bodies = ["print('one')\n", "x = 2\nprint(x)\n", "import os\n"]
    src = _fake_source(tmp_path, bodies)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv",
                        ["b", str(src), "--out", str(out), "--allow-md5-drift"])
    assert builder.main() == 0

    nb = json.loads(next(out.glob("*.ipynb")).read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # probe cell first, then every source code cell unchanged
    assert len(code) == len(bodies) + 1
    assert "HW_PROBE" in "".join(code[0]["source"])
    assert [("".join(c["source"])) for c in code[1:]] == bodies
    # the two upstream markdown cells are replaced by exactly one of ours
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert len(md) == 1 and "reproduction of other people" in "".join(md[0]["source"])

    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == builder.KERNEL_ID
    assert meta["dataset_sources"] == ["a/b", "c/d"]
    assert meta["model_sources"] == ["e/f/PyTorch/g/1"]
    assert meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert meta["enable_internet"] is False


def test_builder_refuses_when_the_two_leading_cells_are_not_markdown(tmp_path, monkeypatch):
    """Guards against an upstream reshuffle silently deleting a code cell."""
    src = _fake_source(tmp_path, ["print('a')\n"])
    path = next(src.glob("*.ipynb"))
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["cells"][1] = {"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": ["load_bearing = True\n"]}
    path.write_text(json.dumps(nb), encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["b", str(src), "--out", str(tmp_path / "out"), "--allow-md5-drift"])
    with pytest.raises(SystemExit) as exc:
        builder.main()
    assert "markdown" in str(exc.value)
