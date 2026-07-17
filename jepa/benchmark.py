"""Stage 1 benchmarking suite.

Evaluates a trained checkpoint against the held-out local Stage 0 corpus
(same split as `eval_stage1.py`) with a per-game breakdown -- the
milestone check in `eval_stage1.py` only reports a single pooled number,
which hides whether the predictor is actually learning some games' dynamics
while failing on others. Results are appended to
`logs/benchmarks/history.jsonl` (one JSON object per line) so runs stay
comparable across experiments (different epoch counts, data mixes, encoder
sizes, ...) instead of being scattered across terminal scrollback.

Also includes a CPU-vs-GPU training throughput benchmark, since "does the
GPU actually help" was an open question before `jepa/device.py` existed.

Usage:
    python -m jepa.benchmark eval --tag "baseline"
    python -m jepa.benchmark eval --checkpoint-dir checkpoints --tag "combined-data-60ep" --notes "..."
    python -m jepa.benchmark throughput
    python -m jepa.benchmark history
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .data.trajectories import TransitionDataset, load_all_transitions
from .device import get_device
from .grid import CANVAS, NUM_CHANNELS
from .losses import per_region_error
from .models import ActionConditionedPredictor, CNNEncoder, MoEPredictor
from .train_predictor import evaluate
from .train_moe_predictor import evaluate as evaluate_moe

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "logs" / "benchmarks" / "history.jsonl"
VAL_FRACTION = 0.1


def _load_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple:
    game_vocab = json.loads((checkpoint_dir / "game_vocab.json").read_text())
    online = CNNEncoder().to(device)
    online.load_state_dict(
        torch.load(checkpoint_dir / "encoder_finetuned.pt", map_location=device)
    )
    online.eval()
    predictor = ActionConditionedPredictor(num_games=len(game_vocab)).to(device)
    predictor.load_state_dict(
        torch.load(checkpoint_dir / "predictor.pt", map_location=device)
    )
    predictor.eval()
    return online, predictor, game_vocab


@torch.no_grad()
def _evaluate_per_game(online, predictor, val_ds, game_vocab: dict, device: torch.device) -> dict:
    """Breaks `train_predictor.evaluate`'s comparison down per game_id, so
    we can see which games the predictor is actually learning dynamics for
    vs which ones drag the pooled aggregate down."""
    idx_to_game = {v: k for k, v in game_vocab.items()}
    per_game = {
        g: {"pred": 0.0, "identity": 0.0, "n": 0, "pred_changed": 0.0, "identity_changed": 0.0, "n_changed": 0}
        for g in game_vocab
    }

    loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        cur_feat = online(cur)
        pred_feat = predictor(cur_feat, action_id, xy, game_idx)
        next_feat = online(nxt)

        pred_err = per_region_error(pred_feat, next_feat)  # (B, 8, 8)
        identity_err = per_region_error(cur_feat, next_feat)

        for b in range(cur.shape[0]):
            g = idx_to_game[int(game_idx[b])]
            stats = per_game[g]
            stats["pred"] += pred_err[b].mean().item()
            stats["identity"] += identity_err[b].mean().item()
            stats["n"] += 1
            m = patch_mask[b]
            if m.any():
                stats["pred_changed"] += pred_err[b][m].mean().item()
                stats["identity_changed"] += identity_err[b][m].mean().item()
                stats["n_changed"] += 1

    results = {}
    for g, s in per_game.items():
        if s["n"] == 0:
            continue
        n_changed = max(s["n_changed"], 1)
        results[g] = {
            "n": s["n"],
            "pred_mse": s["pred"] / s["n"],
            "identity_mse": s["identity"] / s["n"],
            "n_changed": s["n_changed"],
            "pred_changed_mse": s["pred_changed"] / n_changed,
            "identity_changed_mse": s["identity_changed"] / n_changed,
        }
    return results


def _load_moe_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple:
    """MoE counterpart to `_load_checkpoint` (stage6-selfplay-bootstrap):
    `jepa/benchmark.py eval` originally only knew how to load Stage 1's
    monolithic predictor (encoder_finetuned.pt/predictor.pt/game_vocab.json)
    -- there was no way to get an apples-to-apples changed-patches number
    for two different `checkpoints*/moe_predictor.pt` checkpoints on a
    shared held-out split (Stage 4's own changed-patches numbers all came
    from train_moe_predictor.py's own per-run eval, which uses whatever
    split *that* run's transitions list happened to produce -- not
    comparable across runs with different corpora). This loads
    encoder_moe.pt + moe_predictor.pt + game_vocab_moe.json instead."""
    game_vocab = json.loads((checkpoint_dir / "game_vocab_moe.json").read_text())
    num_experts = 8
    meta_path = checkpoint_dir / "moe_training_meta.json"
    if meta_path.exists():
        num_experts = json.loads(meta_path.read_text()).get("num_experts", 8)
    online = CNNEncoder().to(device)
    online.load_state_dict(
        torch.load(checkpoint_dir / "encoder_moe.pt", map_location=device)
    )
    online.eval()
    predictor = MoEPredictor(num_games=len(game_vocab), num_experts=num_experts).to(device)
    predictor.load_state_dict(
        torch.load(checkpoint_dir / "moe_predictor.pt", map_location=device)
    )
    predictor.eval()
    return online, predictor, game_vocab


@torch.no_grad()
def _evaluate_per_game_moe(online, predictor, val_ds, game_vocab: dict, device: torch.device) -> dict:
    """MoE counterpart to `_evaluate_per_game` -- `MoEPredictor.forward`
    returns (pred_feat, gate_weights), unlike the monolith's plain tensor,
    otherwise identical."""
    idx_to_game = {v: k for k, v in game_vocab.items()}
    per_game = {
        g: {"pred": 0.0, "identity": 0.0, "n": 0, "pred_changed": 0.0, "identity_changed": 0.0, "n_changed": 0}
        for g in game_vocab
    }

    loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        cur_feat = online(cur)
        pred_feat, _gate_weights = predictor(cur_feat, action_id, xy, game_idx)
        next_feat = online(nxt)

        pred_err = per_region_error(pred_feat, next_feat)  # (B, 8, 8)
        identity_err = per_region_error(cur_feat, next_feat)

        for b in range(cur.shape[0]):
            g = idx_to_game[int(game_idx[b])]
            stats = per_game[g]
            stats["pred"] += pred_err[b].mean().item()
            stats["identity"] += identity_err[b].mean().item()
            stats["n"] += 1
            m = patch_mask[b]
            if m.any():
                stats["pred_changed"] += pred_err[b][m].mean().item()
                stats["identity_changed"] += identity_err[b][m].mean().item()
                stats["n_changed"] += 1

    results = {}
    for g, s in per_game.items():
        if s["n"] == 0:
            continue
        n_changed = max(s["n_changed"], 1)
        results[g] = {
            "n": s["n"],
            "pred_mse": s["pred"] / s["n"],
            "identity_mse": s["identity"] / s["n"],
            "n_changed": s["n_changed"],
            "pred_changed_mse": s["pred_changed"] / n_changed,
            "identity_changed_mse": s["identity_changed"] / n_changed,
        }
    return results


def run_eval_moe(checkpoint_dir: Path, tag: str, notes: str) -> dict:
    """MoE counterpart to `run_eval`. Builds the val split from whatever
    local recordings are on disk *right now* (`load_all_transitions`,
    sorted-glob deterministic order) with the same VAL_FRACTION and the
    same fixed `manual_seed(0)` split `train_moe_predictor.py`'s own
    `_make_loaders` uses -- so as long as the recordings directory's file
    set doesn't change between two calls to this function, both get the
    exact same held-out examples, making a `checkpoints/` vs
    `checkpoints_selfplay/` comparison genuinely apples-to-apples. Caveat
    worth stating plainly: this only guarantees the *split indices* match
    across runs, not that neither checkpoint ever trained on any given
    held-out example -- a checkpoint trained before some of today's
    recording files existed obviously never saw them either way, but for
    files that predate *both* checkpoints, this val split is not
    guaranteed identical to whatever split that checkpoint's own training
    run happened to draw (different random_split call, same seed, only
    identical if the input list order was also identical at both times)."""
    device = get_device()
    transitions = load_all_transitions(REPO_ROOT)
    online, predictor, game_vocab = _load_moe_checkpoint(checkpoint_dir, device)

    dataset = TransitionDataset(transitions, game_vocab)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    _, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    overall = evaluate_moe(online, predictor, val_loader, device=device)
    per_game = _evaluate_per_game_moe(online, predictor, val_ds, game_vocab, device)

    training_meta = {}
    meta_path = checkpoint_dir / "moe_training_meta.json"
    if meta_path.exists():
        training_meta = json.loads(meta_path.read_text())

    changed_improvement = (
        (overall["identity_changed"] - overall["pred_changed"]) / overall["identity_changed"] * 100
    )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "notes": notes,
        "checkpoint_dir": str(checkpoint_dir),
        "model_type": "moe",
        "device": str(device),
        "n_val_transitions": len(val_ds),
        "n_total_transitions": len(dataset),
        "overall": overall,
        "changed_patches_improvement_pct": changed_improvement,
        "milestone_pass": overall["pred_changed"] < overall["identity_changed"],
        "per_game": per_game,
        "training_meta": training_meta,
    }

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    _print_eval_summary(result)
    return result


def run_eval(checkpoint_dir: Path, tag: str, notes: str) -> dict:
    device = get_device()
    transitions = load_all_transitions(REPO_ROOT)
    game_vocab = json.loads((checkpoint_dir / "game_vocab.json").read_text())
    dataset = TransitionDataset(transitions, game_vocab)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    _, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    online, predictor, _ = _load_checkpoint(checkpoint_dir, device)

    overall = evaluate(online, predictor, val_loader, device=device)
    per_game = _evaluate_per_game(online, predictor, val_ds, game_vocab, device)

    training_meta = {}
    meta_path = checkpoint_dir / "training_meta.json"
    if meta_path.exists():
        training_meta = json.loads(meta_path.read_text())

    changed_improvement = (
        (overall["identity_changed"] - overall["pred_changed"]) / overall["identity_changed"] * 100
    )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "notes": notes,
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "n_val_transitions": len(val_ds),
        "overall": overall,
        "changed_patches_improvement_pct": changed_improvement,
        "milestone_pass": overall["pred_changed"] < overall["identity_changed"],
        "per_game": per_game,
        "training_meta": training_meta,
    }

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    _print_eval_summary(result)
    return result


def _print_eval_summary(result: dict) -> None:
    o = result["overall"]
    print(f"=== benchmark: {result['tag']} ({result['timestamp']}) ===")
    print(f"device: {result['device']}  val transitions: {result['n_val_transitions']}")
    if result["training_meta"]:
        print(f"training config: {result['training_meta']}")
    print(f"overall improvement: {(o['identity'] - o['pred']) / o['identity'] * 100:+.1f}%")
    print(
        f"changed-patches improvement: {result['changed_patches_improvement_pct']:+.1f}%  "
        f"({'PASS' if result['milestone_pass'] else 'FAIL'})"
    )
    print()
    print(f"{'game':<18} {'n':>5} {'pred_chg':>10} {'ident_chg':>10} {'improve%':>9}")
    rows = sorted(
        result["per_game"].items(),
        key=lambda kv: (kv[1]["identity_changed_mse"] - kv[1]["pred_changed_mse"])
        / max(kv[1]["identity_changed_mse"], 1e-9),
    )
    for game, s in rows:
        imp = (
            (s["identity_changed_mse"] - s["pred_changed_mse"])
            / max(s["identity_changed_mse"], 1e-9)
            * 100
        )
        print(
            f"{game:<18} {s['n']:>5} {s['pred_changed_mse']:>10.5f} "
            f"{s['identity_changed_mse']:>10.5f} {imp:>8.1f}%"
        )


def run_throughput(batch_size: int = 32, n_batches: int = 50) -> dict:
    """Times encoder+predictor forward+backward passes on CPU and (if
    available) GPU with random dummy data -- quantifies the GPU speedup
    that CLAUDE.md's Next-steps section flagged as unmeasured before
    `jepa/device.py` existed."""
    results = {}
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        online = CNNEncoder().to(device)
        predictor = ActionConditionedPredictor(num_games=25).to(device)
        opt = torch.optim.AdamW(
            list(online.parameters()) + list(predictor.parameters()), lr=3e-4
        )

        cur = torch.randn(batch_size, NUM_CHANNELS, CANVAS, CANVAS, device=device)
        nxt = torch.randn(batch_size, NUM_CHANNELS, CANVAS, CANVAS, device=device)
        action_id = torch.randint(0, 8, (batch_size,), device=device)
        xy = torch.rand(batch_size, 2, device=device)
        game_idx = torch.randint(0, 25, (batch_size,), device=device)

        def _step():
            cur_feat = online(cur)
            pred_feat = predictor(cur_feat, action_id, xy, game_idx)
            loss = (pred_feat - online(nxt)).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        for _ in range(5):
            _step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_batches):
            _step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        results[device.type] = {
            "batches": n_batches,
            "batch_size": batch_size,
            "seconds": elapsed,
            "batches_per_sec": n_batches / elapsed,
            "transitions_per_sec": n_batches * batch_size / elapsed,
        }
        print(
            f"{device.type:>4}: {n_batches} batches x {batch_size} in {elapsed:.2f}s "
            f"({results[device.type]['transitions_per_sec']:.0f} transitions/sec)"
        )

    if "cpu" in results and "cuda" in results:
        speedup = results["cuda"]["transitions_per_sec"] / results["cpu"]["transitions_per_sec"]
        print(f"GPU speedup: {speedup:.1f}x")
        results["gpu_speedup"] = speedup

    return results


def print_history(limit: int = 20) -> None:
    if not HISTORY_PATH.exists():
        print(f"no benchmark history yet at {HISTORY_PATH}")
        return
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines[-limit:]]
    print(f"{'timestamp':<26} {'tag':<24} {'device':<6} {'changed_imp%':>12} {'pass':>5}")
    for r in rows:
        print(
            f"{r['timestamp']:<26} {r['tag']:<24} {r['device']:<6} "
            f"{r['changed_patches_improvement_pct']:>11.1f}% {'PASS' if r['milestone_pass'] else 'FAIL':>5}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 benchmarking suite.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint, per-game + overall, append to history.")
    p_eval.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    p_eval.add_argument("--tag", type=str, default="untagged")
    p_eval.add_argument("--notes", type=str, default="")
    p_eval.add_argument(
        "--moe",
        action="store_true",
        help="Evaluate a Stage 4 MoE checkpoint (encoder_moe.pt/moe_predictor.pt/game_vocab_moe.json) "
        "instead of the Stage 1 monolith (stage6-selfplay-bootstrap addition).",
    )

    p_tp = sub.add_parser("throughput", help="Benchmark CPU vs GPU training throughput.")
    p_tp.add_argument("--batch-size", type=int, default=32)
    p_tp.add_argument("--n-batches", type=int, default=50)

    p_hist = sub.add_parser("history", help="Print recent benchmark history.")
    p_hist.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    if args.command == "eval":
        if args.moe:
            run_eval_moe(args.checkpoint_dir, args.tag, args.notes)
        else:
            run_eval(args.checkpoint_dir, args.tag, args.notes)
    elif args.command == "throughput":
        run_throughput(args.batch_size, args.n_batches)
    elif args.command == "history":
        print_history(args.limit)
