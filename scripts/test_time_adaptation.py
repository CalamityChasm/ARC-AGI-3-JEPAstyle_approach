"""Stage 6 test-time adaptation experiment: does letting the MoE predictor
take a few real gradient steps on a held-out game's own observed
transitions -- DURING simulated play, not at training time -- close any of
the zero-shot generalization gap documented in experiments/
stage6_game_holdout.md and the multi-fold CV in CLAUDE.md's Stage 6
addendum?

Every mechanism tried earlier this session (categorical game-id embedding,
three different *continuous* embedding/context mechanisms including one
that pools multiple in-episode transitions) was still a frozen forward
pass at eval time -- the model infers a representation of "which game is
this" but never updates its weights on this game's own data. This script
tests a mechanistically different idea: stream a held-out game's recorded
transitions in order (simulating "playing" it), and after every K
observed transitions, take a handful of real AdamW steps on a small,
deliberately-restricted subset of the MoE predictor's parameters, fit
only to the transitions observed so far from THIS game.

Adapted parameters (see ADAPT_PARAM_FILTER below and the module docstring
in "Design notes" for the reasoning): each expert's LAST Conv2d layer
(the one directly producing the residual) plus the gate's LAST Linear
layer. Frozen: the encoder (both online forward passes use it purely for
feature extraction, no gradient ever flows into it), the action/xy/game
embeddings, and every expert's FIRST Conv2d layer. This is deliberately a
last-layer/adapter-style update (per the task's own suggestion that this
is safer than full fine-tuning on a handful of examples), not a full
fine-tune -- ~33.8K trainable parameters out of the predictor's much
larger total, low risk of catastrophic overfitting to a few dozen-to-few-
hundred examples at a tiny learning rate.

Two things are measured for the primary game (r11l, matching this
session's fold-1 held-out set):
  1. changed-patches improvement over identity on a FIXED, never-adapted-
     on eval split of r11l's own transitions, at cumulative adaptation-
     stream sizes of 0/10/50/200 (the task's own checkpoints), for two
     update-cadence conditions (K=10, K=50).
  2. Catastrophic interference: the SAME final adapted model (after
     streaming the whole adaptation buffer) re-evaluated on a sample of
     TRAINED games' transitions, compared against the pre-adaptation
     (0-step) numbers on those same games.

Usage:
    python scripts/test_time_adaptation.py
    python scripts/test_time_adaptation.py --games r11l bp35 m0r0 tr87 ka59
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset, _load_frame_lines
from jepa.device import get_device
from jepa.losses import per_region_error, weighted_prediction_loss
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints_holdout_baseline"
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"

# Same fold-1 held-out set as experiments/stage6_game_holdout.md.
HELDOUT_GAMES_DEFAULT = ["r11l"]
ALL_HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]
# Broader trained-game control set for the catastrophic-interference
# check than eval_game_holdout.py's original 3 -- CLAUDE.md's Stage 1
# item 5 flags ft09/s5i5/vc33 specifically as having tiny identity-
# baseline MSE (~1e-5), which makes the improvement PERCENTAGE swing
# wildly from tiny absolute changes. Keeping those three (for continuity
# with eval_game_holdout.py) but adding 5 more games spanning a wider
# range of typical MSE scale so the interference read isn't dominated by
# that known-noisy trio.
TRAINED_PROBE_GAMES = ["ft09", "s5i5", "vc33", "ar25", "cd82", "cn04", "lp85", "sp80"]

EVAL_CHECKPOINTS = [0, 10, 50, 200]
K_CONDITIONS = [10, 50]
N_STEPS = 3  # gradient steps per update event, middle of the task's 1-5 range
LR = 5e-5  # middle of the task's suggested 1e-5..1e-4 range, well below the 3e-4 training LR
# Both overridable via --n-steps/--lr for a quick robustness check outside
# the default (3 steps, 5e-5) point -- see main().
ADAPT_BATCH_SIZE = 16


# --- Data loading (mirrors jepa/data/trajectories.py's own logic, but
# keeps per-file boundaries so we can split "stream" vs. fixed "eval-only"
# data by whole episode, and preserves within-file order to genuinely
# simulate streaming observations during play). ---------------------------


def load_game_transitions_per_file(game: str) -> list:
    files = sorted(RECORDINGS_DIR.glob(f"{game}-*.recording.jsonl"))
    files = [f for f in files if ".random.80." in f.name]
    per_file = []
    for path in files:
        frames = _load_frame_lines(path)
        transitions = []
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x = xy.get("x", 0)
            y = xy.get("y", 0)
            changed = cur["frame"] != nxt["frame"]
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], changed, game_id))
        per_file.append(transitions)
    return per_file


def load_sample_transitions(games: list, max_per_game: int = 200) -> list:
    """For the catastrophic-interference probe: a bounded sample of
    transitions from TRAINED games (not the held-out ones)."""
    out = []
    for g in games:
        per_file = load_game_transitions_per_file(g)
        flat = [t for f in per_file for t in f]
        out.extend(flat[:max_per_game])
    return out


# --- Model setup -----------------------------------------------------------


def load_baseline_checkpoint(device):
    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab_moe.json").read_text())
    meta = json.loads((CHECKPOINT_DIR / "moe_training_meta.json").read_text())
    num_experts = meta.get("num_experts", 8)
    feature_channels = 64

    online = CNNEncoder(out_channels=feature_channels).to(device)
    online.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_moe.pt", map_location=device))
    online.eval()
    for p in online.parameters():
        p.requires_grad = False

    predictor_state = torch.load(CHECKPOINT_DIR / "moe_predictor.pt", map_location=device)
    return online, predictor_state, game_vocab, num_experts, feature_channels


def build_predictor(state_dict, game_vocab, num_experts, feature_channels, device):
    predictor = MoEPredictor(
        num_games=len(game_vocab), num_experts=num_experts,
        feature_channels=feature_channels, expert_hidden=feature_channels,
    ).to(device)
    predictor.load_state_dict(copy.deepcopy(state_dict))
    predictor.eval()
    return predictor


def set_adapter_trainable(predictor: MoEPredictor) -> list:
    """Freeze everything except each expert's LAST Conv2d layer and the
    gate's LAST Linear layer. Returns the list of now-trainable params.

    Design reasoning (see module docstring): a per-game embedding-only
    adapter would almost certainly reproduce this session's earlier
    negative results (three different continuous embedding mechanisms,
    fully trained for 60 epochs each, already failed to help) for a
    different, less interesting reason (too few gradient steps on a tiny
    parameter to matter) -- it wouldn't be a fair test of whether SGD on
    real observed data can extract signal a frozen forward pass can't.
    Full fine-tuning of the whole predictor risks catastrophic
    overfitting/forgetting on a few dozen-to-few-hundred examples, and
    directly convolves "does adaptation help" with "did we just break the
    model." Last-layer/expert-output adaptation is the standard ANIL-style
    middle ground: expressive enough to plausibly matter, small enough
    (~33.8K params here) to be a low-risk, fast fit at a handful of steps.
    """
    for p in predictor.parameters():
        p.requires_grad = False
    trainable = []
    for expert in predictor.experts:
        last_conv = expert[-1]
        for p in last_conv.parameters():
            p.requires_grad = True
            trainable.append(p)
    gate_last = predictor.gate[-1]
    for p in gate_last.parameters():
        p.requires_grad = True
        trainable.append(p)
    return trainable


# --- Eval / adaptation step -------------------------------------------------


@torch.no_grad()
def changed_patches_eval(online, predictor, transitions: list, game_vocab: dict, device, per_game: bool = False) -> dict:
    """changed-patches pred-vs-identity MSE, pooled over `transitions`
    (and, if per_game=True, broken down per short game code -- needed for
    the interference check so one noisy tiny-denominator game can't hide
    behind a pooled average, per CLAUDE.md's own Stage 1 item 5 warning).
    Unknown game_ids fall back to vocab index 0, mirroring production's
    real behavior on a novel Kaggle game."""
    if not transitions:
        return {"n_changed": 0}
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    predictor.eval()

    total = {"pred": 0.0, "identity": 0.0, "n": 0}
    per_game_totals = defaultdict(lambda: {"pred": 0.0, "identity": 0.0, "n": 0})
    idx = 0
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        b = cur.shape[0]
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask = nxt.to(device), patch_mask.to(device)
        game_idx = game_idx.to(device)
        cur_feat = online(cur)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
        next_feat = online(nxt)
        pred_err = per_region_error(pred_feat, next_feat)
        identity_err = per_region_error(cur_feat, next_feat)
        for i in range(b):
            m = patch_mask[i]
            if m.any():
                p = pred_err[i][m].mean().item()
                idn = identity_err[i][m].mean().item()
                total["pred"] += p
                total["identity"] += idn
                total["n"] += 1
                if per_game:
                    g = transitions[idx + i][6].split("-")[0]
                    per_game_totals[g]["pred"] += p
                    per_game_totals[g]["identity"] += idn
                    per_game_totals[g]["n"] += 1
        idx += b

    if total["n"] == 0:
        return {"n_changed": 0}
    result = {
        "n_changed": total["n"],
        "pred_changed_mse": total["pred"] / total["n"],
        "identity_changed_mse": total["identity"] / total["n"],
        "improvement_pct": (total["identity"] - total["pred"]) / total["identity"] * 100,
    }
    if per_game:
        result["per_game"] = {}
        for g, s in per_game_totals.items():
            if s["n"] == 0:
                continue
            result["per_game"][g] = {
                "n_changed": s["n"],
                "pred_changed_mse": s["pred"] / s["n"],
                "identity_changed_mse": s["identity"] / s["n"],
                "improvement_pct": (s["identity"] - s["pred"]) / s["identity"] * 100,
            }
    return result


def adaptation_step(online, predictor, opt, buffer: list, game_vocab: dict, device, n_steps: int):
    """n_steps AdamW updates on random mini-batches drawn (with
    replacement across steps) from `buffer` -- everything observed so far
    from this one held-out game."""
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(buffer, fallback_vocab)
    bs = min(ADAPT_BATCH_SIZE, len(buffer))
    predictor.train()
    for _ in range(n_steps):
        loader = DataLoader(ds, batch_size=bs, shuffle=True)
        cur, action_id, xy, nxt, patch_mask, game_idx = next(iter(loader))
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask = nxt.to(device), patch_mask.to(device)
        game_idx = game_idx.to(device)

        with torch.no_grad():
            cur_feat = online(cur)
            next_feat = online(nxt)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
        loss = weighted_prediction_loss(pred_feat, next_feat, patch_mask)

        opt.zero_grad()
        loss.backward()
        opt.step()
    predictor.eval()


def run_adaptation_trajectory(
    online, predictor_state, game_vocab, num_experts, feature_channels,
    stream: list, eval_set: list, k: int, device,
) -> dict:
    """Fresh predictor from the pristine checkpoint; stream `stream`'s
    transitions one at a time, firing an adaptation update every k
    observations; record changed-patches on the FIXED `eval_set` at each
    of EVAL_CHECKPOINTS."""
    predictor = build_predictor(predictor_state, game_vocab, num_experts, feature_channels, device)
    trainable = set_adapter_trainable(predictor)
    opt = torch.optim.AdamW(trainable, lr=LR)

    results = {}
    if 0 in EVAL_CHECKPOINTS:
        results[0] = changed_patches_eval(online, predictor, eval_set, game_vocab, device)

    buffer = []
    max_needed = max(EVAL_CHECKPOINTS)
    for i, t in enumerate(stream):
        if i >= max_needed:
            break
        buffer.append(t)
        n_observed = i + 1
        if n_observed % k == 0:
            adaptation_step(online, predictor, opt, buffer, game_vocab, device, N_STEPS)
        if n_observed in EVAL_CHECKPOINTS:
            results[n_observed] = changed_patches_eval(online, predictor, eval_set, game_vocab, device)

    return results, predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=HELDOUT_GAMES_DEFAULT)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    global N_STEPS, LR
    if args.n_steps is not None:
        N_STEPS = args.n_steps
    if args.lr is not None:
        LR = args.lr

    device = get_device()
    print(f"Device: {device}")
    online, predictor_state, game_vocab, num_experts, feature_channels = load_baseline_checkpoint(device)
    print(f"Loaded checkpoints_holdout_baseline (game_vocab has {len(game_vocab)} entries, num_experts={num_experts})")

    trained_probe = load_sample_transitions(TRAINED_PROBE_GAMES, max_per_game=200)
    print(f"Trained-game interference probe: {len(trained_probe)} transitions across {TRAINED_PROBE_GAMES}")

    all_results = {}

    for game in args.games:
        print(f"\n{'=' * 70}\nGAME: {game}\n{'=' * 70}")
        per_file = load_game_transitions_per_file(game)
        if len(per_file) < 2:
            print(f"  SKIPPING {game}: fewer than 2 recording files found")
            continue
        eval_set = per_file[-1]  # last episode file, never touched by adaptation
        stream = [t for f in per_file[:-1] for t in f]  # remaining episodes, in order
        print(f"  stream: {len(stream)} transitions (adaptation buffer), eval: {len(eval_set)} transitions (fixed, never adapted on)")

        game_results = {"trajectories": {}, "interference": {}}

        # Pre-adaptation trained-game baseline (same for every K condition,
        # since it's the pristine checkpoint -- computed once).
        fresh_predictor = build_predictor(predictor_state, game_vocab, num_experts, feature_channels, device)
        pre_trained_eval = changed_patches_eval(online, fresh_predictor, trained_probe, game_vocab, device, per_game=True)
        print(f"\n  [pre-adaptation] trained-games changed-patches: "
              f"improvement={pre_trained_eval.get('improvement_pct', float('nan')):+.1f}% "
              f"(pred={pre_trained_eval.get('pred_changed_mse', float('nan')):.6f}, "
              f"identity={pre_trained_eval.get('identity_changed_mse', float('nan')):.6f}, n={pre_trained_eval.get('n_changed', 0)})")
        game_results["interference"]["pre_adaptation_trained_games"] = pre_trained_eval

        for k in K_CONDITIONS:
            print(f"\n  --- K={k} (adapt every {k} observed transitions, {N_STEPS} steps, lr={LR}) ---")
            traj, adapted_predictor = run_adaptation_trajectory(
                online, predictor_state, game_vocab, num_experts, feature_channels,
                stream, eval_set, k, device,
            )
            for n_obs in EVAL_CHECKPOINTS:
                r = traj.get(n_obs)
                if r and r.get("n_changed", 0) > 0:
                    print(f"    n_observed={n_obs:4d}: improvement={r['improvement_pct']:+.1f}% "
                          f"(pred={r['pred_changed_mse']:.6f}, identity={r['identity_changed_mse']:.6f}, n={r['n_changed']})")
                else:
                    print(f"    n_observed={n_obs:4d}: no changed-patch examples in eval set")
            game_results["trajectories"][f"K={k}"] = traj

            # Catastrophic interference: re-evaluate the SAME final adapted
            # model (after the full stream) on the trained-game probe set.
            post_trained_eval = changed_patches_eval(online, adapted_predictor, trained_probe, game_vocab, device, per_game=True)
            print(f"    [post-adaptation, K={k}] trained-games changed-patches: "
                  f"improvement={post_trained_eval.get('improvement_pct', float('nan')):+.1f}% "
                  f"(pred={post_trained_eval.get('pred_changed_mse', float('nan')):.6f}, "
                  f"identity={post_trained_eval.get('identity_changed_mse', float('nan')):.6f})")
            for g in TRAINED_PROBE_GAMES:
                pre_g = pre_trained_eval.get("per_game", {}).get(g)
                post_g = post_trained_eval.get("per_game", {}).get(g)
                if pre_g and post_g:
                    delta_pct = (post_g["pred_changed_mse"] - pre_g["pred_changed_mse"]) / pre_g["pred_changed_mse"] * 100
                    print(f"      {g}: pred_mse {pre_g['pred_changed_mse']:.6f} -> {post_g['pred_changed_mse']:.6f} ({delta_pct:+.1f}%)")
            game_results["interference"][f"post_adaptation_K={k}"] = post_trained_eval

        all_results[game] = game_results

    out_path = REPO_ROOT / "logs" / "test_time_adaptation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
