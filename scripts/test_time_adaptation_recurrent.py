"""Does test-time adaptation -- real gradient steps on a held-out game's
own observed transitions, taken DURING simulated play -- fix the
recurrent predictor's residual collapse specifically on `bp35`/`ka59`,
the same way it gave a real (if modest) representation-level improvement
for the MoE predictor (`scripts/test_time_adaptation.py`,
`experiments/stage6_test_time_adaptation.md`)?

Motivation: `experiments/stage6_recurrent_exploration.md`'s own
follow-up diagnostic found the recurrent predictor's residual branch is
~8x weaker relative to the true observed change on held-out games than
on trained ones (ratio 0.041 vs 0.325) -- it coasts to identity on
unfamiliar games. That's a direct, mechanistic explanation for why THREE
different exploration strategies built on top of this frozen world model
(Hypothesis's InfoGain/value blend, RecurrentCuriosity's retrospective
surprise ranking, RecurrentSearch's genuine MPC lookahead) all came back
a clean 0/48 on `bp35`/`ka59` specifically: a search or ranking mechanism
can only discriminate candidates the underlying model actually predicts
differently. TTA is the one lever this project has already validated
gives the world model real signal on a held-out game without retraining
from scratch -- this script checks whether it also closes (or narrows)
the residual-collapse gap for THIS predictor, on THESE two specific
games, before spending time wiring it into a live agent.

Adapted parameter subset: only `predictor.net[-1]` (the final Conv2d
that directly produces the residual, zero-initialized at training time,
same "last layer only" ANIL-style restriction the MoE version already
validated) -- small (~4.2K params for feature_channels=64), low risk of
overfitting a handful of examples.

Because RecurrentActionConditionedPredictor's hidden state only has
meaning across an ordered sequence (not i.i.d. transitions like the MoE
predictor), adaptation and eval both operate on SEQ_LEN-length chunks
with a hidden state reset to zero at the start of each chunk -- this
exactly mirrors `jepa/train_recurrent_predictor.py`'s own chunked-BPTT
training methodology (hidden state does NOT persist across chunks, a
simplification already accepted at training time, not a new one
introduced here) rather than inventing a different convention for TTA
specifically.

Usage:
    python scripts/test_time_adaptation_recurrent.py
    python scripts/test_time_adaptation_recurrent.py --games bp35 ka59
"""

import argparse
import copy
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.test_time_adaptation as tta  # noqa: E402 -- reuse its per-file loader
from jepa.device import get_device  # noqa: E402
from jepa.grid import arc3_frame_to_tensor, patch_change_mask  # noqa: E402
from jepa.losses import per_region_error, weighted_prediction_loss  # noqa: E402
from jepa.models import CNNEncoder, RecurrentActionConditionedPredictor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints_recurrent_holdout"

SEQ_LEN = 16  # matches train_recurrent_predictor.py's own chunk length
EVAL_CHECKPOINTS = [0, 10, 50, 200]
K = 5  # adapt every K newly observed transitions -- matches the MoE production op point
N_STEPS = 8  # gradient steps per adaptation event -- matches the MoE production op point
LR = 5e-5


def load_checkpoint(device, checkpoint_dir: Path):
    game_vocab = json.loads((checkpoint_dir / "game_vocab_recurrent.json").read_text())
    online = CNNEncoder().to(device)
    online.load_state_dict(torch.load(checkpoint_dir / "encoder_recurrent.pt", map_location=device))
    online.eval()
    for p in online.parameters():
        p.requires_grad = False
    predictor_state = torch.load(checkpoint_dir / "recurrent_predictor.pt", map_location=device)
    return online, predictor_state, game_vocab


def build_predictor(state_dict, game_vocab, device):
    predictor = RecurrentActionConditionedPredictor(num_games=max(len(game_vocab), 1)).to(device)
    predictor.load_state_dict(copy.deepcopy(state_dict))
    predictor.eval()
    return predictor


def set_adapter_trainable(predictor) -> list:
    for p in predictor.parameters():
        p.requires_grad = False
    trainable = list(predictor.net[-1].parameters())
    for p in trainable:
        p.requires_grad = True
    return trainable


def _chunk_tensors(chunk: list, game_vocab: dict, device):
    curs, actions, xys, nxts, masks = [], [], [], [], []
    game_id = chunk[0][6]
    short = game_id.split("-")[0]
    game_idx = game_vocab.get(short, 0)
    for frame_t, action_id, x, y, frame_t1, _changed, _gid in chunk:
        curs.append(arc3_frame_to_tensor(frame_t))
        actions.append(action_id)
        xys.append([x / 63.0, y / 63.0])
        nxts.append(arc3_frame_to_tensor(frame_t1))
        masks.append(patch_change_mask(frame_t, frame_t1))
    import numpy as np

    cur = torch.from_numpy(np.stack(curs)).to(device)  # (T, 17, 64, 64)
    nxt = torch.from_numpy(np.stack(nxts)).to(device)
    action_t = torch.tensor(actions, dtype=torch.long, device=device)
    xy_t = torch.tensor(xys, dtype=torch.float32, device=device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    game_t = torch.full((len(chunk),), game_idx, dtype=torch.long, device=device)
    return cur, action_t, xy_t, nxt, mask_t, game_t


def run_chunk(online, predictor, chunk: list, game_vocab: dict, device, grad: bool):
    """Runs one SEQ_LEN chunk through encoder+recurrent predictor with a
    FRESH (zero) hidden state, matching train_recurrent_predictor.py's own
    chunked-BPTT convention. Returns per-step (pred_feat, next_feat,
    cur_feat, mask) tuples."""
    cur, action_t, xy_t, nxt, mask_t, game_t = _chunk_tensors(chunk, game_vocab, device)
    hidden = predictor.init_hidden(1, device)
    outputs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for step in range(len(chunk)):
            with torch.no_grad():
                cur_feat = online(cur[step : step + 1])
                next_feat = online(nxt[step : step + 1])
            pred_feat, hidden = predictor(
                cur_feat, action_t[step : step + 1], xy_t[step : step + 1], hidden, game_t[step : step + 1]
            )
            outputs.append((pred_feat, next_feat, cur_feat, mask_t[step]))
    return outputs


def adaptation_step(online, predictor, opt, buffer: list, game_vocab: dict, device, n_steps: int):
    seq_len = min(SEQ_LEN, len(buffer))
    predictor.train()
    for _ in range(n_steps):
        start = 0 if len(buffer) <= seq_len else torch.randint(0, len(buffer) - seq_len + 1, (1,)).item()
        chunk = buffer[start : start + seq_len]
        outputs = run_chunk(online, predictor, chunk, game_vocab, device, grad=True)
        loss = sum(
            weighted_prediction_loss(pred, tgt, mask.unsqueeze(0))
            for pred, tgt, _cur, mask in outputs
        ) / len(outputs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    predictor.eval()


@torch.no_grad()
def eval_metrics(online, predictor, eval_chunks: list, game_vocab: dict, device) -> dict:
    """changed-patches improvement AND the residual/true-delta commitment
    ratio (same measurement as diagnose_recurrent_residual_holdout.py),
    on a FIXED set of chunks never adapted on."""
    total_pred, total_identity, n_changed = 0.0, 0.0, 0
    residual_sq, delta_sq, n_steps = 0.0, 0.0, 0
    for chunk in eval_chunks:
        outputs = run_chunk(online, predictor, chunk, game_vocab, device, grad=False)
        for pred_feat, next_feat, cur_feat, mask in outputs:
            if mask.any():
                pred_err = per_region_error(pred_feat, next_feat)[0][mask]
                identity_err = per_region_error(cur_feat, next_feat)[0][mask]
                total_pred += pred_err.mean().item()
                total_identity += identity_err.mean().item()
                n_changed += 1
            residual_sq += (pred_feat - cur_feat).pow(2).mean().item()
            delta_sq += (next_feat - cur_feat).pow(2).mean().item()
            n_steps += 1
    result = {"n_changed": n_changed, "n_steps": n_steps}
    if n_changed > 0:
        result["pred_changed_mse"] = total_pred / n_changed
        result["identity_changed_mse"] = total_identity / n_changed
        result["improvement_pct"] = (total_identity - total_pred) / total_identity * 100
    if n_steps > 0:
        result["residual_sq"] = residual_sq / n_steps
        result["delta_sq"] = delta_sq / n_steps
        result["commitment_ratio"] = (residual_sq / n_steps) / (delta_sq / n_steps) if delta_sq > 0 else float("nan")
    return result


def run_adaptation_trajectory(
    online, predictor_state, game_vocab, stream: list, eval_chunks: list, k: int, device,
) -> dict:
    predictor = build_predictor(predictor_state, game_vocab, device)
    trainable = set_adapter_trainable(predictor)
    opt = torch.optim.AdamW(trainable, lr=LR)

    results = {}
    if 0 in EVAL_CHECKPOINTS:
        results[0] = eval_metrics(online, predictor, eval_chunks, game_vocab, device)

    buffer = []
    max_needed = max(EVAL_CHECKPOINTS)
    for i, t in enumerate(stream):
        if i >= max_needed:
            break
        buffer.append(t)
        n_observed = i + 1
        if n_observed % k == 0 and len(buffer) >= SEQ_LEN:
            adaptation_step(online, predictor, opt, buffer, game_vocab, device, N_STEPS)
        if n_observed in EVAL_CHECKPOINTS:
            results[n_observed] = eval_metrics(online, predictor, eval_chunks, game_vocab, device)

    return results


def main() -> None:
    global K, N_STEPS, LR

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["bp35", "ka59"])
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--k", type=int, default=K, help="Adapt every K newly observed transitions.")
    parser.add_argument("--steps", type=int, default=N_STEPS, help="AdamW steps per adaptation event.")
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()
    K, N_STEPS, LR = args.k, args.steps, args.lr

    device = get_device()
    print(f"Device: {device}")
    online, predictor_state, game_vocab = load_checkpoint(device, args.checkpoint_dir)
    print(f"Loaded {args.checkpoint_dir} (game_vocab has {len(game_vocab)} entries)")
    print(f"Operating point: K={K} STEPS={N_STEPS} LR={LR} SEQ_LEN={SEQ_LEN}")

    for game in args.games:
        print(f"\n{'=' * 70}\nGAME: {game}\n{'=' * 70}")
        per_file = tta.load_game_transitions_per_file(game)
        if len(per_file) < 2:
            print(f"  SKIPPING {game}: fewer than 2 recording files found")
            continue
        eval_file = per_file[-1]
        eval_chunks = [
            eval_file[s : s + SEQ_LEN] for s in range(0, len(eval_file) - SEQ_LEN + 1, SEQ_LEN)
        ]
        stream = [t for f in per_file[:-1] for t in f]
        print(f"  stream: {len(stream)} transitions, eval: {len(eval_chunks)} chunks of {SEQ_LEN}")

        traj = run_adaptation_trajectory(online, predictor_state, game_vocab, stream, eval_chunks, K, device)
        for n_obs in EVAL_CHECKPOINTS:
            r = traj.get(n_obs)
            if not r:
                continue
            cp = f"{r['improvement_pct']:+.2f}%" if "improvement_pct" in r else "n/a"
            cr = f"{r['commitment_ratio']:.3f}" if "commitment_ratio" in r else "n/a"
            pm = f"{r['pred_changed_mse']:.6f}" if "pred_changed_mse" in r else "n/a"
            im = f"{r['identity_changed_mse']:.6f}" if "identity_changed_mse" in r else "n/a"
            print(
                f"    n_observed={n_obs:4d}: changed-patches={cp:>9}  commitment_ratio={cr}  "
                f"pred_mse={pm}  identity_mse={im}"
            )


if __name__ == "__main__":
    main()
