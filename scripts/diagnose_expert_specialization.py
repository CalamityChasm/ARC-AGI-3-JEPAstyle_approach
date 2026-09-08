"""Does per-expert disagreement in the production MoE predictor track real,
stable per-context competence -- or is it noise that happens to have nonzero
variance?

Direct user-proposed diagnostic: for each (game, action) pair, does one
expert consistently predict *changed* patches better than the others? If the
"best expert" for a given (game, action) is essentially a coin flip from one
sample to the next, the Bayesian hypothesis-confidence machinery
(jepa/hypothesis_bundle.py) is tracking noise, not genuine causal hypotheses,
even though raw InfoGain variance across experts doesn't collapse (see
scripts/diagnose_infogain_holdout.py -- that measured *whether* experts
disagree, not whether the disagreement is meaningful).

Primary metric: split-half agreement. For each (game, action) group with
enough changed-transition samples, split into two independent halves, find
each half's most-common best-expert (by lowest changed-patch MSE), and check
whether the two halves agree. Chance rate for K=8 experts is 1/8 = 12.5% if
"best expert" is unrelated across samples; a rate well above that is evidence
of real, stable specialization. Secondary metrics: per-group winner
concentration (mode share, entropy) and whether one expert just dominates
globally (a generalist, not specialization) vs different groups favoring
different experts.

Usage:
    python scripts/diagnose_expert_specialization.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import _load_frame_lines
from jepa.device import get_device
from jepa.grid import arc3_frame_to_tensor, patch_change_mask
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = Path("E:/ARC-AGI-3-JEPAstyle_data/recordings_archive")
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
MIN_GROUP_SIZE = 16  # need >=16 changed transitions per (game,action) to split-half meaningfully
SEED = 0


def load_changed_transitions() -> list:
    """(frame_t, action_id, x, y, frame_t1, game_id) for transitions where the
    frame actually changed -- unchanged ones trivially favor "predict no
    change" for every expert alike and carry no specialization signal."""
    files = sorted(ARCHIVE_DIR.glob("*.random.80.*.recording.jsonl"))
    assert len(files) == 150, f"expected 150 verified corpus files, found {len(files)}"
    transitions = []
    for path in files:
        frames = _load_frame_lines(path)
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            if cur["frame"] == nxt["frame"]:
                continue
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x, y = xy.get("x", 0), xy.get("y", 0)
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], game_id))
    return transitions


def load_checkpoint(device):
    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab_moe.json").read_text())
    meta = json.loads((CHECKPOINT_DIR / "moe_training_meta.json").read_text())
    num_experts = meta.get("num_experts", 8)
    feature_channels = meta.get("feature_channels", 64)
    encoder = CNNEncoder(out_channels=feature_channels).to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_moe.pt", map_location=device))
    encoder.eval()
    predictor = MoEPredictor(
        num_games=len(game_vocab), num_experts=num_experts,
        feature_channels=feature_channels, expert_hidden=feature_channels,
    ).to(device)
    predictor.load_state_dict(torch.load(CHECKPOINT_DIR / "moe_predictor.pt", map_location=device))
    predictor.eval()
    return encoder, predictor, game_vocab, num_experts


@torch.no_grad()
def compute_per_expert_errors(transitions, encoder, predictor, game_vocab, device, num_experts):
    """Returns dict[(game_id, action_id)] -> list of (K,) per-expert
    changed-patch MSE arrays, one per transition in that group."""
    grouped = defaultdict(list)
    for frame_t, action_id, x, y, frame_t1, game_id in transitions:
        mask = patch_change_mask(frame_t, frame_t1)  # (8, 8) bool
        if not mask.any():
            continue  # shouldn't happen given the "changed" filter, but be defensive

        cur_t = torch.from_numpy(arc3_frame_to_tensor(frame_t)).unsqueeze(0).to(device)
        nxt_t = torch.from_numpy(arc3_frame_to_tensor(frame_t1)).unsqueeze(0).to(device)
        feat = encoder(cur_t)  # (1, C, H, W)
        next_feat = encoder(nxt_t)  # (1, C, H, W)

        game_idx = torch.full((1,), game_vocab.get(game_id, 0), dtype=torch.long, device=device)
        xy_norm = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)
        action_t = torch.full((1,), action_id, dtype=torch.long, device=device)

        expert_preds = predictor.predict_all_experts(feat, action_t, xy_norm, game_idx)  # (1, K, C, H, W)
        expert_preds = expert_preds.squeeze(0)  # (K, C, H, W)
        target = next_feat.squeeze(0)  # (C, H, W)

        mask_t = torch.from_numpy(mask).to(device)  # (8, 8) bool, matches predictor's 8x8 feature map
        # per-expert MSE at changed patches only
        diff2 = (expert_preds - target.unsqueeze(0)).pow(2).mean(dim=1)  # (K, 8, 8) -- mean over channels
        per_expert_err = diff2[:, mask_t].mean(dim=1)  # (K,) -- mean over changed patches

        grouped[(game_id.split("-")[0], action_id)].append(per_expert_err.cpu().numpy())
    return grouped


def split_half_agreement(errors_list: list, num_experts: int, rng: random.Random):
    """errors_list: list of (K,) arrays for one (game,action) group.
    Returns (agree: bool, winner_a: int, winner_b: int) or None if too few samples."""
    n = len(errors_list)
    if n < MIN_GROUP_SIZE:
        return None
    idxs = list(range(n))
    rng.shuffle(idxs)
    half = n // 2
    a_idxs, b_idxs = idxs[:half], idxs[half:]

    a_argmins = [int(np.argmin(errors_list[i])) for i in a_idxs]
    b_argmins = [int(np.argmin(errors_list[i])) for i in b_idxs]

    winner_a = max(set(a_argmins), key=a_argmins.count)
    winner_b = max(set(b_argmins), key=b_argmins.count)
    return (winner_a == winner_b, winner_a, winner_b)


def mode_share_and_entropy(errors_list: list, num_experts: int):
    argmins = [int(np.argmin(e)) for e in errors_list]
    counts = np.bincount(argmins, minlength=num_experts).astype(np.float64)
    probs = counts / counts.sum()
    mode_share = probs.max()
    nonzero = probs[probs > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    max_entropy = np.log(num_experts)
    return mode_share, entropy / max_entropy, int(probs.argmax())


def chance_split_half_agreement(n: int, num_experts: int, n_trials: int, rng: random.Random) -> float:
    """Monte Carlo null: if the per-transition best-expert were i.i.d.
    uniform random (no real per-context signal at all), what split-half
    agreement rate would we see purely from finite-sample mode noise at this
    group size n?"""
    agreements = 0
    half = n // 2
    for _ in range(n_trials):
        labels = [rng.randrange(num_experts) for _ in range(n)]
        a, b = labels[:half], labels[half:]
        wa = max(set(a), key=a.count)
        wb = max(set(b), key=b.count)
        agreements += int(wa == wb)
    return agreements / n_trials


def main():
    device = get_device()
    print(f"device: {device}")
    rng = random.Random(SEED)

    print("loading verified changed-transition corpus...")
    transitions = load_changed_transitions()
    print(f"{len(transitions)} changed transitions loaded")

    print("loading production checkpoint (checkpoints/{encoder_moe,moe_predictor}.pt)...")
    encoder, predictor, game_vocab, num_experts = load_checkpoint(device)
    print(f"num_experts={num_experts}, num_games={len(game_vocab)}")

    print("computing per-expert changed-patch error for every transition, grouped by (game, action)...")
    grouped = compute_per_expert_errors(transitions, encoder, predictor, game_vocab, device, num_experts)

    rows = []
    for (game, action), errors_list in sorted(grouped.items()):
        n = len(errors_list)
        result = split_half_agreement(errors_list, num_experts, rng)
        mode_share, norm_entropy, mode_expert = mode_share_and_entropy(errors_list, num_experts)
        chance = None
        if result is not None:
            chance = chance_split_half_agreement(n, num_experts, n_trials=2000, rng=rng)
        rows.append({
            "game": game, "action": action, "n": n,
            "mode_share": mode_share, "norm_entropy": norm_entropy, "mode_expert": mode_expert,
            "split_half_agree": None if result is None else result[0],
            "winner_a": None if result is None else result[1],
            "winner_b": None if result is None else result[2],
            "chance_agreement_rate": chance,
        })

    print(f"\n{len(rows)} (game, action) groups found "
          f"({sum(1 for r in rows if r['n'] >= MIN_GROUP_SIZE)} with n>={MIN_GROUP_SIZE}, eligible for split-half test)")

    eligible = [r for r in rows if r["split_half_agree"] is not None]
    print(f"\n{'game':<10} {'act':>3} {'n':>5} {'mode%':>7} {'norm_H':>7} {'mode_exp':>8} "
          f"{'split_agree':>11} {'chance%':>8}")
    for r in sorted(rows, key=lambda r: -r["n"])[:60]:
        agree_str = "" if r["split_half_agree"] is None else ("YES" if r["split_half_agree"] else "no")
        chance_str = "" if r["chance_agreement_rate"] is None else f"{r['chance_agreement_rate']*100:.1f}%"
        print(f"{r['game']:<10} {r['action']:>3} {r['n']:>5} {r['mode_share']*100:>6.1f}% "
              f"{r['norm_entropy']:>7.3f} {r['mode_expert']:>8} {agree_str:>11} {chance_str:>8}")

    if eligible:
        observed_rate = sum(1 for r in eligible if r["split_half_agree"]) / len(eligible)
        mean_chance = sum(r["chance_agreement_rate"] for r in eligible) / len(eligible)
        mean_mode_share = sum(r["mode_share"] for r in eligible) / len(eligible)
        mean_norm_entropy = sum(r["norm_entropy"] for r in eligible) / len(eligible)

        winner_counts = defaultdict(int)
        for r in eligible:
            winner_counts[r["mode_expert"]] += 1
        winner_dist = {k: v / len(eligible) for k, v in sorted(winner_counts.items())}

        print(f"\n=== SUMMARY ({len(eligible)} eligible (game,action) groups, n>={MIN_GROUP_SIZE} each) ===")
        print(f"observed split-half agreement rate: {observed_rate*100:.1f}%")
        print(f"mean chance-level agreement rate (Monte Carlo null, matched group sizes): {mean_chance*100:.1f}%")
        print(f"mean per-group mode share (winner's fraction of samples): {mean_mode_share*100:.1f}%  (chance = {100/num_experts:.1f}%)")
        print(f"mean per-group normalized winner-entropy: {mean_norm_entropy:.3f}  (0=one expert always wins this group, 1=uniform/random)")
        print(f"distribution of which expert is the group-mode winner across groups: {winner_dist}")
        max_share = max(winner_dist.values())
        print(f"  -> most common single winning expert accounts for {max_share*100:.1f}% of groups "
              f"({'looks like a global generalist, not context specialization' if max_share > 0.4 else 'no single expert dominates -- consistent with per-context specialization if agreement rate is also high'})")

        print("\nVERDICT:")
        if observed_rate > mean_chance + 0.15:
            print(f"  Split-half agreement ({observed_rate*100:.1f}%) clearly exceeds the chance band "
                  f"({mean_chance*100:.1f}%) -- there IS a real, stable best-expert-per-context signal.")
        elif observed_rate < mean_chance + 0.05:
            print(f"  Split-half agreement ({observed_rate*100:.1f}%) is statistically indistinguishable from "
                  f"chance ({mean_chance*100:.1f}%) -- the 'best expert' for a given (game,action) is close to "
                  f"a coin flip from sample to sample. The Bayesian hypothesis-confidence machinery is most "
                  f"likely tracking noise, not genuine, distinct causal hypotheses about dynamics.")
        else:
            print(f"  Split-half agreement ({observed_rate*100:.1f}%) is modestly above chance "
                  f"({mean_chance*100:.1f}%) -- weak, partial evidence of real specialization, not clean either way.")
    else:
        print("\nNo (game, action) groups had enough changed transitions for a split-half test "
              f"(need n>={MIN_GROUP_SIZE}). Consider lowering MIN_GROUP_SIZE or pooling more data.")


if __name__ == "__main__":
    main()
