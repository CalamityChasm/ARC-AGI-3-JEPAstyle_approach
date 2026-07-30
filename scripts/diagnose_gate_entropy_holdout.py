"""Stage 6 residual-commitment-fix, step 1: does the MoE gate get MORE
uniform/hedged on held-out (unfamiliar) games than on trained games?

Motivation (see experiments/stage6_encoder_holdout_diagnostic.md, the
immediate predecessor to this experiment): the encoder's basic
change-sensitivity does NOT collapse on the 5 games held out of training
(r11l, bp35, m0r0, tr87, ka59) -- if anything it's stronger there. But the
predictor's residual-commitment ratio (bonus check in that same
experiment) collapses to ~0.000 on held-out games vs 0.235/0.010 on
trained games -- the predictor defaults to "predict identity" specifically
on unfamiliar games even though the encoder handed it a clear change
signal. This script checks one candidate mechanism for *why*: does the
MoE gate itself get pushed toward a more uniform blend on held-out games
(consistent with "the model doesn't know which expert to trust here, so
it hedges by blending everyone a little," and a uniform blend of several
small, not-necessarily-aligned per-expert residuals nets out closer to
zero than any single confident expert's output would)?

Stage 4's own finding (CLAUDE.md) is the baseline to compare against: gate
specialization was ALREADY "a minority behavior, not the norm" even on
trained games (mean entropy ~98.6% of the uniform max, ~4% of examples
with a dominant expert). If held-out games show entropy pushed even
closer to 100% of max (and dominant-expert fraction drops further toward
0%), that's mechanistic support for the residual-collapse finding, not
just correlational.

Reuses the exact checkpoints, data split, and archive corpus as
scripts/diagnose_encoder_holdout_predictor_check.py (stage6-encoder-
holdout-diag) for an apples-to-apples comparison with that experiment's
own diagnostic-C numbers.

Usage:
    python scripts/diagnose_gate_entropy_holdout.py
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset, _load_frame_lines
from jepa.device import get_device
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = Path("E:/ARC-AGI-3-JEPAstyle_data/recordings_archive")
HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

CHECKPOINTS = {
    "baseline-holdout": Path(
        "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
        "agent-a0f09770086c096a6/checkpoints_holdout_baseline"
    ),
    "object-identity-holdout": Path(
        "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
        "agent-a0f09770086c096a6/checkpoints_holdout_objid"
    ),
    "residual-commitment-fix": (
        Path(__file__).resolve().parent.parent / "checkpoints_holdout_rescommit"
    ),
}
TRAINED_SAMPLE_N = 2000
DOMINANT_THRESHOLD = 0.3  # matches Stage 4's own "one expert clearly dominant" convention


def load_verified_transitions() -> list:
    transitions = []
    files = sorted(ARCHIVE_DIR.glob("*.random.80.*.recording.jsonl"))
    assert len(files) == 150, f"expected 150 verified recording files, found {len(files)}"
    for path in files:
        frames = _load_frame_lines(path)
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x, y = xy.get("x", 0), xy.get("y", 0)
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], cur["frame"] != nxt["frame"], game_id))
    assert len(transitions) == 12000
    return transitions


def load_moe_checkpoint(checkpoint_dir: Path, device):
    game_vocab = json.loads((checkpoint_dir / "game_vocab_moe.json").read_text())
    num_experts, feature_channels = 8, 64
    meta_path = checkpoint_dir / "moe_training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        num_experts = meta.get("num_experts", 8)
        feature_channels = meta.get("feature_channels", 64)
    online = CNNEncoder(out_channels=feature_channels).to(device)
    online.load_state_dict(torch.load(checkpoint_dir / "encoder_moe.pt", map_location=device))
    online.eval()
    predictor = MoEPredictor(
        num_games=len(game_vocab), num_experts=num_experts,
        feature_channels=feature_channels, expert_hidden=feature_channels,
    ).to(device)
    predictor.load_state_dict(torch.load(checkpoint_dir / "moe_predictor.pt", map_location=device))
    predictor.eval()
    return online, predictor, game_vocab


@torch.no_grad()
def gate_entropy_stats(online, predictor, game_vocab, transitions: list, device, num_experts: int) -> dict:
    """Per-transition gate entropy + dominant-expert fraction, pooled over
    ALL transitions in the given population (not just changed patches --
    the gate is computed once per transition from pooled features, so
    there's no per-patch breakdown to restrict to)."""
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    max_entropy = math.log(num_experts)
    entropies = []
    max_weights = []
    for cur, action_id, xy, nxt, _patch_mask, game_idx in loader:
        cur = cur.to(device)
        action_id, xy, game_idx = action_id.to(device), xy.to(device), game_idx.to(device)

        cur_feat = online(cur)
        _pred_feat, gate_weights = predictor(cur_feat, action_id, xy, game_idx)  # (B, K)

        p = gate_weights.clamp(min=1e-12)
        ent = -(p * p.log()).sum(dim=-1)  # (B,)
        entropies.append(ent.cpu())
        max_weights.append(gate_weights.max(dim=-1).values.cpu())

    entropies = torch.cat(entropies)
    max_weights = torch.cat(max_weights)
    return {
        "n": entropies.numel(),
        "mean_entropy": entropies.mean().item(),
        "max_entropy": max_entropy,
        "mean_entropy_pct_of_max": (entropies.mean().item() / max_entropy) * 100,
        "entropy_std": entropies.std().item(),
        "frac_dominant_expert_gt_0.3": (max_weights > DOMINANT_THRESHOLD).float().mean().item(),
        "mean_max_weight": max_weights.mean().item(),
    }


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    all_transitions = load_verified_transitions()
    heldout_prefixes = tuple(f"{g}-" for g in HELDOUT_GAMES)
    heldout_transitions = [t for t in all_transitions if t[6].startswith(heldout_prefixes)]
    trained_transitions_full = [t for t in all_transitions if not t[6].startswith(heldout_prefixes)]
    random.Random(0).shuffle(trained_transitions_full)
    trained_sample = trained_transitions_full[:TRAINED_SAMPLE_N]
    print(f"held-out transitions: {len(heldout_transitions)}  trained sample: {len(trained_sample)}")

    results = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nCHECKPOINT: {name}\n{'=' * 70}")
        online, predictor, game_vocab = load_moe_checkpoint(ckpt_dir, device)
        num_experts = predictor.num_experts

        g_heldout = gate_entropy_stats(online, predictor, game_vocab, heldout_transitions, device, num_experts)
        print(f"\n[gate entropy] HELD-OUT games (n={g_heldout['n']}):")
        print(
            f"    mean entropy: {g_heldout['mean_entropy']:.4f} nats "
            f"({g_heldout['mean_entropy_pct_of_max']:.2f}% of max={g_heldout['max_entropy']:.4f})"
        )
        print(f"    entropy std: {g_heldout['entropy_std']:.4f}")
        print(
            f"    frac with dominant expert (weight>{DOMINANT_THRESHOLD}): "
            f"{g_heldout['frac_dominant_expert_gt_0.3'] * 100:.2f}%  "
            f"mean max weight: {g_heldout['mean_max_weight']:.4f}"
        )

        g_trained = gate_entropy_stats(online, predictor, game_vocab, trained_sample, device, num_experts)
        print(f"\n[gate entropy] TRAINED games sample (n={g_trained['n']}):")
        print(
            f"    mean entropy: {g_trained['mean_entropy']:.4f} nats "
            f"({g_trained['mean_entropy_pct_of_max']:.2f}% of max={g_trained['max_entropy']:.4f})"
        )
        print(f"    entropy std: {g_trained['entropy_std']:.4f}")
        print(
            f"    frac with dominant expert (weight>{DOMINANT_THRESHOLD}): "
            f"{g_trained['frac_dominant_expert_gt_0.3'] * 100:.2f}%  "
            f"mean max weight: {g_trained['mean_max_weight']:.4f}"
        )

        results[name] = {"heldout": g_heldout, "trained": g_trained}

    out_path = REPO_ROOT / "logs" / "gate_entropy_holdout_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
