"""Stage 6 follow-up: does the ENCODER's basic change-sensitivity (Stage
1 item 8 / scripts/diagnose_encoder_vs_predictor.py's diagnostic A) --
does encoder feature-space distance between frame_t and frame_t+1 differ
at patches that actually changed vs ones that didn't -- survive on games
the encoder never saw during training?

Motivation (see CLAUDE.md's Stage 6 section and
experiments/stage6_game_holdout.md): stage6-game-holdout found the whole
MoE predictor's changed-patches advantage over identity collapses to ~0%
on 5 held-out games, and stage6-gameid-ablation refuted the predictor's
game-id conditioning as the cause (ablating it entirely didn't fix the
collapse). The next suspect is the ENCODER itself -- if it has learned
game-specific visual shortcuts rather than general shape/motion
primitives, no predictor-side fix would help. diagnostic B (object-
identity) already showed a representation collapse on held-out games;
this script tests a more fundamental property -- does the encoder even
register that a pixel changed, on a game it's never seen -- directly,
reusing diagnostic A's original methodology (Stage 1 item 8 found 12x on
the standard held-out-transitions eval).

Compares, for both the baseline and object-identity checkpoints trained
in stage6-game-holdout (checkpoints_holdout_baseline / _objid):
  - diagnostic A ratio (changed/unchanged mean per-patch feature delta)
    computed ONLY on the 5 held-out games' transitions (r11l, bp35, m0r0,
    tr87, ka59 -- games neither checkpoint ever trained on).
  - the SAME ratio computed on a sample of the 20 TRAINED games'
    transitions, as the comparison baseline.
  - raw changed/unchanged patch counts behind each ratio, so the result
    is interpretable rather than a bare number.

Usage:
    python scripts/diagnose_encoder_holdout.py
"""

import json
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

# Same verified 150-file / 12,000-transition corpus used by
# scripts/diagnose_encoder_vs_predictor.py and (for local recordings)
# scripts/eval_game_holdout.py -- stable, worktree-independent path.
ARCHIVE_DIR = Path("E:/ARC-AGI-3-JEPAstyle_data/recordings_archive")

HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

# Checkpoints trained in stage6-game-holdout (worktree
# agent-a0f09770086c096a6), on the identical 20-game corpus (held-out
# games entirely excluded from both local + external training data).
CHECKPOINTS = {
    "baseline-holdout": Path(
        "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
        "agent-a0f09770086c096a6/checkpoints_holdout_baseline"
    ),
    "object-identity-holdout": Path(
        "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
        "agent-a0f09770086c096a6/checkpoints_holdout_objid"
    ),
}

TRAINED_SAMPLE_N = 2000  # matches diagnose_encoder_vs_predictor.py's SAMPLE_N


def load_verified_transitions() -> list:
    """All 150 files / 12,000 transitions, tagged with short game_id."""
    transitions = []
    files = sorted(ARCHIVE_DIR.glob("*.random.80.*.recording.jsonl"))
    assert len(files) == 150, f"expected 150 verified random-corpus files, found {len(files)}"
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
    assert len(transitions) == 12000, f"expected 12000 transitions, found {len(transitions)}"
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
def diagnostic_a(online, game_vocab: dict, transitions: list, device) -> dict:
    """Diagnostic A only: per-patch feature-space delta at changed vs
    unchanged patches. game_vocab lookups fall back to index 0 for any
    game_id not in the checkpoint's vocab (matters for held-out games;
    matches hypothesis_agent.py's real novel-game behavior) -- note this
    only affects which game embedding conditions the ENCODER's forward
    pass not at all (the encoder takes no game_idx), so the fallback here
    is inert for diagnostic A specifically. Kept anyway via TransitionDataset
    for interface consistency with the other diagnostic scripts.
    """
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    changed_deltas, unchanged_deltas = [], []
    for cur, _action_id, _xy, nxt, patch_mask, _game_idx in loader:
        cur, nxt = cur.to(device), nxt.to(device)
        patch_mask = patch_mask.to(device)  # (B, 8, 8) bool

        cur_feat = online(cur)
        nxt_feat = online(nxt)
        true_delta = (nxt_feat - cur_feat).pow(2).mean(dim=1)  # (B, 8, 8)

        changed_deltas.append(true_delta[patch_mask].cpu())
        unchanged_deltas.append(true_delta[~patch_mask].cpu())

    changed_deltas = torch.cat(changed_deltas)
    unchanged_deltas = torch.cat(unchanged_deltas)

    return {
        "n_changed_patches": changed_deltas.numel(),
        "n_unchanged_patches": unchanged_deltas.numel(),
        "changed_delta_mean": changed_deltas.mean().item(),
        "unchanged_delta_mean": unchanged_deltas.mean().item(),
        "ratio": (changed_deltas.mean() / unchanged_deltas.mean().clamp(min=1e-12)).item(),
    }


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    print("\nLoading verified 150-file / 12,000-transition corpus...")
    all_transitions = load_verified_transitions()
    game_ids_full = sorted({t[6] for t in all_transitions})
    print(f"  {len(all_transitions)} transitions across {len(game_ids_full)} games")

    heldout_prefixes = tuple(f"{g}-" for g in HELDOUT_GAMES)
    heldout_transitions = [t for t in all_transitions if t[6].startswith(heldout_prefixes)]
    trained_transitions_full = [t for t in all_transitions if not t[6].startswith(heldout_prefixes)]
    games_seen_heldout = sorted({t[6].split("-")[0] for t in heldout_transitions})
    games_seen_trained = sorted({t[6].split("-")[0] for t in trained_transitions_full})
    assert set(games_seen_heldout) == set(HELDOUT_GAMES), (
        f"expected exactly {HELDOUT_GAMES} in held-out split, found {games_seen_heldout}"
    )
    assert len(games_seen_trained) == 20, f"expected 20 trained games, found {len(games_seen_trained)}"
    print(f"  Held-out split: {len(heldout_transitions)} transitions across {games_seen_heldout}")
    print(f"  Trained split:  {len(trained_transitions_full)} transitions across {len(games_seen_trained)} games")

    random.Random(0).shuffle(trained_transitions_full)
    trained_sample = trained_transitions_full[:TRAINED_SAMPLE_N]

    results = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nCHECKPOINT: {name} ({ckpt_dir})\n{'=' * 70}")
        online, _predictor, game_vocab = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(HELDOUT_GAMES)} "
              f"held-out games present (should be 0 -- confirms true holdout)")

        print(f"\n[diagnostic A] on the 5 HELD-OUT games ({len(heldout_transitions)} transitions):")
        a_heldout = diagnostic_a(online, game_vocab, heldout_transitions, device)
        print(f"    changed patches n={a_heldout['n_changed_patches']}  mean delta={a_heldout['changed_delta_mean']:.6f}")
        print(f"    unchanged patches n={a_heldout['n_unchanged_patches']}  mean delta={a_heldout['unchanged_delta_mean']:.6f}")
        print(f"    ratio (changed/unchanged): {a_heldout['ratio']:.2f}x")

        print(f"\n[diagnostic A] on a sample of the 20 TRAINED games ({len(trained_sample)} transitions):")
        a_trained = diagnostic_a(online, game_vocab, trained_sample, device)
        print(f"    changed patches n={a_trained['n_changed_patches']}  mean delta={a_trained['changed_delta_mean']:.6f}")
        print(f"    unchanged patches n={a_trained['n_unchanged_patches']}  mean delta={a_trained['unchanged_delta_mean']:.6f}")
        print(f"    ratio (changed/unchanged): {a_trained['ratio']:.2f}x")

        results[name] = {"heldout": a_heldout, "trained": a_trained}

    out_path = REPO_ROOT / "logs" / "encoder_holdout_diagnostic_a_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
