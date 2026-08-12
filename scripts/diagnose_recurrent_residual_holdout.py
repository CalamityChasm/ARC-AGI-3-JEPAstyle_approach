"""Does the recurrent predictor's residual branch collapse to ~0 on
held-out games, the same "coasts to identity via the skip connection"
failure Stage 6's own residual-commitment diagnostic already found for
the MoE predictor (CLAUDE.md's Stage 6 addendum, item 3)? If so, that
would explain why RecurrentSearch's novelty-lookahead came back a clean
0/32 on bp35/ka59 even with real multi-step search: every candidate's
imagined trajectory would look equally (un)novel if the model just
predicts "nothing changes" regardless of action, since the search can
only be as discriminating as the model doing the imagining.

Streams REAL local-recording transitions for bp35/ka59 through the
recurrent predictor with a real accumulated hidden state (not a fresh
zero state per transition -- matches how the agent actually uses it),
measuring ||net(x)|| (the residual pre-skip-add) at each step, and
compares against the same measurement on trained games.

Usage:
    python scripts/diagnose_recurrent_residual_holdout.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.sequences import load_all_episodes  # noqa: E402
from jepa.device import get_device  # noqa: E402
from jepa.models import CNNEncoder, RecurrentActionConditionedPredictor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints_recurrent_holdout"
HELD_OUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]


def load_checkpoint(device):
    import json

    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab_recurrent.json").read_text())
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_recurrent.pt", map_location=device))
    encoder.eval()
    predictor = RecurrentActionConditionedPredictor(num_games=max(len(game_vocab), 1)).to(device)
    predictor.load_state_dict(
        torch.load(CHECKPOINT_DIR / "recurrent_predictor.pt", map_location=device)
    )
    predictor.eval()
    return encoder, predictor, game_vocab


@torch.no_grad()
def measure_residual(encoder, predictor, episodes, game_vocab, device, label):
    from jepa.grid import arc3_frame_to_tensor

    residual_norms = []
    true_delta_norms = []
    for ep in episodes:
        hidden = predictor.init_hidden(1, device)
        for frame_t, action_id, x, y, frame_t1, _changed, game_id in ep:
            short = game_id.split("-")[0]
            game_idx = torch.tensor([game_vocab.get(short, 0)], dtype=torch.long, device=device)
            cur = torch.from_numpy(arc3_frame_to_tensor(frame_t)).unsqueeze(0).to(device)
            nxt = torch.from_numpy(arc3_frame_to_tensor(frame_t1)).unsqueeze(0).to(device)
            cur_feat = encoder(cur)
            next_feat = encoder(nxt)
            action_t = torch.tensor([action_id], dtype=torch.long, device=device)
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)

            a_embed = predictor.action_embed(action_t)
            xy_embed = predictor.coord_mlp(xy_t)
            g_embed = predictor.game_embed(game_idx)
            cond = torch.cat([a_embed, xy_embed, g_embed], dim=-1)
            pooled_feat = cur_feat.mean(dim=(2, 3))
            gru_input = torch.cat([pooled_feat, cond], dim=-1)
            new_hidden = predictor.gru_cell(gru_input, hidden)
            b, _, h, w = cur_feat.shape
            cond_spatial = cond.view(b, -1, 1, 1).expand(-1, -1, h, w)
            hidden_spatial = new_hidden.view(b, -1, 1, 1).expand(-1, -1, h, w)
            net_input = torch.cat([cur_feat, cond_spatial, hidden_spatial], dim=1)
            residual = predictor.net(net_input)

            residual_norms.append(residual.pow(2).mean().item())
            true_delta_norms.append((next_feat - cur_feat).pow(2).mean().item())
            hidden = new_hidden

    import statistics

    print(f"{label}: n={len(residual_norms)} transitions")
    print(f"  mean ||residual||^2 = {statistics.mean(residual_norms):.6e}")
    print(f"  mean ||true delta||^2 = {statistics.mean(true_delta_norms):.6e}")
    ratio = statistics.mean(residual_norms) / statistics.mean(true_delta_norms) if true_delta_norms else float("nan")
    print(f"  residual / true-delta ratio = {ratio:.3f}  (1.0 = commits fully, ~0 = coasts to identity)")
    return ratio


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    encoder, predictor, game_vocab = load_checkpoint(device)
    print(f"game_vocab has {len(game_vocab)} entries: {sorted(game_vocab.keys())}")

    all_episodes = load_all_episodes(REPO_ROOT)
    held_out_eps = [ep for ep in all_episodes if ep[0][6].split("-")[0] in HELD_OUT_GAMES]
    trained_eps = [ep for ep in all_episodes if ep[0][6].split("-")[0] not in HELD_OUT_GAMES]
    print(f"held-out episodes: {len(held_out_eps)}, trained episodes: {len(trained_eps)}")

    print("\n--- held-out games (r11l, bp35, m0r0, tr87, ka59) ---")
    ho_ratio = measure_residual(encoder, predictor, held_out_eps, game_vocab, device, "held-out")

    print("\n--- trained games (the 20 the checkpoint saw) ---")
    tr_ratio = measure_residual(encoder, predictor, trained_eps[:60], game_vocab, device, "trained (60-episode sample)")

    print(f"\nheld-out/trained residual-commitment ratio comparison: {ho_ratio:.3f} vs {tr_ratio:.3f}")
    if ho_ratio < 0.3 * tr_ratio:
        print("VERDICT: residual collapses on held-out games relative to trained games -- ")
        print("the recurrent predictor is coasting to identity on unfamiliar games, same as the MoE predictor.")
    else:
        print("VERDICT: no strong held-out-specific collapse relative to trained games.")


if __name__ == "__main__":
    main()
