"""Standalone VRAM probe for ScaledCNNEncoder + ScaledMoEPredictor: measures
real peak GPU memory for one forward+backward step at increasing
width_mult/blocks_per_stage/num_experts/batch_size configs, on dummy data
(no dataset loading -- isolates pure model-size VRAM cost).

Usage: python -m scripts.probe_scaled_vram
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.models.encoder_scaled import ScaledCNNEncoder, count_params as count_enc_params, make_ema_target
from jepa.models.moe_predictor_scaled import ScaledMoEPredictor, count_params as count_pred_params
from jepa.losses import variance_regularizer, weighted_prediction_loss
from jepa.models.moe_predictor import load_balance_loss
from jepa.grid import NUM_CHANNELS, CANVAS


def probe(width_mult, blocks_per_stage, num_experts, expert_hidden, batch_size, feature_channels=64, num_games=30, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(device)

    encoder = ScaledCNNEncoder(out_channels=feature_channels, width_mult=width_mult, blocks_per_stage=blocks_per_stage).to(device)
    target = make_ema_target(encoder)
    predictor = ScaledMoEPredictor(
        feature_channels=encoder.out_channels,
        num_games=num_games,
        num_experts=num_experts,
        expert_hidden=expert_hidden,  # already the actual desired width -- configs table below sets it directly
        expert_depth=1,
    ).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=3e-4)

    n_enc = count_enc_params(encoder)
    n_pred = count_pred_params(predictor)

    cur = torch.rand(batch_size, NUM_CHANNELS, CANVAS, CANVAS, device=device)
    nxt = torch.rand(batch_size, NUM_CHANNELS, CANVAS, CANVAS, device=device)
    action_id = torch.randint(0, 8, (batch_size,), device=device)
    xy = torch.rand(batch_size, 2, device=device)
    game_idx = torch.randint(0, num_games, (batch_size,), device=device)
    patch_mask = torch.rand(batch_size, 8, 8, device=device) > 0.7

    import time as _time

    try:
        # Warm-up step (first CUDA kernel launches / cuDNN algo selection are
        # slow and would otherwise pollute the timing below).
        cur_feat = encoder(cur)
        pred_feat, gate_weights = predictor(cur_feat, action_id, xy, game_idx)
        with torch.no_grad():
            target_feat = target(nxt)
        lb_loss = load_balance_loss(gate_weights)
        loss = weighted_prediction_loss(pred_feat, target_feat, patch_mask) + variance_regularizer(cur_feat) + 0.001 * lb_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        n_steps = 5
        t0 = _time.time()
        for _ in range(n_steps):
            cur_feat = encoder(cur)
            pred_feat, gate_weights = predictor(cur_feat, action_id, xy, game_idx)
            with torch.no_grad():
                target_feat = target(nxt)
            lb_loss = load_balance_loss(gate_weights)
            loss = weighted_prediction_loss(pred_feat, target_feat, patch_mask) + variance_regularizer(cur_feat) + 0.001 * lb_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        step_time_ms = (_time.time() - t0) / n_steps * 1000
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        ok = True
        err = None
    except torch.cuda.OutOfMemoryError as e:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        step_time_ms = -1.0
        ok = False
        err = str(e).splitlines()[0]

    del encoder, target, predictor, opt, cur, nxt, action_id, xy, game_idx, patch_mask
    torch.cuda.empty_cache()

    return {
        "width_mult": width_mult,
        "blocks_per_stage": blocks_per_stage,
        "num_experts": num_experts,
        "expert_hidden": expert_hidden,
        "batch_size": batch_size,
        "n_enc_params": n_enc,
        "n_pred_params": n_pred,
        "n_total_params": n_enc + n_pred,
        "peak_vram_gb": round(peak_gb, 3),
        "reserved_vram_gb": round(reserved_gb, 3),
        "step_time_ms": round(step_time_ms, 1),
        "ok": ok,
        "err": err,
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no CUDA GPU available")
        sys.exit(1)

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_properties(0).name}, total VRAM: {total_vram:.2f} GB\n")

    def eh(width_mult):  # expert_hidden convention matching train_scaled_curriculum.py: round(64 * width_mult)
        return int(round(64 * width_mult))

    configs = [
        # (width_mult, blocks_per_stage, num_experts, expert_hidden, batch_size)
        (1.0, 0, 8, eh(1.0), 32),
        (2.0, 1, 8, eh(2.0), 32),
        (3.0, 2, 12, eh(3.0), 32),
        (4.0, 2, 16, eh(4.0), 32),
        (4.0, 2, 16, eh(4.0), 16),
        (6.0, 3, 16, eh(6.0), 16),
        (8.0, 3, 24, eh(8.0), 16),
        (8.0, 3, 24, eh(8.0), 32),
        (10.0, 4, 24, eh(10.0), 16),
        (10.0, 4, 24, eh(10.0), 32),
        (11.0, 4, 28, eh(11.0), 16),
        (12.0, 4, 32, eh(12.0), 16),
    ]

    header = (
        f"{'width':>6} {'blocks':>7} {'experts':>8} {'exp_hid':>8} {'batch':>6} {'params':>12} "
        f"{'peak_GB':>9} {'resv_GB':>9} {'ms/step':>8} {'ok':>4}"
    )
    print(header)
    print("-" * len(header))
    results = []
    for width_mult, blocks, num_experts, expert_hidden, batch_size in configs:
        try:
            r = probe(width_mult, blocks, num_experts, expert_hidden, batch_size)
        except Exception as e:
            r = {
                "width_mult": width_mult, "blocks_per_stage": blocks, "num_experts": num_experts,
                "expert_hidden": expert_hidden, "batch_size": batch_size, "n_total_params": -1,
                "peak_vram_gb": -1, "reserved_vram_gb": -1, "step_time_ms": -1, "ok": False,
                "err": f"{type(e).__name__}: {e}",
            }
        results.append(r)
        print(
            f"{r['width_mult']:>6.1f} {r['blocks_per_stage']:>7} {r['num_experts']:>8} {r['expert_hidden']:>8} "
            f"{r['batch_size']:>6} {r['n_total_params']:>12,} {r['peak_vram_gb']:>9.3f} "
            f"{r['reserved_vram_gb']:>9.3f} {r['step_time_ms']:>8.1f} {str(r['ok']):>4}"
        )
        if not r["ok"]:
            print(f"    -> {r['err']}")

    import json
    Path("logs").mkdir(exist_ok=True)
    Path("logs/scaled_vram_probe.json").write_text(json.dumps(results, indent=2))
    print("\nfull results written to logs/scaled_vram_probe.json")
