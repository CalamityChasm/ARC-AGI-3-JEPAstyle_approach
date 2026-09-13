"""What the multimodal grid image actually costs, and why MULTIMODAL_UPSCALE is inert.

The brief's leading hypothesis was that the 4x-upscaled 64x64 grid dominates
tokens-per-request and that lowering `MULTIMODAL_UPSCALE` would be the lever.
Both halves are wrong, and this script shows why from the model's own
`preprocessor_config.json` rather than from a guess.

    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16, "merge_size": 2,
    "image_processor_type": "Qwen2VLImageProcessorFast"

For Qwen2/3-VL, `size.shortest_edge` is `min_pixels` and `size.longest_edge` is
`max_pixels`, and `smart_resize` rounds each side to a multiple of
`patch_size * merge_size` = 32 while forcing total pixels into
[min_pixels, max_pixels]. One vision token then covers a 32x32 block.

A 64x64 ARC grid at upscale 4 is 256x256 = 65,536 px, which is *exactly*
min_pixels. Anything smaller gets scaled back up to it. So upscale 1, 2, 3 and
4 all produce the identical 8x8 = 64 vision tokens (+2 markers), and only
upscale >= 5 costs more. The setting is already at its floor.

Usage:
    python scripts/vision_token_cost.py [--grid 64] [--requests 1339] [--images 9828]
"""

from __future__ import annotations

import argparse
import math

PATCH, MERGE = 16, 2
FACTOR = PATCH * MERGE            # 32 px per vision token
MIN_PIXELS = 65_536               # preprocessor_config.json: size.shortest_edge
MAX_PIXELS = 16_777_216           # preprocessor_config.json: size.longest_edge
MARKERS = 2                       # vision_start_token_id + vision_end_token_id


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def vision_tokens(height, width):
    h, w = smart_resize(height, width)
    return (h // FACTOR) * (w // FACTOR) + MARKERS, (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--requests", type=int, default=1339,
                    help="vllm:request_prompt_tokens_count from the baseline run")
    ap.add_argument("--images", type=int, default=9828,
                    help="vllm:mm_cache_queries_total from the baseline run")
    ap.add_argument("--image-hits", type=int, default=9170,
                    help="vllm:mm_cache_hits_total from the baseline run")
    ap.add_argument("--mean-prompt", type=float, default=20126.0)
    args = ap.parse_args()

    per_request = args.images / args.requests

    print(f"{'MULTIMODAL_UPSCALE':>20}{'rendered':>12}{'after smart_resize':>22}"
          f"{'tokens/image':>14}{'tokens/request':>16}{'% of prompt':>13}")
    for u in (1, 2, 3, 4, 5, 6, 8, 16):
        side = args.grid * u
        n, (h, w) = vision_tokens(side, side)
        total = n * per_request
        flag = "  <- current" if u == 4 else ""
        print(f"{u:>20}{f'{side}x{side}':>12}{f'{h}x{w}':>22}{n:>14}"
              f"{total:>16,.0f}{100 * total / args.mean_prompt:>12.1f}%{flag}")

    print()
    print(f"images per request (server-side, VERIFIED): "
          f"{args.images:,} / {args.requests:,} = {per_request:.2f}")
    print(f"multimodal-cache hit rate: {args.image_hits:,}/{args.images:,} = "
          f"{100 * args.image_hits / args.images:.1f}%")
    print("  -> the vision encoder is not re-running on stale images; they cost")
    print("     KV footprint and prompt tokens, not encoder compute.")
    print()
    print("CONCLUSION: upscale 1-4 are token-identical (min_pixels forces a")
    print("scale-up back to 256x256), so MULTIMODAL_UPSCALE cannot be lowered to")
    print("save anything. The only multimodal lever is dropping images entirely,")
    print("which is a net LOSS -- see experiments/stage7_context_budget.md.")


if __name__ == "__main__":
    main()
