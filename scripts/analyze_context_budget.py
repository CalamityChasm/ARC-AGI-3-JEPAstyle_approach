"""Decompose the Duck/TAAF analyzer request into its token components.

Answers: where do the 21,608 resident tokens per request actually go?

Inputs
------
1. A Duck NVFP4 run's unzipped `kaggle kernels output` directory, which must
   contain `prompts/*.log` (the harness's own "LATEST MODEL CALL SNAPSHOT" of
   the final request for each game), `artifacts/*_events.jsonl` (the real
   boards, used to render the real multimodal images) and
   `vllm-metrics-final.prom` (the ground-truth mean prompt length to check
   against).
2. The model's `tokenizer.json` (Qwen3.8-Flash-Next-NVFP4's own tokenizer),
   so text token counts are exact rather than a chars/N guess.

Usage
-----
    python scripts/analyze_context_budget.py <output_dir> --tokenizer <tokenizer.json> \
        [--json out.json]

Method notes (read before trusting a number)
--------------------------------------------
* `prompts/*.log`'s `[MODEL INPUT]` block is written by the harness itself
  (`tool_agent.py: _render_prompt_log_message`) and renders every message that
  was actually sent, in order. The one thing it drops is the image part of a
  multimodal user message -- it renders only the `{"type": "text"}` part -- so
  images are accounted for separately below.
* Image cost is computed two ways, because the two differ by ~30x and that gap
  is the finding:
    - REAL model cost: Qwen3-VL vision tokens. `preprocessor_config.json` gives
      patch_size=16, merge_size=2 => one vision token per 32x32 px block, plus
      the vision_start / vision_end markers.
    - HARNESS-ESTIMATED cost: `tool_agent.py: _estimate_tokens` is
      `len(json.dumps(payload)) // 3`, and `json.dumps` of a multimodal message
      includes the full base64 `data:image/png;base64,...` URL. That string is
      measured here by re-rendering the real board exactly as
      `vision_context.py: frame_to_png_data_url` does.
* The rendered log is a faithful but not byte-exact stand-in for the wire
  payload (the chat template adds role markup; the tool schema is sent as JSON).
  Both are accounted for; the total is cross-checked against the server's own
  mean prompt length.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
import statistics
import sys

# --------------------------------------------------------------------------
# Constants read from the bundle / model, quoted with source in the write-up
# --------------------------------------------------------------------------

# vision_context.py: ARC_COLOR_MAP
ARC_COLOR_MAP = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}

# preprocessor_config.json: patch_size 16, merge_size 2 -> 32 px per vision token
VISION_PIXELS_PER_TOKEN_SIDE = 32
VISION_MARKER_TOKENS = 2  # vision_start_token_id + vision_end_token_id

# tool_agent.py:149-152
REQUEST_SAFETY_MARGIN_TOKENS = 512
REPLY_RESERVE_TOKENS = 512  # _max_output_tokens is None (MAX_OUTPUT=0) -> 512
DEFAULT_CONTEXT_WINDOW = 32768


def estimate_tokens_harness(value) -> int:
    """Verbatim port of tool_agent.py: _estimate_tokens."""
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, (len(rendered) + 2) // 3)


# --------------------------------------------------------------------------
# Prompt-log parsing
# --------------------------------------------------------------------------

_HEADER = re.compile(
    r"^\[(SYSTEM|USER|ASSISTANT|TOOL RESULT: [^\]]+|REASONING|ASSISTANT TOOL CALL: [^\]]+)\]$"
)


def parse_prompt_log(path):
    """Return (meta, [(kind, text), ...]) for the [MODEL INPUT] block."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    meta = {}
    for key in ("analysis_step", "action", "message_count", "model"):
        m = re.search(rf"^{key}: (.*)$", raw, re.M)
        if m:
            meta[key] = m.group(1).strip()

    # [AVAILABLE TOOLS] ... [MODEL INPUT] ... [TURN TRANSCRIPT SO FAR]
    start = raw.index("[MODEL INPUT]") + len("[MODEL INPUT]")
    end = raw.index("\n[TURN TRANSCRIPT SO FAR]")
    body = raw[start:end]

    sections = []
    cur_kind = None
    cur_lines = []
    for line in body.split("\n"):
        m = _HEADER.match(line.strip())
        if m:
            if cur_kind is not None:
                sections.append((cur_kind, "\n".join(cur_lines).strip()))
            tag = m.group(1)
            if tag.startswith("TOOL RESULT"):
                cur_kind = "tool_result"
            elif tag.startswith("ASSISTANT TOOL CALL"):
                cur_kind = "tool_call"
            else:
                cur_kind = tag.lower()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_kind is not None:
        sections.append((cur_kind, "\n".join(cur_lines).strip()))

    # An [ASSISTANT] header immediately followed by [REASONING] produces an
    # empty "assistant" section; fold those away.
    sections = [(k, t) for k, t in sections if t or k == "assistant"]
    return meta, sections


# --------------------------------------------------------------------------
# The repeated per-turn user boilerplate
# --------------------------------------------------------------------------

def split_user_boilerplate(user_texts):
    """Split each user message into (variable_prefix, shared_boilerplate_lines).

    The harness rebuilds the same instruction block on every turn. Find the
    lines that appear in *every* user message of this game and treat them as
    fixed per-turn overhead; the rest is genuinely new information.
    """
    if not user_texts:
        return [], set()
    line_sets = [set(t.split("\n")) for t in user_texts]
    common = set.intersection(*line_sets)
    # Drop trivial lines
    common = {ln for ln in common if len(ln.strip()) > 20}
    out = []
    for t in user_texts:
        shared, uniq = [], []
        for ln in t.split("\n"):
            (shared if ln in common else uniq).append(ln)
        out.append(("\n".join(uniq), "\n".join(shared)))
    return out, common


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def render_data_url(board, upscale):
    """Verbatim behaviour of vision_context.py: frame_to_png_data_url."""
    from PIL import Image

    rows = len(board)
    cols = max((len(r) for r in board), default=0)
    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for r_i, row in enumerate(board):
        for c_i in range(cols):
            v = row[c_i] if c_i < len(row) else 0
            pixels[c_i, r_i] = ARC_COLOR_MAP.get(int(v), ARC_COLOR_MAP[0])
    if upscale > 1:
        image = image.resize((cols * upscale, rows * upscale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return (
        "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
        image.size,
    )


def vision_tokens(size):
    w, h = size
    return (w // VISION_PIXELS_PER_TOKEN_SIDE) * (
        h // VISION_PIXELS_PER_TOKEN_SIDE
    ) + VISION_MARKER_TOKENS


def sample_boards(events_path, limit):
    """Return up to `limit` distinct boards from the tail of a game's events."""
    boards = []
    with open(events_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for line in lines[-limit:]:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        b = d.get("board")
        if b:
            boards.append(b)
    return boards


# --------------------------------------------------------------------------
# Ground truth from the server
# --------------------------------------------------------------------------

def prom_mean_prompt(path):
    total = count = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("vllm:prompt_tokens_total"):
                total = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:request_prompt_tokens_sum"):
                total = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:request_prompt_tokens_count"):
                count = float(line.rsplit(" ", 1)[1])
    if total and count:
        return total / count, total, count
    return None, total, count


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)

    def ntok(text):
        if not text:
            return 0
        return len(tok.encode(text, add_special_tokens=False).ids)

    root = args.output_dir
    logs = sorted(glob.glob(os.path.join(root, "prompts", "*.log")))
    if not logs:
        sys.exit(f"no prompts/*.log under {root}")

    per_game = []
    for log in logs:
        game = os.path.basename(log).replace(".log", "")
        meta, sections = parse_prompt_log(log)

        buckets = {
            "system": 0,
            "user_boilerplate": 0,
            "user_variable": 0,
            "assistant_reasoning": 0,
            "assistant_toolcall": 0,
            "tool_result": 0,
        }
        counts = {"user": 0, "assistant": 0, "tool_result": 0}

        user_texts = [t for k, t in sections if k == "user"]
        split, _common = split_user_boilerplate(user_texts)
        ui = 0
        for kind, text in sections:
            if kind == "system":
                buckets["system"] += ntok(text)
            elif kind == "user":
                uniq, shared = split[ui]
                ui += 1
                counts["user"] += 1
                buckets["user_variable"] += ntok(uniq)
                buckets["user_boilerplate"] += ntok(shared)
            elif kind == "reasoning":
                buckets["assistant_reasoning"] += ntok(text)
            elif kind == "tool_call":
                buckets["assistant_toolcall"] += ntok(text)
                counts["assistant"] += 1
            elif kind == "tool_result":
                buckets["tool_result"] += ntok(text)
                counts["tool_result"] += 1

        n_images = counts["user"]

        # Real board -> real image cost, measured not assumed
        ev = os.path.join(root, "artifacts", f"{game}_events.jsonl")
        img_real = img_b64_chars = 0
        img_px = None
        if os.path.exists(ev):
            boards = sample_boards(ev, n_images)
            for b in boards:
                url, size = render_data_url(b, args.upscale)
                img_px = size
                img_real += vision_tokens(size)
                img_b64_chars += len(url)
            if boards and len(boards) < n_images:
                scale = n_images / len(boards)
                img_real = int(img_real * scale)
                img_b64_chars = int(img_b64_chars * scale)

        per_game.append(
            {
                "game": game,
                "message_count": int(meta.get("message_count", 0) or 0),
                "n_user_turns": n_images,
                **buckets,
                "image_real_tokens": img_real,
                "image_b64_chars": img_b64_chars,
                "image_harness_estimate": img_b64_chars // 3,
                "image_px": img_px,
            }
        )

    # ---- report -----------------------------------------------------------
    keys = [
        "system",
        "user_boilerplate",
        "user_variable",
        "assistant_reasoning",
        "assistant_toolcall",
        "tool_result",
        "image_real_tokens",
    ]
    means = {k: statistics.mean(g[k] for g in per_game) for k in keys}
    text_total = sum(means[k] for k in keys if k != "image_real_tokens")
    grand = text_total + means["image_real_tokens"]

    print("=" * 78)
    print("PER-REQUEST TOKEN DECOMPOSITION  (mean over the 25 final-request snapshots)")
    print("=" * 78)
    print(f"{'component':<28}{'tokens':>10}{'share':>9}   note")
    order = [
        ("system", "system prompt (1x, static)"),
        ("user_boilerplate", "per-turn instruction block, REPEATED VERBATIM"),
        ("user_variable", "genuinely new per-turn info (state, world model)"),
        ("assistant_reasoning", "retained model reasoning"),
        ("assistant_toolcall", "retained python code"),
        ("tool_result", "retained tool stdout"),
        ("image_real_tokens", "Qwen3-VL vision tokens for the retained images"),
    ]
    for k, note in order:
        print(f"{k:<28}{means[k]:>10.0f}{100*means[k]/grand:>8.1f}%   {note}")
    print(f"{'TOTAL (this method)':<28}{grand:>10.0f}{100.0:>8.1f}%")

    mean_prompt, tot, cnt = prom_mean_prompt(os.path.join(root, "vllm-metrics-final.prom"))
    if mean_prompt:
        print()
        print(f"server ground truth: mean prompt = {mean_prompt:,.0f} tokens "
              f"({tot:,.0f} / {cnt:,.0f} requests)")
        print(f"this decomposition   = {grand:,.0f}  "
              f"({100*grand/mean_prompt:.0f}% of ground truth)")
        print("  (final-request snapshots are the deepest-history requests of each")
        print("   game, so they sit above the run-wide mean by construction)")

    print()
    print("=" * 78)
    print("THE IMAGE ACCOUNTING GAP")
    print("=" * 78)
    px = next((g["image_px"] for g in per_game if g["image_px"]), None)
    mb64 = statistics.mean(g["image_b64_chars"] for g in per_game)
    mest = statistics.mean(g["image_harness_estimate"] for g in per_game)
    mreal = means["image_real_tokens"]
    n_img = statistics.mean(g["n_user_turns"] for g in per_game)
    print(f"images retained per request (mean)      : {n_img:.1f}")
    print(f"rendered image size                     : {px}")
    print(f"base64 data-URL chars per request       : {mb64:,.0f}")
    print(f"  -> harness _estimate_tokens (chars/3) : {mest:,.0f} 'tokens'")
    print(f"  -> real Qwen3-VL vision tokens        : {mreal:,.0f} tokens")
    print(f"  -> OVERCOUNT FACTOR                   : {mest/max(1,mreal):.1f}x")
    budget = DEFAULT_CONTEXT_WINDOW - REPLY_RESERVE_TOKENS - REQUEST_SAFETY_MARGIN_TOKENS
    print(f"harness context budget                  : {budget:,} tokens")
    print(f"  phantom share of that budget          : {100*mest/budget:.1f}%")

    print()
    print("=" * 78)
    print("PER GAME")
    print("=" * 78)
    hdr = f"{'game':<18}{'msgs':>5}{'usr':>5}{'sys':>7}{'boiler':>8}{'var':>7}{'reason':>8}{'code':>7}{'toolout':>8}{'img':>6}{'b64est':>8}"
    print(hdr)
    for g in sorted(per_game, key=lambda x: -x["message_count"]):
        print(
            f"{g['game'][:17]:<18}{g['message_count']:>5}{g['n_user_turns']:>5}"
            f"{g['system']:>7}{g['user_boilerplate']:>8}{g['user_variable']:>7}"
            f"{g['assistant_reasoning']:>8}{g['assistant_toolcall']:>7}"
            f"{g['tool_result']:>8}{g['image_real_tokens']:>6}{g['image_harness_estimate']:>8}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "per_game": per_game,
                    "means": means,
                    "total_mean": grand,
                    "server_mean_prompt": mean_prompt,
                    "image_b64_chars_mean": mb64,
                    "image_harness_estimate_mean": mest,
                    "context_budget": budget,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
