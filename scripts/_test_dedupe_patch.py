"""Replay the dedupe patch over the baseline run's REAL messages before spending GPU time.

The patch body lives inside a notebook cell, which is exactly the kind of code
that never gets tested until it fails 25 minutes into a run. This extracts that
body from the generated notebook, executes it verbatim against a stub
`inference.agent.tool_agent`, then feeds it the 25 real request snapshots
reconstructed from `prompts/*.log` and checks two things:

  1. the invariant the patch claims -- every line it removes from a historical
     user message is present verbatim in the newest user message, so the model
     still reads it this turn. Nothing is lost, only de-duplicated.
  2. the saving it claims, in real tokenizer tokens.

Usage:
    python scripts/_test_dedupe_patch.py <baseline_output_dir> --tokenizer <tokenizer.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_context_budget import parse_prompt_log  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "kaggle_submission_duck_nvfp4" / "notebook" / "duck-qwen3-8-anim-base.ipynb"
DATA = REPO / "experiments" / "stage7_context_budget_data.json"

# One real 256x256 grid PNG measured on the baseline run is ~1,580 base64 chars
# per image (13,590 chars across 8.6 images); the exact value does not matter to
# the invariant, only to the reported saving.
B64_CHARS_PER_IMAGE = 1580


def load_patch_body():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][14]["source"])
    marker = "# [calamitychasm] stage7-context-budget"
    if marker not in src:
        raise SystemExit("cell 14 carries no stage7-context-budget patch; generate it first")
    body = src[src.index(marker):]
    if "_dd_lean" not in body:
        raise SystemExit("cell 14 carries a different candidate patch, not the dedupe one")
    return body


def exec_patch(body):
    fake = types.ModuleType("inference.agent.tool_agent")

    class ToolAgent:
        def _chat_completion(self, messages, **kw):
            return ("SENT", messages)

    fake.ToolAgent = ToolAgent
    sys.modules.setdefault("inference", types.ModuleType("inference"))
    sys.modules.setdefault("inference.agent", types.ModuleType("inference.agent"))
    sys.modules["inference.agent.tool_agent"] = fake
    ns = {"json": json, "os": os, "__name__": "nbcell"}
    exec(compile(body, "<patch>", "exec"), ns)
    return ns


def rebuild_messages(sections):
    msgs = []
    for kind, text in sections:
        if kind == "system":
            msgs.append({"role": "system", "content": text})
        elif kind == "user":
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + "X" * B64_CHARS_PER_IMAGE}},
                ],
            })
        elif kind in ("reasoning", "tool_call"):
            msgs.append({"role": "assistant", "content": text})
        elif kind == "tool_result":
            msgs.append({"role": "tool", "content": text})
    return msgs


def text_of(message):
    c = message.get("content")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    ntok = lambda t: len(tok.encode(t, add_special_tokens=False).ids) if t else 0

    ns = exec_patch(load_patch_body())
    lean_fn = ns["_dd_lean"]
    print("patch body executed; its own self-probe assertions passed")

    data = json.loads(DATA.read_text(encoding="utf-8"))
    bg = {g["game"]: g for g in data["per_game"]}

    before, after, violations, stale_vision = [], [], [], []
    logs = sorted(glob.glob(os.path.join(args.output_dir, "prompts", "*.log")))
    for log in logs:
        game = os.path.basename(log).replace(".log", "")
        _meta, sections = parse_prompt_log(log)
        msgs = rebuild_messages(sections)
        lean = lean_fn(msgs)

        assert len(lean) == len(msgs), "message count changed"
        user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        newest_lines = set(text_of(msgs[user_idx[-1]]).split("\n"))

        for i, (a, b) in enumerate(zip(msgs, lean)):
            if a["role"] != "user" or i == user_idx[-1]:
                assert a == b, f"{game}: non-target message {i} was modified"
                continue
            removed = [ln for ln in text_of(a).split("\n") if ln not in text_of(b).split("\n")]
            for ln in removed:
                if ln.strip() and ln not in newest_lines:
                    violations.append((game, i, ln[:70]))

        before.append(sum(ntok(text_of(m)) for m in msgs))
        after.append(sum(ntok(text_of(m)) for m in lean))
        g = bg[game]
        n = max(1, g["n_user_turns"])
        stale_vision.append(g["image_real_tokens"] * (n - 1) / n)

    mb, ma, mv = statistics.mean(before), statistics.mean(after), statistics.mean(stale_vision)
    total = (mb - ma) + mv
    print(f"replayed {len(logs)} real request snapshots")
    print(f"  text tokens before      : {mb:,.0f}")
    print(f"  text tokens after       : {ma:,.0f}")
    print(f"  text saved              : {mb - ma:,.0f}")
    print(f"  stale vision tokens cut : {mv:,.0f}")
    print(f"  TOTAL real tokens saved : {total:,.0f}  ({100 * total / 20126:.1f}% of the 20,126 prompt)")
    print(f"  invariant violations    : {len(violations)}")
    for v in violations[:10]:
        print("    ", v)
    if violations:
        raise SystemExit("FAIL: a line was removed that the model is not reading elsewhere")
    print("PASS: every removed line is present verbatim in the newest user message")


if __name__ == "__main__":
    main()
