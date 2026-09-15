"""Measure the on-disk size of candidate Kaggle Model mounts.

Why size, specifically
----------------------
The kernel has a 95 GiB card and runs offline, so a model is only a candidate
if its *served* weights plus a workable KV pool fit. The two reference points,
both read from real run logs by ``scripts/analyze_model_tradeoff.py``:

* ``keithtyser/qwen3-8-flash-next-nvfp4`` -- 135.25 GB on disk, but only
  **81.8 GiB resident**, because the fork's bundle CPU-offloads the FP8 PLE
  embedding layers. Disk size therefore over-states VRAM need for this one
  model, and only for it: the offload lives in a sealed third-party bundle
  (see ``experiments/stage7_model_search_artifacts/nvfp4_bundle_model_pin.txt``)
  that cannot be pointed at any other checkpoint.
* ``foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`` -- 28.95 GiB resident.

For every other candidate, assume resident ~= on-disk, because we would be
serving it from our own stock-vLLM stack with no offload path.

``models/<ref>/get`` does not report bytes. The per-instance file listing does,
so this walks ``.../files`` per instance and sums ``totalBytes``. That endpoint
pages, and a 400-file NVFP4 checkpoint needs several pages -- forgetting the
cursor silently under-counts by an order of magnitude, which is exactly the
kind of confident-wrong number this project has paid for before, so the page
count is reported alongside the total.

Usage
-----
    venv/Scripts/python.exe scripts/size_kaggle_models.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

# ref -> short note used in the write-up's candidate table.
CANDIDATES: dict[str, str] = {
    "keithtyser/qwen3-8-flash-next-nvfp4": "INCUMBENT: Qwen3.8-Flash-Next, 125B MoE/6B active, NVFP4",
    "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1": "PRIOR: Qwen3.8-27B dense FP8 (our 2.57 stack)",
    "google/gemma-4": "Gemma-4 family (31b-it used by two LB teams)",
    "impactganyu/qwen38-27b-radixark-nvfp4": "Qwen3.8-27B dense at NVFP4",
    "michaelpoluektov/qwen3-8-27b-nvfp4": "Qwen3.8-27B dense at NVFP4 (alt)",
    "woochangsim/qwen38-flash-next-w4a16-autoround-4c67bf6": "Flash-Next at W4A16 (AutoRound)",
    "ram2121/qwen3-8-flash-next-gptq-4bit": "Flash-Next at GPTQ-4bit",
    "surasan092/qwen3-5-122b-a10b-nvfp4": "Qwen3.5-122B-A10B NVFP4",
    "barnobarno/nvidia-nemotron-3-super-120b-a12b-nvfp4": "Nemotron-3-Super-120B-A12B NVFP4",
    "pranshubahadur/deepseek-v3.2-reap-4bf-86b-a10b-sgptq14-w4a16": "DeepSeek-V3.2-REAP-86B-A10B W4A16",
    "kekshibata/qwen3-next-80b-awq-4bit": "Qwen3-Next-80B-A3B-Instruct AWQ4",
    "konstantinboyko/qwen3-next-80b-a3b-thinking-awq-4bit-cpatonn": "Qwen3-Next-80B-A3B-Thinking AWQ4",
    "russcore/glm53-flash-nvfp4-redhatai-240131d6": "GLM-5.3-Flash 320B/18B NVFP4",
    "qwen-lm/qwen3-coder-next": "Qwen3-Coder-Next (official)",
    "michaelpoluektov/qwen3-6-35b-a3b-nvfp4": "Qwen3.6-35B-A3B NVFP4",
    "cryptozenith/qwen-27b-nvfp4": "Qwen-27B NVFP4",
    "qwen-lm/qwen-3-5": "Qwen3.5 family (official)",
}

API = "https://www.kaggle.com/api/v1"


def creds() -> tuple[str, str]:
    for path in (Path.home() / ".kaggle" / "kaggle.json",
                 Path.home() / ".kaggle" / "credentials.json"):
        if path.is_file():
            blob = json.loads(path.read_text(encoding="utf-8"))
            return blob["username"], blob["key"]
    raise SystemExit("no Kaggle credentials found under ~/.kaggle")


def instance_bytes(auth, owner, slug, framework, inst, version) -> tuple[int, int, int]:
    """(total_bytes, n_files, n_pages) summed across every page of the listing."""
    url = f"{API}/models/{owner}/{slug}/{framework}/{inst}/{version}/files"
    total = files = pages = 0
    token = None
    while True:
        params = {"page_size": 200}
        if token:
            params["page_token"] = token
        resp = requests.get(url, auth=auth, params=params, timeout=90)
        if resp.status_code != 200:
            return -resp.status_code, files, pages
        body = resp.json()
        for entry in body.get("files", []) or []:
            size = entry.get("totalBytes") or entry.get("size") or 0
            total += int(size)
            files += 1
        pages += 1
        token = body.get("nextPageToken") or None
        if not token or pages > 60:
            break
    return total, files, pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--only", default="", help="comma-separated substring filter on refs")
    args = parser.parse_args()
    auth = creds()
    only = [s for s in args.only.split(",") if s]

    out = []
    for ref, note in CANDIDATES.items():
        if only and not any(s in ref for s in only):
            continue
        owner, slug = ref.split("/", 1)
        meta = requests.get(f"{API}/models/{ref}/get", auth=auth, timeout=60)
        if meta.status_code != 200:
            print(f"{meta.status_code:>4} {ref}  (not accessible)")
            out.append({"ref": ref, "note": note, "http": meta.status_code})
            continue
        instances = meta.json().get("instances") or []
        print(f" 200 {ref}  [{note}]  {len(instances)} instance(s)")
        for entry in instances:
            framework = (entry.get("framework") or "").lower()
            inst = entry.get("instanceSlug") or entry.get("slug")
            version = entry.get("versionNumber") or 1
            if not inst:
                continue
            total, files, pages = instance_bytes(auth, owner, slug, framework, inst, version)
            if total < 0:
                print(f"        {framework}/{inst}/{version}  HTTP {-total}")
                continue
            gib = total / 2 ** 30
            print(f"        {framework}/{inst}/{version:<3} {gib:9.2f} GiB  "
                  f"{files:>4d} files  {pages} page(s)")
            out.append({
                "ref": ref, "note": note, "framework": framework, "instance": inst,
                "version": version, "bytes": total, "gib": round(gib, 2),
                "files": files, "pages": pages,
                "mount": f"{ref}/{framework}/{inst}/{version}",
            })

    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
