"""One-time Procgen corpus generator -- run from inside a **dedicated
Python 3.10 venv**, not this project's main venv. See
`jepa/data/procgen_data.py`'s module docstring for the full reasoning:
Procgen has no PyPI wheel for Python 3.13 (confirmed against PyPI's file
listing -- newest available wheels, procgen 0.10.7, top out at `cp310`),
so this script's *output* (a pickled transitions list + a saved palette),
not `procgen` itself, is what the main training venv ever consumes.

Dedicated-venv setup used on the dev box this was built on (repeat on a
new machine if this cache needs regenerating):
    1. Install Python 3.10 (e.g. from
       https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
       -- silent/no-admin: `<installer>.exe /quiet InstallAllUsers=0
       PrependPath=0 TargetDir=<dir>`).
    2. <dir>\\python.exe -m venv C:\\pgvenv   (a short path -- a long/nested
       venv path hit a real Windows long-path pip install failure with
       numpy's bundled OpenBLAS .dll during development; C:\\pgvenv sidesteps
       it entirely, not worth chasing the long-paths registry fix for a
       throwaway venv).
    3. C:\\pgvenv\\Scripts\\python.exe -m pip install procgen numpy

Usage (from the repo root, using the dedicated venv's interpreter):
    C:\\pgvenv\\Scripts\\python.exe scripts\\generate_procgen_corpus.py \\
        --envs maze,heist --steps-per-env 33600 --out data/procgen_corpus.pkl

Writes:
    <out>                              pickled list of
                                        (frame_t, action_id, x, y, frame_t1,
                                        changed, game_id) tuples, exactly
                                        the shape TransitionDataset expects.
    <out>.palette.npy                  the (16, 3) float32 RGB palette used
                                        for quantization (kept for
                                        documentation/reproducibility, not
                                        needed to *load* the cache).
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from jepa.data.procgen_data import DEFAULT_ENVS, fit_palette, generate_transitions, sample_palette_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, default=",".join(DEFAULT_ENVS))
    parser.add_argument("--steps-per-env", type=int, default=33_600)
    parser.add_argument("--num-parallel", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--palette-frames-per-env", type=int, default=400)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "procgen_corpus.pkl")
    args = parser.parse_args()

    envs = args.envs.split(",")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"fitting a shared 16-color palette on {args.palette_frames_per_env}/env sample frames "
          f"across {envs}...")
    palette_frames = sample_palette_frames(envs, n_frames_per_env=args.palette_frames_per_env, seed=args.seed)
    palette = fit_palette(palette_frames, seed=args.seed)
    palette_path = args.out.with_suffix(args.out.suffix + ".palette.npy")
    np.save(palette_path, palette)
    print(f"  saved palette ({palette.shape}) to {palette_path}")

    print(f"generating transitions: envs={envs} steps_per_env={args.steps_per_env} "
          f"num_parallel={args.num_parallel} seed={args.seed}")
    transitions = generate_transitions(
        envs=envs,
        steps_per_env=args.steps_per_env,
        num_parallel=args.num_parallel,
        palette=palette,
        seed=args.seed,
    )
    print(f"  generated {len(transitions)} total transitions")

    # Sanity checks before ever trusting this data downstream -- CLAUDE.md's
    # own gotcha (Sokoban's action-id-9 CUDA crash) is exactly "check this
    # BEFORE a training run, not after a confusing CUDA assert."
    action_ids = [t[1] for t in transitions]
    action_counts = Counter(action_ids)
    print(f"  action id distribution: {dict(sorted(action_counts.items()))}")
    assert max(action_ids) < 8, f"action id {max(action_ids)} >= NUM_ACTIONS=8 -- would crash training"
    assert min(action_ids) >= 0

    game_id_counts = Counter(t[6] for t in transitions)
    print(f"  per-game_id counts: {dict(game_id_counts)}")

    n_changed = sum(1 for t in transitions if t[5])
    print(f"  frame-level changed rate: {n_changed}/{len(transitions)} = {n_changed / len(transitions):.1%}")

    with open(args.out, "wb") as f:
        pickle.dump(transitions, f)
    print(f"saved {len(transitions)} transitions to {args.out}")


if __name__ == "__main__":
    main()
