"""Stage 5: value-target extraction from local ARC-3 recordings, for
training the decoupled value head (architecture.md's "Decoupled Heads"
component -- `V(s)` estimates expected future progress, kept separate
from the dynamics predictor, same rationale as DIAMOND's decoupled
reward/termination networks per architecture.md).

Target: discounted Monte-Carlo return using `levels_completed` increases
as the only reward signal -- the harness gives us no other score signal
at this granularity, so a level-up is reward=1, everything else is 0.
`GAMMA` close to 1 since episodes are short (~80 steps) and level-ups are
rare; a low gamma would make the value target ~0 almost everywhere.
"""

from pathlib import Path

from .trajectories import RECORDINGS_DIR, _load_frame_lines

GAMMA = 0.95


def load_value_targets(repo_root: Path) -> list:
    """Returns a list of `(frame, value_target, game_id)` tuples. `frame`
    is a `FrameData.frame` (list of one layer), matching
    `arc3_frame_to_tensor`'s expected input directly."""
    recordings_dir = repo_root / RECORDINGS_DIR
    samples = []
    for path in sorted(recordings_dir.glob("*.recording.jsonl")):
        frames = _load_frame_lines(path)
        if not frames:
            continue
        n = len(frames)
        rewards = [0.0] * n
        for i in range(1, n):
            if frames[i].get("levels_completed", 0) > frames[i - 1].get("levels_completed", 0):
                rewards[i] = 1.0

        returns = [0.0] * n
        running = 0.0
        for i in range(n - 1, -1, -1):
            running = rewards[i] + GAMMA * running
            returns[i] = running

        game_id = frames[0].get("game_id", "unknown")
        for i in range(n):
            if frames[i].get("frame"):
                samples.append((frames[i]["frame"], returns[i], game_id))
    return samples
