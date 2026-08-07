"""Move (or restore) the 5 held-out games' recording files out of
ARC-AGI-3-Agents/recordings/ so jepa.train_value_head's load_value_targets
(which has no exclude-games filter) doesn't see them.

Usage:
  python move_holdout_recordings.py out   # move holdout files to temp dir
  python move_holdout_recordings.py back  # move them back
"""
import sys
import shutil
from pathlib import Path

RECORDINGS = Path(r"C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\ARC-AGI-3-Agents\recordings")
STASH = Path(r"C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\ARC-AGI-3-Agents\recordings_holdout_stash")
GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

def main():
    mode = sys.argv[1]
    if mode == "out":
        STASH.mkdir(exist_ok=True)
        moved = 0
        for f in RECORDINGS.iterdir():
            if f.is_file() and any(f.name.startswith(g + "-") for g in GAMES):
                shutil.move(str(f), str(STASH / f.name))
                moved += 1
        print(f"moved {moved} files to stash")
    elif mode == "back":
        moved = 0
        if STASH.exists():
            for f in STASH.iterdir():
                shutil.move(str(f), str(RECORDINGS / f.name))
                moved += 1
        print(f"restored {moved} files from stash")
    else:
        raise SystemExit("usage: move_holdout_recordings.py out|back")

if __name__ == "__main__":
    main()
