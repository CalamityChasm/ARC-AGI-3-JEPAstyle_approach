"""`GraphExplorerAgent` (the training-free reproduction of Rudakov, Shock &
Cowley, arXiv:2512.24156 -- see `graph_explorer_agent.py`'s own module
docstring for the full port/attribution story) layered with this project's
own JEPA world model.

**Scope, stated upfront:** this exists to answer a narrower question than
"replace the graph agent with the JEPA agent" or vice versa -- both of
those already exist and are already compared in CLAUDE.md. The question
here is: can the JEPA world model make the *coverage-first,
frontier-distance-routing* algorithm that made `GraphExplorerAgent` work
make *better* choices at the specific points where it currently has no
basis for choice beyond "pick uniformly at random" -- without touching the
algorithm's own priority-tier/frontier-distance discipline, which is what
the local backtest evidence (see CLAUDE.md's "GraphExplorerAgent" section)
says is actually doing the work.

**What changed, precisely:** `graph_explorer_core.py: GraphExplorer` grew
an optional `tie_break_fn(node, mode, candidates) -> edge_idx` hook, called
from `choose_edge` in place of `random.choice` when set (`None` by default
-- exactly reproduces upstream's own behavior, verified by the existing
unit tests). This class supplies that hook for both regimes `choose_edge`
can be in:
- `mode="explore"` (picking which *untested* action to try next within the
  current priority tier): score each candidate via
  `jepa/hypothesis_bundle.py: info_gain` -- the same expert-disagreement
  signal Stage 5's `Hypothesis` agent already uses -- and sample among them
  weighted toward the one the world model is most uncertain about.
- `mode="travel"` (navigating toward a known frontier once the current
  node has nothing left to explore): among edges already confirmed
  successful and tied for the shortest known distance, sample weighted
  toward whichever destination the value head rates most promising. Needs
  a frame cache (`self._frame_cache: dict[hash -> pixel array]`), since
  `GraphExplorer` itself only ever stores hashes, never pixel data (by
  design, to stay lightweight, matching upstream) -- populated here as a
  side effect of every `tie_break_fn` call (every frame we ever actually
  stand on gets cached the moment we make a decision from it).

The graph engine still decides *which set* of actions is even eligible in
either mode; this only re-ranks within that set.

**A real bug found and fixed in the first draft of this file, via direct
evidence, not guesswork.** The original `explore`-mode scoring used a hard
`argmax` over InfoGain. A live n=1 sweep on `r11l` showed the JEPA variant
completing fewer levels than the pure agent (4 vs. 5-7) and hit the
graph's "choose_edge called on a still-unconfirmed node" failure mode far
more often (179/301 actions vs. 0 for the pure agent on the same game).
The original writeup speculated this meant InfoGain was systematically
*seeking out* disruptive/game-ending actions. Directly checked instead of
left as speculation: instrumented `_jepa_tie_break` with DEBUG-gated
per-decision logging (`.env`'s `DEBUG=True`), replayed `r11l`, and found
the real story is different -- across 89 real explore-mode decisions,
candidate InfoGain scores clustered within roughly 0.0002-0.0006 of each
other on a ~0.017 baseline (i.e. **near-flat**, not meaningfully
differentiated), and the *chosen* click location landed on just 2 of 39
distinct spots 22/89 times (~25%). This is the exact "deterministic
argmax over a flat/near-flat map defaults to the same handful of indices"
failure this project has already hit and fixed twice before --
`Curiosity._sample_click` and `Hypothesis`'s `PATCH_SAMPLE_TEMPERATURE` --
just newly discovered in a third context (exploration-order tie-breaking,
not click-pixel selection). **Fixed the same way**: replaced hard argmax
with temperature-weighted softmax sampling (`_weighted_sample`,
`TIE_BREAK_TEMPERATURE`), z-normalizing scores first so the temperature is
scale-invariant regardless of `info_gain`'s or the value head's raw output
magnitude for a given checkpoint.

**Real n=8x25 backtest result (see CLAUDE.md for the full table and
analysis): a clear, decisive regression, not an improvement.** 34 total
levels / 6 distinct games / mean pooled score 0.121, vs. the pure
`GraphExplorerAgent`'s 46 / 8 / 0.242 on identical protocol. Every single
one of the 8 repeats scored below the pure agent's *worst* repeat --
directly checked and ruled out "more crashes/fallbacks" as the cause
(this class actually had *fewer* total fallback events across the 8
repeats, 2796 vs. 5025). The working (not yet directly verified) read:
this algorithm's strength is broad, roughly-uniform coverage within a
fixed action budget, and any systematic per-step bias in *which* untested
action gets tried first -- even a well-motivated, non-degenerate one --
changes *when* within an episode a GAME_OVER-triggering action gets hit,
which can mean earlier resets and less depth reached per episode, a cost
this class's own per-decision metrics don't capture. **Do not treat this
class as an improvement over the pure agent, and do not submit it** --
see CLAUDE.md's "Recommended next steps" for what a follow-up attempt
should try differently (most likely: bias *away* from high-uncertainty
candidates, or drop per-step bias entirely in favor of an exact-recall/
exploit layer instead).

**Remaining known limitations** (real, not hidden):
- No exact-recall/exploit-on-win bias is added on top of the graph's own
  coverage-first behavior (unlike Stage 3/5's `TransitionGraph`-based
  agents, which explicitly recall known winning paths). The graph here
  already reaches a known-good edge deterministically once discovered
  (it becomes a "travel" edge from then on, now with value-head-informed
  routing); an *explicit* preference for exploiting it over continuing
  exploration is future work.
- No held-out-game novelty handling (unlike `Hypothesis`'s
  `NOVELTY_BETA_CAP`) -- an unseen `game_id` silently falls back to
  embedding index 0, same as every other JEPA-based agent in this project
  before that cap was added.
- Travel-mode scoring falls back to uniform random among any candidates
  whose destination frame isn't yet cached (can happen if that node was
  only ever reached via `_choose_action_inner`'s concurrent-modification
  defensive path, never as this agent's own current position -- rare, not
  directly measured how rare).

If JEPA model construction/loading fails for any reason, this degrades
gracefully to exactly `GraphExplorerAgent`'s own already-verified behavior
(`tie_break_fn` simply stays unset).
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from arcengine import GameAction

from .graph_explorer_agent import GraphExplorerAgent

logger = logging.getLogger()

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.device import get_device  # noqa: E402
from jepa.grid import grid_to_tensor  # noqa: E402
from jepa.hypothesis_bundle import info_gain  # noqa: E402
from jepa.models import CNNEncoder, MoEPredictor, ValueHead  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"


class GraphExplorerJepaAgent(GraphExplorerAgent):
    """See module docstring. Registered as `graphexplorerjepaagent`."""

    # Not swept -- a starting point deliberately between near-argmax (low
    # temperature, reintroduces the near-flat-signal concentration bug
    # this class exists to fix) and near-uniform (high temperature, throws
    # away genuine signal when a real gap exists). Scores are z-normalized
    # before this is applied (see _weighted_sample), so this value is
    # scale-invariant across info_gain's and the value head's differing
    # raw output magnitudes.
    TIE_BREAK_TEMPERATURE = 0.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._jepa_ready = False
        self._frame_cache: dict[str, np.ndarray] = {}
        try:
            self._init_jepa_models()
            self.graph_explorer.tie_break_fn = self._jepa_tie_break
            self._jepa_ready = True
        except Exception:
            logger.exception(
                f"{self.game_id} - GraphExplorerJepaAgent: JEPA model init "
                "failed, falling back to plain GraphExplorerAgent behavior"
            )

    def _init_jepa_models(self) -> None:
        # Mirrors hypothesis_agent.py's own _init_models loading pattern
        # exactly (same checkpoint files, same device helper) -- no reason
        # to load these any differently here.
        self.device = get_device()

        self.encoder = CNNEncoder().to(self.device)
        self.encoder.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "encoder_moe.pt", map_location=self.device)
        )
        self.encoder.eval()

        game_vocab: dict[str, int] = {}
        vocab_path = _CHECKPOINT_DIR / "game_vocab_moe.json"
        if vocab_path.exists():
            game_vocab = json.loads(vocab_path.read_text())
        self.game_idx = game_vocab.get(self.game_id, 0)
        num_games = max(len(game_vocab), 1)

        self.predictor = MoEPredictor(num_games=num_games, num_experts=8).to(self.device)
        self.predictor.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "moe_predictor.pt", map_location=self.device)
        )
        self.predictor.eval()

        self.value_head = ValueHead().to(self.device)
        value_path = _CHECKPOINT_DIR / "value_head.pt"
        if value_path.exists():
            self.value_head.load_state_dict(torch.load(value_path, map_location=self.device))
        else:
            logger.warning(
                f"{self.game_id} - no value_head.pt found at {value_path}, "
                "using an untrained value head for travel-mode scoring"
            )
        self.value_head.eval()

    @torch.no_grad()
    def _encode_frame(self, frame_np: np.ndarray) -> torch.Tensor:
        tensor = grid_to_tensor(frame_np)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return self.encoder(x)  # (1, C, 8, 8)

    def _encode_current_frame(self) -> torch.Tensor:
        # self._current_frame_np is stashed by GraphExplorerAgent._choose_
        # action_inner right before it calls choose_edge -- see that
        # method's own comment. It's the status-bar-masked (64, 64) grid
        # for the frame we're currently standing on, already the same
        # array the graph itself just hashed.
        return self._encode_frame(self._current_frame_np)

    @torch.no_grad()
    def _predict_experts(self, feat: torch.Tensor, action_id: int, xy: tuple[int, int] | None) -> torch.Tensor:
        b = feat.shape[0]
        action_t = torch.full((b,), action_id, dtype=torch.long, device=self.device)
        if xy is not None:
            x, y = xy
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=self.device)
        else:
            xy_t = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
        game_t = torch.full((b,), self.game_idx, dtype=torch.long, device=self.device)
        return self.predictor.predict_all_experts(feat, action_t, xy_t, game_t)[0]  # (K, C, 8, 8)

    def _edge_to_action_xy(self, edge_idx: int) -> tuple[int, tuple[int, int] | None]:
        """Maps a GraphExplorer edge index back to a real (action_id, xy)
        pair the predictor can condition on, using the frame-segments/
        arrow-actions context GraphExplorerAgent._choose_action_inner
        stashes right before calling choose_edge."""
        if edge_idx < self._current_num_click_actions:
            segment = self._current_frame_segments[edge_idx]
            x1, y1, x2, y2 = segment["bounding_box"]
            # Segment centroid -- a representative point for *scoring*
            # only. The real click point is still sampled uniformly within
            # the segment later in _choose_action_inner, unaffected by
            # this (that logic doesn't know or care how action_id was
            # chosen, only which segment index it names).
            xy = ((x1 + x2) // 2, (y1 + y2) // 2)
            return GameAction.ACTION6.value, xy
        game_action = self._current_arrow_actions[edge_idx - self._current_num_click_actions]
        return game_action.value, None

    def _weighted_sample(self, edge_scores: list[tuple[int, float]]) -> int:
        """Temperature-weighted softmax sample over (edge_idx, score)
        pairs -- see module/class docstring for why this replaced a hard
        argmax. Z-normalizes scores first so TIE_BREAK_TEMPERATURE stays
        meaningful regardless of the raw magnitude of whatever produced
        the scores (info_gain vs. the value head are on different scales
        from each other, and info_gain's own scale varies by checkpoint)."""
        if len(edge_scores) == 1:
            return edge_scores[0][0]
        scores = np.array([s for _, s in edge_scores], dtype=np.float64)
        std = scores.std()
        if std < 1e-9:
            # genuinely flat (or only one distinct value) -- softmax would
            # be exactly uniform anyway; skip straight to it and avoid a
            # division by ~0.
            return edge_scores[random.randrange(len(edge_scores))][0]
        z = (scores - scores.mean()) / std
        logits = z / self.TIE_BREAK_TEMPERATURE
        logits = logits - logits.max()  # numerical stability
        weights = np.exp(logits)
        weights = weights / weights.sum()
        idx = np.random.choice(len(edge_scores), p=weights)
        return edge_scores[idx][0]

    def _explore_tie_break(self, candidates: list) -> int:
        feat = self._encode_current_frame()
        scored = []
        for edge_idx in candidates:
            action_id, xy = self._edge_to_action_xy(edge_idx)
            top_k = 8 if action_id == GameAction.ACTION6.value else None
            expert_preds = self._predict_experts(feat, action_id, xy)
            score = info_gain(expert_preds, top_k_patches=top_k).item()
            scored.append((edge_idx, score))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.game_id} - tie_break[explore] scored={scored}")
        return self._weighted_sample(scored)

    def _travel_tie_break(self, node, candidates: list) -> int:
        """Among edges already confirmed successful and tied for the
        shortest known distance to the frontier, sample toward whichever
        destination the value head rates most promising."""
        node_info = self.graph_explorer._nodes[node]
        scored = []
        for edge_idx in candidates:
            target_hash = str(node_info.edge_data["target"][edge_idx])
            target_frame = self._frame_cache.get(target_hash)
            if target_frame is None:
                continue
            with torch.no_grad():
                feat = self._encode_frame(target_frame)
                value = self.value_head(feat).item()
            scored.append((edge_idx, value))

        if not scored:
            # No candidate's destination has ever been cached (see class
            # docstring's "remaining known limitations") -- degrade to
            # exactly what the pure agent would have done.
            return random.choice(candidates)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.game_id} - tie_break[travel] scored={scored} of {len(candidates)} candidates")
        return self._weighted_sample(scored)

    def _jepa_tie_break(self, node, mode: str, candidates: list) -> int:
        if self._jepa_ready:
            # `node` is always the exact frame we're currently standing on
            # (choose_edge is called right after _choose_action_inner
            # computed hashed_frame for it) -- caching here means every
            # frame we ever physically visit becomes available for
            # travel-mode value scoring once we're back at a node having
            # actually explored from it.
            self._frame_cache[node] = self._current_frame_np.copy()

        if not self._jepa_ready or len(candidates) <= 1:
            return random.choice(candidates)

        try:
            if mode == "explore":
                return self._explore_tie_break(candidates)
            return self._travel_tie_break(node, candidates)
        except Exception:
            logger.exception(
                f"{self.game_id} - GraphExplorerJepaAgent: tie_break scoring "
                f"failed (mode={mode}), falling back to random choice for this decision"
            )
            return random.choice(candidates)
