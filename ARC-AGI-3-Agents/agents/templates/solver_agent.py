"""Stage 6: systematic, policy-free search harvester ("Solver").

Motivation (see experiments/stage6_selfplay_bootstrap.md): using the
current best *policy* (`Hypothesis`) to harvest a bigger/harder training
corpus was a real, verified regression -- the harvester's own learned
biases narrowed action-space and click-space coverage on several games
even while unlocking a few others (an echo-chamber effect: a policy that
is still learning reflects its own gaps back into the data it produces).

The fix this module implements: replace "a learned policy harvests data"
with "systematic, exhaustive-within-budget search harvests data." The
core property that makes this bias-free where a policy isn't: at any
given *discovered* state, this agent either has tried a given
(action, click-patch) option from that exact state or it hasn't -- there
is no learned preference that can cause it to systematically avoid an
option the way a policy's Q-ranking can. The only place a learned,
possibly-biased ranking signal is used at all is choosing *which already-
discovered state to push toward next* once the current one is fully
explored -- never *what* to try once there. That's a deliberate,
narrow use of `Hypothesis`'s existing Q/InfoGain/value machinery (see
`_pick_frontier` below), not a re-introduction of policy bias into what
actually gets tried.

**Honesty about scope**: this is *systematic, budget-bounded* search, not
literal exhaustive/complete search -- the true state space is far larger
than any real action budget could cover. What it guarantees is narrower
but real: every option it tries from a state it commits to exploring is
chosen deterministically, not by a learned preference, and once a state's
action space (at 8x8-patch click granularity, not raw pixels) is
exhausted, it moves on rather than re-trying known outcomes.

Design, reusing existing infrastructure rather than inventing new pieces:

- **Exact state tracking**: `jepa/memory.py: TransitionGraph`, unchanged
  -- `record`/`tried_actions` already do exactly the per-state
  (action, xy) -> outcome bookkeeping this agent needs; see that module's
  own docstring for why exact-hash keying is the right tool here (ARC-3
  is deterministic, so anything seen once is recallable forever).
- **Click granularity**: reuses `Hypothesis`/`Curiosity`'s 8x8-patch
  abstraction (64 options, not 4,096 raw pixels) for ACTION6 -- picking a
  fixed representative pixel (the patch center) per patch, so "has this
  patch been tried from this state" is a well-defined, stable question
  (an exact-pixel key would almost never repeat across draws, which is
  exactly why `Hypothesis`'s own *scoring* uses patch granularity too).
- **Replay-to-frontier**: ARC-3 only supports resetting to a game's exact
  initial state, not arbitrary save/restore (rules.md: "RESET | Start or
  restart the game"). To keep exploring from a previously-discovered
  state once the current one is exhausted, this agent tracks the exact
  action sequence from the post-RESET root to every discovered state
  (`_path_to_state`), issues a deliberate RESET (not just the automatic
  one on GAME_OVER/NOT_PLAYED -- RESET is a normal, anytime-legal action
  per rules.md), then replays that exact sequence action-for-action. This
  means total actions taken scales with sum-over-discovered-states of
  (depth to reach it), not just the number of new states explored --
  budgeted for explicitly, see `MAX_ACTIONS`'s docstring below.
- **Frontier choice**: once a state has no untried options left, which
  already-discovered state to push toward next is chosen via the *same*
  entropy-gated `Q = (1-beta)*InfoGain + beta*V` blend `Hypothesis` uses
  for its own action selection (`jepa/hypothesis_bundle.py`), scored
  against a small, comparable path-length penalty (states are cheaper to
  reach have a real advantage -- reaching them costs real budget). This
  is the one place a ranking signal legitimately matters: it can only
  affect the *order* frontier states get pushed toward within a bounded
  budget, never *whether* an option at an already-committed-to state gets
  tried.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.device import get_device  # noqa: E402
from jepa.grid import CANVAS, PATCH, arc3_frame_to_tensor  # noqa: E402
from jepa.hypothesis_bundle import HypothesisBundle, info_gain  # noqa: E402
from jepa.memory import TransitionGraph  # noqa: E402
from jepa.models import CNNEncoder, MoEPredictor, ValueHead  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
_PATCHES_PER_SIDE = CANVAS // PATCH  # 8
_NUM_PATCHES = _PATCHES_PER_SIDE * _PATCHES_PER_SIDE  # 64


class Solver(Agent):
    """Systematic, policy-free search: exhausts each discovered state's
    action space (at 8x8-patch click granularity) before replaying a known
    path to push exploration toward another discovered, not-yet-exhausted
    state."""

    # SOLVER_MAX_ACTIONS env var override, mirroring hypothesis_agent.py's
    # HYPOTHESIS_MAX_ACTIONS pattern (stage6-selfplay-bootstrap). Default
    # (300) matches every other agent's harness-safe default; a real
    # harvest run needs far more, since total actions = new-state
    # exploration *plus* the sum of replay-path lengths back to every
    # frontier state ever pushed toward. This project's own Memory/
    # Hypothesis teacher-policy precedent used 2500 for a single long
    # rollout with no replay overhead at all -- a systematic version that
    # deliberately RESETs and replays back to older states needs
    # meaningfully more. Chose 4000 as the harvesting default (set via env
    # var per-game, not this class default) after a smoke test on a
    # handful of games showed most of a state's local action space
    # (<=70 options: up to 6 simple actions + 64 click patches) gets
    # exhausted well before that, with remaining budget going toward
    # deeper frontier states -- see experiments/stage6_search_harvest.md
    # for the actual per-game numbers this produced.
    MAX_ACTIONS = int(os.getenv("SOLVER_MAX_ACTIONS", "300"))
    TAU = 0.01
    DECAY = 0.8
    # Same rationale as Hypothesis.TOP_K_PATCHES: a flat mean over all 64
    # patches underrates ACTION6 (whose value is one good location, not an
    # average over 63 irrelevant ones); a flat max overrates it via
    # extreme-value inflation. Reused unswept, same value.
    TOP_K_PATCHES = 8
    # How much a frontier candidate's replay cost (path length from root)
    # discounts its Q score, normalized to [0, DEPTH_PENALTY_WEIGHT) via a
    # half-life so it never fully swamps IG/V (which sit at a similar
    # O(1e-2 to 1e-1) scale per this project's own Stage 5 numbers) but
    # still meaningfully favors cheaper-to-reach frontiers when scores are
    # otherwise close. Unswept starting point, like TOP_K_PATCHES above.
    DEPTH_PENALTY_WEIGHT = 0.05
    DEPTH_PENALTY_HALFLIFE = 50.0
    # Scoring every frontier candidate on every exhaustion event is one
    # small forward pass each -- cheap individually, but the frontier can
    # grow into the hundreds over a long harvest. Capping the candidate
    # pool bounds worst-case per-decision cost; which candidates get
    # sampled when capped is itself randomized (self._rng), not a bias
    # toward any particular kind of state.
    FRONTIER_SAMPLE_CAP = 200
    # SOLVER_DIAG_MODE env var (mirrors HYPOTHESIS_DIAG_MODE): skip real
    # logic, always return a safe random action, while __init__ still runs
    # in full -- isolates "does setup/checkpoint loading work" from "does
    # the real search logic work."
    DIAG_MODE = os.getenv("SOLVER_DIAG_MODE") == "1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random()
        self._init_failed = False

        try:
            self._init_models()
        except Exception:
            logger.exception(
                f"{self.game_id} - solver agent: model init failed, "
                "falling back to random-action-only mode for this game"
            )
            self._init_failed = True

        self._last_levels_completed = 0

        self._prev_feat: torch.Tensor | None = None
        self._prev_action_id: int | None = None
        self._prev_xy: tuple[int, int] | None = None
        self._prev_state_key: str | None = None

        # Systematic-search bookkeeping, persists for the agent's whole
        # lifetime (every RESET within this game, deliberate or forced --
        # same rationale as Memory/Hypothesis's own self.graph).
        self._available_by_state: dict[str, list[int]] = {}
        self._feat_by_state: dict[str, torch.Tensor] = {}
        self._path_to_state: dict[str, list[tuple[int, tuple[int, int] | None]]] = {}
        self._frontier: set[str] = set()
        self._replay_queue: list[tuple[int, tuple[int, int] | None]] = []
        self._replay_target: str | None = None
        # A target whose recorded path has already failed to reproduce the
        # expected state once is evicted here and never retried -- without
        # this, _pick_frontier has no way to learn "this path doesn't
        # actually lead where it's supposed to" and can re-select the same
        # cheap (low-depth), now-provably-wrong target forever, burning the
        # whole rest of the action budget on identical failed 1-2-step
        # replays (observed directly: 1560+ consecutive identical failed
        # travels to the same target on one game during this session's own
        # harvest run, before this fix -- see
        # experiments/stage6_search_harvest.md for the full story).
        self._unreachable: set[str] = set()

        # Diagnostics -- cheap counters, logged in a summary line at
        # cleanup() so a harvest run's log is enough to sanity-check the
        # replay-to-frontier mechanic actually fired (and how often replay
        # landed where expected -- ARC-3 is documented as deterministic,
        # but this is worth verifying empirically rather than assuming).
        self._n_travels = 0
        self._n_replay_arrivals = 0
        self._n_replay_mismatches = 0

    def _init_models(self) -> None:
        self.device = get_device()

        self.encoder = CNNEncoder().to(self.device)
        self.encoder.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "encoder_moe.pt", map_location=self.device)
        )
        self.encoder.eval()

        import json

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
        self.num_experts = self.predictor.num_experts

        self.value_head = ValueHead().to(self.device)
        value_path = _CHECKPOINT_DIR / "value_head.pt"
        if value_path.exists():
            self.value_head.load_state_dict(torch.load(value_path, map_location=self.device))
        else:
            logger.warning(
                f"{self.game_id} - no value_head.pt found at {value_path}, "
                "using an untrained value head (V will be ~noise)"
            )
        self.value_head.eval()

        self.graph = TransitionGraph()
        self.hypotheses = HypothesisBundle(num_hypotheses=self.num_experts, tau=self.TAU, decay=self.DECAY)

    def cleanup(self, scorecard: Any = None) -> None:
        already_done = not getattr(self, "_cleanup", True)
        super().cleanup(scorecard)
        if already_done or self._init_failed:
            return
        logger.info(
            f"{self.game_id} - solver agent: summary -- {len(self._available_by_state)} states discovered, "
            f"{len(self.graph)} (state, action) edges tried, {len(self._frontier)} states still have untried "
            f"options, {self._n_travels} deliberate travels ({self._n_replay_arrivals} landed on target, "
            f"{self._n_replay_mismatches} diverged, {len(self._unreachable)} distinct targets evicted as unreachable)"
        )

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        try:
            return latest_frame.state is GameState.WIN
        except Exception:
            logger.exception(f"{self.game_id} - solver agent: is_done raised, treating as not-done")
            return False

    @torch.no_grad()
    def _encode(self, latest_frame: FrameData) -> torch.Tensor:
        tensor = arc3_frame_to_tensor(latest_frame.frame)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return self.encoder(x)  # (1, C, 8, 8)

    @torch.no_grad()
    def _predict_experts(
        self, feat: torch.Tensor, action_id: int, xy: tuple[int, int] | None
    ) -> torch.Tensor:
        b = feat.shape[0]
        action_t = torch.full((b,), action_id, dtype=torch.long, device=self.device)
        if xy is not None:
            x, y = xy
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=self.device)
        else:
            xy_t = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
        game_t = torch.full((b,), self.game_idx, dtype=torch.long, device=self.device)
        return self.predictor.predict_all_experts(feat, action_t, xy_t, game_t)[0]  # (K, C, 8, 8)

    @staticmethod
    def _patch_center(patch_idx: int) -> tuple[int, int]:
        row, col = divmod(patch_idx, _PATCHES_PER_SIDE)
        return col * PATCH + PATCH // 2, row * PATCH + PATCH // 2

    def _sample_random_xy(self) -> tuple[int, int]:
        return self._rng.randrange(CANVAS), self._rng.randrange(CANVAS)

    def _untried_options(self, state_key: str) -> list[tuple[int, tuple[int, int] | None]]:
        """Every (action_id, xy) option not yet tried from this exact
        state, at 8x8-patch click granularity -- deterministic order
        (ascending action id; ACTION6 patches in row-major order), so
        "the next untried option" is a well-defined, reproducible choice,
        not a random one."""
        available = self._available_by_state.get(state_key)
        if not available:
            return []
        tried = self.graph.tried_actions(state_key)
        options: list[tuple[int, tuple[int, int] | None]] = []
        for a in available:
            if a == GameAction.RESET.value:
                continue
            if a == GameAction.ACTION6.value:
                for p in range(_NUM_PATCHES):
                    xy = self._patch_center(p)
                    if (a, xy) not in tried:
                        options.append((a, xy))
            else:
                if (a, None) not in tried:
                    options.append((a, None))
        return options

    def _refresh_frontier(self, state_key: str) -> None:
        if state_key not in self._available_by_state:
            return
        if self._untried_options(state_key):
            self._frontier.add(state_key)
        else:
            self._frontier.discard(state_key)

    def _cache_state_if_new(self, state_key: str, feat: torch.Tensor, latest_frame: FrameData) -> None:
        if state_key in self._available_by_state:
            return
        available = latest_frame.available_actions or [
            a.value for a in GameAction if a is not GameAction.RESET
        ]
        self._available_by_state[state_key] = sorted(set(available))
        self._feat_by_state[state_key] = feat.detach().to("cpu")
        if state_key not in self._path_to_state:
            if self._prev_state_key is not None:
                prior_path = self._path_to_state.get(self._prev_state_key, [])
                self._path_to_state[state_key] = prior_path + [(self._prev_action_id, self._prev_xy)]
            else:
                self._path_to_state[state_key] = []  # the post-RESET root
        self._refresh_frontier(state_key)

    def _update_from_last_turn(self, feat: torch.Tensor, latest_frame: FrameData) -> None:
        """Records the observed transition into the exact graph (marking
        that (state, action, xy) tried, whatever the outcome -- including
        a GAME_OVER outcome, which is real, informative data: "this option
        ends the game" is exactly the kind of thing systematic coverage
        should discover once and never retry pointlessly), refreshes that
        state's frontier membership, and updates the Bayesian hypothesis
        confidence off the real observed prediction error -- same update
        rule as Hypothesis._update_hypotheses."""
        if self._prev_feat is None or self._prev_action_id is None:
            return

        if self._prev_state_key is not None:
            levels_delta = latest_frame.levels_completed - self._last_levels_completed
            self.graph.record(
                self._prev_state_key, self._prev_action_id, self._prev_xy, latest_frame.frame, levels_delta
            )
            self._refresh_frontier(self._prev_state_key)

        if self._prev_action_id == GameAction.RESET.value:
            return

        with torch.no_grad():
            expert_preds = self._predict_experts(self._prev_feat, self._prev_action_id, self._prev_xy)
            errors = (expert_preds - feat[0].unsqueeze(0)).pow(2).mean(dim=(1, 2, 3)).cpu()  # (K,)
        self.hypotheses.update(errors)

    def _pick_frontier(self, exclude: str) -> str | None:
        """Which already-discovered, not-yet-exhausted state to push
        toward next -- the one place a learned ranking signal is used
        (see module docstring). Scores each candidate by the same
        entropy-gated Q = (1-beta)*InfoGain + beta*V blend Hypothesis uses
        for its own action selection, evaluated against that state's own
        first untried option (the option that would actually be tried
        next once we arrive), minus a small path-length penalty (reaching
        a frontier costs real replay actions)."""
        candidates = [s for s in self._frontier if s != exclude and s not in self._unreachable]
        if not candidates:
            return None
        if len(candidates) > self.FRONTIER_SAMPLE_CAP:
            candidates = self._rng.sample(candidates, self.FRONTIER_SAMPLE_CAP)

        beta = self.hypotheses.beta()
        weights = self.hypotheses.weights.to(self.device)

        best_score, best = -1e18, None
        for s in candidates:
            untried = self._untried_options(s)
            if not untried:
                continue
            rep_action, _rep_xy = untried[0]
            feat = self._feat_by_state[s].to(self.device)
            with torch.no_grad():
                expert_preds = self._predict_experts(feat, rep_action, (32, 32))
                ig = info_gain(expert_preds, top_k_patches=self.TOP_K_PATCHES).item()
                v_per_expert = self.value_head(expert_preds)
            v = (weights * v_per_expert).sum().item()
            depth = len(self._path_to_state.get(s, []))
            depth_penalty = self.DEPTH_PENALTY_WEIGHT * (depth / (depth + self.DEPTH_PENALTY_HALFLIFE))
            score = (1.0 - beta) * ig + beta * v - depth_penalty
            if score > best_score:
                best_score, best = score, s
        return best

    def _build_action(self, action_id: int, xy: tuple[int, int] | None) -> GameAction:
        action = GameAction.from_id(action_id)
        if action.is_complex():
            x, y = xy if xy is not None else (32, 32)
            action.set_data({"x": x, "y": y})
        return action

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        try:
            if self._init_failed or self.DIAG_MODE:
                return self._safe_fallback_action(latest_frame)
            return self._choose_action_inner(frames, latest_frame)
        except Exception:
            logger.exception(
                f"{self.game_id} - solver agent: choose_action raised, falling back to a safe random action"
            )
            return self._safe_fallback_action(latest_frame)

    def _safe_fallback_action(self, latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "solver agent: reset (fallback)"
            return action
        available = latest_frame.available_actions or [
            a.value for a in GameAction if a is not GameAction.RESET
        ]
        action_id = self._rng.choice(available)
        action = GameAction.from_id(action_id)
        if action.is_complex():
            action.set_data(
                {"x": self._rng.randrange(CANVAS), "y": self._rng.randrange(CANVAS)}
            )
        action.reasoning = "solver agent: safe fallback after internal error"
        return action

    def _choose_action_inner(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            # A GAME_OVER frame (unlike the very first NOT_PLAYED call) is
            # still a valid, encodable outcome -- record the transition
            # that led here before resetting, so that (state, action)
            # pair counts as tried and won't be retried pointlessly next
            # time this state is reached.
            if self._prev_state_key is not None:
                try:
                    feat = self._encode(latest_frame)
                    self._update_from_last_turn(feat, latest_frame)
                except Exception:
                    logger.exception(f"{self.game_id} - solver agent: failed to record terminal transition")
            action = GameAction.RESET
            action.reasoning = "solver agent: reset"
            self._last_levels_completed = 0
            self._prev_feat = None
            self._prev_action_id = None
            self._prev_xy = None
            self._prev_state_key = None
            self._replay_queue = []
            self._replay_target = None
            return action

        feat = self._encode(latest_frame)
        self._update_from_last_turn(feat, latest_frame)
        self._last_levels_completed = latest_frame.levels_completed

        state_key = TransitionGraph.key_for(latest_frame.frame)
        self._cache_state_if_new(state_key, feat, latest_frame)

        if self._replay_target is not None and not self._replay_queue:
            # A deliberate travel's replay queue just ran out -- check
            # whether we actually landed on the intended frontier state.
            if state_key == self._replay_target:
                self._n_replay_arrivals += 1
            else:
                self._n_replay_mismatches += 1
                # Evict the failed target permanently rather than leaving it
                # in the frontier -- its recorded path just proved wrong
                # once, and _pick_frontier has no other signal to avoid
                # re-selecting the same (likely cheapest, lowest-depth)
                # target again next time a state gets exhausted. See the
                # _unreachable field's own docstring for why this matters.
                self._unreachable.add(self._replay_target)
                self._frontier.discard(self._replay_target)
                logger.info(
                    f"{self.game_id} - solver agent: replay landed on an unexpected state "
                    f"(wanted {self._replay_target[:8]}, got {state_key[:8]}) -- marking that target "
                    f"unreachable and treating the arrival as newly discovered"
                )
            self._replay_target = None

        if self._replay_queue:
            action_id, xy = self._replay_queue.pop(0)
            action = self._build_action(action_id, xy)
            action.reasoning = f"solver agent: replaying toward frontier ({len(self._replay_queue)} steps left)"
        else:
            untried = self._untried_options(state_key)
            if untried:
                action_id, xy = untried[0]
                action = self._build_action(action_id, xy)
                action.reasoning = f"solver agent: systematic probe ({len(untried) - 1} untried remain here)"
            else:
                target = self._pick_frontier(exclude=state_key)
                if target is None:
                    # No known state anywhere still has an untried option
                    # -- systematic coverage of everything discovered so
                    # far is complete. Spend remaining budget on a bounded
                    # random action rather than idling (harmless: games
                    # are deterministic, so this can at most retrace a
                    # known edge, never corrupt the graph).
                    available = latest_frame.available_actions or [
                        a.value for a in GameAction if a is not GameAction.RESET
                    ]
                    action_id = self._rng.choice(available)
                    xy = self._sample_random_xy() if action_id == GameAction.ACTION6.value else None
                    action = self._build_action(action_id, xy)
                    action.reasoning = "solver agent: search space exhausted, random fallback"
                else:
                    self._replay_queue = list(self._path_to_state.get(target, []))
                    self._replay_target = target
                    self._n_travels += 1
                    action_id, xy = GameAction.RESET.value, None
                    action = GameAction.RESET
                    action.reasoning = (
                        f"solver agent: traveling to frontier state "
                        f"({len(self._replay_queue)}-step replay queued)"
                    )
                    logger.info(
                        f"{self.game_id} - solver agent: travel #{self._n_travels} to frontier state "
                        f"{target[:8]} (replay depth {len(self._replay_queue)}, frontier size {len(self._frontier)})"
                    )

        self._prev_feat = feat
        self._prev_action_id = action_id
        self._prev_xy = xy
        self._prev_state_key = state_key
        return action
