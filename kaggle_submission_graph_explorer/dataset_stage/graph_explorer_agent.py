"""GraphExplorer agent: a training-free, exact-state graph-exploration agent
ported from Evgenii Rudakov, Ryan Shock, and Nathan Cowley's "Graph-Based
Exploration for ARC-AGI-3 Interactive Reasoning Tasks" (arXiv:2512.24156,
AAAI 2026 Workshop on AI for Scientific Research) -- the system that placed
3rd on the ARC-AGI-3 Preview Challenge private leaderboard, solving a median
of 30/52 levels across six games with no learned model at all.

Source: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
(`agents/heuristic_agent.py`, `HeuristicAgent`/`FrameProcessor`, commit
including their post-evaluation RESET-loop fix, aad8145). Used under the MIT
License -- see `graph_explorer_THIRD_PARTY_LICENSE` in this directory for
the exact license text and copyright notice required by that license's
terms.

Deliberately does NOT touch this project's own JEPA world model, MoE
predictor, or hypothesis bundle -- this is a from-scratch reproduction of
the paper's own method, kept isolated so it can be tested and (optionally)
submitted on its own merits before any integration work starts.

Ported with the following adaptations from the original (algorithm logic --
`FrameProcessor.segment_frame`/`identify_status_bars*`/
`frame_segments_to_action_groups`, and `choose_action`'s action-selection
body -- is otherwise unchanged):
- This project's vendored `arcengine` package has 8 actions (RESET +
  ACTION1-7), one more than the original's local `agents/structs.py` (RESET
  + ACTION1-6) -- `SIMPLE_ACTION_ID2GAME_ACTION` is built dynamically from
  `GameAction` instead of the original's hardcoded 1-5 dict, so ACTION7
  (a simple action here) isn't silently dropped.
- `FrameData` here has `levels_completed`/`win_levels`, not the original
  framework's `score` field -- level-up detection compares
  `levels_completed` instead.
- The original's `HeuristicAgent.main()` override read the *next* frame
  directly after `take_action` inside its own main loop to detect a level-up
  before the following `choose_action` call. This project's base `Agent`
  (see `agents/agent.py`) doesn't call a per-agent `main()` override for any
  of the other agents in this project (`Hypothesis`/`Memory`/`Curiosity` all
  rely on the shared `Agent.main()`), so the same level-up check is done at
  the top of `choose_action` instead, comparing `latest_frame.levels_completed`
  against the value observed on the previous call -- behaviorally identical
  (both run exactly once per action, using the frame that resulted from the
  previous action), but avoids a custom `main()` override and stays
  consistent with how the rest of this project's agents integrate with the
  harness.
- `choose_action`'s body is wrapped in a top-level try/except, falling back
  to `GameAction.RESET` on any exception -- the original's `main()` override
  had an equivalent try/except around its call to `choose_action`; since
  this port has no `main()` override, the same robustness has to live inside
  `choose_action` itself. This also matches this project's own established
  "heartbeat" pattern for real Kaggle submissions (see `hypothesis_agent.py`
  and CLAUDE.md's Kaggle submission section) -- a single uncaught exception
  must never kill the whole multi-game run.
- Matplotlib-dependent debug/visualization methods (`visualize_last_frame`,
  `visualize_connected_components`, `FrameProcessor.visualize_components`)
  are dropped -- they write PNGs to disk for the paper's own figures and
  don't participate in real gameplay; keeping them would add a hard
  matplotlib dependency this project doesn't otherwise need for headless
  offline/Kaggle runs.
- The original's live-server rate limiter (`minimal_step_time = 0.31`
  seconds between actions, sized to avoid overwhelming the real ARC-AGI-3
  HTTP API when polling it directly every step) is kept but defaults to 0.0
  here -- no other agent in this project self-throttles, and the harness's
  own gateway-wait/heartbeat mechanics (see CLAUDE.md) already handle live
  pacing at a different layer. Settable via `MIN_STEP_TIME` if a real
  submission ever needs it.
- Otherwise-unconditional `print(...)` debug statements (state dumps on
  every single decision) are gated behind `verbose_level` so a real
  multi-game backtest doesn't flood stdout; this changes log volume only,
  not any decision logic.
"""

import hashlib
import logging
import random
import time
from collections import deque
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .graph_explorer_core import GraphExplorer

logger = logging.getLogger()


class FrameProcessor:
    OFFSETS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
    OFFSETS8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self):
        self.connectivity_rank = 4
        self.status_bar_mode = "rule"
        self.status_bar_distance_threshold = 3
        self.status_bar_ratio_threshold = 5
        self.status_bar_twins_threshold = 3
        self.frame_shape = (64, 64)

        self.status_bar_color = 16
        self.minimal_width = 2
        self.maximal_width = 32
        self.non_salient_color = set([0, 1, 2, 3, 4, 5])
        self.salient_color = set([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    def segment_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Segment `frame` into {self.connectivity_rank}-connected components (same color).

        NOTE (upstream): the twins identification increases complexity of the algorithm to O(n^2)

        Returns
        -------
        list[dict]
            One dict per component with keys
            - bounding_box : (x1, y1, x2, y2)   # inclusive pixel coords
            - color        : int                # original greyscale value
            - area         : int                # pixel count
            - is_rectangle : bool               # fully fills its bounding box
            - number_of_twins : int             # number of other components considered twins
            - twin_ids     : list[int]          # ids (1-based) of those twins
                NOTE: here we don't check shapes of the twins thoroughly
        """

        h, w = frame.shape
        label_map = np.zeros((h, w), dtype=int) - 1  # -1 = unvisited
        components: list[dict] = []
        cid = -1  # component id counter

        offsets = self.OFFSETS4 if self.connectivity_rank == 4 else self.OFFSETS8

        # --- first pass: flood-fill each blob ---------------------------------
        for y in range(h):
            for x in range(w):
                if label_map[y, x] != -1:  # already labelled
                    continue
                cid += 1
                color = int(frame[y, x])
                q = deque([(y, x)])
                label_map[y, x] = cid

                min_x = max_x = x
                min_y = max_y = y
                area = 0

                while q:  # BFS
                    cy, cx = q.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)

                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h and 0 <= nx < w
                            and label_map[ny, nx] == -1  # not visited
                            and frame[ny, nx] == color
                        ):
                            label_map[ny, nx] = cid
                            q.append((ny, nx))

                rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                is_rect = area == rect_area

                components.append(
                    dict(
                        bounding_box=(min_x, min_y, max_x, max_y),
                        color=color,
                        area=area,
                        is_rectangle=is_rect,
                    )
                )

        # --- second pass: identify twins --------------------------------------
        # here: simple rule -> same area, same rectangle status, and same color
        for i, comp in enumerate(components):
            twins = [
                j
                for j, other in enumerate(components)
                if i != j
                and other["area"] == comp["area"]
                and other["is_rectangle"] == comp["is_rectangle"]
                and other["color"] == comp["color"]
            ]
            comp["number_of_twins"] = len(twins)
            comp["twin_ids"] = twins

        return label_map, components

    def identify_status_bars(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]] | None, np.ndarray]:
        """
        Identify the status bars from the frame segments.
        Return a list of dictionaries and a frame mask.
        """
        if self.status_bar_mode == "crude":
            status_bar_mask = self.identify_status_bars_crude()
            status_bar_segments_list = None
        elif self.status_bar_mode == "rule" or self.status_bar_mode == "move":
            status_bar_segments_list, status_bar_mask = self.identify_status_bars_with_rule(segmented_frame, frame_segments)
            if self.status_bar_mode == "move":
                raise NotImplementedError("'move' mode is not implemented yet")
        else:
            raise ValueError(f"Invalid status bar mode: {self.status_bar_mode}")
        return status_bar_segments_list, status_bar_mask

    def identify_status_bars_crude(self) -> np.ndarray:
        status_bar_mask = np.zeros(self.frame_shape)
        status_bar_mask[:self.status_bar_distance_threshold, :] = 1
        status_bar_mask[-self.status_bar_distance_threshold:, :] = 1
        status_bar_mask[:, :self.status_bar_distance_threshold] = 1
        status_bar_mask[:, -self.status_bar_distance_threshold:] = 1
        return status_bar_mask

    def identify_status_bars_with_rule(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]], np.ndarray]:
        """
        Identify the status bars from the frame segments.

        The rules (upstream):
            - the status bars are close to the edges of the screen
            - they can be in any orientation
            - they can be duplicated from both sides of the screen
            - there are 2 types of status bars:
                1. the line
                2. the dots (need at least 3 twins)
        """

        checked_segment_ids = set()
        status_bar_segment_ids_list = []  # list[list[int]]
        for i, segment in enumerate(frame_segments):

            status_bar_segment_ids = [i]

            if i in checked_segment_ids:
                continue
            checked_segment_ids.add(i)
            on_edge_list = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(on_edge_list) == 0:
                continue
            directions = []
            if 'left' in on_edge_list or 'right' in on_edge_list:
                directions.append('vertical')
            if 'top' in on_edge_list or 'bottom' in on_edge_list:
                directions.append('horizontal')
            if len(directions) == 2:
                direction = 'any'
            else:
                direction = directions[0]
            is_long_ratio = self.check_segment_ratio(segment, direction=direction)

            if not is_long_ratio:
                twin_ids_on_edge_list = self.segment_twins_on_edge(segment, frame_segments)
                for twin_id in twin_ids_on_edge_list:
                    checked_segment_ids.add(twin_id)
                if len(twin_ids_on_edge_list) + 1 < self.status_bar_twins_threshold:
                    continue
                status_bar_segment_ids.extend(twin_ids_on_edge_list)

            status_bar_segment_ids_list.append(status_bar_segment_ids)

        status_bar_segments_list = []
        status_bar_mask = np.zeros(segmented_frame.shape, dtype=bool)

        for i, status_bar_segment_ids in enumerate(status_bar_segment_ids_list):
            status_bar_segments = []
            for status_bar_segment_id in status_bar_segment_ids:
                status_bar_mask[segmented_frame == status_bar_segment_id] = 1

                status_bar_segments.append(frame_segments[status_bar_segment_id])
            status_bar_segments_list.append(status_bar_segments)

        return status_bar_segments_list, status_bar_mask

    def check_segment_fully_on_edge(self, segment: dict, edges: list[str] | None = None) -> list[str]:
        """Check if the segment is fully on the edge of the screen."""
        x1, y1, x2, y2 = segment["bounding_box"]
        if edges is None:
            edges = ['any']
        for edge in edges:
            assert edge in ['any', 'left', 'right', 'top', 'bottom']

        result = []

        if 'left' in edges or 'any' in edges:
            max_x = max(x1, x2)
            if max_x < self.status_bar_distance_threshold:
                result.append('left')
        if 'right' in edges or 'any' in edges:
            min_x = min(x1, x2)
            if min_x > self.frame_shape[1] - self.status_bar_distance_threshold:
                result.append('right')
        if 'top' in edges or 'any' in edges:
            max_y = max(y1, y2)
            if max_y < self.status_bar_distance_threshold:
                result.append('top')
        if 'bottom' in edges or 'any' in edges:
            min_y = min(y1, y2)
            if min_y > self.frame_shape[0] - self.status_bar_distance_threshold:
                result.append('bottom')
        return result

    def check_segment_ratio(self, segment: dict, direction: str | None = None) -> bool:
        """Check if the segment is shaped like a status bar (long/thin)."""
        if direction is None:
            direction = 'any'
        assert direction in ['any', 'horizontal', 'vertical']

        x_length, y_length = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
        x_to_y_ratio = x_length / y_length
        if x_to_y_ratio >= self.status_bar_ratio_threshold and direction in ('any', 'horizontal'):
            return True
        if x_to_y_ratio <= 1 / self.status_bar_ratio_threshold and direction in ('any', 'vertical'):
            return True
        return False

    def segment_twins_on_edge(self, segment: dict, frame_segments: list[dict], edges: list[str] | None = None) -> list[int]:
        """Check if the segment has twins on the same edge."""

        if edges is None:
            edges = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(edges) == 0:
                return []

        twins = []
        for twin_id in segment["twin_ids"]:
            twin = frame_segments[twin_id]
            twin_edges = self.check_segment_fully_on_edge(twin, edges=edges)
            if len(twin_edges) > 0:
                twins.append(twin_id)

        return twins

    def hash_frame(self, frame: np.ndarray) -> str:
        """
        Deterministic 128-bit hash for an integer-valued NumPy array whose
        elements are in the range 0-15 (4 bits).
        """
        frame = np.asarray(frame, dtype=np.uint8, order='C')

        flat = frame.ravel()
        if flat.size & 1:  # pad to even length
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
        payload = packed.tobytes()

        shape_tag = frame.shape.__repr__().encode()
        return hashlib.blake2b(payload,
                                digest_size=16,
                                person=shape_tag
                                ).hexdigest()

    def frame_segments_to_action_groups(self, frame_segments: list[dict], n_groups: int) -> list[set[int]]:
        """Assign click-segment candidates to one of 5 visual-salience priority
        tiers (paper's Frame Processor): salient-color + medium-sized segments
        first (most likely to be a real button/object), down to probable
        status-bar segments last (least likely to be interactive)."""
        group_0_segments = set()
        group_1_segments = set()
        group_2_segments = set()
        group_3_segments = set()
        group_4_segments = set()

        for segment_id, segment in enumerate(frame_segments):
            x_width, y_width = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
            is_salient = segment["color"] in self.salient_color
            is_medium_width = self.minimal_width <= x_width <= self.maximal_width and self.minimal_width <= y_width <= self.maximal_width
            is_status_bar = segment["color"] == self.status_bar_color

            assert n_groups == 5, "Only 5 groups are supported for now"

            if is_salient and is_medium_width:
                group_0_segments.add(segment_id)
            elif is_medium_width:
                group_1_segments.add(segment_id)
            elif is_salient:
                group_2_segments.add(segment_id)
            elif not is_status_bar:
                group_3_segments.add(segment_id)
            else:
                group_4_segments.add(segment_id)

        return [group_0_segments, group_1_segments, group_2_segments, group_3_segments, group_4_segments]


class GraphExplorerAgent(Agent):
    """Training-free, exact-state graph-exploration agent (Rudakov, Shock &
    Cowley, arXiv:2512.24156) -- see module docstring for provenance and
    port adaptations. No learned components at all."""

    MAX_ACTIONS: int = 300

    N_GROUPS: int = 5
    TOTAL_TIME_ALLOWED = 7.9 * 60 * 60  # 7.9 hours, matches upstream's real-competition safety margin
    MIN_STEP_TIME: float = 0.0  # see module docstring; upstream default was 0.31s (live-server rate limit)

    DEBUG_PRINTS: bool = False
    verbose_level: int = 0

    SIMPLE_ACTION_ID2GAME_ACTION: dict[int, GameAction] = {
        a.value: a for a in GameAction if a.is_simple() and a is not GameAction.RESET
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        self.frame_processor = FrameProcessor()

        self.status_bar_mask = None  # np.ndarray | None

        self.hashed_frame2action_results: dict[str, np.ndarray] = {}
        self.hashed_frame2transitions: dict[str, list] = {}

        self.last_hashed_frame = None
        self.last_action = None  # int | None

        self.arrow_control = True
        self.favor_new_actions = False
        self.favor_frontier_search = True

        self.graph_explorer = GraphExplorer(verbose_level=self.verbose_level, n_groups=self.N_GROUPS)

        self.level_first_frame = None

        self.failed = False
        self.level_up = True

        self.last_action_object = GameAction.RESET

        self.time_start = time.time()
        self.last_time = time.time()

        self.last_transition_suspicious = False

        self._prev_levels_completed = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def get_frame_transition_data(self, hashed_frame: str, num_actions: int) -> tuple[np.ndarray, list]:
        curr_frame_action_results = self.hashed_frame2action_results.get(hashed_frame, None)
        if curr_frame_action_results is None:
            self.hashed_frame2action_results[hashed_frame] = np.zeros(num_actions)  # 0 = untried, -1 = no transition, 1 = transition
            curr_frame_action_results = self.hashed_frame2action_results[hashed_frame]

        curr_frame_transitions = self.hashed_frame2transitions.get(hashed_frame, None)
        if curr_frame_transitions is None:
            self.hashed_frame2transitions[hashed_frame] = [0] * num_actions
            curr_frame_transitions = self.hashed_frame2transitions[hashed_frame]

        return curr_frame_action_results, curr_frame_transitions

    def _update_level_up_state(self, latest_frame: FrameData) -> None:
        """Runs once per `choose_action` call, using the frame that resulted
        from the previous action -- see module docstring for why this
        replaces upstream's per-`main()`-loop check."""
        levels_completed = latest_frame.levels_completed
        if levels_completed > self._prev_levels_completed:
            self.level_up = True
            self.status_bar_mask = None
        elif self.status_bar_mask is not None:
            self.level_up = False
        self._prev_levels_completed = levels_completed

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as e:
            import traceback
            logger.warning(f"GraphExplorerAgent.choose_action failed: {e}, falling back\n{traceback.format_exc()}")
            self.failed = True
            self.level_up = True
            return self.last_action_object

    def _choose_action_inner(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:

        time_diff = time.time() - self.last_time
        if time_diff < self.MIN_STEP_TIME:
            time.sleep(self.MIN_STEP_TIME - time_diff)
        self.last_time = time.time()

        if latest_frame.state in [GameState.NOT_PLAYED]:
            action = GameAction.RESET
            self.last_hashed_frame = None
            self.last_action = None
            if self.failed:
                self.level_up = True
                self.failed = False
            self._prev_levels_completed = 0
            return action

        self._update_level_up_state(latest_frame)

        if latest_frame.state in [GameState.GAME_OVER]:
            action = GameAction.RESET
            self.last_transition_suspicious = True
            return action

        latest_frame_np = np.array(latest_frame.frame, dtype=np.uint8)

        if latest_frame_np.size == 0:
            # Empty frame: fall back to a safe random non-reset action.
            action = random.choice(list(self.SIMPLE_ACTION_ID2GAME_ACTION.values()))
            self.last_action_object = action
            return action

        num_frames = latest_frame_np.shape[0]
        latest_frame_np = latest_frame_np[-1]  # sometimes there are multiple frames; take the last one

        if self.level_up:
            segmented_frame_for_status_bars, frame_segments_for_status_bars = self.frame_processor.segment_frame(latest_frame_np)
            _status_bar_segments_list, status_bar_mask = self.frame_processor.identify_status_bars(segmented_frame_for_status_bars, frame_segments_for_status_bars)
            self.status_bar_mask = status_bar_mask

            self.hashed_frame2action_results = {}
            self.hashed_frame2transitions = {}

        latest_frame_np[self.status_bar_mask] = self.frame_processor.status_bar_color
        segmented_frame, frame_segments = self.frame_processor.segment_frame(latest_frame_np)
        available_actions = latest_frame.available_actions

        num_click_actions = 0
        num_actions = 0
        arrow_actions: list[GameAction] = []
        if GameAction.ACTION6.value in available_actions:
            num_actions += len(frame_segments)
            num_click_actions += len(frame_segments)
            action_groups = self.frame_processor.frame_segments_to_action_groups(frame_segments, n_groups=self.N_GROUPS)
        else:
            action_groups = [set() for _ in range(self.N_GROUPS)]

        for action_id in available_actions:
            if action_id in self.SIMPLE_ACTION_ID2GAME_ACTION:
                arrow_actions.append(self.SIMPLE_ACTION_ID2GAME_ACTION[action_id])
                action_groups[0].add(num_actions)
                num_actions += 1

        latest_frame_np[latest_frame_np == self.frame_processor.status_bar_color] = 0  # avoid overflow in the hash
        hashed_frame = self.frame_processor.hash_frame(latest_frame_np)

        if self.level_up and self.favor_frontier_search:
            self.level_first_frame = hashed_frame
            self.graph_explorer.reset()
            self.graph_explorer.initialize(start_node=hashed_frame, num_candidates=num_actions, group2remaining_candidate_ids=action_groups)

        transition = False
        suspicious_transition = False
        if self.last_hashed_frame is not None and not self.level_up:
            transition = hashed_frame != self.last_hashed_frame
            suspicious_transition = hashed_frame == self.level_first_frame and num_frames > 1

            if self.last_transition_suspicious:
                suspicious_transition = True
                self.last_transition_suspicious = False

            if transition:
                self.hashed_frame2action_results[self.last_hashed_frame][self.last_action] = 1
                self.hashed_frame2transitions[self.last_hashed_frame][self.last_action] = hashed_frame
            else:
                self.hashed_frame2action_results[self.last_hashed_frame][self.last_action] = -1
                self.hashed_frame2transitions[self.last_hashed_frame][self.last_action] = None

            if self.favor_frontier_search:
                self.graph_explorer.record_test(
                    self.last_hashed_frame, self.last_action, transition, hashed_frame,
                    target_num_candidates=num_actions,
                    group2remaining_candidate_ids=action_groups,
                    suspicious_transition=suspicious_transition,
                )

        curr_frame_action_results, _curr_frame_transitions = self.get_frame_transition_data(hashed_frame, num_actions)

        if self.favor_frontier_search and hashed_frame not in self.graph_explorer._nodes:
            # Defensive (matches upstream): reachable whenever the primary
            # record_test call above either didn't run (last_hashed_frame is
            # None / level_up) or was itself ignored by the suspicious-
            # transition dedup below its own threshold -- in the latter case
            # this frame genuinely isn't in the graph yet. Must reuse the
            # same `suspicious_transition` value computed above (not a
            # hardcoded False): passing False here would bypass the "ignore
            # until N consistent observations" protection that
            # last_transition_suspicious/suspicious_transition exists for
            # (see GraphExplorer.record_test) -- exactly the protection a
            # GAME_OVER-triggering edge relies on, since GAME_OVER short-
            # circuits below skip recording it while last_hashed_frame/
            # last_action still point at that edge. Hardcoding False here
            # let a single, possibly-wrong GAME_OVER-adjacent transition get
            # permanently recorded on its first (unconfirmed) observation,
            # which crashed later with "Edge result must be untested before
            # recording a test" once the same edge was genuinely retested
            # and produced a different result.
            self.graph_explorer.record_test(
                self.last_hashed_frame, self.last_action, transition, hashed_frame,
                target_num_candidates=num_actions,
                group2remaining_candidate_ids=action_groups,
                suspicious_transition=suspicious_transition,
            )

        available_action_ids = np.where(curr_frame_action_results != -1)[0]
        new_actions = np.where(curr_frame_action_results == 0)[0]

        if len(available_action_ids) == 0:
            raise ValueError(f'No available actions found for frame {hashed_frame}')

        if self.favor_frontier_search:
            action_id, reasoning = self.graph_explorer.choose_edge(hashed_frame, return_reasoning=True)
        elif len(new_actions) > 0 and self.favor_new_actions:
            action_id = random.choice(new_actions)
            reasoning = ""
        else:
            action_id = random.choice(available_action_ids)
            reasoning = ""

        arrow_control = action_id >= num_click_actions

        if not arrow_control:
            segment_mask = segmented_frame == action_id
            segment_points = np.argwhere(segment_mask)
            segment_point = segment_points[random.randint(0, len(segment_points) - 1)]
            y, x = segment_point

            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"{reasoning}\nClicking on segment {action_id}-{frame_segments[action_id]}, x: {x}, y: {y}",
            }
        else:
            action = arrow_actions[action_id - num_click_actions]
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"{reasoning}Arrow control: {action} for frame {hashed_frame}",
            }

        self.last_hashed_frame = hashed_frame
        self.last_action = action_id
        self.last_action_object = action

        if self.verbose_level >= 1:
            print(action.reasoning)

        return action
