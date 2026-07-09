"""Stage 5 (plan.md): Bayesian hypothesis-confidence tracking over Stage
4's MoE experts, treated as N parallel hypotheses about ARC-3 dynamics.

Scoped down from architecture.md's full "N parallel hypotheses = candidate
(action/tile -> expert+gate) assignments, each predicting a full
trajectory" -- this reuses the already-trained MoE experts directly as the
hypothesis pool (each expert IS a hypothesis about "what this action
does") rather than building a separate hypothesis-search structure from
scratch. `jepa/models/moe_predictor.py: MoEPredictor.predict_all_experts`
exposes the ungated per-expert predictions this module consumes.

`p(H_i) *= exp(-error_i / tau)`, renormalized each update
(architecture.md's `p(Hi) ∝ p(Hi) · exp(-prediction_error_i / τ)`).
Entropy of this distribution drives the explore/exploit blend `beta_t`:
confident (low entropy, one hypothesis/expert has clearly been right)
-> exploit (trust the value head more); uncertain (high entropy, experts
still disagree about what's reliable) -> explore (chase InfoGain instead).
"""

import math

import torch


class HypothesisBundle:
    def __init__(self, num_hypotheses: int, tau: float = 1.0, decay: float = 0.8):
        """
        decay: per-update multiplier applied to accumulated log_weights
            *before* folding in the new evidence (a forgetting factor, not
            just the numerical-stability max-subtraction below). Without
            it, log_weights is a pure running sum of (-error/tau) over
            every transition since episode start -- with tau small enough
            that per-step latent MSE differences between experts are
            already an appreciable fraction of tau (see TAU's docstring in
            hypothesis_agent.py), even tiny/spurious per-step differences
            compound without bound over a ~300-step episode. Measured
            directly (jepa/../scripts/diagnose_hypothesis_beta.py, replaying
            real episodes through this update): with decay=1.0 (no
            forgetting), beta exceeded 0.5 in 86% of transitions and
            averaged 0.76 -- entropy was collapsing to "confident" within
            the first few dozen steps of essentially every episode and
            staying there, regardless of whether later evidence still
            supported that confidence. That hands control to V(next_state)
            for nearly the whole episode instead of the intended
            entropy-gated blend, which is a problem given the value head's
            own training data is ~98% zero-target (sparse level-completion
            reward) and only marginally beats a zero baseline (see
            CLAUDE.md's Stage 5 status) -- a near-noise value signal
            driving action selection for most of an episode. Geometric
            decay keeps confidence tracking which hypothesis has been
            reliable *recently*, not just cumulatively since the episode's
            first few actions. Swept decay in {1.0, 0.95, 0.8, 0.6, 0.4,
            0.2} on the same replayed episodes: 0.95 only partially helped
            (mean beta 0.61, still >0.5 62% of the time); 0.8 was the best
            balance found (mean beta 0.37, >0.5 only 10% of the time) --
            low enough to stop runaway certainty from swamping InfoGain,
            without decaying so hard (0.6 and below effectively zeroed out
            beta almost everywhere, >0.5 under 8% of the time) that V
            never gets to matter at all, which would just make this
            equivalent to Curiosity's own ranking with extra steps.
        """
        self.num_hypotheses = num_hypotheses
        self.tau = tau
        self.decay = decay
        self.log_weights = torch.zeros(num_hypotheses)  # uniform prior

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def update(self, errors: torch.Tensor) -> None:
        """errors: (K,) prediction error per hypothesis for the transition
        just observed (lower error -> that hypothesis gets more confident)."""
        self.log_weights = self.decay * self.log_weights + (-errors / self.tau)
        # Softmax is shift-invariant -- subtract the max each update purely
        # to keep log_weights numerically bounded over a long episode
        # (unbounded cumulative subtraction would eventually under/overflow).
        self.log_weights = self.log_weights - self.log_weights.max()

    def entropy(self) -> float:
        w = self.weights
        return -(w * torch.log(w + 1e-12)).sum().item()

    def max_entropy(self) -> float:
        return math.log(self.num_hypotheses)

    def beta(self) -> float:
        """Entropy -> explore/exploit blend in [0, 1]. High entropy
        (still uncertain which hypothesis/expert is trustworthy here) ->
        low beta (explore via InfoGain). Low entropy (confident) -> high
        beta (exploit via the value head)."""
        normalized_entropy = self.entropy() / self.max_entropy()
        return 1.0 - normalized_entropy


def info_gain(expert_predictions: torch.Tensor) -> torch.Tensor:
    """expert_predictions: (K, C, H, W) raw per-expert predicted
    next-features for one (state, candidate action) pair (see
    `MoEPredictor.predict_all_experts`, called with batch size 1 and
    squeezed). Returns a scalar: variance across the K hypotheses,
    averaged over channels and spatial positions -- how much the experts
    disagree about what this action does, i.e. how informative actually
    taking it would be.
    """
    return expert_predictions.var(dim=0).mean()
