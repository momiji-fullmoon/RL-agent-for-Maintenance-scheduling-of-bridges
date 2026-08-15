from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = x.sum(axis=-1, keepdims=True)
    d[d <= 0] = 1.0
    return x / d


@dataclass
class HiddenShiftEnv:
    """Interactive evaluator with an optional hidden dynamics shift.

    This environment is intentionally separated from the agent world model.
    It enables genuine Predict -> Act -> Observe -> Revise evaluation.
    """

    transition_pre: np.ndarray
    transition_post: Optional[np.ndarray] = None
    shift_step: Optional[int] = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.transition_pre = _normalize_rows(self.transition_pre)
        if self.transition_post is None:
            self.transition_post = self.transition_pre.copy()
        else:
            self.transition_post = _normalize_rows(self.transition_post)
        if self.transition_pre.shape != self.transition_post.shape:
            raise ValueError("pre/post transition tensors must have same shape")
        self.rng = np.random.default_rng(self.seed)
        self.t = 0

    @property
    def n_actions(self) -> int:
        return int(self.transition_pre.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.transition_pre.shape[1])

    def current_transition(self) -> np.ndarray:
        if self.shift_step is not None and self.t >= self.shift_step:
            return self.transition_post
        return self.transition_pre

    def step(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=int)
        actions = np.asarray(actions, dtype=int)
        if states.shape != actions.shape:
            raise ValueError("states and actions must have identical shape")
        tm = self.current_transition()
        nxt = np.empty_like(states)
        for i, (s, a) in enumerate(zip(states, actions)):
            nxt[i] = self.rng.choice(self.n_states, p=tm[a, s])
        self.t += 1
        return nxt


def make_stress_shift(base: np.ndarray, severity: float = 0.25) -> np.ndarray:
    """Create a controlled deterioration shift for benchmark stress testing.

    Probability mass is moved from equal/better states toward the immediately
    worse state. The transformation is deterministic and preserves row sums.
    """
    p = _normalize_rows(base).copy()
    severity = float(np.clip(severity, 0.0, 1.0))
    a_count, s_count, _ = p.shape
    out = p.copy()
    for a in range(a_count):
        for s in range(s_count):
            if s == 0:
                continue
            move = severity * out[a, s, s:].sum()
            if move <= 0:
                continue
            donor = out[a, s, s:].copy()
            donor_sum = donor.sum()
            if donor_sum > 0:
                out[a, s, s:] -= move * donor / donor_sum
                out[a, s, s - 1] += move
    return _normalize_rows(out)
