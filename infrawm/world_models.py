from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


Transition = Tuple[int, int, int]  # (state, action, next_state)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    denom = x.sum(axis=-1, keepdims=True)
    denom[denom <= 0] = 1.0
    return x / denom


@dataclass
class FixedMatrixWorldModel:
    """Static categorical dynamics P(s'|s,a).

    Expected shape is [n_actions, n_states, n_states].
    """

    transition_probs: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.transition_probs, dtype=float)
        if p.ndim != 3 or p.shape[1] != p.shape[2]:
            raise ValueError("transition_probs must have shape [A, S, S]")
        self.transition_probs = _normalize_rows(p)

    @property
    def n_actions(self) -> int:
        return int(self.transition_probs.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.transition_probs.shape[1])

    def predict_proba(self, state: int, action: int) -> np.ndarray:
        return self.transition_probs[action, state].copy()

    def expected_next_state(self, state: int, action: int) -> float:
        probs = self.predict_proba(state, action)
        return float(np.dot(probs, np.arange(self.n_states)))

    def sample_next_state(self, state: int, action: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_states, p=self.predict_proba(state, action)))

    def uncertainty(self, state: int, action: int) -> float:
        probs = self.predict_proba(state, action)
        nz = probs[probs > 0]
        return float(-(nz * np.log(nz)).sum())

    def update(self, state: int, action: int, next_state: int) -> None:
        # Static model by definition.
        return None


class BayesianMatrixWorldModel(FixedMatrixWorldModel):
    """Dirichlet-adaptive transition model for L3 Evolver experiments.

    The posterior parameters alpha[a,s,s'] are updated after each observation.
    """

    def __init__(self, alpha: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim != 3 or alpha.shape[1] != alpha.shape[2]:
            raise ValueError("alpha must have shape [A, S, S]")
        if np.any(alpha <= 0):
            raise ValueError("Dirichlet alpha values must be positive")
        self.alpha = alpha.copy()
        super().__init__(_normalize_rows(self.alpha))

    @classmethod
    def from_transitions(
        cls,
        transitions: Iterable[Transition],
        n_states: int = 4,
        n_actions: int = 4,
        prior: float = 1.0,
    ) -> "BayesianMatrixWorldModel":
        alpha = np.full((n_actions, n_states, n_states), float(prior), dtype=float)
        for s, a, sp in transitions:
            alpha[int(a), int(s), int(sp)] += 1.0
        return cls(alpha)

    def _refresh(self) -> None:
        self.transition_probs = _normalize_rows(self.alpha)

    def update(self, state: int, action: int, next_state: int) -> None:
        self.alpha[action, state, next_state] += 1.0
        self._refresh()

    def uncertainty(self, state: int, action: int) -> float:
        # Posterior concentration is a direct epistemic proxy here.
        concentration = float(self.alpha[action, state].sum())
        return 1.0 / concentration

    def posterior_concentration(self, state: int, action: int) -> float:
        return float(self.alpha[action, state].sum())


def transitions_to_counts(
    transitions: Iterable[Transition], n_states: int = 4, n_actions: int = 4
) -> np.ndarray:
    counts = np.zeros((n_actions, n_states, n_states), dtype=float)
    for s, a, sp in transitions:
        counts[int(a), int(s), int(sp)] += 1.0
    return counts
