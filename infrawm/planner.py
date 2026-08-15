from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class PlanResult:
    actions: np.ndarray
    expected_values: np.ndarray
    total_cost: float


class ConstrainedMPCPlanner:
    """Independent world-model scoring + centralized hard-budget allocation.

    Each bridge is scored locally under a learned world model. A centralized
    multiple-choice knapsack dynamic program then selects one action per bridge
    while satisfying the annual budget exactly.
    """

    def __init__(
        self,
        world_model,
        action_costs: Sequence[float],
        horizon: int = 5,
        gamma: float = 0.99,
        uncertainty_penalty: float = 0.0,
        cost_unit: float = 100.0,
    ) -> None:
        self.world_model = world_model
        self.action_costs = np.asarray(action_costs, dtype=float)
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.cost_unit = float(cost_unit)
        if len(self.action_costs) != world_model.n_actions:
            raise ValueError("action_costs length must equal world_model.n_actions")

    def _state_value(self, state: int) -> float:
        # Higher categorical condition is better.
        return float(state)

    def _rollout_value(self, state: int, first_action: int) -> float:
        dist = np.zeros(self.world_model.n_states, dtype=float)
        dist[state] = 1.0
        total = 0.0
        for t in range(self.horizon):
            action = first_action if t == 0 else 0
            next_dist = np.zeros_like(dist)
            for s, p_s in enumerate(dist):
                if p_s <= 0:
                    continue
                next_dist += p_s * self.world_model.predict_proba(s, action)
            expected_health = float(np.dot(next_dist, np.arange(self.world_model.n_states)))
            total += (self.gamma ** t) * expected_health
            dist = next_dist
        total -= self.uncertainty_penalty * self.world_model.uncertainty(state, first_action)
        return total

    def score_actions(self, states: Iterable[int]) -> np.ndarray:
        states = np.asarray(list(states), dtype=int)
        scores = np.zeros((len(states), self.world_model.n_actions), dtype=float)
        for i, state in enumerate(states):
            for action in range(self.world_model.n_actions):
                scores[i, action] = self._rollout_value(int(state), action)
        # Use no-action as local reference so the allocator spends only for gain.
        scores = scores - scores[:, [0]]
        return scores

    def _multiple_choice_knapsack(self, scores: np.ndarray, budget: float) -> np.ndarray:
        n, n_actions = scores.shape
        unit = self.cost_unit
        weights = np.ceil(self.action_costs / unit).astype(int)
        capacity = int(np.floor(float(budget) / unit))
        neg_inf = -1e30
        dp = np.full((n + 1, capacity + 1), neg_inf, dtype=float)
        dp[0, 0] = 0.0
        choice = np.full((n, capacity + 1), -1, dtype=int)
        prev_cap = np.full((n, capacity + 1), -1, dtype=int)

        for i in range(n):
            for c in range(capacity + 1):
                if dp[i, c] <= neg_inf / 2:
                    continue
                for a in range(n_actions):
                    nc = c + weights[a]
                    if nc > capacity:
                        continue
                    val = dp[i, c] + scores[i, a]
                    if val > dp[i + 1, nc]:
                        dp[i + 1, nc] = val
                        choice[i, nc] = a
                        prev_cap[i, nc] = c

        c = int(np.argmax(dp[n]))
        actions = np.zeros(n, dtype=int)
        for i in range(n - 1, -1, -1):
            a = choice[i, c]
            if a < 0:
                a = 0
            actions[i] = a
            pc = prev_cap[i, c]
            c = 0 if pc < 0 else pc
        return actions

    def plan(self, states: Iterable[int], budget: float) -> PlanResult:
        states = np.asarray(list(states), dtype=int)
        scores = self.score_actions(states)
        actions = self._multiple_choice_knapsack(scores, budget)
        total_cost = float(self.action_costs[actions].sum())
        if total_cost > budget + 1e-9:
            raise RuntimeError("planner violated hard budget")
        values = scores[np.arange(len(actions)), actions]
        return PlanResult(actions=actions, expected_values=values, total_cost=total_cost)
