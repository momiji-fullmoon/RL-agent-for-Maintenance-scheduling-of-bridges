#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infrawm import BayesianMatrixWorldModel, ConstrainedMPCPlanner, FixedMatrixWorldModel, HiddenShiftEnv
from infrawm.envs import make_stress_shift


ACTION_COSTS = np.array([0.0, 1148.81, 2317.70, 3004.33])


def default_transition() -> np.ndarray:
    # State order: critical, poor, fair, good. Action order: none/minor/major/replace.
    return np.array(
        [
            [[1.00, 0.00, 0.00, 0.00], [0.02, 0.98, 0.00, 0.00], [0.00, 0.02, 0.98, 0.00], [0.00, 0.00, 0.04, 0.96]],
            [[0.02, 0.56, 0.33, 0.09], [0.00, 0.30, 0.57, 0.13], [0.00, 0.00, 0.84, 0.16], [0.00, 0.00, 0.00, 1.00]],
            [[0.50, 0.33, 0.17, 0.00], [0.00, 0.18, 0.65, 0.17], [0.00, 0.00, 0.65, 0.35], [0.00, 0.00, 0.00, 1.00]],
            [[0.50, 0.38, 0.12, 0.00], [0.00, 0.45, 0.36, 0.19], [0.00, 0.00, 0.68, 0.32], [0.00, 0.00, 0.00, 1.00]],
        ],
        dtype=float,
    )


def model_nll(model, observations):
    total = 0.0
    for s, a, sp in observations:
        total -= np.log(max(model.predict_proba(int(s), int(a))[int(sp)], 1e-12))
    return float(total / max(len(observations), 1))


def run(adaptive: bool, seed: int, n_bridges: int, years: int, shift_year: int, budget: float):
    rng = np.random.default_rng(seed)
    base = default_transition()
    shifted = make_stress_shift(base, severity=0.35)

    # Deliberately initialize the agent on pre-shift dynamics.
    if adaptive:
        alpha = 1.0 + 40.0 * base
        wm = BayesianMatrixWorldModel(alpha)
    else:
        wm = FixedMatrixWorldModel(base)

    planner = ConstrainedMPCPlanner(
        wm,
        ACTION_COSTS,
        horizon=5,
        gamma=0.97,
        uncertainty_penalty=0.2 if adaptive else 0.0,
        cost_unit=100.0,
    )
    env = HiddenShiftEnv(base, shifted, shift_step=shift_year, seed=seed + 100)

    states = rng.choice(4, size=n_bridges, p=[0.05, 0.20, 0.45, 0.30])
    initial_health = float(states.mean())
    observations = []
    health_curve = [initial_health]
    costs = []
    prediction_error = []

    for _year in range(years):
        plan = planner.plan(states, budget)
        actions = plan.actions
        next_states = env.step(states, actions)

        year_nll = 0.0
        for s, a, sp in zip(states, actions, next_states):
            probs = wm.predict_proba(int(s), int(a))
            year_nll -= np.log(max(probs[int(sp)], 1e-12))
            observations.append((int(s), int(a), int(sp)))
            if adaptive:
                wm.update(int(s), int(a), int(sp))

        prediction_error.append(year_nll / n_bridges)
        costs.append(plan.total_cost)
        states = next_states
        health_curve.append(float(states.mean()))

    post = prediction_error[shift_year:]
    first_post = float(np.mean(post[: min(3, len(post))])) if post else 0.0
    last_post = float(np.mean(post[-min(3, len(post)) :])) if post else 0.0
    return {
        "adaptive": adaptive,
        "seed": seed,
        "n_bridges": n_bridges,
        "years": years,
        "shift_year": shift_year,
        "budget": budget,
        "initial_mean_health": initial_health,
        "final_mean_health": float(states.mean()),
        "mean_health": float(np.mean(health_curve)),
        "total_cost": float(np.sum(costs)),
        "budget_violations": int(sum(c > budget + 1e-9 for c in costs)),
        "mean_prediction_nll": float(np.mean(prediction_error)),
        "post_shift_initial_nll": first_post,
        "post_shift_final_nll": last_post,
        "adaptation_nll_gain": first_post - last_post,
        "health_curve": health_curve,
        "prediction_nll_curve": prediction_error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bridges", type=int, default=40)
    ap.add_argument("--years", type=int, default=30)
    ap.add_argument("--shift-year", type=int, default=10)
    ap.add_argument("--budget", type=float, default=18000.0)
    ap.add_argument("--output", default="results/infrawm_smoke.json")
    args = ap.parse_args()

    fixed = run(False, args.seed, args.n_bridges, args.years, args.shift_year, args.budget)
    adaptive = run(True, args.seed, args.n_bridges, args.years, args.shift_year, args.budget)
    result = {"fixed": fixed, "adaptive": adaptive}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "fixed_final_health": fixed["final_mean_health"],
        "adaptive_final_health": adaptive["final_mean_health"],
        "fixed_budget_violations": fixed["budget_violations"],
        "adaptive_budget_violations": adaptive["budget_violations"],
        "adaptive_nll_gain": adaptive["adaptation_nll_gain"],
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
