import numpy as np

from infrawm import BayesianMatrixWorldModel, ConstrainedMPCPlanner, FixedMatrixWorldModel


def toy_transition():
    p = np.zeros((4, 4, 4), dtype=float)
    for a in range(4):
        for s in range(4):
            p[a, s, min(3, s + (1 if a > 0 else 0))] = 1.0
    return p


def test_budget_is_hard():
    wm = FixedMatrixWorldModel(toy_transition())
    planner = ConstrainedMPCPlanner(wm, [0, 100, 200, 300], horizon=2, cost_unit=100)
    plan = planner.plan([0, 1, 2, 3], budget=300)
    assert plan.total_cost <= 300


def test_bayesian_update_increases_observed_probability():
    alpha = np.ones((4, 4, 4), dtype=float)
    wm = BayesianMatrixWorldModel(alpha)
    before = wm.predict_proba(1, 2)[3]
    wm.update(1, 2, 3)
    after = wm.predict_proba(1, 2)[3]
    assert after > before
