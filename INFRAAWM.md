# InfraAWM: Agentic World Models for constrained bridge maintenance

This branch is a clean re-framing of the InfraRL task around **Agentic World Models (AWM)** rather than model-free policy learning.

## Research question

Can an agent learn an action-conditioned infrastructure world model from historical data, use it to imagine feasible long-horizon maintenance plans under a hard annual budget, and revise the world model after distribution shift using prediction error?

## L1 / L2 / L3 task decomposition

- **L1 Predictor**: estimate `P(s_{t+1} | s_t, a_t)` and evaluate predictive accuracy/calibration.
- **L2 Simulator**: roll the model forward for planning and choose actions under a hard budget.
- **L3 Evolver**: after each new observation, update the world model and measure adaptation under hidden dynamics shift.

The current implementation deliberately starts with categorical transition models because they provide a strong, interpretable baseline before adding larger neural world models.

## Implemented components

### Static world model

`infrawm.FixedMatrixWorldModel`

A fixed categorical transition model. This corresponds to the empirical transition-matrix style evaluator used by InfraRL and serves as the non-adaptive baseline.

### Bayesian L3 Evolver

`infrawm.BayesianMatrixWorldModel`

Each state-action transition row has a Dirichlet posterior. After observing `(s, a, s')`, the posterior is updated online. Posterior concentration is exposed as an epistemic-uncertainty proxy.

### Hard-budget MPC planner

`infrawm.ConstrainedMPCPlanner`

The planner:

1. evaluates each bridge/action locally by multi-step world-model rollout;
2. subtracts the no-action value to obtain intervention value;
3. optionally penalizes uncertain actions;
4. solves a centralized multiple-choice knapsack problem;
5. guarantees executed cost is at or below the annual budget.

This separates weak physical coupling between assets from strong resource coupling through the shared budget.

### Interactive hidden-shift evaluator

`infrawm.HiddenShiftEnv`

The evaluation environment owns transition dynamics that are hidden from the agent. A configurable shift occurs at a selected year. This permits a genuine

`Predict -> Act -> Observe -> Revise -> Re-plan`

loop rather than counterfactual replay on logged actions only.

## Run the smoke benchmark

```bash
pip install -r requirements-infrawm.txt
python scripts/run_infrawm_smoke.py
```

The output is written to `results/infrawm_smoke.json` and reports:

- final/mean network health;
- total maintenance cost;
- hard-budget violation count;
- prediction negative log-likelihood (NLL);
- post-shift NLL adaptation gain.

A static matrix model and Bayesian adaptive model are evaluated under the same shift.

## Unit tests

```bash
pip install pytest
pytest -q tests/test_infrawm.py
```

The tests currently verify hard-budget feasibility and that Bayesian updating increases posterior probability of the observed transition.

## FHWA NBI raw data downloader

```bash
python scripts/download_fhwa_nbi.py --state CA --start-year 1992 --end-year 2023
```

This downloads the same California/year range targeted by InfraRL into `data/fhwa_nbi/`.

Important: historical NBI schemas vary by year. The downloader intentionally does **not** claim that the downloaded files are immediately learning-ready. They still require harmonization/action verification before being used as transitions.

## Connection to InfraRL data

The cleanest integration path is:

1. run the original InfraRL data pipeline to obtain verified bridge-year records and transition matrices;
2. export transitions as `(state, action, next_state)` triples;
3. initialize `BayesianMatrixWorldModel.from_transitions(...)` from the training period only;
4. reserve later years for chronological L1/L3 adaptation evaluation;
5. use `HiddenShiftEnv` only for controlled counterfactual/interactive experiments.

Do not treat the cleaned InfraRL transition matrix as causal ground truth. The original pipeline removes some physically inconsistent records and fills unsupported rows with defaults, so it is best treated as a benchmark dynamics model.

## Next model tier

After the matrix baseline is validated, the next model should be a probabilistic sequence model:

`history + action -> categorical next-health distribution`

Recommended first neural baseline: GRU or small Transformer encoder plus categorical transition head, trained as an ensemble. Keep the same planner, hard-law layer, hidden-shift evaluator, and L1/L2/L3 metrics so model capacity is the only major change.

## Required evaluation table

| Model | L1 Predictor | L2 Simulator | L3 Evolver | Hard budget |
|---|---:|---:|---:|---:|
| Fixed Matrix | yes | yes | no | yes |
| Bayesian Matrix | yes | yes | yes | yes |
| Neural WM | yes | yes | no | yes |
| Ensemble Neural WM | yes | yes | no | yes |
| Ensemble Neural WM + online revision | yes | yes | yes | yes |

Key L3 metrics should include prediction-error reduction, cumulative regret after shift, time-to-recovery, and catastrophic forgetting on pre-shift transitions.
