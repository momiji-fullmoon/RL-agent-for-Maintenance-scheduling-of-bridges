# InfraAWM socio-technical extension

This document records an **optional second-stage extension** of InfraAWM from a purely physical infrastructure world model to a socio-technical world model. It is intentionally separated from the core benchmark so that the main claim remains falsifiable: first prove physical prediction, constrained planning, and online world-model revision; only then add stakeholder and institutional simulation.

## Why this extension is plausible

Several recent social/urban simulation systems suggest a common architecture: a structured environment, heterogeneous agents grounded in data or personas, explicit interaction protocols, and validation against real-world aggregate outcomes.

### SocioVerse: alignment as a first-class design problem

SocioVerse frames social simulation as a world-model problem and separates four alignment components:

1. **Social Environment**: updated external context and social structure;
2. **User Engine**: target-user reconstruction from a large real-world user pool;
3. **Scenario Engine**: scenario-specific interaction structure;
4. **Behavior Engine**: behavior generation conditioned on the other three components.

The paper validates simulations against real outcomes using value- and distribution-level metrics such as RMSE/NRMSE and KL divergence, rather than treating plausible text as sufficient evidence.

Source: https://arxiv.org/abs/2504.10157

### LLM-enabled Social Agents: roles need personas, norms, memory, and hybrid control

The 2026 conceptual baseline argues that a social agent should be role-enacting rather than merely a generic LLM wrapper. A role is operationalized through a persona-like representation containing goals, constraints, relationships, commitments, and dynamic state. The LLM is only one component of a **hybrid architecture**; memory, planning, normative structures, and control mechanisms remain explicit.

For InfraAWM this is important because budget, safety rules, agency responsibilities, and institutional boundaries should not be left to unconstrained language generation.

Source: https://arxiv.org/html/2605.02335v1

### LLM agents for smart-city management: use tools/RAG/validation for evidence access, not as the world model itself

Kalyuzhnaya et al. demonstrate a multi-agent smart-city decision-support architecture that routes urban questions to document databases and service APIs, uses Retrieval-Augmented Generation (RAG), and adds validation agents. Their experiments report high routing accuracy and better response quality when city data sources are integrated.

This pattern is useful for an **evidence/orchestration layer** in InfraAWM, but it should not be conflated with physical transition modeling. RAG can retrieve current regulations, costs, inspection records, or service constraints; it does not replace `P(s_{t+1}|s_t,a_t)`.

Source: https://www.mdpi.com/2624-6511/8/1/19

### SimCity: separate environment, interaction protocol, and heterogeneous agents

SimCity explicitly separates three layers: environment, interaction protocol, and agents. Its heterogeneous household, firm, government, and central-bank agents interact through structured markets and are tested against a checklist of macroeconomic stylized facts and exogenous shocks. The paper also warns that LLM agents can produce abnormal actions and therefore applies explicit heuristic checks; it further states that the simulation is not quantitatively calibrated as a prediction of reality.

For InfraAWM, the transferable idea is the **validation protocol**: a simulator should reproduce known regularities and respond coherently to controlled shocks before it is trusted for novel scenarios.

Source: https://arxiv.org/html/2510.01297v3

### Audience simulation: synthetic populations require external validation

The AIMultiple review/benchmark is not primary scientific evidence, but it is useful as a failure warning. Its benchmark notes that many LLMs remain close to the 50% chance baseline on a binary engagement-prediction task, and it explicitly recommends validating virtual-audience predictions against actual outcomes. It also highlights bias, generalization, interpretability, and computational-cost risks.

InfraAWM should therefore treat synthetic stakeholder populations as **hypothesis generators unless calibrated and externally validated**.

Source: https://aimultiple.com/audience-simulation

## Proposed architecture

The extension should preserve the current physical core and add social/institutional layers around it.

```text
Historical NBI / inspection / cost data
              |
              v
+----------------------------------+
| Physical World Model             |
| P(condition', cost | state, act) |
| + calibrated uncertainty         |
+----------------+-----------------+
                 |
           imagined futures
                 |
                 v
+----------------------------------+
| Constrained Planner              |
| MPC + centralized budget solver  |
+----------------+-----------------+
                 |
        candidate maintenance plan
                 |
                 v
+----------------------------------+
| Institutional / Evidence Layer   |
| regulations, records, APIs, RAG  |
| explicit validation              |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
| Stakeholder Simulation Layer     |
| role/persona-grounded agents     |
| norms, commitments, memory       |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
| Joint evaluation                 |
| physical + fiscal + social       |
+----------------------------------+
```

The key separation is deliberate:

- **physical dynamics** are learned from infrastructure data;
- **hard laws** such as annual budget and action feasibility remain symbolic constraints;
- **urban information systems** supply evidence and current context;
- **stakeholder agents** model heterogeneous institutional/social responses;
- no LLM is allowed to overwrite physical laws or budget feasibility.

## Candidate agent roles (design proposal, not source-derived ground truth)

A minimal socio-technical experiment could use four role types:

- maintenance engineer: prioritizes structural condition and intervention effectiveness;
- budget authority: enforces fiscal constraints and portfolio-level trade-offs;
- mobility/service representative: evaluates disruption/access consequences;
- citizen/stakeholder agents: represent heterogeneous acceptance and perceived service impacts.

Each role should be represented as structured state rather than a one-line prompt:

```text
role
stable goals
formal responsibilities
allowed actions
forbidden actions
resources
current observations
memory / commitments
relationships
uncertainty
```

This follows the role/persona + norms + memory separation suggested by the LLM-enabled Social Agents paper.

## Benchmark redesign

### Track A: Physical AWM -- mandatory

This remains the current InfraAWM core.

- L1: next-state distribution prediction;
- L2: long-horizon constrained simulation/planning;
- L3: online revision after dynamics shift.

No social-agent result should compensate for weak performance on this track.

### Track B: Institutional decision support

Goal: test whether the planner can incorporate current rules, records, and service data through explicit tools/RAG without hallucinating constraints.

Suggested metrics:

- routing/tool-selection accuracy;
- evidence coverage;
- unsupported-claim rate;
- constraint extraction accuracy;
- final plan budget violation rate (must remain zero).

The MDPI smart-city paper's routing and response-quality experiments motivate this track, but InfraAWM should add stronger evidence-consistency metrics because answer fluency is not sufficient for maintenance decisions.

### Track C: Stakeholder world model

Goal: evaluate whether simulated stakeholder responses reproduce held-out aggregate observations.

When suitable ground truth exists, use both point and distribution metrics:

- RMSE / NRMSE for aggregate quantities;
- KL divergence or Jensen-Shannon divergence for distributions;
- calibration error for probabilistic responses;
- role/persona coherence across repeated interactions;
- norm/commitment violation rate.

The first three are motivated by SocioVerse-style external validation; the latter two follow the evaluation agenda in LLM-enabled Social Agents.

### Track D: Shock and adaptation

Create controlled exogenous changes, analogous to the shock tests used in SimCity:

- deterioration-rate shift;
- intervention-cost inflation;
- sudden budget cut;
- regulatory/action-feasibility change;
- service-demand or stakeholder-preference shift.

Measure:

- cumulative regret after shift;
- time-to-recovery;
- prediction-error reduction after online revision;
- change in stakeholder-distribution error;
- catastrophic forgetting on pre-shift regimes.

## Validation hierarchy

A result should only be promoted to a stronger claim when it passes the previous level.

1. **Plausibility**: outputs are syntactically/semantically reasonable.
2. **Internal consistency**: actions satisfy hard laws and agent roles remain coherent.
3. **Predictive validity**: held-out physical and social outcomes are predicted accurately.
4. **Distributional validity**: aggregate distributions match real data, not only averages.
5. **Intervention validity**: controlled policy/action changes produce externally supported responses.
6. **Adaptive validity**: performance recovers after distribution shift through model revision.

Synthetic audience results that lack level 3 or higher should be labeled exploratory, not predictive.

## What not to do

- Do not replace the physical world model with an LLM-only simulator.
- Do not claim that persona realism implies behavioral validity.
- Do not use RAG response quality as evidence that long-horizon infrastructure dynamics are correct.
- Do not treat a synthetic population as a substitute for empirical stakeholders without held-out validation.
- Do not add many stakeholder agents before the physical L1/L2/L3 baseline is stable; otherwise the source of failure becomes unidentifiable.

## Implementation order

1. Finish the current Fixed Matrix vs Bayesian Evolver benchmark and real NBI transition pipeline.
2. Add a generic `EvidenceProvider` interface for regulations/records/API-derived context.
3. Add structured `RoleState` and `NormConstraint` data classes without an LLM dependency.
4. Add one LLM-backed stakeholder role behind a replaceable interface.
5. Build a small held-out stakeholder validation dataset before scaling agent count.
6. Only after external validation, add population sampling and multi-agent interaction.

The 80/20 objective is therefore **not** to reproduce SocioVerse or SimCity at full scale. It is to import their strongest design principles -- alignment, heterogeneity, explicit interaction protocols, external validation, and shock testing -- while preserving the testability of the core InfraAWM task.
