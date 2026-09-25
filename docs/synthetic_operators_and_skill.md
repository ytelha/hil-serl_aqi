# Synthetic Operators and Automatic Skill Assessment

Implementation of the two-part methodology ("Synthetic Operator
Behaviour and Automatic Skill Classification", working document, and
"Synthetic Operator Models and Automatic Skill Assessment for
HIL-SERL", v1.0 — both 22 Jul 2026) on top of AQI-HIL-SERL. Part A
generates operator behaviour at graded, ground-truth-labelled skill
levels for both operator roles (demonstrations, interventions); Part B
types any operator — synthetic or human — from interaction logs.

> **Deep-dive companion docs.** For a thorough, standalone treatment of each part
> see [oracle_and_synthetic_operator.md](oracle_and_synthetic_operator.md) (Part A —
> the oracle → synthetic-operator design: perception → decision → motor, where
> noise enters, geometric vs policy oracle) and
> [operator_skill_assessment.md](operator_skill_assessment.md) (Part B — the idea
> and usage of automatic skill classification from interaction logs).

## What is NEW relative to AQI-HIL-SERL

| Piece | File | Role (doc section) |
|---|---|---|
| Operator model | `serl_launcher/serl_launcher/aqi/operator_model.py` | theta (Tables 2+4), tier anchors, theta(s) interpolation, jitter, profiles op1–op4, perception→decision→motor `SyntheticOperator`, `PhaseOracle`/`PolicyOracle` (S3–S4) |
| Intervention gate | `serl_launcher/serl_launcher/aqi/operator_gate.py` | `MonitoringSignal` z_t, SDT `SyntheticInterventionGate` (δ_in/m/L_react, p_miss, ρ_fa, δ_out/k, D_max), `SyntheticOperatorWrapper` at the SpaceMouse integration point (S6) |
| Toy study | `serl_launcher/serl_launcher/aqi/toy_insertion.py` | the 2-D worked example env + oracle + population harness (S7) |
| Skill pipeline | `serl_launcher/serl_launcher/aqi/skill.py` | Table-7 features (LDJ, SPARC, KM median, DTW, Â aggregates, ΔV_on…), Methods A–D, LOSO/QWK/ICC validation utilities, `features_from_aqi_logs` (S9–S11) |
| Demo collector | `examples/collect_synthetic_demos.py` | Role 1 → repo-format demo pkl + ground-truth sidecar; filtered/unfiltered (S5) |
| Skill CLI | `examples/operator_skill.py` | `--mode assess` (reports) and `--mode validate-synthetic` (the S11 loop) |

Touched existing files: `examples/train_aqi_rlpd.py` (flags
`--synthetic_operator`, `--synthetic_seed`, `--oracle_checkpoint`,
`--synthetic_env_mm_per_unit`; env wrapping in the actor; theta into
`meta.json`), `examples/experiment_pipeline.py` (`assess` subcommand +
`operators.csv` annotation), `serl_launcher/serl_launcher/aqi/__init__.py`
(exports), this file and pointers in README/walkthrough/protocol.
Everything downstream of the wrapper — intervention routing, buffers,
learner, the AQI gate and quality score — is untouched (requirement R5);
`git diff` against the vanilla repo shows the cumulative AQI + operator
delta, and `synthetic_operators_skill.patch` shows only this feature.

## Requirements R1–R6 → where enforced

Ordinal validity (R1): calibration test, three seeds, non-overlapping
Wilson CIs and ordered secondaries. Ground truth (R2): theta/s/tier in
every sidecar, `meta.json`, and `info["synthetic_operator"]`.
Human-plausible errors (R3): latency deque, per-APPROACH bias resampled
only on retract (the documented per-step-resampling pitfall is
structurally impossible), signal-dependent noise, OU tremor, geometric
pauses, whole-episode lapses ×2.5, q_rec recovery style, p_early
premature commits. Both roles (R4): the same theta drives
`collect_synthetic_demos.py` and the gate. Non-invasiveness (R5): the
wrapper emits `info["intervene_action"]` with the executed action, byte-
compatible with the SpaceMouse path. Determinism (R6): every random
draw flows through seeded `np.random.Generator`s; the calibration test
asserts byte-identical episodes per seed.

## Calibration on the toy study (doc Table 5 pattern)

200 episodes/tier, seeds 0–2: SR 0.29–0.33 < 0.78–0.80 < 0.99–1.00
(all adjacent Wilson CIs disjoint); jam events 6.7–7.3 > 3.6–3.8 >
0.6; alignment-at-commit 3.5–4.1 > 1.1 > 0.4 mm; s-sweep SR 0.34 →
0.47 → 0.77 → 0.99 → 1.00 (doc: 0.34 → 1.00, monotone). Known
divergence from the doc's own env: absolute jam counts differ (jam
events here are contact rising edges + one event per 0.5 s of sustained
grinding), and the novice successes-only median time lands at 5.8–7.9 s
(doc 3.9 s) — the censoring direction differs but the LESSON (successes-
only time is selection-biased; use the KM median) is identical and the
KM median is what the pipeline reports. Per doc S7/S8 the absolute
anchors are environment-specific; re-run the calibration in any new env
and adjust theta anchors until this separation pattern reappears.

## Classifier validation (doc S11 loop, 30 jittered ops × 12 episodes)

Method A composite: tertile cuts −0.49/+0.54 (doc −0.49/+0.44),
tier means −1.07/−0.01/+1.08, accuracy 1.00. Method B k-means: accuracy
1.00, ARI 1.00, bootstrap stability 1.00 (doc 0.97/0.90). Method C LOSO
ordinal logit: accuracy/macro-F1/QWK 1.00. Spearman ρ(S, true s) over a
9-point sweep: 0.983. Episodes-needed curve (2/4/8/12): 0.72 → 0.85 →
0.95 → 1.00 (doc shape 0.67 → 0.97). Split-battery test-retest ICC
0.909. Method D doc vectors reproduce exactly: Beta(8,4) mean 0.667
CI90 [0.44, 0.86]; Rasch θ̂=1.05 with P̂ 0.89/0.51. Reproduce with:
`python examples/operator_skill.py --mode validate-synthetic`.

## Usage

Calibrate/validate the generator: run the module tests or
`--mode validate-synthetic` above. Collect Role-1 demos:
`python examples/collect_synthetic_demos.py --profile novice
--episodes 20 --nofiltered --out demos/nov.pkl` (sidecar JSON carries
theta and outcomes; `--filtered` reproduces the HIL-SERL successes-only
regime and reports attempts-per-kept-demo). Drive training with a
synthetic operator (Exp-1 cells): `python examples/train_aqi_rlpd.py
--actor --exp_name=... --aqi_mode=aqi --synthetic_operator op3
--synthetic_seed 5 --operator_id op3` — sim envs expose a
`phase_oracle` attribute (doc option A); otherwise pass
`--oracle_checkpoint` (option B) plus `--synthetic_env_mm_per_unit`
for normalized action spaces. Assess operators from campaign logs:
`python examples/experiment_pipeline.py assess --campaign <dir>` or the
standalone `operator_skill.py --mode assess --log_root <runs>
[--demo_glob 'demos/*.pkl'] [--labels_json truth.json]`.

## Pitfalls the code enforces or surfaces (doc S13)

Per-approach (not per-step) perceptual bias is structural. Censoring:
capped means + KM medians everywhere; time comparisons flagged.
Matched-phase intervention comparisons and critic-warm-up caveats are
noted in reports (Â aggregates use the σ_Q-gated quality machinery).
LOSO only, never episode splits. Clustering warns below 15 operators.
Synthetic validation licenses ordinal claims; human thresholds must be
re-anchored (report footer says so). Goodhart: keep assessed operators
blind to the scoring details.
