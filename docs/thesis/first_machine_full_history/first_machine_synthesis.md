# The first training machine — synthesis

**Master analysis of everything that trained on the first training machine** (4 September – 31 October 2026),
what it showed, what it means, and how well each number is checked. Companion of
[second_machine_synthesis.md](../second_machine_synthesis.md); both machines are indexed in [README.md](../README.md).

**Explained in depth (written here, from the code, configs and reports), in the order the campaigns ran:**
- [first_machine_teacher_and_ensemble_results_and_qa_log.md](first_machine_teacher_and_ensemble_results_and_qa_log.md): how the teacher effect is measured, the first week's ruler bug, the two oracles, and the critic ensemble (A1–A4, B1, B2);
- [first_machine_grading_and_asking_results_and_qa_log.md](first_machine_grading_and_asking_results_and_qa_log.md): quality replay, the counterfactual and behavioural scores and the timing audit, σ_Q against physical trouble, the first asking experiment (C1–C6, D1, D2);
- [first_machine_asking_rate_results_and_qa_log.md](../first_machine_asking_rate_results_and_qa_log.md): what the critics' disagreement σ_Q is, how it becomes a request, how the operator answers, and the asking-rate campaign D3;
- [first_machine_oracle_quality_results_and_qa_log.md](../first_machine_oracle_quality_results_and_qa_log.md): what the quality oracle computed, how a score enters training, the oracle campaign C11 and the demonstration probe.

Per-campaign detail also lives in the machine's own logs, in this folder:

| log | campaigns |
|---|---|
| [rq1_campaign_engineering_log.md](../rq1_campaign_engineering_log.md), [rq1_oracle_setup_and_qa_log.md](../rq1_oracle_setup_and_qa_log.md) | A1–A4: does HIL-SERL depend on its operator? |
| [rq2_quality_replay_and_qa_log.md](../rq2_quality_replay_and_qa_log.md) | C1–C3: quality-prioritised replay |
| [rq3_structural_ablation_and_qa_log.md](../rq3_structural_ablation_and_qa_log.md) | B1: the ensemble itself |
| [rq4_ensemble_convergence_and_qa_log.md](../rq4_ensemble_convergence_and_qa_log.md) | B2, C4, C5: the fixed ensemble, the score probes |
| [active_ratematch_results_and_qa_log.md](../active_ratematch_results_and_qa_log.md) | D3: how often to ask |
| [aqi_oracle_quality_results_and_qa_log.md](../aqi_oracle_quality_results_and_qa_log.md), [active_quality_campaigns_synthesis.md](../active_quality_campaigns_synthesis.md) | C11 and the demonstration probe |
| [wandb_guide_and_run_cleanup.md](../wandb_guide_and_run_cleanup.md) | the W&B record (169 runs) |

The numbers below come from the reports copied into `presentation_bundle/results/`, the W&B export
(`wandb_analysis/`) and, where nothing else came back, the machine's own logs. Section 9 says which is which.
Campaign IDs are the thesis's (`tab:programme`); the logs use older names (`rq2_*` = C1–C3, `rq3_*` = B1,
`rq4_*` = B2).

---

## 0. Executive summary

- **What ran:** 16 training campaigns, **459 cells**, on the original platform (an operator who cannot turn the peg;
  rolling-window convergence; no orientation limit). Nothing is running now. E2HiL and HiRIL never ran.
- **What holds:**
  1. **HIL-SERL depends on its operator,** clearly in three campaigns (ICC 0.60, 0.99, 0.66; p 0.035, 0.004, 0.001),
     but not in every campaign (0.00–0.30 elsewhere). A single campaign's ICC is unstable; the pooled analysis is not done.
  2. **The default critic ensemble hurts** (−18.5 points, ICC 0.03 → 0.74) **and is fixable to parity** (−1.9 points,
     p 0.30, 16 of 18 cells converge).
  3. **No learned quality score works.** Quality replay lost to plain HIL-SERL every time (C3: −13.0 points, p 0.031;
     B1: −35.8, p 0.006); the counterfactual score is anti-predictive (AUROC 0.18); the behavioural score was timing
     leakage (0.78 → 0.50 on clean takeovers).
  4. **Asking works.** The uncertainty gate matches the operator's own monitor at one sixth of the help (71 vs 78 %,
     p 0.58; 9 vs 54 % help) and, asking in the 55 % most uncertain steps, beats it (87 vs 74 %, p 0.021). Against plain
     HIL-SERL it gains 8 points, not significant (p 0.18).
- **What is not settled:** the oracle campaign's "even a perfect score amplifies dependence" (its arms also changed
  hard filters and imitation; the novice test decides); whether the gate's timing beats random requests (+5 to +15
  points, never significant).
- **What the second machine changed:** this machine's operator could rarely rescue a twisted cuboid (7–13 %). The paired
  comparisons stay fair, but the slow start and possibly part of the dependency come from it. The next job here is to
  test whether the conclusions survive with an operator who can.

## 1. The machine

| | |
|---|---|
| hardware | one Linux workstation: 8 GB Blackwell GPU (shared; VRAM checked with `nvidia-smi` before each campaign), 30 GB RAM, 1.9 TB disk |
| software | Linux, Python 3.12.3, JAX 0.11.0, MuJoCo 3.10.0: the package lock the second machine and the cluster copied |
| repo | `/home/user/Downloads/hil-serl_aqi`; baseline `1259581` (24 Sep snapshot) plus patches 0001–0010 (applied 30 Sep) plus its own local code: quantile trigger, random requests, per-arm trigger keys, `oracle` quality score, milder novices, HiRIL re-calibration, W&B groups (all imported into the development repo as patch 0020). It lacks 0011–0029. |
| how campaigns run | `run_campaign.py` from a YAML, in a `systemd-run --user` unit with `ManagedOOMPreference=avoid`; W&B online |
| pace | the actor runs at 10 Hz; the learner makes about **0.45 updates per environment step** for HIL-SERL and 0.45–0.63 for the ensemble arms (W&B); about 0.8 h of training per HIL-SERL cell, 1.2–1.5 h per ensemble cell, plus the ruler |
| storage | about 2.5 GB per cell after replay dumps were switched off |

## 2. What ran

| when | ID | config | question | arms | cells | evidence |
|---|---|---|---|---|---|---|
| 4–8 Sep | A1 | `rq1_3seed_nopretrain` | does HIL-SERL depend on its operator? | `hilserl`, geometric operators, 1 per tier × 3 seeds | 9 | report |
| Sep | A2 | `rq1_s075_5seed` | the same with 5 seeds | `hilserl` | 15 | log |
| Sep | A3 | `rq1_3seed_pretrain` | with demonstration pre-training | `hilserl` | 9 | log |
| 10 Sep | A4 | `po_rq1_campaign` | with policy-oracle demonstrations | `hilserl` | 9 | report |
| 10–18 Sep | C1–C3 | `rq2_quality_replayonly*` | does quality-prioritised replay help? | `hilserl`, `quality` (three stabilisation stages) | 18 + 12 + 12 | log; C3 also W&B |
| 18–21 Sep | B1 | `rq3_plain_logged_quality` | what does the critic ensemble itself do? | `hilserl`, ensemble extended (K 10, M 2), `quality`; 2 per tier × 2 seeds | 36 | report, W&B |
| 21–23 Sep | B2 | `rq4_ensemble_convergence` | can the ensemble be fixed? | `hilserl`, fixed ensemble (K 8, M 1, warm start, LR / 2); 3 per tier × 2 seeds | 36 | report, W&B |
| 23 Sep | C4, C5 | `quality_on_rq4_diag`, `quality_behavioral_diag` | does a healthy critic rescue the score? is the behavioural score valid? | `quality` | 3 + 3 | reports |
| 30 Sep | C6 | `quality_signal_audit.py` | timing-leakage audit of C5 | no training | – | report |
| 1–2 Oct | D1 | `active_validity` | does the critics' disagreement σ_Q track physical trouble? | `active` with simulator ground truth | 9 | report, W&B |
| 2–6 Oct | D2 | `active_effect` | does asking teach a good policy? | ensemble extended, `active`, rate-matched random, `active-loud` | 72 | report, W&B |
| to 21 Oct | D3 | `active_ratematch` | how often should the robot ask? | `hilserl`, ensemble extended, asks 10 / 30 / 55 %, random 30 % | 108 | report |
| to 29 Oct | C11 | `aqi_oracle_quality` | does a perfect quality score help? | ensemble extended, asks 30 %, quality-oracle, asks + oracle, asks + counterfactual | 90 | its log |
| to 31 Oct | – | `quality_starvation_probe` | do more demonstrations rescue the novice? | novices only, 20 demonstrations | 18 | its log |

All on the cuboid, the frozen 9-pose ruler, and 3 per tier × 2 seeds unless stated. Caps: 40 k steps (35 k in D3, 50 k in B1).
Total 459 cells. Smoke tests, stability probes and superseded runs are not counted
(`presentation_bundle/01_RESULTS_INDEX.md`, last section).

**On the RWTH cluster, not this machine** (separate copy, never pooled): the method test of 26 Sep (3 000-step cells of
every arm, including E2HiL and HiRIL) and the counterfactual-score study C9 (1 cell, 29 Sep; result in
[../counterfactual_score_study.md](../../counterfactual_score_study.md) §5).

## 3. How it went: the incidents that shaped the study

| when | what looked wrong | real cause | fix | lesson |
|---|---|---|---|---|
| 4–8 Sep | RQ1 showed no operator signal (ICC ≈ 0) | the ruler built a cylinder while training used the cuboid | `--peg_shape` on the ruler; signal reappeared (ICC 0.60 / 0.99) | check that the evaluator evaluates the trained task |
| Sep | early quality campaign: ICC 0.79, novices at 0 % | only novice cells ran out of GPU memory in the counterfactual batch (longer takeovers) | fixed-shape chunking with a CPU fallback | a crash correlated with the independent variable looks like an effect |
| Sep | identical resets in every replicate | a fixed random state for resets | `--env_seed` per replicate | honest seed noise |
| Sep | the last checkpoint of a cuboid policy is random (0 ↔ 78 % within 1 000 steps) | oscillating policies | the ruler scores the best window of 3 checkpoints | report sustained peak competence |
| Sep | a grid filled the 1.9 TB disk | replay dumps of 12–17 GB per cell | dumps off by default; about 2.5 GB per cell | – |
| Sep | the OOM killer took down the editor and the run | they shared one cgroup | each campaign in its own `systemd-run` unit | – |
| 23–30 Sep | the behavioural score looked valid (AUROC 0.78) | 61 % of takeovers end at the duration cap: timing leakage | leakage audit; reported as a null | audit any signal against the confound it resembles |
| 26 Sep (cluster) | the operator roster changed between processes | the original code iterated a set | patch 0007 (applied 30 Sep); the roster audit on this machine found all finished campaigns consistent | verify what the code really ran, not what the config says |
| 21 Oct | report headline had a sign error, a wrong operator count, and a frontier plot with HIL-SERL at 0 % help | display bugs; the tables were correct | patch 0013, not yet on this machine | – |
| 29 Oct | "even a perfect score amplifies dependence" | its quality arms also ran `outcome_filter` (drops every takeover of a failed episode) and a BC loss of 0.1 | novice test prepared (`docs/novice_ablation.md`), not run | change one thing at a time |

The full reasoning per decision is in `presentation_bundle/02_DECISION_LOG.md` (phases A–G).

## 4. What came out

| question | result | strength |
|---|---|---|
| Does HIL-SERL depend on its operator? (A1–A4, D3) | geometric operators: ICC 0.60, p 0.035, tiers 1.00 / 0.82 / 0.59; policy-oracle demonstrations: ICC 0.99, p 0.004, novice 6 %; D3: ICC 0.66, p 0.001 | real, but campaign-dependent |
| Is it stable across campaigns? | A2 (5 seeds) 0.30, p 0.10; A3 (pre-training) 0.00; B1 0.03; B2 0.00; seed variance large (C2 and C3 hilserl 82 vs 95 %) | no: single-campaign ICC is unstable; pooled model A5 not done |
| Does the default ensemble work? (B1) | −18.5 points vs hilserl (86.7 → 68.2 %), ICC 0.03 → 0.74, 7 of 12 converge | clear |
| Can it be fixed? (B2) | 82.3 vs 84.2 %, −1.9 points, p 0.30, ICC 0.00 both, 16 of 18 converge each | parity; which of the four fixes matters is unknown (B3 not run) |
| Does quality replay help? (C1–C3, B1) | always worse than hilserl: C1 −21.0, C2 −33.3 (intermediate collapses to 6 %), C3 −13.0 (p 0.031, 6 of 6 cells); B1 quality −35.8 (p 0.006), novice 4.6 % | clear negative, with the counterfactual score |
| Is the counterfactual score valid? (C4, C9) | AUROC 0.175 vs episode success on the fixed critic; on the cluster, against ground truth, every variant below chance → abandon by the fixed rule | clear negative |
| Is the behavioural score valid? (C5, C6) | 0.78 live, 0.499 on clean takeovers; takeover start time alone 0.69–0.83 | timing leakage |
| Does σ_Q detect trouble? (D1) | jam in the contact zone 0.51; takeovers 0.62 only because the operator obeys the gate | no |
| Does asking teach a good policy? (D2) | 70.8 vs 78.2 % (p 0.58) at 9 vs 54 % help; ICC 0.84 → 0.00; random at a matched rate 56.2 % (p 0.007 vs the monitor) | yes, statistically equal at one sixth of the help |
| Does more asking help? (D3) | 70 → 81 → 87 % at 16 / 28 / 42 % help; asks 55 % vs ensemble extended +13.2 (p 0.021); vs hilserl +8.0 (p 0.18), almost all among intermediates; asks 30 % matches hilserl at half the help | yes against the same ensemble; not yet against hilserl |
| Is the gate's timing better than chance? (D2, D3) | vs random at a matched rate: +14.6 (9 / 8 / 1, sign test p 1.00; Wilcoxon 0.40) and +4.9 points (10 / 5 / 3, p 0.30) | suggestive, not shown |
| Does a perfect quality score help? (C11) | ensemble extended 79.0 % / ICC 0.18; quality-oracle 73.3 % / 0.63; asks + oracle 78.8 % / 0.93 (novice 36 %); asks + counterfactual 73.4 % / 0.97 | undecided: three changes at once; raw data not checked |
| Do more demonstrations rescue the novice? (probe) | 20 instead of 6: asks + oracle novice 36 → 27 % | no; raw data not checked |

## 5. What this means

1. **The thesis's first claim stands with a caveat.** The operator matters to plain HIL-SERL, most strongly when
   demonstrations come from a realistic (policy) oracle. Whether it matters in a given campaign varies, so the thesis
   reports paired designs and needs the pooled model (A5).
2. **The ensemble is a prerequisite, not a cost.** Fixed, it matches HIL-SERL, so the active and quality arms are not
   confounded with a weaker learner.
3. **Grading corrections failed with every learned score**, for diagnosable reasons. The oracle campaign does not yet show
   that grading is wrong in principle; the novice test decides.
4. **Asking is the positive result.** It saves human effort at equal success and removes the measurable dependency. Part of
   that is mechanical: novices answer few of the 10-step requests (7–13 % of the steps under the gate against 43–47 %),
   so their corrections largely leave the data.
5. **Threats to validity, all reported in the thesis:**
   - the operator could rarely rescue a twisted cuboid (second machine; the slow start of every cuboid campaign);
   - the gated arms converge less often under the rolling-window rule (8–14 of 18 against 15 for hilserl);
   - the learner makes more updates per step for some arms than others (0.45 for hilserl, up to 0.63 for ensemble arms);
   - single-campaign ICC is unstable; compare dependency only within one campaign.

## 6. What did not run here

| | why it matters |
|---|---|
| A5 pooled dependency model (no training) | a test of RQ1 with far more power than any single campaign |
| A6 ladder (27) and A7 original vs new stack (54) | does the main result survive an operator who can turn the peg? |
| B3 attribution of the ensemble fixes (72) | which fix matters |
| C12 novice test (24) | which part of the oracle campaign's arms collapses the novice |
| C13 clean filter test (36) | can removing bad corrections help at all? |
| D4 asking follow-up | more seeds against random; OR arm; held requests |
| E1–E3 E2HiL and HiRIL comparison (54+) | the state of the art; never ran here |

The order, cost and reading rules are in [second_machine_synthesis.md](../second_machine_synthesis.md) §6 and
`handoff/first_machine/HANDOFF_first_machine.md`. The bundle `first_machine_bundle_2026-10-31.tgz` carries the novice test
and patches 0011–0021; patches 0024–0029 are not in it yet.

## 7. Against the second machine

| | first machine | second machine |
|---|---|---|
| dates | 4 Sep – 31 Oct | 24–29 Oct |
| cells | 459 | 58 (57 cuboid) |
| platform | original: operator moves the peg but cannot turn it; rolling-window convergence; unbounded orientation | new stack: rotation control, orientation box and channel, mini-eval stop, update ratio 1 |
| operator | consistent | inconsistent (hidden careful / careless state) in 57 cells |
| learner updates per env step | about 0.45 (hilserl) | 1 |
| what it answered | RQ1–RQ4 | can the operator help at all; do scores notice carelessness; discard vs re-weight |
| first success without help, ensemble extended | about 18 300–20 200 steps | about 4 600 |

**Never pooled.** Updates per environment step differ between machines, so every comparison has its reference on its own
machine. Results without learning (the operator test, the tolerance sweep) carry over.

## 8. State today

Nothing is running. The machine holds 0001–0010 plus its local code. Two small outputs it was asked for never came
back: `first_machine_state.txt` and the count of what the oracle campaign's filters dropped. The next step there is the
novice test, then the patches and the ladder.

## 9. Artifacts, and how well each number is checked

| level | campaigns | where |
|---|---|---|
| report copied and read | A1, A4, B1, B2, C4, C5, C6, D1, D2, D3 | `presentation_bundle/results/{rq1_geometric, rq1_policy_oracle, rq3_structural, rq4_convergence, quality_on_rq4_diag, quality_behavioral, active_validity, active_effect, active_ratematch}/` |
| raw training logs checked | B1, B2, C3, D1, D2 (169 W&B runs, every episode) | `wandb_analysis/analysis/` (REPORT.md, tables, episodes.csv.gz) |
| machine's own log only | A2, A3, C1, C2, C11, demonstration probe | the logs listed at the top |
| source and configs | the machine's code on 31 Oct | `Downloads/first_machine_code.tgz` |

The D3 pairings against hilserl were computed here from the report's per-cell tables. No first-machine number has
been re-derived from its checkpoints.
