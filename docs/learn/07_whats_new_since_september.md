# 07 — What changed since September: the new stack, the campaigns, where the results stand

*Added 2026-11-08; **updated 2026-11-10** with the latest runs on both machines (the comprehensive study and the
ensemble-fix ablation) and the data home `results/`. Parts 01–05 were written on 29 September. Read this part
after part 02. Every item points to the file or the doc that holds the details.*

---

## 1. New in the code: the "new stack" and the operator model

The **new stack** is the set of switches the second machine has run since October, and both machines ran in
November. All of them are off by default, so older campaigns stay reproducible.

| feature | what it does | switch | code |
|---|---|---|---|
| rotation control | the synthetic operator also turns the tool back to an insertable orientation, and pushes only once position *and* orientation are aligned | `--synthetic_rotation_control` | `aqi/operator_model.py` |
| orientation box | the environment limits how far the tool can tilt and twist (±0.1 / ±0.1 / ±0.4 rad), as on the real robot | campaign config | `examples/experiments/sim_peg_insertion/config.py`, `wrapper.py` |
| twist channel | the operator's monitor also reacts to tilt or yaw errors, so a twisted but centred peg is taken over before it jams | `--synthetic_orientation_monitor` | `aqi/operator_gate.py` (`MonitoringSignal`) |
| mini-eval stop rule | a run stops when a small autonomous test (5 poses, every 2 000 steps) passes ≥ 0.6 twice in a row; independent of how often the operator helped | `--converge_mode eval` | `examples/train_aqi_rlpd.py` |
| **fixed update ratio** | exactly one learner update per robot step; the actor waits for the learner. Without it the ratio depends on the GPU (about 0.45 on the first machine) | `--utd_ratio 1` | `serl_launcher/utils/update_ratio.py` |
| takeover channels | logs which monitor channel caused each takeover and how it ended (logging only) | `--log_takeover_channels` | `train_aqi_rlpd.py` |
| sometimes-careless operator | a hidden careful / careless state per takeover, logged as ground truth (see part 06) | `--synthetic_inconsistent`, `--synthetic_bad_share`, `--synthetic_bad_persistence` | `aqi/operator_model.py` |
| discard filters | drop the lowest-scored share of takeovers after a warm-up; also by the hidden label (perfect filter) or at random (control) | `quality_filters: score_filter / oracle_filter / random_filter`, `quality_drop_fraction` | `aqi/runtime.py` → `_discard` |
| quality oracle | a takeover score from the simulator's truth (trouble removed, jam, outcome): a ceiling, not deployable | `quality_score_mode: oracle` | `aqi/quality.py` → `oracle_q` |
| OR arm, held requests, smooth hand-over | the operator may step in unasked while the robot asks; a request stays on until answered; operator and robot actions are blended at the hand-over | `--synthetic_request_or`, `--synthetic_request_hold_steps`, `--synthetic_handover_steps` | `train_aqi_rlpd.py`, `aqi/operator_gate.py` |
| shorter ruler | skip the pre-training checkpoints and score only the newest 8: 36 % of the ruler time, no score changed on 57 runs | `eval: skip_pretrain`, `newest: 8` in the campaign config; re-score a finished campaign with `examples/ruler_reselect.py --skip_pretrain --newest 8` | `examples/ruler_select.py` (patch 0037; the first machine doesn't have it) |
| E2HiL and HiRIL on the new stack | both baselines run with rotation control, the twist channel, the mini-eval stop and the fixed ratio, also with the careless operator | E2HiL: mode `e2hil`; HiRIL: its own trainer | `examples/train_hiril.py`; docs `e2hil_baseline.md`, `hiril_baseline.md` |
| synthetic-operator test suite | one driver for ladders, pilots, the careless-operator experiments and the comprehensive study, including the export of results | `examples/synthetic_suite.py --suite …` | `docs/synthetic_suite.md` |

**Exercise:** for each row, open the file and find where the switch is read. Then explain in one sentence why it is
off by default.

## 2. What ran, and what came out — in time order

All numbers are for **experts and intermediates**. Since 8 November the novice is out of every result except the
teacher dependency. **Compare methods only within one campaign** (§3).

### 2.1 First machine, old stack (September–October)

Asynchronous learner (about 0.45 updates per step), an operator that couldn't turn the peg.

| campaign | question | result | read it as |
|---|---|---|---|
| teacher dependency | does the robot inherit its teacher? | yes: ICC 0.60 with geometric demos, 0.99 with policy-oracle demos (all three tiers; this result keeps the novice) | holds for the old stack with novices |
| first vs fixed ensemble | can 8–10 critics learn as well as 2? | the first version: 79 vs 92 %, but 3 wins, 3 losses (p = 1.0); it converged later in 7 of 8 pairs. The fixed version: 95 vs 89 %, p = 0.18 | the first version's main problem was slower convergence, not lower success |
| first asking experiment | does asking at the right time help? | a fifth of the help, but 75 vs 91 % success; random timing only 56 % | asking about 1 step in 10 was too little |
| asking rate | how often should the robot ask? | asking in the 55 % most uncertain steps: 94 % against the fixed ensemble's 75 % (p = 0.021) and HIL-SERL's 81 % (p = 0.039), with the same help | a within-campaign result on the old stack, against references that were weak in that campaign; **not replicated on the new stack** (§2.4) |
| grading with the critics' comparison | does the counterfactual advantage grade corrections? | no: AUROC 0.15 (inverted), −5 points in training | confirmed by every later test |
| quality oracle | would a perfect score help? | 85.8 → 94.1 % (operator decides) and 95.7 → 100 % (robot asks) | three things changed at once; for novices the novice test settled it (§2.4b) |

Docs: [`thesis/first_machine_latest_experiments_guide.md`](../thesis/first_machine_latest_experiments_guide.md),
[`thesis/active_ratematch_results_and_qa_log.md`](../thesis/active_ratematch_results_and_qa_log.md),
[`thesis/aqi_oracle_quality_results_and_qa_log.md`](../thesis/aqi_oracle_quality_results_and_qa_log.md).

### 2.2 Second machine, new stack (24–29 October, 58 runs)

| campaign | result |
|---|---|
| operator alone | rotation control: twisted-cuboid rescue 10–13 → 100 %. The expert tolerates 8° of yaw; at a takeover the peg is off by about 21° |
| training starts sooner | first success without help after about 4 300 instead of 16 600–20 300 steps. Five settings changed at once; the first machine's ladder attributed the gain later (§2.4b) |
| careless operator (40 %), scores | behavioural score 0.70 within a run, length 0.69, hindsight 0.65; outcome, progress and the critics' comparison at about chance (part 06) |
| careless operator, training | discard vs re-weight by the behavioural score: 93 vs 94 % success, but discarding converged faster (10 of 12 pairs, p = 0.039) |

Docs: [`thesis/second_machine_latest_experiments_guide.md`](../thesis/second_machine_latest_experiments_guide.md),
[`thesis/second_machine_synthesis.md`](../thesis/second_machine_synthesis.md),
[`thesis/inconsistent_operator_results_and_qa_log.md`](../thesis/inconsistent_operator_results_and_qa_log.md).

### 2.3 RWTH cluster

- **Counterfactual study, 29 September:** better-grounded critic scores all rank below chance against the
  simulator's truth ([`counterfactual_score_study.md`](../counterfactual_score_study.md)).
- **Small E2HiL campaign, 5 November:** failed technically (a rendering stall at step 201). No result.

### 2.4 The latest runs: both machines on the new stack (November)

| | first machine: ensemble-fix ablation | second machine: comprehensive study |
|---|---|---|
| runs | 96: 8 arms × 6 operators × 2 seeds | 168: Study A (critics, asking) 96, Study B (grading, 20 % careless expert) 72 |
| what came back | the report, then the slim run folder (13 Nov) | the full export |
| details | [`thesis/ensemble_fix_ablation_results.md`](../thesis/ensemble_fix_ablation_results.md) | [`thesis/comprehensive_study_results.md`](../thesis/comprehensive_study_results.md) |

**Both together** ([`thesis/latest_runs_both_machines.md`](../thesis/latest_runs_both_machines.md)):

1. **The task is at the ceiling.**
   - Every method reaches about 90–99 % on the ruler within about 6 000–9 000 steps.
   - There is no teacher dependency left (ICC ≈ 0).
   - Final success no longer separates methods; speed and human effort do.
2. **The machines agree.** HIL-SERL reaches 93.5 % with 39 % help on both. With one update per step, the level no
   longer depends on the machine.
3. **Ensembles.**
   - Any number of critics (4, 8, 10, 16) matches HIL-SERL, and converges faster.
   - The first version no longer fails, so no part of the fix is needed for the final policy (96–100 % at the last
     checkpoint in every arm).
   - The warm start is the one change that makes ensembles faster: 78 against 42 % (first version) and 23 % (HIL-SERL)
     on the ruler after 4 000 steps. De-pessimisation adds nothing, and the warm start slows plain HIL-SERL.
   - Ensembles need slightly less help (10 of 12 pairs, p = 0.039).
4. **Asking does not replicate.** Asking at 40–70 % gives the same success as the operator deciding, with more help
   (45–56 vs 38 %); random requests do as well.
5. **Grading.**
   - With a 20 % careless expert the behavioural score reaches 0.76 within a run, which is "valid". Hindsight reaches
     0.74, and the score keeps 0.72 without its length term.
   - No training gain is visible at the ceiling.
   - The critics' comparison stays at chance against carelessness. In the ablation it points the wrong way whenever
     the critics are optimistic (one-critic target or warm start).
6. **Baselines.**
   - E2HiL: same success, slower.
   - HiRIL: not fairly tested. Its requests last 2.7 steps on average, and the operator answered only 7.5 % of them.
7. **Measurement problems found:**
   - the ruler's 3-checkpoint window meets the early stop badly: a policy still unstable at the stop scores low;
   - one E2HiL run crashed and was counted as finished.

### 2.4b First machine, early November: the novice test and the ladder

Found on 14 Nov in the first machine's full run folder; until then the docs called both "not run". Details:
[`thesis/first_machine_novice_test_and_ladder.md`](../thesis/first_machine_novice_test_and_ladder.md).

| campaign | question | result | read it as |
|---|---|---|---|
| novice test (4–5 Nov, novices only, 6 runs per arm) | which part of the perfect-score arms collapses the novice? | imitation alone: 13 %; score + imitation without filters: 8 %; score as replay priority only: 68 %; filters only: 80 %; ensemble extended: 65 % | **imitating a novice's corrections** collapses the novice, not grading them |
| ladder (5–6 Nov, HIL-SERL, 3 runs per setting) | which new-stack change removes the slow start? | first unaided success 18 275 steps (original) → 7 118 (+ rotation control) → 4 464 (full stack with one update per step); the box and the stopping rule don't speed learning | the operator who can turn the peg can rescue (7–13 → 70–100 %), and one update per step does the rest |

### 2.5 Still open

| what | status on 10 Nov |
|---|---|
| raw run folders of the first machine's older campaigns (above all the quality oracle) | done 14 Nov: the 15 valid campaigns are in `results/first_machine/campaigns/` |
| a measure that isn't at the ceiling, and a harder task | proposed, see [`thesis/latest_runs_both_machines.md`](../thesis/latest_runs_both_machines.md) §4 |
| novice test | **ran 4–5 Nov** (found 14 Nov): the imitation loss collapses the novice, not the score; see [`thesis/first_machine_novice_test_and_ladder.md`](../thesis/first_machine_novice_test_and_ladder.md) |
| porting AQI to the lab's real UR5 | planned; waiting for the lab's repo |

## 3. Five lessons that change how you read every result

1. **On the first machine a seed doesn't fix a run (finding F3).**
   - The robot acts in real time while the learner updates as fast as the GPU allows. Small timing differences
     change which network acts in which episode.
   - The fixed ensemble with the operator deciding scored **95, 91 and 75 %** in three campaigns with identical
     settings. The same operator and seed moved by 16.5 points on average.
   - So: compare methods only within one campaign, run by run with the same operator and seed. Never pool machines.
   - With the fixed update ratio, both machines now give the same level (§2.4). Still run each comparison on one
     machine.
2. **The synthetic novice is too weak.**
   - It answers 2–9 of 100 ten-step requests: it would take over at step 14.
   - 97 % of its takeovers lie in failed episodes.
   - Its careless state hardly differs from its normal one.
   - So it drives several "novice" results, and it is left out of the results.
3. **Change one thing at a time.**
   - The quality-oracle runs changed the score, the outcome filter and the imitation loss together.
   - The move to the new stack changed five settings together.
   - The fixed ensemble changed four.
   - Each time, the effect is real but not attributed. The ablation then showed that on the new stack none of the four
     is needed, and *why* the old stack needed them is still open.
4. **A better operator made the task easy, so measure speed and effort.**
   - When every method reaches 90–99 %, final success can't show a difference.
   - What still differs is how fast the robot learns and how much the person has to help.
   - The next campaigns need a measure for that, such as ruler success at fixed step budgets, and a harder task.
   - Also: a ruler that averages 3 checkpoints punishes runs that stop right after they learned.
5. **Check that the test can work before reading its result.**
   - HiRIL "lost" because its operator couldn't answer requests that short.
   - The novice "collapsed" partly because a filter removed its corrections.
   - One E2HiL run crashed but counted as finished.
   - Look at how much help each arm actually got, and whether every run produced a result.

## 4. How the thesis story changed

| question | old stack (first machine, Sept–Oct) | new stack (both machines, Nov) |
|---|---|---|
| Does the robot inherit its teacher? | yes, clearly, with novices and policy-oracle demonstrations | not with experts and intermediates: ICC ≈ 0 |
| Can many critics learn? | the fixed version matches HIL-SERL | every version matches HIL-SERL, and faster |
| Does asking help? | asking more often beat the operator's own judgement, within one campaign | no gain at the ceiling; it costs more help |
| Can a score grade corrections? | the critics' comparison fails | the behavioural score detects carelessness (valid); the critics' comparison fails; no training gain at the ceiling |

## 5. Where to look now

| you want | open |
|---|---|
| the latest runs on both machines, and the plan | [`thesis/latest_runs_both_machines.md`](../thesis/latest_runs_both_machines.md) |
| **every dataset, where it is, which slide or claim it supports** | [`thesis/data_inventory.md`](../thesis/data_inventory.md); the data itself in `results/` at the repo root (git-ignored) |
| the scripts behind the deck numbers | `results/analysis/scripts/` (outputs in `results/analysis/outputs/`) |
| the mid-term deck (66 slides) | `presentation_bundle/output/AQI-HIL-SERL_Siemens_short_updated.pptx`; generator `presentation_bundle/output/_siemens/build_sie.py` |
| both machines in one file, to present | [`thesis/presentation_summary_both_machines.md`](../thesis/presentation_summary_both_machines.md) (written before the November runs) |
| every question you were asked, with the answer's location | [`thesis/questions_index.md`](../thesis/questions_index.md) |
| one training iteration and the statistics (ICC, sign test) with numbers | [`thesis/training_iteration_and_statistics.md`](../thesis/training_iteration_and_statistics.md) |
| the hub of all thesis docs | [`thesis/README.md`](../thesis/README.md) |

## 6. Check yourself

- Name the switches of the new stack, and say which of them made training start sooner. (Trick question: the gain
  isn't attributed yet; say why.)
- Why can't you compare the fixed ensemble's 75 % on the asking-rate slide with its 91 % on the slide before?
- The asking-rate result was significant on the old stack but not on the new one. Give two reasons that could explain
  this, and one experiment that would tell them apart.
- The first ensemble version no longer fails. What can you now say about the four changes of the fix, and what can't
  you say?
- Why did HiRIL get only 8 % help, and why does that make its result unusable?
- What measure would you use when every method reaches 95 %, and how would you compute it from the checkpoints?
