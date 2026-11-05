# Presenting the training machines

A guide for presenting what ran where and what came out, with the slide numbers of both decks
(`presentation_bundle/output/AQI-HIL-SERL_Siemens_short.pptx`, 59 slides; `AQI-HIL-SERL_Midterm_Telha.pptx`,
227 slides, 49 hidden). Sources: [first_machine_synthesis.md](first_machine_full_history/first_machine_synthesis.md) and
[second_machine_synthesis.md](second_machine_synthesis.md). Numbers as of 3 November 2026.

**The file to show:** [presentation_summary_both_machines.md](presentation_summary_both_machines.md), both machines in one
file, with formulas, worked examples, the questions asked while preparing and a glossary.

## The story in one minute

The first training machine answered the four research questions on the original platform (459 runs). The teacher
matters to plain HIL-SERL; a critic ensemble can be made to learn as well as HIL-SERL; no learned score can grade
corrections; letting the robot ask for help saves human effort and removes the measurable teacher effect. Then a
second machine showed that the operator of all those runs could rarely rescue a twisted peg. With rotation control
it can, and training starts about four times sooner. It also tested an operator who is sometimes careless: a score
notices carelessness only moderately, and using it showed no gain in a test that was too weak to decide. Next: find
out whether the main results survive with the operator who can help.

## What ran where

| | first training machine | second training machine | RWTH cluster |
|---|---|---|---|
| when | 4 Sep – 31 Oct | 24–29 Oct | 26 Sep, 29 Sep, 5 Nov |
| runs | 459, in 16 campaigns | 58, in 5 campaigns | method test; 1-cell counterfactual study; 4-cell E2HiL test (ran 5 Nov and failed technically: in all 4 runs the camera rendering stalled once the policy took over (step 201) and aborted; no result) |
| platform | original: the operator moves the peg but cannot turn it | new stack: rotation control, orientation limit and channel, new stop rule, fixed update ratio | – |
| answered | does the teacher matter; can many critics learn; can corrections be graded; does asking help | can the operator help at all; does a score notice a careless operator | can the counterfactual score be repaired (no) |
| state | idle; next: novice test, then the new stack | idle; its copy was deleted after the export | idle; the rendering stall is to be diagnosed (`presentation_bundle/results/e2hil_small_claix/RESULT.md`) |

**Never compare runs across machines.** The learner makes about 0.45 updates per environment step on the first
machine and 1 on the second, so each comparison has its reference on its own machine.

## The storyline, with slides

| # | beat | Siemens deck | long deck |
|---|---|---|---|
| 1 | how we test: synthetic operators, runs, campaigns, the ruler | 9, 13, 14 | 79–93 (programme 93) |
| 2 | does the teacher matter? yes | 17 | 102–107 |
| 3 | can many critics learn? first no, then yes | 18, 19 | 110, 117–122 |
| 4 | can corrections be graded? no | 20 | 125–131; study result 168 |
| 5 | asking for help: same success, a sixth of the help; asking more beats the operator | 21, 22 | 146–155 |
| 6 | the turn: the operator could not turn the peg | 23 | 49–51 |
| 7 | the sometimes-careless operator | 24 | 133–137 |
| 8 | what we learned, what comes next | 25 | 174–176 |
| – | where it all ran | – | 161 |

Backups to have ready: Siemens 35 (the operator in asking mode), 36 (all asking variants, both references), 37 (how
the robot decides to ask: a worked example), 38 (the asking line adapts), 40 (what ran on the second machine), 41
(where a run's time goes, the fixed update ratio, the shorter evaluation), 42 (scores), 43 (discard or re-weight),
44 (perfect score and novice test), 52 (score ideas), 57 (the request gate in formulas); long deck 223 (how the robot
decides to ask), 224 (the asking line adapts), 225 (runs on the second machine), 227 (where a cell's time goes), 168
(counterfactual study).

**The first machine's latest experiments in one guide:** [first_machine_latest_experiments_guide.md](first_machine_latest_experiments_guide.md).

**If asked how σ_Q or the quality oracle work:** section 1–3 of
[first_machine_asking_rate_results_and_qa_log.md](first_machine_asking_rate_results_and_qa_log.md) and section 1–2 of
[first_machine_oracle_quality_results_and_qa_log.md](first_machine_oracle_quality_results_and_qa_log.md).

## Numbers to know

| claim | number | where |
|---|---|---|
| teacher matters (geometric operators) | ICC 0.60, p 0.035; tiers 100 / 82 / 59 % | first machine |
| teacher matters (human-like demonstrations) | ICC 0.99, p 0.004; novice 6 % | first machine |
| but not in every campaign | ICC 0.00–0.30 in four others | first machine |
| first ensemble | −18.5 points, ICC 0.03 → 0.74 | first machine |
| fixed ensemble | 82 vs 84 %, p 0.30, 16 of 18 converge | first machine |
| quality replay | always worse than HIL-SERL; −13.0 points, p 0.031 after stabilisation | first machine |
| counterfactual score | AUROC 0.18; in the study every variant below chance → abandoned | first machine, cluster |
| behavioural score | 0.78 → 0.50 on clean takeovers (timing leakage) | first machine |
| asking, first experiment | 71 vs 78 %, p 0.58, at 9 vs 54 % help; ICC 0.84 → 0 | first machine |
| asking more | 87 vs 74 %, p 0.021 against the same ensemble; 87 vs 79 % against HIL-SERL, p 0.18 | first machine |
| operator alone, rescue a twisted peg | 13 / 10 / 7 % → 100 / 100 / 70 % with rotation control | second machine, no learning |
| why | peg 21 degrees twisted at the takeover; the expert tolerates 8 | second machine, no learning |
| training starts sooner | first success without help ≈ 18 300–20 200 → 4 600 steps, same method | across machines, not attributed |
| a whole run | ≈ 30 min training + 55 min ruler ≈ 1.5 h | second machine |
| a shorter ruler | ≈ 20 min, all 57 scores unchanged (skip pre-training, newest 8; not yet used) | second machine, re-selected |
| training time split | first machine 20 min pre-training + 56–59 min online (9.0–9.4 steps/s); second 5 + 23–25 min (6.2–6.4 steps/s) | both machines, same method |
| why training fell less than the steps | 2.5 × shorter against 3.4 × fewer steps: each step 1.45 × slower, pre-training fixed | second machine |
| learner updates per env step without the fixed ratio | 0.45 first machine; 2.5–2.7 on the H100; E2HiL 1.3 there | finding F3 |
| careless operator: best online score | AUROC 0.66 (0.71 / 0.67 / 0.52) | second machine |
| discard vs re-weight | 93.2 vs 90.3 %, −2.9 points, p 0.39 | second machine |

## Say this aloud

- **The finished cuboid results stay fair as paired comparisons,** because every method had the same operator. The
  open question is whether the size of the teacher effect survives an operator who can help.
- **The "four times sooner" is not attributed:** another machine, another operator, five changes at once. The ladder
  separates them.
- **"Asking beats the operator"** means the same ensemble with the operator deciding. Against plain HIL-SERL the gain
  is 8 points and not significant, almost all among intermediates.
- **The ICC of 0 under asking is partly mechanical:** novices answer few of the short requests, so their corrections
  largely leave the data.
- **The careless-operator test was weak:** the filter started late, all runs shared two condition sequences, success
  sits near the ceiling, and no control ran at scale. And no gate cleared it: after two no-go gates the automation was
  told to run to the end.
- **The perfect-score campaign is not yet a verdict on grading:** its arms also used hard filters and an imitation loss.
  The novice test separates them.

## Likely questions

| question | short answer |
|---|---|
| Why two training machines? | The first ran everything for RQ1–RQ4; the second was available for a week and tested the new stack and the careless operator. Never mixed. |
| Can I compare the speed-up? | Only as the size of what can be gained. Same method on both: ≈ 19 k → 4.6 k steps. Different machine, operator and five settings. |
| Is the teacher effect real if ICC varies so much? | Yes where it is measured with power (0.60, 0.99, 0.66, all significant). Single-campaign ICC is unstable; a pooled model over all HIL-SERL runs is planned. |
| Does asking beat plain HIL-SERL? | Not yet significantly: +8 points, p 0.18. It does beat the same ensemble with the operator deciding (p 0.021), with less help. |
| What does "asks 55 %" do, and why do novices answer so little? | At every step the 8 critics value the robot's next action; their spread σ_Q, as a z-score against the run so far, opens a 10-step request when it is above the 45th percentile of the last 2 000 steps, then 5 steps of pause. The operator sees the request through a smoothed, noisy signal and must stay above its threshold and then react: an expert takes over at step 6, an intermediate at step 10, a novice would need step 14, so the request is over first (simulated on the tier anchors: 87.5 / 54.3 / 0.4 % answered). |
| Is the asking threshold adaptive? Is σ_Q fitted to a distribution? | Adaptive, yes: the line is the 45th percentile of the last 2 000 steps' scores, recomputed every step, so it follows σ_Q as the critics converge. Nothing is fitted; it counts. The price: a rank line never stops asking, so the rate stays fixed even when the robot is confident. "Asks 55 %" means 55 % of the recent steps are above the line, not 55 % requested; the operator ended up controlling 41.8 %. |
| Is the gate's timing better than random? | Suggestive, not shown: +15 and +5 points vs random at a matched rate, sign-test p 1.00 and 0.30. More seeds are planned. |
| Why did grading fail? | The critic compares the operator's move with the robot's own, which the actor optimised: the score measures distance from the policy, not quality. A pre-registered study confirmed it. Critic-free scores were timing artefacts. |
| Did the perfect score prove grading is wrong? | No: three things changed at once. 24 runs decide which one. |
| What did the careless operator show? | Scores notice carelessness moderately (0.66, mostly the takeover's length); discarding vs re-weighting made no detectable difference. The novices cannot test it in this model. |
| How is the careless operator built? | A hidden state per takeover: it keeps the last state with probability 0.7, otherwise it is redrawn, careless with probability 0.4 (40 % careless, spells of about 5–6 takeovers). Careless means novice motor noise, tremor, speed and recovery, a premature push half the time, and an aim point 6 mm beside the hole. When it steps in is unchanged, and the demonstrations are always careful. |
| What exactly is the behavioural score? | B = z(ΔV) − z(T) − z(F_peak): the critics' value at the end of the takeover minus at its start, minus its length, minus its peak force, each standardised against the run's earlier takeovers. Discard drops a takeover below the 40th percentile of the last 200 scores; re-weight replays it more often the higher max(B, 0). Formulas of all scores: inconsistent-operator log, section 3.0. |
| Why is 0.66 not "valid"? | The bar fixed in advance was 0.70. |
| What runs next? | The novice test, then the new stack on the first machine, its ladder, and the repeat at thesis scale; the 4-run E2HiL test is repeated once the cluster's rendering stall is fixed, or on a training machine. |
| Why does a run take 1.5 h if training is 30 min? | The ruler scores every saved checkpoint, and the early ones fail every trial, each running to the 300-step limit: about 30 000 simulated steps, three times the training. Skipping the pre-training checkpoints and scoring only the newest 8 cuts it to about 20 min without changing any score; it is implemented and goes to the first machine, to be measured there first. |
| What is the fixed update ratio, and why? | The learner makes exactly one update per environment step. Before, the robot stepped at 10 Hz and the learner trained as fast as its GPU allowed, so learning per step depended on the machine (0.45 on the first, 2.5–2.7 on the H100) and on the method (E2HiL got about half). Fixing it makes machines and methods comparable. |
| Does it make convergence slower? | In steps, probably not: each step is used for more learning. In clock time it depends: on the second machine it cost little (the robot waited for the learner half a minute per run); on the first machine each step takes about twice as long, so a run breaks even only if it needs about half the steps. The ladder measures it. |
| Why did training time fall less than the steps? | The 4 × is the first success; training stops later (3.4 × fewer steps). Each step is about 1.45 × slower, and the 5-minute pre-training does not shrink. Result: 77 → 31 minutes. |
| Why not the cluster for everything? | The student quota is 3 GPU-hours a month; one campaign needs 80–135. The Thesis project (1 500 GPU-hours) would change that. |

## Numbers that changed recently (do not use the old ones)

| old | now |
|---|---|
| first success without help ≈ 17 000 steps | ≈ 18 700 for plain HIL-SERL; same-method comparison ≈ 18 300–20 200 → 4 600 |
| three to four times earlier | about four times earlier |
| Q1 took 9.1 h; 1.7 GB per run; 57 runs | 10.1 h; up to 2.4 GB; 58 runs (57 on the cuboid) |
| smoothness fails for novices | a property of the model: its novices have more tremor than the careless setting |
| a perfect filter gained nothing | no clear gain: the order flips at the last checkpoint, 3 pairs |
| counterfactual study queued | done: abandon |
| 13 campaigns, ≈ 250 runs | about 520 runs |
| student quota ≈ 20 GPU-h | 3 GPU-h per month |
| 7.7–8.4 environment steps per second on the second machine | 6.2–6.4 on average (7.7–8.4 was the progress bar's smoothed rate) |
