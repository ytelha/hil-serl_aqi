# Finalising the experiments: the short plan

*15 Nov 2026. This replaces the longer plan of 14 Nov, which was too long for the time left.*
**Principle:** use what the 600+ runs already answer, and add one experiment of about a week for what they cannot.

> **As built (15 Nov), see `docs/final_study.md`.** The run is unattended, so the pilot's decisions were fixed in
> advance instead:
> - asking at a rank line of 15 % (`asks-15`, `aqi-15`, `random-15` at rate 0.05);
> - the discard share matches the careless share (40 %);
> - the task question is answered by running **two campaigns** that differ only in the demonstrations:
>   `F1_hard` (3) first, then `F2_standard` (6), each 84 runs;
> - the novice is at skill 0.3 for execution and monitoring;
> - the ensemble base is the ablation's fastest (10 critics, warm start);
> - every kept checkpoint (the newest 8 plus every 2 000th step) is scored, for success at fixed budgets.

---

## 1. What the runs already answer

### Does asking (active) help?

| where | result (experts and intermediates unless noted) | what it means |
|---|---|---|
| original platform, first experiment | robot asks in about 1 step in 10: 75 % against 91 % when the operator decides, at a fifth of the help; random requests 56 %; teacher effect ICC 0.84 → 0.00 (all tiers) | asking can save most of the help, at a cost in success, and removes the teacher effect |
| original platform, asking rate | asking in the 55 % most uncertain steps: 94 % against 75 % (same ensemble) and 81 % (HIL-SERL), same help (p 0.021) | when the operator's own judgement is poor (rescues 7–13 %), the robot asking more often is better |
| new platform, comprehensive study | asking at 40 / 55 / 70 %: the same 93–98 % as the operator deciding, with more help (45–56 % against 38 %); random requests as good; asking at 55 % converges sooner (10 of 12) | at the ceiling, asking only moves work to the person |
| both platforms | σ_Q-timed against random at the same rate: +15 and +5 pp, never significant | the *timing* has not been shown to matter |

**Answer today:** asking changes *how much* help is used and *who decides when*. It has not been shown to beat
random timing, and on the new platform it has not saved help. **Open:** can asking at *less* help than the
operator's own share keep success, on a setting with room to learn?

### Does grading (quality) work?

| where | result | what it means |
|---|---|---|
| critics' comparison (counterfactual) | AUROC 0.15–0.18, inverted; below chance in every variant on the cluster | dropped for good |
| behavioural score, consistent operators | 0.78 live, but 0.50 on clean takeovers: timing leakage | no ground truth there |
| behavioural score, sometimes-careless operator | 0.70 (40 % careless, experts and intermediates); 0.76 (20 % careless expert) | **detection works** ("valid") |
| training with it, new platform | discard ≈ re-weight ≈ keep-all ≈ perfect filter (93–100 %); hints only: discard converged faster (10 of 12, p 0.039), the perfect filter used fewer operator steps (8 of 9) | carelessness cost almost nothing, so **no score could show a gain** |
| perfect score, original platform | +8 pp for experts and intermediates, but with filters and imitation; the novice test: imitation, not the score, collapsed the novice | the training value of grading is still untested |

**Answer today:** the score finds careless takeovers, but its use in training has never been tested where careless
takeovers actually hurt. **Open:** when keeping everything loses against a perfect filter, how much of that loss
does the behavioural score win back?

### AQI (both)

Never tested cleanly: its only runs were on the original platform with the confounded perfect-score setup.
**Open:** do the two modules add up, or get in each other's way?

---

## 2. The one remaining experiment

A 2 × 2 of asking × grading on the same operators and seeds, with two controls. **One campaign, second
machine, about 2–3 days.**

| arm | asking | grading | why |
|---|---|---|---|
| HIL-SERL | – | – | the reference everyone knows |
| ensemble | – | – | the operator decides, keeps everything |
| asking | ✓ at rate r | – | can asking save help? |
| grading | – | behavioural score, discard the lowest 20 % | does the score win back what carelessness costs? |
| **AQI** | ✓ at rate r | behavioural score, discard 20 % | do they add up? |
| perfect filter | – | drops exactly the careless takeovers | how much there is to win back at all |
| random requests | random at rate r | – | does σ_Q timing beat chance? |

**Size:** 7 arms × 3 tiers × 2 operators × 2 seeds = **84 runs** (12 pairs per arm).
- Optional: the first machine runs the same campaign with seeds 2–3 as an independent replication, analysed
  separately.
- **Left out:** E2HiL and HiRIL. Their comprehensive-study results stay, with the caveats (HiRIL's requests could not
  be answered).

### The operators: sometimes careless, three tiers, held requests

- **Every operator is careless in 40 % of takeovers** (persistence 0.7), as in the October test. 20 % was too mild.
  Every grading result needs this hidden label as ground truth, and asking works with it too.
- **Tiers:**
  - expert s = 1.0;
  - intermediate 0.5 (as before);
  - **novice v2 at s ≈ 0.3** (config `tier_s_exec`).
- **Why bring the novice back:** both modules claim to help most with a weaker operator, and the teacher effect
  needs a spread of skill.
- **Why only now:** three of the old novice's problems are fixed:
  - rotation control lets it rescue (7 → 70 %);
  - held requests let it answer;
  - no imitation loss.
- **Why s ≈ 0.3:** its careless mode (novice-level execution at s = 0) then differs from its careful mode.
- **Do not shift the other tiers.** The expert was set below superhuman in September on purpose.
- **Held requests:** a request stays up to 30 steps (3 s) until the operator answers (`synthetic_request_hold_steps`).
  This is fairer to slower operators than the old 10-step requests.
- **Requests replace the operator's own monitor,** as in every earlier asking result, so the results stay comparable.
- **Demonstrations and corrections** from the geometric oracle.

### The rate r: less than the operator's own share

On the new platform the operator decided to help in 38 % of the steps. Asking at the same or more help is already
answered. **The pilot measures the share s_op with these operators, and r = ½ s_op** (probably about 20 %).

### Harder task: only if the pilot says so

A task change costs the most time and comparability, so the pilot decides whether it is needed:
- **First choice:** the current cuboid. With the novice v2 and 40 % carelessness, it may already leave room.
- **If not:** the cuboid with 3 demonstrations instead of 6 (config only).
- **Not in this plan:**
  - a wider start range (needs a code change);
  - the hex peg (needs new orientation limits and a new operator test).

### What is measured (with the existing scripts)

- **Ruler success at 4 000 / 6 000 / 8 000 environment steps**, value carried forward after the stop, as in
  `study_budget.py`.
  - Keep all checkpoints (`checkpoint_keep 40`) and score them all on the ruler.
- **Operator steps:**
  - in total;
  - until the first success without help.
- **Final:** the ruler and the last checkpoint.
- **Teacher effect:** ICC per arm.
- **Detection:** behavioural-score AUROC within a run, in the grading arm and in AQI. Under asking the robot picks
  the start of each takeover, which removes the timing confound that misled the score on the first machine.

### How it is read (fixed before the runs)

All comparisons are paired by operator and seed; sign test and Wilcoxon.

| question | comparison |
|---|---|
| is there anything for grading to win? | perfect filter − ensemble |
| does grading win it? | grading − ensemble, and against the perfect filter |
| does asking save help? | asking against ensemble: success within 5 pp at about half the operator steps |
| does timing matter? | asking − random |
| do they add up? | (AQI − grading) − (asking − ensemble), and AQI − HIL-SERL |

### How to test AQI, specifically

- **In the same campaign** as the other cells: same operators, seeds, rate and score settings. Its value is the
  interaction, which needs all four cells side by side.
- **No imitation loss, no outcome filter** (the novice test showed why).
- **Check in the pilot** that the discard filter sees enough takeovers under asking: at least about 40 per run.
  Otherwise lower its window from 200.
- **Two hypotheses to state in advance:**
  - grading matters more under asking, because each requested correction weighs more when help is scarcer;
  - or less, because dropping a requested correction removes data exactly where the robot was unsure.

---

## 3. Steps and time

| day | what | who |
|---|---|---|
| 1 | campaign config; guard the zero-norm-quaternion crash; mark crashed runs as failed; operator-alone check for novice v2 | me |
| 2 | **stage 0** (operator alone, about 2 h): novice v2 solo ≥ 45 %, answers ≥ 80 % of held requests, careful − careless gap ≥ 15 pp; otherwise s = 0.4, or drop the novice | your machine |
| 2–3 | **pilot, 12 runs (about 6 h):** ensemble keep-all and perfect filter, 3 tiers, current cuboid and 3-demo cuboid; measures s_op | your machine |
| 3 | pick the setting: the current cuboid if perfect filter − keep-all ≥ 8 pp at 6 000 steps and keep-all ≤ 85 % there; else 3 demos; set r = ½ s_op | both |
| 3–6 | **main campaign, 84 runs** | second machine |
| 7 | analysis, docs, deck, thesis results | me |

**About a week from go to results.** Writing runs in parallel: the exploratory results (original platform), the
platform (ladder) and the earlier new-platform results are final already.

## 4. Every outcome gives a thesis statement

| outcome | statement |
|---|---|
| perfect filter ≈ ensemble | carelessness costs little even here; grading is shown as detection (valid), with no training benefit to win |
| perfect filter > ensemble, grading ≈ perfect filter | grading works: the score recovers the loss |
| perfect filter > ensemble, grading ≈ ensemble | detection at 0.7–0.76 is not enough to help training |
| asking ≈ ensemble at half the help | asking saves help: the efficiency claim |
| asking < ensemble at half the help | asking cannot replace the operator's judgement with less help on this platform; the original-platform gain came from poor operator rescue |
| asking > random | the timing matters |
| AQI ≥ both single modules | the modules add up |
| AQI < the better single module | they interfere; report which and why |
