# A perfect quality score? The oracle campaign and the demonstration probe — results & QA log

Companion to [first_machine_synthesis.md](first_machine_synthesis.md). This log explains **what the quality oracle
computed, how a quality score enters training, and what the campaign actually changed**. It then gives what
`aqi_oracle_quality` (thesis C11) and `quality_starvation_probe` showed, and why their conclusion does not follow yet. It
is written from the code and configs; the numbers of both campaigns come from the first machine's own log, because no
report or raw data of them has come back (section 10).

- Machine: the first training machine, original platform, cuboid, consistent operators.
- Campaigns: `runs/aqi_oracle_quality` (90 cells, finished 2026-10-29, run in two parts) and
  `runs/quality_starvation_probe` (18 novice cells, finished 2026-10-31).
- Configs: `examples/campaigns/aqi_oracle_quality.yaml`, `quality_starvation_probe.yaml`.
- Code: `serl_launcher/aqi/quality.py` (scores, filters, weights), `config.py` (defaults), `runtime.py`
  (`end_episode`), `examples/train_aqi_rlpd.py` (buffers, BC loss).
- The machine's own log: [aqi_oracle_quality_results_and_qa_log.md](aqi_oracle_quality_results_and_qa_log.md).
- The follow-up test: [../novice_ablation.md](../novice_ablation.md).

---

## 0. Headline

- **The quality arms collapsed the novice and raised the teacher effect.** Novice success 65 % with ensemble extended
  against 32 % with the oracle score, 36 % with asking plus the oracle, 10 % with asking plus the counterfactual score; ICC
  0.18 against 0.63, 0.93 and 0.97. Experts and intermediates rose to the ceiling.
- **Twenty demonstrations instead of six did not rescue the novice** (asks + oracle 36 → 27 %).
- **The campaign does not show that the score did it.** Its quality arms changed three things at once: the oracle score,
  a hard filter that drops every takeover from a failed episode, and an imitation loss.
- **That filter alone removes almost all novice corrections.** On the asking-rate campaign (same operators, same
  platform), 97 % of a novice's takeovers under its own monitor fall in failed episodes, against 67 % for experts
  (measured here). With the filter, a novice run keeps about 7 takeovers in its demonstration half; an expert run about 200.
- **The novice test (24 cells, ready, not run) separates the three.** Until it reads out, the thesis does not say "even a
  perfect score amplifies dependence".

## 1. How a quality score enters training

**What gets graded.** Each takeover is a *segment*: from the step the operator takes over to the hand-back. It is graded
once, at the end of the episode (`runtime.end_episode`).

**Where takeovers go.** Every learner batch is half online experience (everything the actor did, takeovers included)
and half from a *demonstration buffer* (the operator's demonstrations plus takeover transitions)
(`train_aqi_rlpd.py`: two iterators of batch size / 2).
- In the modes without quality (`hilserl`, ensemble extended, `active`), **every** takeover step goes into the
  demonstration buffer.
- In `quality` and `aqi` modes, only the takeovers that survive the filters go there, each tagged with its score. The
  others stay in the online half only.

**What the score does to a kept takeover** (`quality.py`):

| use | formula | effect |
|---|---|---|
| replay priority | p = (ε + \|TD error\| + η · max(q, 0))^α, η = 1 | a positively scored takeover is sampled more often; a negative one gets the baseline priority |
| imitation weight | w = clip(sigmoid(2 · q), 0.05, 1) | q = 0 → 0.5, q = +1 → 0.88, q = −1 → 0.12, very negative → 0.05 |
| imitation loss | L_actor = L_RL + 0.1 · mean(w · L_imitate) | the actor also copies the kept takeovers, weighted by w |

The imitation loss (`bc_loss_weight: 0.1`) is switched on only in the quality arms (`bc_weight = … if quality_on else 0`).

**The filters.** The quality arms did not set `quality_filters`, so the defaults were on (`config.py`):

| filter | drops a takeover from the demonstration buffer when | on? |
|---|---|---|
| `safety_veto` | the force/torque limit was breached during it | yes |
| `jerk_veto` | its action jerk exceeds `jerk_hard` | no-op (`jerk_hard` unset) |
| `outcome_filter` | **its episode failed** | **yes** |
| `unsolicited` | the robot had not asked for it | off (needs `unsolicited_policy: rl_only`) |

A dropped takeover keeps its steps in the online half; it only loses its place in the demonstration half, its priority
and its imitation weight.

## 2. What the quality oracle computed

`quality_score_mode: oracle` (`QualityScorer.oracle_q`). It needs `quality_cf_diag`, which records the simulator's
ground truth at every step. For one takeover:

    resolved = z_task at the takeover − z_task at the hand-back
    jam_end  = 1 if the peg is physically jammed at the hand-back, else 0
    outcome  = +1 if the episode succeeded, −1 if it failed
    q_raw    = resolved − jam_end + 0.5 · outcome
    q        = q_raw / (running mean of |q_raw|)          (sign kept)

**z_task** is the synthetic operator's own trouble signal, computed from the *true* state: the lateral error relative
to the clearance inside the corridor above the hole, the error while hovering, 1 for a peg pressed beside the hole,
stagnation, force and elapsed time. "Resolved" is therefore how much the takeover reduced the real trouble.

**Examples.**
- An expert takes over with the peg 6 mm off the bore (z ≈ 1.2), centres it (z ≈ 0.1), no jam, the episode succeeds:
  q_raw = 1.1 − 0 + 0.5 = 1.6 → high priority, weight near 1.
- A novice takes over at z ≈ 0.8, ends at 0.9 with the peg jammed, the episode fails: q_raw = −0.1 − 1 − 0.5 = −1.6 →
  baseline priority, weight 0.05. But this takeover never gets graded in practice, because `outcome_filter` has already
  dropped it.

**What it is.** A grade from information no real system has: the true alignment, the true jam flag and the episode's
outcome. It is a research instrument, the upper bound that a learned score could approximate. It never drops a takeover
itself; it only ranks the ones that reach the buffer. Its log reports a positive score for 67 % of the segments under
asks + oracle and 45 % under quality-oracle.

**What the oracle grades in effect.** With `outcome_filter` on, every graded takeover comes from a successful episode, so
its outcome term is always +0.5. Among those, the oracle orders by trouble resolved and jams.

## 3. What each quality arm changed against its reference

| change | quality-oracle vs ensemble extended | asks + oracle vs asks 30 % |
|---|---|---|
| the score sets priority and imitation weight | yes | yes |
| `outcome_filter` and `safety_veto` drop takeovers | yes | yes |
| imitation loss 0.1 | yes | yes |
| asks + counterfactual: the same, with the counterfactual score instead of the oracle | | yes |

The oracle arm and the counterfactual arm share the filter and the imitation loss, and both collapse the novice. That
is the reason to suspect them, not the score.

## 4. Design

| arm | timing of help | grading |
|---|---|---|
| ensemble extended (`logged`) | the operator's own monitor | none |
| asks 30 % (`active-q30`) | the critics' disagreement σ_Q, top 30 % | none |
| quality-oracle | the operator's own monitor | oracle + filters + imitation |
| asks + oracle (`aqi-oracle`) | σ_Q, top 30 % | oracle + filters + imitation |
| asks + counterfactual (`aqi-cf`) | σ_Q, top 30 % | counterfactual + filters + imitation |

9 operators (3 per tier, sample seed 0, the same roster as the asking-rate campaign), 2 seeds, 6 demonstrations each:
90 cells. Cap 40 000 steps. Fixed ensemble base (8 critics, 5 000 pre-training steps, learning rate 1.5e-4), request
cool-down 5, no quality warm-up. Run in two parts across travel (a 72-hour cap, then resumed).

## 5. Results (from the machine's log)

| arm | ruler success | ICC | expert / intermediate / novice | operator share | converged |
|---|---|---|---|---|---|
| ensemble extended | 79.0 % | 0.18 | 93.8 / 77.8 / **65.4** | 54.6 % | 15 / 18 |
| asks 30 % | 82.9 % | 0.23 | 100 / 91.4 / 57.4 | 28.4 % | 9 / 18 |
| quality-oracle | 73.3 % | 0.63 | 96.9 / 91.4 / **31.5** | 37.9 % | 8 / 18 |
| asks + oracle | 78.8 % | 0.93 | 100 / 100 / **36.4** | 27.7 % | 0 / 18 |
| asks + counterfactual | 73.4 % | 0.97 | 100 / 100 / **9.6** | 29.1 % | 2 / 18 |

Paired against ensemble extended (W/L/T, sign test): asks 30 % +3.9 (10/4/4, p 0.18); quality-oracle −5.8 (9/6/3, 0.61);
asks + oracle −0.2 (10/4/4, 0.18); asks + counterfactual −6.8 (8/5/4, 0.58; one cell failed to be scored).

The log checked that the collapsed novice cells really trained (130 and more episodes, 22–23 k steps) and score 0–7 % on
the ruler, so this is not the old out-of-memory artefact. "Converged" uses the rolling-window rule, which the gated arms
rarely satisfy; the ruler scores every cell anyway. Why quality-oracle needed less help than ensemble extended with the
same monitor (37.9 against 54.6 %) is not explained in the log.

## 6. How much the outcome filter removes (measured here)

No count came back from the oracle campaign. The asking-rate campaign has the same 9 operators, the same platform and
base, and logs every takeover with its episode's outcome (`quality_audit/segments.csv`). Its share of takeovers in
successful episodes is what `outcome_filter` would have kept:

| timing (asking-rate campaign arm) | tier | takeovers per run | in successful episodes = kept | kept per run |
|---|---|---|---|---|
| the operator's own monitor (ensemble extended) | expert | 615 | 33 % | ≈ 202 |
| | intermediate | 350 | 11 % | ≈ 37 |
| | novice | 206 | **3 %** | **≈ 7** |
| asks 30 % | expert | 857 | 55 % | ≈ 471 |
| | intermediate | 335 | 20 % | ≈ 67 |
| | novice | 71 | **8 %** | **≈ 6** |

So in the quality arms the novice's demonstration half held its 6 demonstrations and a handful of takeovers; the
expert's held hundreds of graded takeovers, mostly good ones. This is a proxy (35 000-step cap instead of 40 000; a
different campaign), but it is large enough that the filter is the first suspect. On the second machine's new stack, where
the operator can turn the peg, 76–78 % of novice takeovers were in failed episodes, against 26 % for experts; there,
score-based discarding without filter and imitation kept novices at 94 %.

## 7. The demonstration probe

`quality_starvation_probe`: novice tier only, the campaign's three novices (their parameters copied in), 2 seeds, **20**
demonstrations instead of 6; arms ensemble extended, asks 30 %, asks + oracle. Otherwise identical.

| arm | novice, 6 demos (oracle campaign) | novice, 20 demos (probe) | change |
|---|---|---|---|
| ensemble extended | 65.4 % | 60.5 % | −4.9 |
| asks 30 % | 57.4 % | 64.2 % | +6.8 |
| asks + oracle | 36.4 % | 27.2 % | −9.3 |

**Reading.** More of the novice's own demonstrations do not rescue it. The log concludes "quality is actively harmful".
What the probe shows is narrower: adding demonstrations does not undo whatever the quality package does. It adds more of
the same novice's mediocre demonstrations (20 barely changed ensemble extended either), and it does not remove the filter
or the imitation loss.

## 8. Why the log's conclusion does not follow yet

The log says that even a perfect score amplifies dependence, so grading corrections is wrong in principle ("a paradigm
problem"). Three reasons it is premature:
1. **Three changes at once** (section 3). The oracle score never drops a takeover; `outcome_filter` does, and it removes
   about 97 % of a novice's corrections (section 6).
2. **The imitation loss** makes the quality arms copy the kept takeovers. For a novice those are few and noisy; the
   reference arms have no imitation loss at all.
3. **The counterfactual arm collapses too,** with a score known to be inverted. Shared ingredients (filter, imitation)
   explain both collapses more simply than "any score".

**A hint the other way:** the intermediate rises from 78 to 91 % with the oracle, and to 100 % with asks + oracle. Grading
may help where enough good corrections survive.

## 9. The novice test that decides

[../novice_ablation.md](../novice_ablation.md): novices only, the campaign's 3 operators, 2 seeds, on the code that ran
the campaign. Two reference arms are copied from it, so every new cell pairs.

| arm | score | filters | imitation | cells |
|---|---|---|---|---|
| ensemble extended | – | – | – | 6 copied |
| quality-oracle | oracle | default | 0.1 | 6 copied |
| oracle without filters | oracle | none | 0.1 | 6 new |
| oracle without filters and imitation | oracle | none | 0 | 6 new |
| filters only | none (uniform) | default | 0 | 6 new |
| imitation only | none (uniform) | none | 0.1 | 6 new |

Reading, fixed before the results (collapsed = below the midpoint of the two references):

| if | the cause is |
|---|---|
| filters only collapses, the oracle without filters recovers | the outcome filter |
| imitation only collapses, the oracle without imitation recovers | imitating novice corrections |
| both harmless, the oracle alone collapses | the oracle's replay priority: the log's conclusion stands |

A count that takes seconds comes first: `novice_ablation_report.py --count` reports what the campaign's filters actually
dropped. 24 new cells, about 1.5 days on the first machine.

## 10. Artifacts and how well it is checked

| what | where | checked |
|---|---|---|
| campaign results (section 5), probe (section 7) | the machine's log | **not checked:** no report, cell table or log of either campaign has come back |
| score, filters, buffers, imitation (sections 1–3) | the code in this repo; the same functions are in the machine's code of 31 Oct | read |
| filter proxy (section 6) | asking-rate campaign, `quality_audit/segments.csv` | computed here |
| report, per-cell results, per-segment scores | `runs/aqi_oracle_quality/{report, eval_runs, aqi_logs}` on the first machine | not received |

## 11. Q&A log

| question | answer |
|---|---|
| What exactly was the "quality oracle"? | A grade per takeover from the simulator's truth: how much the real trouble fell during it, minus 1 if it ended jammed, plus or minus 0.5 for the episode's outcome. |
| Did the oracle throw away bad corrections? | No. It only set priority and imitation weight. The `outcome_filter`, on by default, threw away every takeover of a failed episode. |
| Why would that hurt novices so much? | Almost all novice takeovers happen in failed episodes (97 % in the same setting), so the filter leaves a novice run with about 7 takeovers to learn from in its demonstration half. |
| Is the oracle then useless? | Not shown. With the filter on, it only ranked takeovers from successful episodes. Its own effect is what the novice test measures. |
| Did more demonstrations help? | No (36 → 27 %), but they were more of the novice's own, and the filter and imitation stayed on. |
| Why does the ICC jump to 0.93? | Experts and intermediates reach 100 % while novices fall to 36 %: the gap between tiers is almost all the variance. |
| Does this contradict the second machine? | No. There, discarding by a score without the outcome filter and without imitation kept novices at 94 %, on another platform. |
| What can the thesis say now? | That this package of score, filter and imitation collapses novices, and that which part does it is open (C12). |
