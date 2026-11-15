# The first machine's last two campaigns: the novice test and the ladder

*Written 2026-11-14 from the raw runs* (`results/first_machine/campaigns/`, received as
`first_machine_runs_slim_2026-11-14.tgz`). Both campaigns ran 1–6 Nov, before the ensemble-fix ablation. Until now the
docs listed both as "not run".

| campaign | runs | question | answer |
|---|---|---|---|
| novice test (`quality_novice_ablation`) | 24 new + 12 copied | which part of the perfect-score arms collapses the novice: the score, the hard filters or the imitation loss? | **the imitation loss** |
| ladder (`synthetic_suite`, `L0`–`L5`, `X_*`) | 27 + stage 0 | which new-stack change removes the slow start? | **rotation control first, one update per step second** |

---

## 1. The novice test

### What it tested

The perfect-score campaign (C11) found that novices collapse in the quality arms:
- ensemble extended: 65 %;
- perfect score: 32 %;
- asks + perfect score: 36 %.

Its log concluded that "even a perfect score amplifies operator dependence". But the quality arms changed three
things at once:
1. the score;
2. the default hard filters (the outcome filter drops every takeover of a failed episode);
3. an imitation loss of weight 0.1.

The test separates them (design: [docs/novice_ablation.md](../novice_ablation.md)). It used novices only, the three
operators of C11 and 2 seeds.

### Results

Each arm has 6 runs; filter-only has 5 (one crashed). "Paired" compares the same operator and seed with ensemble
extended (wins / losses / ties, exact sign test).

| arm | score | hard filters | imitation | ruler | last checkpoint | converged | takeovers dropped | paired with ensemble extended |
|---|---|---|---|---|---|---|---|---|
| ensemble extended (`logged`) | – | – | – | 65.4 % | 87.0 % | 4 / 6 | none (see note) | – |
| perfect score as in C11 (`quality-oracle`) | oracle | yes | 0.1 | 31.5 % | 20.4 % | 0 / 6 | 97.6 % | −34.0 (2 / 4 / 0, p 0.69) |
| score, no filters (`oracle-nofilter`) | oracle | – | 0.1 | **8.0 %** | 1.9 % | 0 / 6 | 0 % | −57.4 (0 / 6 / 0, **p 0.031**) |
| score only, as replay priority (`oracle-nobc`) | oracle | – | – | 67.9 % | 70.4 % | 3 / 6 | 0 % | +2.5 (2 / 3 / 1, p 1.00) |
| filters only (`filter-only`) | – | yes | – | 80.0 % | 100 % | 5 / 5 | 94.4 % | +13.3 (3 / 2 / 0, p 1.00) |
| imitation only (`bc-only`) | – | – | 0.1 | **13.0 %** | 9.3 % | 0 / 6 | 0 % | −52.5 (0 / 6 / 0, **p 0.031**) |

**Note on the dropped column.** The report gives 97.3 % for ensemble extended. That number is only what the filters
*would* have dropped. In this mode every takeover step goes straight to replay (`examples/train_aqi_rlpd.py:1076`); the
filters only write their verdict to the log. This is a display bug in `examples/novice_ablation_report.py`.

### Reading

The rule was fixed in advance: if the collapse needs the imitation loss, the result is about imitating corrections,
not about the score. That is what happened.

1. **The imitation loss alone collapses the novice:** 13 % against 65 %, 0 of 6 pairs better.
2. **Without the imitation loss, the perfect score does no harm:** 67.9 % against 65.4 %. Used only as a replay
   priority, it neither helps nor hurts the novice.
3. **The filters soften the collapse:** with imitation, the score reaches 31.5 % with filters and 8.0 % without. The
   outcome filter drops 94–98 % of a novice's takeovers, so there is little left to imitate.
4. **Filters alone did best** (80 %, every run converged), but on 5 pairs this is not significant (3 / 2).

The demonstration probe agrees. With 20 instead of 6 demonstrations:
- asks + perfect score, which has the imitation loss, reached 27 %;
- ensemble extended reached 60.5 %;
- asks 30 % reached 64.2 %.

So the novices' problem in C11 was copying their corrections, not grading them.

**What it changes:**
- "Even a perfect score amplifies operator dependence" is withdrawn.
- The correct statement: imitating a novice's corrections collapses the novice. Grading them as a replay priority
  changes nothing for the novice.
- For experts and intermediates, the perfect-score lift (85.8 → 94.1 %) still mixes the three parts. This test used
  novices only.

### Caveats

- **Six runs per arm.** Only the two collapses are significant (p 0.031).
- **Old and new runs mixed.** The 12 copied runs (ensemble extended and the original perfect-score arm) ran on 22–26
  Oct. The 24 new runs ran on 4–5 Nov, on newer code (`c0bc257`, new-stack features off). The commands are identical
  apart from the arm's own flags and `--checkpoint_keep` (25 against 10). More kept checkpoints give the ruler's best
  window more chances, which favours the new arms a little.
- The main reading does not depend on that mix. Among the new runs alone, imitation only (13 %) and the score without
  filters (8 %) are far below the score without imitation (67.9 %) and the filters alone (80 %).
- **One crash:** `filter-only_nov02_s1` hit the zero-norm-quaternion crash (`panda_peg_insert_gym_env.py:287`) before
  training. This is the same crash as E2HiL's on the second machine and `aqi-cf_nov02_s1` in C11.

---

## 2. The ladder

### What it tested

The ladder runs HIL-SERL on the original platform (`L0`) and adds the new-stack changes one at a time. Three more
settings each remove one change from the full stack (`X_*`).

The changes:
- **rot:** the operator can turn the peg;
- **box:** orientation limits;
- **chan:** the orientation takeover channel;
- **conv:** the stopping rule, a rolling window or a mini-evaluation;
- **utd:** one learner update per environment step.

Each setting has one operator per tier and one seed. This gives the direction of an effect, not its significance.

### Results

Table from `results/first_machine/campaigns/synthetic_suite/report.md`. This repo's `examples/synthetic_suite.py report`
regenerates the same table.

| setting | rot | box | chan | stop rule | utd | first unaided success (median env steps) | time-outs ≤ 10 k | operator share | ruler | last checkpoint | actor minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L0_original` | – | – | – | rolling | async | 18 275 | 96 % | 0.55 | 72 % | 81 % | 46 |
| `L1_rotation` | x | – | – | rolling | async | 7 118 | 36 % | 0.47 | 73 % | 93 % | 23 |
| `L2_box` | x | x | – | rolling | async | 9 209 | 43 % | 0.38 | 75 % | 100 % | 24 |
| `L3_channel` | x | x | x | rolling | async | 6 811 | 42 % | 0.39 | 94 % | 100 % | 23 |
| `L4_evalconv` | x | x | x | mini-eval | async | 7 594 | 39 % | 0.41 | 63 % | 63 % | 25 |
| `L5_full` | x | x | x | mini-eval | 1 | 4 464 | 20 % | 0.38 | 97 % | 96 % | 36 |
| `X_nobox` | x | – | x | mini-eval | 1 | 4 642 | 21 % | 0.41 | 95 % | 89 % | 28 |
| `X_nochannel` | x | x | – | mini-eval | 1 | 7 246 | 22 % | 0.38 | 88 % | 100 % | 34 |
| `X_noeval` | x | x | x | rolling | 1 | 3 866 | 20 % | 0.44 | 94 % | 100 % | 38 |

**Why rotation matters.** Stage 0 on the same machine tested the operators alone on the cuboid (30 episodes each). In
the "rescue" test the operator takes over a peg that is already twisted.

| tier | solo, cannot turn | solo, can turn | rescue, cannot turn | rescue, can turn |
|---|---|---|---|---|
| expert | 97 % | 100 % | **13 %** | **100 %** |
| intermediate | 87 % | 100 % | **10 %** | **100 %** |
| novice | 40 % | 83 % | **7 %** | **70 %** |

### Reading

1. **Rotation control removes most of the slow start.**
   - First unaided success: 18 275 → 7 118 steps (−61 %).
   - Early time-outs: 96 → 36 %.
   - Actor time: halved.

   On the original platform an operator who takes over a twisted peg cannot untwist it (rescues 7–13 %), so most early
   takeovers end in a time-out and teach little.
2. **One update per step does the rest:** 7 594 → 4 464 steps (`L4` → `L5`); the ruler rose from 63 to 97 %. Each
   step costs more learner time (actor 25 → 36 min), so the saving in clock time is smaller than in steps.
3. **The box and the stopping rule do not speed up learning.**
   - Without the box: 4 642 steps, against 4 464 with the full stack.
   - With the rolling stop instead of the mini-evaluation: 3 866.
   - `L4`'s low ruler score (63 %; the intermediate scored 30 %) shows the mini-evaluation stop landing mid-swing.
4. **The channel is unclear.** Removing it slowed the full stack (7 246 against 4 464), but adding it on the
   asynchronous learner barely helped (9 209 → 6 811). One run per tier cannot settle this.

**What it changes:** the ladder answers the question the second machine left open:
- The new stack reaches the ceiling mainly because the operator can turn the peg, and because the learner makes one
  update per step instead of about 0.45.
- This explains why the old-stack differences (the slow first ensemble, the help from asking) shrink or vanish on the
  new stack. Those methods had room to help because the operator could not rescue.

### Caveats

- Three runs per setting, one seed. The confirmation stage (more seeds, the asking arm) was never run.
- `L0`–`L4` compare against `L0` in the same operators; `int00` and `nov00` here differ from the 3-per-tier rosters of
  other campaigns.
- "Ruler" is the best window of three checkpoints. With the early stop it can sit far from the last checkpoint (`L4`).

---

## 3. Consequences for the deck and the thesis

**Done on 14 Nov:**
- In the Siemens deck: backup 51 (exact numbers, the test's result instead of "planned", the takeovers kept
  counted in these runs' own logs); backup 52 (an expert kept about 110, not 200; the collapse came from imitation,
  not from the filter); the notes of pages 25 and 27.
- `examples/novice_ablation_report.py` reports "0 % (would 97 %)" for arms whose mode does not apply the filters.
  The corrected tables are in `results/analysis/outputs/first_machine_campaigns/`.

Before that, the list read:

- **Deck page 25, backup "a perfect score on the first machine", next-steps text:**
  - The novice test is no longer "planned".
  - The deck's backup card describes it as "experts and intermediates". It actually ran on novices.
  - Its answer is "imitating the corrections".
- **Backup numbers.** Backup 51's 86.0 % and 95.5 % come from rounded tier means; the exact values from the raw runs
  are 85.8 % and 95.7 %. The other rows match (94.1, 100, 100).
- **Thesis, quality chapter:** replace "a perfect score amplifies dependence" with the imitation result. Keep the
  limit: experts and intermediates were not tested.
- **Thesis, platform chapter:** the ladder is the evidence for why the new stack is at the ceiling.

## 4. Files

| what | where |
|---|---|
| raw runs | `results/first_machine/campaigns/quality_novice_ablation/`, `…/synthetic_suite/` |
| the machine's own reports | `…/quality_novice_ablation/report/novice_ablation.md`, `…/synthetic_suite/report.md` |
| how they ran | `…/_orchestrator_logs/first_machine_chain.sh` and `.out` (stage 0, roster audit, cell copy, novice test, ladder) |
| per-arm tier means from the raw runs | `results/analysis/outputs/first_machine_campaigns/tier_means_from_raw.txt` (script `fm_check_numbers.py`) |
| which campaigns were kept and why | `results/first_machine/campaigns/README.md` |
