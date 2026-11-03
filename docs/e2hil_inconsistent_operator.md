# E2HiL with the inconsistent operator

**Question.** E2HiL (Deng et al., RA-L 2026; [e2hil_baseline.md](e2hil_baseline.md)) drops the batch states
whose estimated influence on the policy entropy is extreme. Does that selection cope with an operator who is
sometimes careless better than plain HIL-SERL? The operator is the inconsistent operator of
[inconsistent_operator.md](inconsistent_operator.md): a hidden condition per takeover, inattentive in 0.4 of
them, persistence 0.7, logged as ground truth. Neither arm reads it.

**Arms.** `hilserl` (plain 2-critic RLPD) and `e2hil` (the same plus the entropy-bounded actor loss, paper
defaults K 8, 5th to 90th percentile, min over critics). Both from scratch, without demo pre-training, as in
the paper. Platform: the new stack (rotation control, orientation box, orientation channel with the stage-0
limits 0.384 / 0.1396 rad, mini-eval convergence, update ratio 1). The fixed update ratio matters here: on
CLAIX without it, E2HiL made about half the learner updates per environment step of HIL-SERL (finding F3).

Configs: `examples/campaigns/e2hil_inconsistent/`. They run through `run_campaign.py` directly, not through
`synthetic_suite.py` (the suite expects quality arms and refuses them).

## 1. Small campaign on CLAIX (within the student quota)

| | |
|---|---|
| config | `e2hil_small_claix.yaml` |
| cells | 2 arms × expert and intermediate (exp00, int00 of the 1-per-tier roster) × seed 0 = **4** |
| cuts against the comprehensive campaign | no novice, 20 k cap instead of 40 k, ruler on the last 8 checkpoints instead of up to 25 |
| cost | about 1–1.7 GPU-h per cell, 4–7 GPU-h with the prepare job (quota: 9 GPU-h) |
| submit | `bash cluster/claix/submit_campaign.sh examples/campaigns/e2hil_inconsistent/e2hil_small_claix.yaml e2hil_small_claix --time 03:00:00` |
| results | `$HPCWORK/aqi_runs/e2hil_small_claix/` (report, cells, `aqi_logs`, `logs/<cell>.*.log`) |

The novice is left out because the inattentive execution uses the novice anchor: a novice's two conditions
barely differ, so a novice cell cannot test the question.

Its numbers are never pooled with the comprehensive campaign: other machine, other cap, other ruler window.

## 2. Go rule (fixed before the small campaign runs)

Run the comprehensive campaign on the second machine if all technical checks pass:

1. All 4 cells train and are scored on the ruler, without a crash.
2. Both arms log a condition for every takeover, and the inattentive share of takeovers lies between 0.2 and 0.6.
3. E2HiL's learner logs a keep fraction (`e2hil_*_keep_frac`) near the expected 0.85 (between 0.75 and 0.95).
4. GPU time per cell is measured (from `sacct`), to plan the second machine's time.

Direction: two pairs cannot decide anything. Report per pair the steps to converge, the operator share, the
first success without help and the ruler SR, and read them as direction only. One exception: if E2HiL fails to
converge within the cap in both pairs while HIL-SERL converges, check the port before spending the second
machine's time.

## 3. Comprehensive campaign on the second training machine

| config | cells | operator |
|---|---|---|
| `e2hil_confirm_new.yaml` | 2 arms × 3 operators per tier × 2 seeds = 36 | inconsistent |
| `e2hil_consistent_new.yaml` | 36, identical otherwise | consistent (control) |

- 40 k cap, standard ruler (best window of 3 over up to 25 checkpoints), same 3-per-tier roster as the
  second machine's confirmation.
- Time: the second machine needed about 1.5 h per cell, run one after another, so 72 cells take about
  4.5 days.
- Primary measures: steps to converge and operator share, paired by operator and seed; ruler SR secondary.
  The test rests on the expert and intermediate tiers (novice model flaw, above).
- The question is answered by the interaction: E2HiL's gain over HIL-SERL with the inconsistent operator
  minus its gain with the consistent one. Each campaign's report gives its own gain; there is no script yet
  that pairs the two campaigns.

**Before it runs there.** The second machine's original copy was deleted after its export. It needs the repo
at `cec765c` or later (at least patch 0021, for the per-operator condition sequence), the same package lock,
and its stage-0 tests. Then, from the repo root:

```bash
mkdir -p runs/e2hil_inconsistent
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/e2hil_inconsistent/e2hil_confirm_new.yaml \
    --campaign runs/e2hil_inconsistent/e2hil_confirm_new >> runs/e2hil_inconsistent/e2hil_confirm_new.out 2>&1
./.venv/bin/python examples/run_campaign.py --config examples/campaigns/e2hil_inconsistent/e2hil_consistent_new.yaml \
    --campaign runs/e2hil_inconsistent/e2hil_consistent_new >> runs/e2hil_inconsistent/e2hil_consistent_new.out 2>&1
```

Never mix its cells with CLAIX cells or with cells of the first machine (F3).
