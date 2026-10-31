# Handoff implementation summary (2026-09-30)

Record of applying the `handoff/` migration bundle (built against the
2026-09-24 snapshot) to the training machine. All ten patches were applied in
order with `git apply` and verified before any campaign was run.

## What was applied

All 10 patches applied cleanly in order, confirming **no drift** from the
snapshot (0001 and 0006 applied cleanly against the live tree; the rest are
sequential and applied once their predecessors were in place).

| # | patch | category |
|---|---|---|
| 0001 | E2HiL external baseline (`--aqi_mode=e2hil`) + intervention-rate report | **new comparison algorithm** |
| 0002 | Named campaign arms (`arms:`) + provenance fixes | infra |
| 0003 | Quality-signal audit + `quality_behavioral_terms` knob | analysis + config |
| 0004 | HiRIL external baseline (`examples/train_hiril.py`, `serl_launcher/hiril/`) | **new comparison algorithm** |
| 0005 | Quality scores v2 (hindsight / progress / outcome / uniform + consistency BC) | new quality scores |
| 0006 | Cluster (RWTH CLAIX) scripts + `--shard/--list_cells` + `AQI_TRAINER_PORT` | infra (defaults unchanged) |
| 0007 | Operator-roster jitter fix (F1) + frozen roster + `roster_audit.py` | **bug fix** |
| 0008 | Comments only (F2/F3/F4 markers) | no behaviour change |
| 0009 | Reuse-recipe docs (copy `operators/` too) | docs + 1 YAML comment |
| 0010 | Counterfactual-score diagnostics logging (off by default) | logging only |

## Verification (all passed)

- **Plumbing (numpy) tests:** `test_campaign_e2hil` (5), `test_campaign_arms`
  (7), `test_behavioral_terms` (3), `test_quality_audit` (4),
  `test_hiril_priorities` (5), `test_campaign_hiril` (7), `test_roster_freeze`
  (4), `test_cluster_support` (4), `test_cf_diag` (4),
  `test_quality_scores_v2` (23).
- **JAX tests (CPU), never run before this machine:**
  - `test_e2hil` (5) — `c(s)` matches the official E2HiL formula; the
    non-e2hil actor path is byte-identical.
  - `test_hiril_agent` (8) — BIQN critic, dual uncertainty, TD3-delayed update.
- **Campaign wiring:** E2HiL and HiRIL configs dry-run (to `/tmp`) with correct
  flags; the existing `rq4_ensemble_convergence` config still produces valid
  commands (backward compatible).
- **F1 fix:** `operator_model.jitter_theta` now visits keys in sorted order;
  three `PYTHONHASHSEED` values yield identical `sigma0` (was divergent).
- `get_errors` clean on all core modified files.

## New comparison algorithms

- **E2HiL** — `--aqi_mode=e2hil`; knobs `--e2hil_K/_q_low/_q_high/_q_agg`.
  Plain 2-critic HIL-SERL plus an entropy-bounded actor loss. Campaign:
  `examples/campaigns/e2hil_comparison.yaml`. Doc: `docs/e2hil_baseline.md`.
- **HiRIL** — `examples/train_hiril.py`; knobs `--hiril_*`. TD3 + BIQN critic,
  risk-aware intervention and imitation. Campaign:
  `examples/campaigns/hiril_comparison.yaml`. Doc: `docs/hiril_baseline.md`.

## Decisions pending before running campaigns

1. **HiRIL trigger rule (F2):** a frozen `tau_risk` calibration made HiRIL
   stop requesting help after ~1700 steps. Choose a rule (periodic
   re-calibration recommended) before the first HiRIL campaign.
2. **Hindsight quality score (F4):** timing artifact before the critic sees
   rewards. Only relevant if running `quality_score_mode: hindsight` (C7b).
3. **Roster audit:** run `examples/roster_audit.py runs/*/` (read-only) to
   check whether any finished campaign was affected by F1.

## Behaviour changes now active (new campaigns only)

- `quality_filters: []` now means **no filters** (was: default filters).
- Unknown keys under `method_overrides:` now **abort** instead of passing
  through.

## Safety

- No patch modified any finished demo, checkpoint, log or report.
- `runs/`, `examples/runs/` and `**/aqi_logs/` remain gitignored.
- The patches also added `handoff/` and `tests/` to `.gitignore`.
