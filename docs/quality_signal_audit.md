# Quality-signal audit — is the behavioral score real, or timing leakage?

Script: [examples/quality_signal_audit.py](../examples/quality_signal_audit.py)
(numpy only; also runs automatically in `run_campaign.py --stages analyze`).
Tests: [tests/test_quality_audit.py](../tests/test_quality_audit.py).

## 1. The claim being audited

RQ4 §11–12 ([rq4 log](thesis/rq4_ensemble_convergence_and_qa_log.md))
scored every intervention segment by **AUROC(score, episode success)**. It
found:
- the counterfactual Q-gap `q_int` is anti-predictive (0.175);
- the critic-free behavioral composite `B = z(d_value) − z(t_int) − z(f_peak)`
  predicts success (0.72 offline, 0.78 live);
- `t_int` (takeover length) carries the skill gradient: expert 31.6 <
  intermediate 45.8 < novice 65.8 steps.

Two gaps motivated this audit:
1. **Not reproducible.** That offline analysis was ad hoc; no script in the
   repo produces it. The script now reproduces it (PART A).
2. **Possible leakage.** The synthetic operator can make `t_int` track the
   outcome mechanically, independently of how good the correction was
   (PART B).

## 2. The leakage mechanism (why the doubt is concrete)

- **A hard takeover cap.** Each synthetic operator hands back after at most
  `d_max` seconds (`SyntheticInterventionGate`, event `handback_dmax`). The
  tier anchors are 3.834 / 5.538 / 8.0 s ([operator_model.py](../serl_launcher/serl_launcher/aqi/operator_model.py)),
  jittered per operator. At 10 Hz that is about **38 / 55 / 80 steps**. The
  RQ4 per-tier mean `t_int` values (31.6 / 45.8 / 65.8) are each about 83% of
  their tier's cap. The "skill gradient" of `t_int` may simply be the
  gradient of `d_max`.
- **Time pressure blocks handback.** The operator's trouble signal
  `z_t = max(…, 0.5·t/budget)` ([operator_gate.py](../serl_launcher/serl_launcher/aqi/operator_gate.py),
  `MonitoringSignal`) grows with episode time. The operator hands back only
  after `z_t` stays below `delta_out ≈ 0.2–0.35` for `k_out` steps. Once
  `t/budget > 2·delta_out` (40–70% of the episode), that cannot happen, so
  late takeovers run to the cap or to the episode end.
- **Late means failing.** Timeouts are failures, and a successful episode
  ends early.

So "short takeover ⇒ success" can mean "**early** takeover ⇒ success" plus
"`t_int` ≈ the operator's `d_max`". Neither is intervention quality. This is
a hypothesis; the audit tests it on existing logs **with no new training**.

## 3. What the script computes

Everything is computed per **arm**: runs are grouped by `meta.json
config.arm`, else by mode. Each row of the AUROC table is one score, and the
columns are:

| column | meaning | why |
|---|---|---|
| **all** | AUROC over all segments + 95% CI from an **episode-cluster bootstrap** | reproduces RQ4; segments of one episode share a label, so episodes are resampled, not segments |
| **clean** | only segments **released by handback**: not capped, not cut by the episode end | removes the two mechanical routes from timing to `t_int` |
| **time-stratified** | pairs compared only within the same quantile bin of start time / episode budget | a score cannot win by encoding *when* the takeover happened |
| **difficulty-stratified** | pairs compared only within the same quantile bin (`--n_diff_bins`, default 3) of V(onset) = μQ at the takeover's first step | a score cannot win by encoding *how easy the state was* (PART C) |
| **window-clean** | only segments whose post-release window (handback + K) closes before the episode ends | the hindsight advantage contains the success reward when the window reaches it (PART C) |
| **post-release** | label = episode succeeded within `--horizon` steps after release (released segments only) | a label closer to "the correction helped" than whole-episode success |
| per tier | AUROC within expert / intermediate / novice | RQ4's within-tier check |

Scores:
- the logged online `q_int` (counterfactual or behavioral, whatever the run
  used);
- `a_hat_mean`, `a_hat_at_trigger`, `d_value`, `d_progress`, `−t_int`,
  `−f_peak`, `−jerk_mean`, `sigma_mean`;
- the offline composites `B` and `B without t_int`;
- two **time-only yardsticks**, `−start_step` and `−end_step`, which know
  nothing about the correction.

The rest of the report:
- **B1/B2 — how takeovers ended.**
  - Each takeover is labelled **cap** (length ≥ cap − `--cap_tolerance`,
    with cap = ⌈`d_max`·10⌉ from the run's `meta.json`), **episode_end**
    (still in control at episode end) or **handback**.
  - Shares are reported per tier, with the success rate of each kind.
  - Mean `t_int` vs mean cap per tier.
  - Per-operator `t_int` vs cap, and their correlation across operators.
- **B4 — logistic fits** of `success ~ −t_int`, `~ time`, and
  `~ −t_int + time`: does `t_int` add anything once start time is known?
- **Observations:** rule-based sentences that quote the numbers. They flag
  what to look at; they do not decide the thesis claim.

## 4. How to run and read it

```bash
# on the existing RQ4 diagnostic logs (no training):
./.venv/bin/python examples/quality_signal_audit.py --campaign runs/quality_behavioral_diag
./.venv/bin/python examples/quality_signal_audit.py --campaign runs/quality_on_rq4_diag
# -> <campaign>/report/quality_audit/{summary.md, audit.json, segments.csv, <arm>_auroc.png}
```

**Reproduction check (PART A).** The `all` column of `q_int (logged)` should
match RQ4:
- ≈ 0.175 on `quality_on_rq4_diag`;
- ≈ 0.78 on `quality_behavioral_diag`.

Offline `B` should be ≈ 0.72 on `quality_on_rq4_diag`, but exact equality is
not expected: offline `B` uses whole-run z-scores, while the online score used
running ones. If the logged `q_int` is far from RQ4, stop and find out why
before interpreting PART B.

**Reading PART B**, rules of thumb:

| pattern | reading |
|---|---|
| `−t_int` high on **all**, near 0.5 on **clean** and **time-stratified**; time-only yardsticks ≥ `−t_int`; B4 `−t_int` coefficient shrinks with time; corr(`t_int`, cap) ≈ 1 | `t_int`'s signal is **timing leakage**. Drop it from the score (`quality_behavioral_terms: [d_value, f_peak]`) and report the leakage in the thesis. |
| `−t_int` stays clearly > 0.5 on clean and stratified, and the CIs exclude 0.5 | `t_int` carries real information beyond timing; the behavioral score stands, with this audit as supporting evidence. |
| `d_value` / `B without t_int` survive the controls | a critic-free-ish signal remains usable either way. |

In a synthetic world with built-in leakage (the unit test), `−t_int` scores
0.83 on all segments but 0.57 on the clean subset, while a genuine quality
feature stays at 0.85–0.98. That is the signature to look for.

## 5. Acting on the result

- **Score terms are a knob now:** `AQIConfig.quality_behavioral_terms`
  (`--quality_behavioral_terms d_value,f_peak`, YAML
  `aqi.quality_behavioral_terms`, or per arm). The default is all three
  terms, i.e. the RQ4 score unchanged.
- **RQ5** ([campaigns/rq5_quality_score.yaml](../examples/campaigns/rq5_quality_score.yaml))
  trains `logged` vs `quality-cf` vs `quality-beh` vs `quality-beh-not`. This
  tests whether any score *improves training* (autonomous SR, convergence,
  ICC, intervention rate), which AUROC alone cannot show.

## 6. Limits of the audit

- **Few runs, correlated segments.** The CI resamples episodes, not runs; with
  1–3 runs per arm, treat it as a lower bound on uncertainty.
- **Cap tolerance.** "At cap" uses ±`--cap_tolerance` steps (default 2) around
  ⌈`d_max`·10⌉ to absorb off-by-one timing between the operator gate and the
  logged `intervening` flag. Widen it if many lengths sit just below the cap.
- **Human operators have no `d_max`.** For humans, B1 reports `released`
  instead of cap/handback, and the clean subset is empty. Use the
  stratified and post-release columns.

## 7. PART C — the quality scores v2

The four scores added from the literature review
([quality_scores_v2.md](quality_scores_v2.md)) get the same treatment,
**before** any of them is trained on:

| score row | source | what to check |
|---|---|---|
| `a_hind (hindsight)` | segments.jsonl if logged with the audit's K and discount; else recomputed from steps.jsonl (μQ, reward, mask per step; pre-v2 logs: sparse reward inferred from the episode outcome) | **window-clean** AUROC. The window of a late takeover contains the success reward, so "all" is inflated by construction |
| `d_progress_post` | same; offline the window ends at the last logged state, online at the true final state | window-clean and difficulty-stratified AUROC |
| `V(onset) (difficulty only)` | μQ at the takeover's first step | a yardstick like the time-only rows: how far does state difficulty alone get? |
| `-sigma_beta (consistency)` | replayed in time order from steps.jsonl with `serl_launcher/aqi/consistency.py` | its **per-tier means** (should rise expert → novice); its success AUROC is secondary, since σβ is an imitation weight, not an outcome predictor |

Also reported: the correlation of each key score with V(onset) (section
"C2") and per-tier means of V(onset), `a_hind`, `d_progress_post` and σβ.

```bash
# PART C on existing logs, with other window lengths:
./.venv/bin/python examples/quality_signal_audit.py --campaign runs/quality_behavioral_diag --post_release_steps 5 --out /tmp/qa_k5
./.venv/bin/python examples/quality_signal_audit.py --campaign runs/quality_behavioral_diag --post_release_steps 20 --out /tmp/qa_k20
# --no_steps skips steps.jsonl (fast; PART C only from segments.jsonl, no sigma_beta)
```

Reading it:
- **Hindsight / progress arm worth training** (C7b,
  `rq5_quality_score_ext.yaml`) if the **window-clean** AUROC is clearly
  above 0.5 and survives the difficulty stratification. If it is ≈ 0.5
  window-clean, the score only re-labels outcomes: drop the arm (the
  outcome arm `quality-outc` already covers that).
- **Consistency weight worth training** (C8, `rq5_quality_imitation.yaml`)
  if σβ rises from expert to novice. If it does not, it does not measure
  operator skill in this simulator.
- **Difficulty:** a key score whose AUROC falls toward 0.5 within V(onset)
  bins mostly says "the takeover started in an easy state".
