# Handoff evaluation + quality-score verdict + active-part kickoff (2026-09-30)

**Author of this note:** on-machine session, at the thesis author's request.
**Scope:** (1) evaluate the `handoff/` experiment bundle against the real repo/data;
(2) finish the quality-score testing; (3) start verifying/evaluating the **active
part (ensemble disagreement)**. The handoff's own experiment plan is treated as a
*hint only*, per the author's instruction.

> **One-line result:** the quality-score thread is **finished with a clean negative
> result** — no replay-priority intervention score survives the leakage/difficulty
> checks — and the one signal that *does* survive is **ensemble disagreement σ_Q**,
> so the thesis energy now moves to the **active gate** (a validity run is already
> launched).

---

## 1. Is the handoff helpful? (evaluation)

The `handoff/` bundle (10 patches) is **already applied and verified** on this
machine ([docs/handoff_implementation_summary.md](handoff_implementation_summary.md))
— code, configs, docs and tests are live. It is a well-designed, pre-registered
program. Verdict per piece, against the two goals (finish quality; start active):

| piece | serves | verdict |
|---|---|---|
| **0003** quality-signal audit + `quality_behavioral_terms` | quality validity (RQ2) | **Essential** — it is what settled the behavioral-score question below |
| **0010** CF diagnostics + **simulator ground truth** (`op_z_task`, true lateral/jam) | quality labels **and** the active-gate label | **Essential** — the only timing-free label; drives the active run |
| **0005** quality scores v2 (`hindsight`/`progress`/`outcome`/`uniform`) + configs | quality alternatives | **High**, but see §2 — most fail the audit; keep `uniform` as a control |
| **0002** named `arms:` + `rq4_attribution_sweep.yaml` | infra + RQ4 attribution | **Essential enabler** |
| **0007** roster fix + `roster_audit.py` | reproducibility (F1) | **Ran it** — all campaigns clean (§2) |
| **0001** E2HiL, **0004** HiRIL baselines | ultimate goal (vs prior HiL methods) | **Later**; HiRIL blocked on a trigger decision (F2) |
| **0006/0008/0009** cluster/comments/reuse | infra | enablers |

**Net:** helpful, but as *tools + labels*, not as a rigid plan. The single most
valuable contribution is the **leakage audit (0003) + ground truth (0010)**, which
turned an ambiguous quality result into a decisive one.

---

## 2. Stage-0 verification (read-only) — what was run and found

### 2.1 Roster audit (F1) — PASS
`roster_audit.py` on `rq4_ensemble_convergence`, `quality_behavioral_diag`,
`quality_on_rq4_diag`: **all consistent** (0 cells trained against a different
operator than their demos; 0 roster rewrites). So RQ4 and the quality diagnostics
are valid at the operator level — no F1 corruption.

### 2.2 Quality-signal audit — the behavioral score was TIMING LEAKAGE
`quality_signal_audit.py --campaign runs/quality_behavioral_diag` (1106 segments,
252 from successful episodes). AUROC vs episode success, with the leakage-clean
column = takeovers the operator actually **handed back** (not ended by the `d_max`
cap or the episode end):

| score | all | **clean (handback only)** | time-strat | window-clean | verdict |
|---|---|---|---|---|---|
| `a_hat` (counterfactual) | 0.148 | 0.147 | 0.170 | 0.170 | **inverted** (confirms RQ4 §11) |
| **B = behavioral** z(dV)−z(t)−z(F) | 0.725 | **0.499** | 0.714 | 0.680 | **leakage — no clean signal** |
| B without `t_int` | 0.632 | 0.490 | 0.654 | 0.522 | no clean signal |
| `d_value` | 0.640 | 0.526 | 0.674 | 0.523 | mostly leakage |
| `a_hind` (hindsight) | 0.628 | — | — | **0.521** | difficulty, not quality (r=−0.81 with V(onset)) |
| `d_progress_post` | 0.472 | 0.417 | — | 0.481 | noise |
| `-t_int` alone | 0.704 | 0.558 | 0.684 | 0.731 | timing |
| `-start_step` (time only) | **0.825** | 0.694 | 0.568 | 0.807 | **pure timing beats every score** |
| **`σ_Q` (ensemble disagreement)** | **0.827** | **0.849** | **0.812** | **0.803** | **survives — the standout** |

**Why the behavioral score failed:** 61% of takeovers end exactly at the tier-fixed
`d_max` cap; `t_int / cap = 0.80` for every tier; `corr(mean t_int, cap) = 1.00`.
So `t_int` just re-encodes the operator's tier and *when* in the episode the
takeover happened, and late takeovers fail. On the clean subset the behavioral
score is **0.499 = a coin flip**. The earlier "AUROC 0.78" was a timing artifact of
the synthetic operator's handback rule.

**The one exception:** `σ_Q` stays at **0.849 on the clean subset** and holds under
time- and difficulty-stratification — it is *not* a timing artifact.

---

## 3. Verdict on the quality part (finished)

**RQ2 (which intervention property predicts help/harm): answered — negatively, and
that is a legitimate thesis result.** On this task + synthetic operator, **no tested
replay-priority quality score carries information about intervention value once
timing and state-difficulty are controlled for:**
- counterfactual Q-gap `A_hat` — *inverted* (off-distribution asymmetry, RQ4 §11–12);
- behavioral composite — *timing leakage* (clean 0.499);
- hindsight — *state difficulty*, not correction quality (r=−0.81 with V(onset), window-clean 0.52);
- progress — *noise* (0.47).

So prioritising replay by any of these is not expected to help — consistent with
RQ2/RQ3's empirical losses. **Recommendation:** do **not** launch the large
`rq5_quality_score_ext` / `rq5_quality_imitation` SR campaigns as headline results;
if run at all, run them only to *demonstrate the null* (and keep the `uniform`
machinery control to show the PER machinery itself is neutral). The quality story
for the thesis is the **diagnosis + the negative result**, already documented in
[critic_ensemble.md §4.2](critic_ensemble.md) and the rq2/rq3/rq4 logs.

---

## 4. Active part (ensemble disagreement) — kicked off

The audit hands us the pivot on a plate: **σ_Q is the only intervention-time signal
that predicts outcome without leakage (clean AUROC 0.849).** σ_Q is exactly the
**active gate's trigger**, so this is where the operator-dependence reduction should
come from — and it aligns with the thesis's ultimate goal (active querying + the
ensemble).

### 4.1 What was launched
`examples/campaigns/active_validity.yaml` → `runs/active_validity`
(`active_val.service`, OOM-protected, wandb off):
- `active` mode, **`trigger_terms: [uncertainty]`** (pure σ_Q), `combiner: max`,
  `query_behavior: continue` (request is a cue → clean autonomous SR);
- the **RQ4 fixed-ensemble base** (K=8, M=1, per-member, warm-start 5000, LR 1.5e-4)
  so σ_Q is epistemic;
- **`quality_cf_diag: true`** → logs the **simulator ground truth** (`op_z_task`,
  true lateral error, jam) per step = a **timing-free** label, unlike episode success;
- 3 operators/tier × 3 tiers × 1 seed = **9 cells**.

### 4.2 The question it answers (RQ3, validity)
Does the disagreement trigger σ_Q — and the full acquisition score A(s_t) — actually
fire when the operator is **genuinely in trouble** (ground truth), not merely late in
the episode? Measured as AUROC/ECE of σ_Q and A(s_t) vs the ground-truth trouble
label, pooled + per tier + time-stratified.

### 4.3 Caveats to keep honest
- The 0.849 above is vs **episode success** (survives stratification, but still not
  the physical label). The launched run fixes this by logging the **ground truth**.
- σ_Q's per-tier means are nearly flat (0.010/0.009/0.009) yet its AUROC is high →
  its power is **within-episode/temporal**, not a skill-tier effect. The
  ground-truth, time-stratified audit will show whether that is real signal.
- **Gate-direction check:** higher σ_Q correlated with success here; the gate
  `g(σ_Q)=1/(1+(σ_Q/σ_ref)²)` *down*-weights high σ_Q. Whether the request/using
  direction is correct must be checked against the ground truth (possible inversion,
  flagged in [critic_ensemble.md §4.2](critic_ensemble.md)).

### 4.4 Next actions (active)
1. **Build the active audit** (analogue of `quality_signal_audit.py`): read the
   per-step σ_Q, A(s_t), request flag and the `gt_*` fields from
   `runs/active_validity/aqi_logs/*/steps.jsonl`; report AUROC/ECE of σ_Q and A(s_t)
   vs ground-truth trouble, per tier + time-stratified, and the request precision/recall.
2. **Read it early** — the signal accrues from the first post-warmup episodes, so we
   can judge validity before the 9 cells finish.
3. **If valid → the active training effect:** `active` vs `logged` vs `aqi` on the
   fixed base (ICC, novice SR, intervention rate). If not valid → that is the RQ3
   answer and we report it.

---

## 5. Monitoring & control

```bash
systemctl --user status active_val.service          # active-validity run
tail -f runs/active_validity.out
find runs/active_validity/checkpoints -maxdepth 2 -name .done | wc -l   # done/9
systemctl --user stop active_val.service            # stop
```

## 6. Files
- Audit output: `runs/quality_behavioral_diag/report/quality_audit/summary.md`.
- Active run: `examples/campaigns/active_validity.yaml` → `runs/active_validity/`.
- Quality verdict context: [critic_ensemble.md §4.2/§7](critic_ensemble.md),
  [thesis/rq4_ensemble_convergence_and_qa_log.md](thesis/rq4_ensemble_convergence_and_qa_log.md) §11–§12.
- Handoff tools: `examples/{quality_signal_audit,cf_score_analysis,roster_audit}.py`.

---

## 7. Quality score — consolidated verdicts (everything, in one place)

The complete, final position on the intervention-quality score, across RQ2/RQ3/RQ4
and the 2026-09-30 leakage audit:

| score / claim | measured | verdict |
|---|---|---|
| **counterfactual** `A_hat = μQ(a_H) − μQ(a_robot)` | AUROC 0.175 (all), 0.147 (clean) | **inverted** — off-policy critic under-values the off-distribution `a_H`; 64% of `A_hat` negative (RQ4 §11) |
| standardising / warm-up / IS / EMA on `A_hat` | still inert / still hurts | rescaling cannot fix a signal that isn't there (RQ2 §7/§18) |
| **behavioral** `z(dV) − z(t_int) − z(f_peak)` | 0.78 live → **0.499 clean** | **timing leakage**, not signal — `t_int ≈ 0.80·d_max(tier)`, `corr(t_int,cap)=1.00`; pure time beats it (RQ4 §12 CORRECTED) |
| behavioral without `t_int` | 0.490 clean | no signal |
| **hindsight** (n-step advantage over V) | 0.52 window-clean, r=−0.81 with V(onset) | measures **state difficulty**, not correction quality |
| **progress** (Δ insertion depth) | 0.47 | **noise** |
| `d_value` | 0.526 clean | mostly leakage |
| quality-prioritised replay (RQ3, extended) | SR 50.9% vs hilserl 84%, ICC 0.887, paired p=0.006 | **hurts** and is the most operator-dependent arm |
| **overall RQ2 answer** | — | **No tested replay-priority intervention score predicts help/harm once timing and state-difficulty are controlled.** A legitimate negative thesis result. |
| **the exception: `σ_Q`** (ensemble disagreement) | clean AUROC **0.849**, time-strat 0.812, diff-strat 0.791 | **survives every leakage/difficulty control** → the active gate is where the signal is |

**Consequences for the thesis writeup:**
- Report the quality thread as **diagnosis → negative result** (the score is the
  problem, not the critic or the rescaling). Sources: [critic_ensemble.md §4.2](critic_ensemble.md),
  rq2/rq3/rq4 logs, and this audit.
- Do **not** present the behavioral score as a fix (it's leakage). The RQ4 §12
  correction banner now says so.
- The large `rq5_quality_score_ext` / `rq5_quality_imitation` SR campaigns are
  **not worth running as headline results**; if run, only to demonstrate the null,
  keeping the `uniform` machinery control. This is the "treat the handoff plan as a
  hint" call — skip the expensive quality SR grid, pivot the compute to the active
  part.

---

## 8. Active learning — the plan (what to run to get an evaluable result)

The audit makes the active gate the obvious next focus, and σ_Q's survival is the
positive lead the whole project has been missing. The plan mirrors the quality
program's discipline (cheap validity gate → payoff campaign), but on the trigger.

### A1 — Trigger validity (**RUNNING**: `active_validity.yaml`, `runs/active_validity`)
Does σ_Q / the acquisition score `A(s_t)` fire when the operator is **physically in
trouble**, using the simulator ground truth (`op_z_task`, `gt_lat_err_mm`,
`gt_jammed`) as a **timing-free** label — the thing the quality audit lacked?
- `active` mode, `trigger_terms: [uncertainty]`, `quality_cf_diag: true`, RQ4 base,
  9 cells (3 ops/tier × 1 seed).
- **Analyzer (built, ready):** `examples/active_signal_audit.py` — reads the per-step
  `steps.jsonl` (`sigma_q`, `a_score`, `request`, `gt_*`, `op_z_task`) and reports
  AUROC of σ_Q / `A(s_t)` vs the physical trouble and the operator's own trouble
  estimate, **per tier + time-stratified**, plus request **precision/recall** and a
  **gate-direction** check (AUROC < 0.5 ⇒ the gate is inverted).
- **Read it early** — σ_Q accrues from the first post-warm-up episodes; run
  `./.venv/bin/python examples/active_signal_audit.py --campaign runs/active_validity`
  as soon as a couple of cells have `steps.jsonl`.

**Go/no-go (commit before looking):** the trigger is **valid** if AUROC(σ_Q,
physical trouble) ≥ 0.65 pooled, ≥ 0.60 time-stratified, and > 0.55 in ≥ 2/3 tiers,
in the **correct direction**. If AUROC < 0.5, the gate is inverted — flip the
request direction / the gate `g(σ_Q)` before A3 (this is the open concern flagged in
[critic_ensemble.md §4.2](critic_ensemble.md)).

### A2 — Gate calibration / direction (same data, no new run)
From A1's output: is the **request** well-targeted (precision/recall vs trouble), and
is `g(σ_Q)=1/(1+(σ_Q/σ_ref)²)` pointing the right way? If σ_Q is *positively*
associated with trouble, the current gate (which down-weights high σ_Q in the quality
path) is backwards; the **request** path (ask when σ_Q high) is the correct use and is
what A3 tests. Decide the τ_A threshold from the precision/recall table.

### A3 — Trigger training effect (**PREPARED, gated**: `active_effect.yaml`)
Does gating interventions by σ_Q **reduce operator-dependence / help the novice
tier** vs no gate? This is the active analogue of RQ4's parity result and the
thesis's RQ3/RQ5 payoff.
- Arms: `logged` (operator's own monitor, no gate) vs `active` (σ_Q gate). RQ4 base,
  **36 cells** (3 ops/tier × 2 seeds). `quality_cf_diag` stays on to grow the trigger
  audit sample.
- Endpoints: autonomous SR + **ICC** + per-tier gradient (does the gate flatten
  operator dependence, especially the weak novice tier?); **intervention rate** +
  request precision/recall (is help better-targeted?).
- **Launch only if A1 passes.** Command in the config header (wandb online, group
  `active_effect`).

### A4 — (Could) threshold / behaviour ablation
τ_A sweep and `query_behavior: continue` vs `pause`, if A3 shows an effect and time
allows. Uses the `arms:` infra (patch 0002).

### The mode ladder (ultimate goal, later)
Once the gate is validated and its effect measured, the full thesis comparison is the
ladder on the fixed base: `hilserl` → `logged` → `active` → `aqi`, plus the external
baselines `e2hil` / `hiril` (handoff 0001/0004). Since quality is dead, `aqi` is
expected to ≈ `active` (the gate carries it); run it to confirm, not to hope.

### What to run, in order

| step | what | cost | gate |
|---|---|---|---|
| **now** | `active_signal_audit.py` on `runs/active_validity` as cells land | minutes | — |
| A1 | let `active_validity` finish (9 cells) | ~hrs | trigger valid? |
| A3 | launch `active_effect.yaml` (36 cells) | ~day | only if A1 passes |
| A4 | τ_A / query-behaviour ablation | ~day | only if A3 shows an effect |
| ladder | `hilserl/logged/active/aqi` (+ e2hil/hiril) | multi-day | for the final chapter |

**Next experiment right now:** nothing new to launch — `active_validity` is running;
the immediate action is to **run `active_signal_audit.py`** on it the moment
`steps.jsonl` appears, apply the A1 go/no-go, and (if it passes) launch
`active_effect.yaml`.

### New files created for the active part
- `examples/active_signal_audit.py` — the trigger-validity analyzer (AUROC/ECE/
  precision-recall of σ_Q & `A(s_t)` vs ground truth).
- `examples/campaigns/active_validity.yaml` — A1 (running).
- `examples/campaigns/active_effect.yaml` — A3 (prepared, gated on A1).

