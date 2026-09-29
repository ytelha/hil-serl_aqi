# E2HiL comparison baseline (`--aqi_mode=e2hil`)

An external baseline for the thesis: **E2HiL** — H. Deng et al., *"E2HiL:
Entropy-Guided Sample Selection for Efficient Real-World Human-in-the-Loop
Reinforcement Learning"*, IEEE RA-L 2026,
[arXiv:2601.19969](https://arxiv.org/abs/2601.19969),
[project page](https://e2hil.github.io/), official code
[E2HiL/E2HiL-project-a1x](https://github.com/E2HiL/E2HiL-project-a1x) (a JAX
HIL-SERL fork, the same `serl_launcher` lineage as this repo).

It is **not part of AQI**. It is a published competitor that also answers
"which intervention samples should the learner trust?", with a
different signal. AQI scores each intervention segment by its value to the
task (counterfactual or behavioural `q_int`). E2HiL scores each replay state by
its influence on policy entropy and drops the extremes.

---

## 1. The method in one screen

E2HiL is a plug-in for HIL-SERL. Buffers, demos, the critic, the temperature
and the actor loop are unchanged. Only the **actor loss** changes. At each actor
update, for every state `s` in the (online + demo) batch:

| step | formula | paper | our code |
|---|---|---|---|
| MC actions | `a_k ~ pi(.|s)`, `k = 1..K`, log-probs `lp_k` | §III-D | `influence_per_state` |
| soft value | `g_k = Q(s,a_k) - alpha*lp_k`, `V(s) = mean_k g_k` | Eq. 7 | same |
| soft advantage | `A_k = g_k - V(s)` | Eq. 6 | same |
| logit update | `y_k = pi(a_k|s) * A_k` (pi clipped at 1e6) | Eq. 8 | same |
| entropy influence | `c(s) = -Cov_k(lp_k, y_k)` (stop-gradient) | Eq. 9–10 | same |
| selection | `I(s) = 1[ P5(|c|) <= |c(s)| <= P90(|c|) ]` over the batch | Eq. 11 | `entropy_bounded_mask` |
| actor loss | `L = -sum_s I(s) l(s) / (sum_s I(s) + eps)`, `l = Q - alpha log pi` | Eq. 12 | `e2hil_actor_loss` |

States above the 90th percentile are **shortcut** samples, which cause sharp
entropy collapse. States below the 5th percentile are **noisy** samples, with
negligible effect. The paper reports that most clipped states come from human
interventions (its Table III / Fig. 8).

## 2. Where it lives

| file | what |
|---|---|
| [serl_launcher/agents/continuous/e2hil.py](../serl_launcher/serl_launcher/agents/continuous/e2hil.py) | the method: influence, mask, masked loss, per-source diagnostics |
| [serl_launcher/agents/continuous/sac.py](../serl_launcher/serl_launcher/agents/continuous/sac.py) | `policy_loss_fn` calls it when `config["e2hil"]` is set |
| [serl_launcher/aqi/config.py](../serl_launcher/serl_launcher/aqi/config.py) | mode `e2hil`, `e2hil_K / e2hil_q_low / e2hil_q_high / e2hil_q_agg`, `baseline_arch` |
| [examples/train_aqi_rlpd.py](../examples/train_aqi_rlpd.py) | `--aqi_mode=e2hil` + the four `--e2hil_*` flags; transition source labels |
| [examples/run_campaign.py](../examples/run_campaign.py) | YAML `aqi.e2hil_*` / CLI `--e2hil_*`; **intervention-rate** report section |
| [examples/campaigns/e2hil_comparison.yaml](../examples/campaigns/e2hil_comparison.yaml) | `hilserl` vs `e2hil`, RQ4-identical settings |
| [tests/test_e2hil.py](../tests/test_e2hil.py), [tests/test_campaign_e2hil.py](../tests/test_campaign_e2hil.py) | formula-vs-official, mask, real `SACAgent.update`, plumbing |

`e2hil` has `hilserl`'s exact architecture: 2 critics, no subsample, no
bootstrap, no AQI runtime and no JSONL training logs. The actor loop is
byte-identical, so `e2hil` and `hilserl` checkpoints are interchangeable
(`eval_ruler.py` needs no change).

## 3. Fidelity to the official code, and every deliberate difference

The port follows the authors' `examples/cov_actor_loss.py`
(`cov_actor_loss_fn` / `_compute_cov_per_state`), not only the paper text.
Two facts are only visible in the code:

1. **`c` is per state, not per stored action.** The paper writes `c(s_t, a_t)`,
   but the official code draws K fresh actions from the current policy at `s`
   and never uses the buffer action. The "sample" being selected is the batch
   state's contribution to the SAC actor objective.
2. **The actor objective uses `min` over critics.** Our `hilserl` actor uses
   the **mean** over critics. With E2HiL's defaults, two things therefore change
   versus `hilserl`: the mask **and** `min` vs `mean`. `--e2hil_q_agg=mean`
   removes the second difference, so the mask is then the only difference.
   Report which one you ran. `min` (the official setting) is the default.

Deliberate differences (none changes the numbers):

| difference | why |
|---|---|
| The K critic passes run as a sequential `lax.map` instead of one `(B*K)`-row forward | The official code repeats every camera image K times (8× activation memory on the ResNet encoder). Ours peaks at one batch-sized pass. Numerically identical (the test checks it against a transcription of the official formula). |
| `eta` in `c = -eta Cov` is dropped | A positive constant cancels in the percentile mask (the official code also drops it). |
| Source labels are logged only in `e2hil` mode | The official code adds `labels` to every buffer. Here other modes stay byte-identical. |
| `SACAgent` only | Learned-gripper (hybrid) agents raise `NotImplementedError`. The sim task uses `SACAgent`. |

Hyperparameter defaults = paper/official: `K=8` (`train_rlpd.py` in their repo
defaults to 4, but their launch script and the paper use 8), band
`[0.05, 0.90]`. `K=1` makes every covariance exactly 0, so nothing is masked
(same as the official code). Keep that in mind if you reproduce their K
ablation.

## 4. Running it

Single run (two shells), exactly like any other mode:

```bash
cd examples && source experiments/sim_peg_insertion/sim_env.sh
python train_aqi_rlpd.py --exp_name=sim_peg_insertion --learner --aqi_mode=e2hil \
    --demo_path=<demos.pkl> --checkpoint_path=<ckpt>
python train_aqi_rlpd.py --exp_name=sim_peg_insertion --actor --aqi_mode=e2hil \
    --checkpoint_path=<ckpt> --synthetic_operator=expert
```

The learner prints `E2HiL actor loss ON: K=8, |c| band = [0.05, 0.90] ...`.

Campaign (36 cells: 2 methods × 3 tiers × 3 operators × 2 seeds):

```bash
WANDB_RUN_GROUP=e2hil_comparison \
systemd-run --user --unit=e2hil_campaign --collect \
  -p ManagedOOMPreference=avoid -p WorkingDirectory=$PWD \
  -- bash -c './.venv/bin/python examples/run_campaign.py \
       --config examples/campaigns/e2hil_comparison.yaml \
       --campaign runs/e2hil_comparison \
       --converge_require_autonomous --wandb online \
       >> runs/e2hil_comparison.out 2>&1'
```

### Reusing the RQ4 `hilserl` cells (halves the compute)

`e2hil_comparison.yaml` matches RQ4's `hilserl` arm in every setting:
- same roster (`sample_seed 0`, `jitter 0.15`, 3/tier) and 6 filtered demos;
- 40k-step cap and the same convergence rule;
- the same ±16 mm ruler, no pretrain, default LR.

This patch leaves the `hilserl` code path unchanged, so RQ4's 18 scored
`hilserl` cells can be copied in. The campaign then trains only the 18 `e2hil`
cells, on the **same demos**, which keeps the design paired:

**The operators must be copied too.** The copied cells and demos were made
with `$SRC`'s synthetic operators. A new folder would otherwise sample its
own roster, which differs from `$SRC`'s (always before patch 0007, whose
fixed key order still draws different values than an old unpatched process).
The new cells would then train against other operators than the copied ones,
silently breaking the paired design. Requirements:
- **patch 0007 applied first.** Without it, the first `run_campaign.py`
  call in `$DST` re-samples and overwrites the copied `operators/*.json`;
- `$SRC` audited: `roster_audit.py` must not report MISMATCH cells for it.

```bash
SRC=runs/rq4_ensemble_convergence; DST=runs/e2hil_comparison
./.venv/bin/python examples/roster_audit.py $SRC       # no MISMATCH, else ask the user
mkdir -p $DST/checkpoints $DST/eval_runs
cp -r $SRC/eval_grid.json $SRC/demos $SRC/operators $SRC/operators.csv $DST/
for d in $SRC/checkpoints/hilserl_*/; do
  c=$(basename $d); mkdir -p $DST/checkpoints/$c
  cp $d/.done $DST/checkpoints/$c/
  cp $d/episodes.jsonl $d/CONVERGED.json $d/manifest.json $DST/checkpoints/$c/ 2>/dev/null
done
cp -r $SRC/eval_runs/hilserl_* $DST/eval_runs/
# repairs an operator file that an old run rewrote, from the copied demos
./.venv/bin/python examples/roster_audit.py --restore $DST
```

(`manifest.json` is copied so the audit can check the copied cells as well.)

Checks:
- The runner must **not** print `a DIFFERENT eval grid was requested`. If it
  does, stop.
- `--plan` should list 36 cells.
- The train stage should log `already trained (.done) -- skip` for every
  `hilserl_*` cell.
- After the first new cells finish, `roster_audit.py $DST` must report
  them consistent with their demos.

If a copied cell has no `episodes.jsonl`, its intervention rate shows as
missing (SR / ICC / convergence are unaffected).

## 5. What to read out

`runs/e2hil_comparison/report/dependency.md` contains everything RQ1–RQ4
reported, plus a new section:

- **Human effort: intervention rate.** Operator steps / env steps per cell
  (the paper's second metric, next to success rate), per tier, and paired
  `e2hil − hilserl` deltas with a sign test. It is computed from
  `checkpoints/<cell>/episodes.jsonl`, which every mode writes, so re-running
  `--stages analyze` on older campaigns (RQ1–RQ4) adds it retroactively.

wandb (learner, `actor/` group):

| key | meaning |
|---|---|
| `entropy` | policy entropy: the paper's main diagnostic (smoother decline vs `hilserl`) |
| `e2hil_keep_ratio` | fraction of batch states kept (≈ 0.85 by construction) |
| `e2hil_abs_c_lo` / `e2hil_abs_c_hi` | the percentile bounds of \|c\| |
| `e2hil_{demo,online,intervention}_abs_c_mean` | mean \|c\| per source (paper Table III) |
| `e2hil_{demo,online,intervention}_keep_frac` | share of each source that survives the mask (paper: interventions are clipped most) |

The last row connects E2HiL directly to this thesis. It shows how often
E2HiL discards intervention states and which tiers they come from. You can
compare that with what AQI's `q_int` would have up- or down-weighted.

## 6. Ablations the paper reports (all flags, no code)

| ablation (paper Fig. 9) | how |
|---|---|
| Monte-Carlo samples `K ∈ {1,4,8,16}` | `--e2hil_K` / YAML `aqi.e2hil_K` |
| clipping band 5–90 / 10–90 / 5–80 | `--e2hil_q_low`, `--e2hil_q_high` |
| mask only (no `min`-Q change) | `--e2hil_q_agg=mean` |

The runner keys cells by mode, so each variant needs its own `--campaign`
directory.

## 7. Tests

```bash
python tests/test_e2hil.py            # needs jax/flax (training env)
python tests/test_campaign_e2hil.py   # plumbing + intervention-rate analysis
```

`test_e2hil.py` checks:
- `c(s)` against a numpy transcription of the official formula;
- the mask against `np.percentile`;
- a real (tiny) `SACAgent.update`: finite E2HiL diagnostics and the actor
  actually updates;
- that with the full band and `q_agg=mean` the loss equals the `hilserl` loss
  exactly, and the critic loss is unchanged in all cases.
