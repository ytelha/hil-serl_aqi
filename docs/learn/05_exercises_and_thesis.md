# 05 — Exercises, and preparing the thesis defence

The exercises go from reading to changing code. Levels 1–3 are safe
anywhere. Level 4 changes code: do those on a branch, behind a flag, with
tests. Several level-4 tasks are real open tasks from the experiment plans,
so learning and thesis progress are the same work.

---

## Level 1 — navigate (no running)

1. **Follow σ_Q.** Open `aqi/uncertainty.py`, put the cursor on
   `q_stats_single`, press `Shift+F12`. Write down every caller, and what
   each does with the value (gate, failure features, log).
2. **Follow `quality_weight`.** From `aqi/runtime.py → end_episode` (where
   segments get `weights`) to the transition dict, the intervention data
   store, the learner batch, and finally `sac.py → policy_loss_fn` (the
   `aqi_bc_weight` block). Which modes does it affect, and why is it inert
   when `bc_loss_weight = 0`?
3. **Modes table from the code.** Read the four properties of `AQIConfig`
   (`active_on`, `quality_on`, `logging_on`, `baseline_arch`). Fill a table
   mode × property, and check it against part 02 §5.1.
4. **The wrapper stack.** Read
   `examples/experiments/sim_peg_insertion/config.py → get_environment`.
   Draw the stack yourself, then compare with part 01 §4.
5. **One learner step.** In `train_aqi_rlpd.py → learner`, find:
   - the two iterators and `concat_batches`;
   - the `cta_ratio` loop;
   - `publish_network`.

   Explain in 3 sentences what one iteration of the `for step` loop does.

## Level 2 — run and observe (cheap)

6. **Run the tests** (training environment, repo root):

   ```bash
   for t in tests/test_*.py; do echo "== $t"; ./.venv/bin/python "$t" 2>&1 | tail -1; done
   ```

   Pick two test files and read them: what exactly do they guarantee?
7. **Dry-run a campaign** into a scratch folder:

   ```bash
   ./.venv/bin/python examples/run_campaign.py --config examples/campaigns/rq5_quality_score.yaml --campaign /tmp/learn_dry --stages train,eval --dry_run | grep DRY | head -4
   ```

   Explain **every flag** of one learner and one actor command. Find where
   each flag is defined (`flags.DEFINE_…` in `train_aqi_rlpd.py`) and where
   it ends up (`AQIConfig` or the train config). Then `rm -rf /tmp/learn_dry`.
8. **Read a finished cell.** On the cluster:
   `runs/method-test/checkpoints/quality-hind_int00_s0/README.md` and
   `manifest.json`, then `eval_runs/…/cell.json` and `trials.jsonl`.
   Recompute the SR from `trials.jsonl` by hand.
9. **Reproduce finding F6.** Write a 15-line script that reads one
   `aqi_logs/<run>/steps.jsonl` and plots mean σ_Q per episode (log
   scale). Compare with the table in `handoff/FINDINGS_cluster_method_test.md`.

## Level 3 — small experiments in a notebook

10. **JAX compilation.**
    - Time a function with and without `jax.jit`: the first call vs later
      calls.
    - Call it with a new input shape and watch it recompile (the time jumps
      again).
    - Then read how `counterfactual_advantage_batch` avoids that by padding.
11. **An ensemble by hand.** Use `ensemblize(MLP, 8)` (part 03 §3) to build
    8 networks. Evaluate them on inputs near and far from the training data
    after a few gradient steps on a tiny regression. Does the spread
    (std over axis 0) grow away from the data? That's the epistemic idea
    behind σ_Q, in 30 lines.
12. **Synthetic operators on the toy task.** With
    `serl_launcher/aqi/toy_insertion.py` (`ToyInsertionEnv`,
    `ToyInsertionOracle`, `rollout_episode`) and `aqi/operator_model.py`
    (`SyntheticOperator`, `make_operator_theta`):
    - roll out 50 episodes each for `"expert"`, `"intermediate"` and
      `"novice"`;
    - compare success rate, time and jam counts.

    You've just reproduced the tier separation behind the whole study.
13. **ICC intuition.** Call `aqi/stats.icc_oneway` on made-up groups:
    - identical operator means → ICC ≈ 0;
    - very different means with small seed noise → ICC → 1.

    Then add one outlier operator and watch `permutation_sigma2` react.

## Level 4 — change the code (real thesis tasks)

The workflow for every change:
1. Create a branch (`git switch -c <name>`).
2. Implement it behind a flag that defaults to off.
3. Write a unit test.
4. Run all tests.
5. Prove that all existing campaign configs emit **identical** commands
   (`GUIDE_campaigns_and_yaml.md` §4 in `handoff/` shows how to dry-run
   them).
6. Commit it.
7. For the training machine, make a handoff patch.

14. **Placebo score** (quality plan task T2): add `quality_score_mode:
    shuffled`, which gives each segment the real score of a random other
    segment. This separates "boosting interventions" from "boosting the
    right interventions".
15. **Sliding-window gate** (active plan task T-A5): an option that makes
    the gate's z-scores use the last N steps instead of all-time
    statistics. First replay the logged `sigma_q` offline and show the
    difference, then add the option.
16. **Time-controlled trigger evaluation** (part of task T-A1): a
    read-only script computing the AUROC of `a_score` / `p_fail` / `sigma_q`
    for "episode ends in failure", **within** bins of the step in the
    episode, with a clock as a yardstick.

## The thesis defence

### The story (a suggested chapter arc)

1. **Problem.** In human-in-the-loop RL, the trained policy depends on who
   trained it. Why that matters for industrial deployment (the Siemens/WZL
   context).
2. **Background.** RL and SAC; RLPD / HIL-SERL; uncertainty from
   ensembles; intervention quality; related work (E2HiL, HiRIL).
3. **Method.**
   - AQI: an active gate (when to ask) and intervention quality (whom to
     trust, how much);
   - the synthetic operator model;
   - the frozen ruler and the dependency statistics.
4. **Experimental infrastructure.** The campaign runner, reproducibility
   (frozen roster and ruler, provenance, pre-registration), the cluster.
5. **Results,** RQ by RQ, including the negative ones.
6. **Discussion,** the lessons with mechanisms:
   - why the counterfactual score is anti-aligned;
   - why the ensemble penalty was fixable (M = 2 → 1);
   - timing leakage (F4, F7);
   - normalisation fade-out (F6);
   - updates per env step (F3);
   - the roster bug (F1) and how it was found.
7. **Conclusion and next steps:** the hardware stages (SO-101 human study,
   UR5e).

**Negative results, explained, are results.** An examiner learns more from
"the counterfactual advantage is non-positive almost by construction,
because the actor maximises the same critic" than from a table of numbers.
Your logs contain several such explanations; they're a strength of the
thesis.

### Figures to prepare (draw them yourself: that's also how you learn them)

- **System architecture:** environment + wrappers → actor (policy, AQI
  runtime) ⇄ agentlace ⇄ learner (two buffers, SAC update). Part 01 §5.
- **The operator model:** perception → decision → motor, and the
  monitoring gate. `docs/methods.md` §1.
- **The mode ladder** (`hilserl` → `logged` → `active` / `quality` →
  `aqi`) and what each contrast isolates.
- **The campaign pipeline:** roster → demos → cells → ruler → analysis,
  with what's frozen.
- **Results:**
  - SR and ICC per method and tier;
  - paired per-cell deltas;
  - signal-validity AUROC with and without timing control;
  - the σ_Q trajectory (F6).

### Likely examiner questions, and where your answers are

| question | where to prepare the answer |
|---|---|
| Why SAC / an off-policy method, not PPO? | part 02 §2–3: sample efficiency, reuse of demos and interventions (RLPD) |
| Are synthetic operators realistic? How do you know tiers differ? | `docs/methods.md`, `operator_skill_assessment.md`, exercise 12; the planned human study |
| How do you know an ICC difference isn't noise? | `dependency_analysis.md`: permutation test, bootstrap, seeds as replicates |
| What does σ_Q measure? Is it really epistemic? | `critic_ensemble.md`; F6; the active plan (AQ1, the diversity control) |
| Why did quality-prioritised replay hurt in RQ2/RQ3? | `docs/thesis/rq2…`, `rq3…`; the machinery control Q3a in the quality plan |
| What is timing leakage, and how did you control it? | `quality_signal_audit.md`; F4, F7; time-stratified and ground-truth labels |
| Is the comparison fair across methods? | frozen ruler, same demos and operators (the F1 fix, patch 0009 recipe), updates per env step (F3) |
| Why is plain HIL-SERL so strong here, and what's left to improve? | RQ4 results; the novice tier; human effort as a second axis |
| Would this transfer to a real robot and real people? | limits of the sim and the synthetic operator; the SO-101/UR5e stages |
| What would you do with six more months? | the experiment plans in `handoff/`, ordered by priority |

### Practise like this

- A **2-minute pitch** of the whole thesis, without slides.
- **60 seconds per figure**, out loud.
- **A whiteboard derivation** of the SAC losses, the ICC, and the gate
  formula.
- **One "why" deep enough per design choice**, e.g.:
  - why half the batch from demos;
  - why frozen rosters;
  - why padding for `jit`;
  - why the time feature makes P_fail a clock.

## What this gives you beyond the thesis

- **ML research engineering:** JAX/Flax, distributed actor–learner systems,
  replay buffers, uncertainty, prioritisation.
- **Experimental science:** controls, confounds, paired designs,
  pre-registration, mixed-effects thinking, honest negative results.
- **Research infrastructure:** idempotent pipelines, provenance, HPC/Slurm,
  git and patch workflows, testing strategies.
- **Communication:** writing documentation other people (and agents) can
  act on. You've been doing it all along in `docs/` and `handoff/`.

Before publishing any code or results outside the university, ask your
supervisor what the Siemens cooperation allows.
