# Learning this repository — start here

A self-study course built around **this** codebase. The goal: by the end
you can explain every part of your thesis system, from the maths down to
the line of code; change it confidently; and defend it in front of
examiners. Along the way you learn the libraries, programming ideas and
Python features you'll keep using after the thesis.

## The six parts

| part | what you learn | time |
|---|---|---|
| [01 — Repository map](01_repo_map.md) | the layers, every directory, **where each algorithm lives**, the environment wrapper chain, how the actor and the learner talk, one training step traced end to end | 1 week |
| [02 — Algorithms](02_algorithms.md) | RL basics → SAC → RLPD / HIL-SERL → critic ensembles → AQI (gate + quality) → E2HiL → HiRIL → synthetic operators → the evaluation statistics. Each with the maths, the file/function and a check question | 2 weeks |
| [03 — Libraries](03_libraries.md) | NumPy, JAX, Flax, Optax, distrax, chex, Gymnasium, MuJoCo, agentlace, absl, PyYAML, wandb: the core concepts, where each is used here, a tiny example, pitfalls | 1 week (and ongoing) |
| [04 — Python and programming paradigms](04_python_and_paradigms.md) | every Python feature this code uses, with a real example location; functional vs object-oriented design, design patterns, distributed actor–learner, configuration, reproducibility engineering, testing | 1 week (and ongoing) |
| [05 — Exercises and the thesis defence](05_exercises_and_thesis.md) | graded exercises (read → run → experiment → change), how to tell the thesis story, the figures to prepare, likely examiner questions and where the answers are | last week, then continuously |

The times assume about 1.5–2 hours a day. Stretch them if you're running
experiments at the same time; the parts don't depend on a calendar.

## How to study (the method matters more than the hours)

1. **Read with the code open.** Every section names files and functions.
   Open them in VS Code next to the text:
   - `Ctrl+P` opens a file by name;
   - `F12` goes to a definition; `Shift+F12` finds every use;
   - `Ctrl+Shift+O` jumps to a function in the current file;
   - `Ctrl+Shift+F` searches the whole repo.
2. **Trace, don't skim.** For each algorithm, follow one piece of data
   through the code: where it's created, transformed, stored and used.
   Write the path down, e.g. `sigma_q: uncertainty.py → runtime.observe_step
   → query_gate.step → steps.jsonl`.
3. **Run small things.** A Python REPL or a Jupyter notebook next to the
   repo is your lab. JAX and Flax especially are learned by typing examples,
   not by reading. The exercises in part 05 tell you what to try.
4. **Explain it back.** After each section, explain it in 3–5 sentences
   without looking, out loud or on paper. If you can't, re-read. This is
   exactly the skill the defence tests.
5. **Keep a glossary and a question list.** Every unknown word goes into
   the glossary. Every "why is it done this way?" goes into the question
   list. Many answers are in `docs/`; the rest are good questions for your
   supervisor, or for Claude.
6. **Connect to your results.** Each algorithm has a finding attached in
   your logs (`docs/thesis/`) or in `handoff/FINDINGS_cluster_method_test.md`.
   Knowing *why* something failed is worth more than knowing how it works.

## Checkpoints — you're ready when you can…

- [ ] draw the system from memory: environment + wrappers, actor, learner,
      the two replay buffers, the AQI runtime, and what flows between them
      (part 01);
- [ ] write down the SAC critic and actor losses, and point to the lines
      that compute them (part 02);
- [ ] explain what `jax.jit`, `jax.grad`, `jax.vmap` and a pytree are, and
      why `SACAgent` is immutable (part 03);
- [ ] explain how a Gymnasium wrapper works and name the wrappers around
      the simulator in order (parts 01 and 04);
- [ ] explain the active gate formula and the counterfactual quality score,
      and what went wrong with each (parts 02 and 05);
- [ ] run a campaign dry run, read a cell's README and manifest, and find
      its logs (part 05);
- [ ] change a small piece of the code behind a flag, write a test for it,
      and prove that old configs produce identical commands (part 05).

## Where the existing documentation fits

This course routes you through the repo's own docs instead of repeating
them:

| doc | read it in |
|---|---|
| [`getting_started.md`](../getting_started.md) | part 01 (first day) |
| [`architecture.md`](../architecture.md) | part 01 |
| [`aqi_walkthrough.md`](../aqi_walkthrough.md) | parts 01–02 |
| [`sim_walkthrough.md`](../sim_walkthrough.md) | part 01 |
| [`methods.md`](../methods.md), [`oracle_and_synthetic_operator.md`](../oracle_and_synthetic_operator.md) | part 02 (synthetic operators) |
| [`critic_ensemble.md`](../critic_ensemble.md) | part 02 (ensembles) |
| [`quality_scores_v2.md`](../quality_scores_v2.md), [`quality_signal_audit.md`](../quality_signal_audit.md), [`counterfactual_score_study.md`](../counterfactual_score_study.md) | part 02 (quality) |
| [`e2hil_baseline.md`](../e2hil_baseline.md), [`hiril_baseline.md`](../hiril_baseline.md) | part 02 (baselines) |
| [`campaign_runner.md`](../campaign_runner.md), [`dependency_analysis.md`](../dependency_analysis.md), [`eval_ruler.md`](../eval_ruler.md) | parts 02 and 05 |
| [`thesis/`](../thesis/README.md) (RQ logs) | part 05 (the story of your results) |
