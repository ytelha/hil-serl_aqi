"""Role 1 - collect synthetic demonstrations (methodology doc S5).

Runs a SyntheticOperator closed-loop from reset to termination and
writes transitions in the SAME pkl format as examples/record_demos.py
(observations / actions / next_observations / rewards / masks / dones /
infos), so the resulting file drops straight into --demo_path of any
training script. Ground truth (R2) - the operator's theta, s, tier and
per-episode outcomes - is written to a sidecar JSON next to the pkl.

The filtering policy (doc S5 step 4) is explicit: --filtered keeps only
successful episodes (the HIL-SERL default; skill then manifests as
quality variation among successes plus attempts-per-kept-demo cost);
--nofiltered keeps everything (the harsher regime for a quality-aware
learner, which must down-weight actively harmful data).

Default environment is the 2-D toy insertion study; pass a different
env via --env module:factory (a zero-arg callable returning a gym env
exposing a `phase_oracle` attribute, or use --oracle_checkpoint-style
integration in train_aqi_rlpd.py for policy oracles).

Examples:
    python collect_synthetic_demos.py --profile expert --episodes 20 \
        --out ./demos/expert_toy.pkl
    python collect_synthetic_demos.py --profile s=0.35 --nofiltered \
        --episodes 30 --seed 7 --out ./demos/s035_unfiltered.pkl
"""

import copy
import importlib
import json
import os
import pickle as pkl

import numpy as np
from absl import app, flags

try:
    import cv2
except ImportError:
    cv2 = None

from serl_launcher.aqi.operator_model import (SyntheticOperator,
                                              make_operator_theta,
                                              theta_from_json)
from serl_launcher.aqi.toy_insertion import (BUDGET_S, ToyInsertionEnv,
                                             ToyInsertionOracle)

FLAGS = flags.FLAGS
flags.DEFINE_string("profile", "expert",
                    "operator profile: tier name, 'op1'..'op4', 's=0.7',"
                    " or 'exec=expert,mon=novice'")
flags.DEFINE_integer("episodes", 20, "episodes to KEEP (filtered) or to"
                     " run (unfiltered)")
flags.DEFINE_integer("max_attempts", 400, "attempt cap in filtered mode")
flags.DEFINE_boolean("filtered", True, "keep only successful episodes"
                     " (doc S5 step 4)")
flags.DEFINE_boolean("jitter", False, "apply per-operator theta jitter")
flags.DEFINE_integer("seed", 0, "seed (R6: same seed => same demos)")
flags.DEFINE_string("out", "./synthetic_demos.pkl", "output pkl path")
flags.DEFINE_string("env", "", "optional 'module:factory' for a custom"
                    " env with a phase_oracle attribute")
flags.DEFINE_boolean("render", False,
                     "show a live window while collecting synthetic demos")
flags.DEFINE_string("theta_json", "",
                    "path to a JSON operator-theta file (operator_model."
                    "theta_to_json). When set it overrides --profile/--jitter "
                    "with an EXACT theta so a jittered operator collects demos "
                    "with the same theta that later drives its interventions.")
flags.DEFINE_string("operator_id", "", "override operator id in logs")


def build_env_and_oracle():
    if FLAGS.env:
        mod, fac = FLAGS.env.split(":")
        env = getattr(importlib.import_module(mod), fac)()
        oracle = getattr(env, "phase_oracle")
        budget = getattr(env, "budget_s", BUDGET_S)
        action_scale = getattr(env, "operator_action_scale", 1.0)
    else:
        env = ToyInsertionEnv(seed=FLAGS.seed)
        oracle = ToyInsertionOracle()
        budget = BUDGET_S
        action_scale = 1.0
    return env, oracle, budget, action_scale


def main(_):
    rng = np.random.default_rng(FLAGS.seed)
    if FLAGS.theta_json:
        theta = theta_from_json(FLAGS.theta_json)
        if FLAGS.operator_id:
            theta.operator_id = FLAGS.operator_id
    else:
        theta = make_operator_theta(FLAGS.profile, rng=rng,
                                    jitter=FLAGS.jitter,
                                    operator_id=FLAGS.operator_id or None)
    env, oracle, budget, action_scale = build_env_and_oracle()
    op = SyntheticOperator(oracle, theta,
                           rng=np.random.default_rng(FLAGS.seed + 17),
                           action_scale=action_scale)
    render_on = bool(FLAGS.render and cv2 is not None)
    if FLAGS.render and cv2 is None:
        print("render requested but opencv-python is not installed; continuing headless")
    win_name = "synthetic_demos"
    if render_on:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    transitions, ep_meta = [], []
    kept = attempts = 0
    while kept < FLAGS.episodes and attempts < FLAGS.max_attempts:
        ep_seed = FLAGS.seed * 100000 + attempts
        obs, info = env.reset(seed=ep_seed)
        op.reset()
        traj, done, ep_ret, n = [], False, 0.0, 0
        while not done:
            a = np.asarray(op.act(obs), dtype=np.float32)
            next_obs, rew, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            if render_on:
                frame = env.render()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(win_name, frame)
                    cv2.waitKey(1)
            traj.append(copy.deepcopy(dict(
                observations=obs, actions=a, next_observations=next_obs,
                rewards=float(rew), masks=1.0 - float(done),
                dones=float(done),
                infos=dict(succeed=bool(info.get("succeeded",
                                                 info.get("succeed", False)))),
            )))
            obs = next_obs
            ep_ret += rew
            n += 1
        success = bool(info.get("succeeded", info.get("succeed", False)))
        attempts += 1
        keep = success or (not FLAGS.filtered)
        if keep:
            transitions.extend(traj)
            kept += 1
        ep_meta.append(dict(attempt=attempts - 1, seed=ep_seed, kept=keep,
                            success=success, steps=n,
                            duration_s=n / 10.0,
                            commit_err=(op.commit_events[0][1]
                                        if op.commit_events else None),
                            attempts_within=op.n_approaches))
        print(f"[{attempts:3d}] success={success} kept={keep} "
              f"steps={n} (kept {kept}/{FLAGS.episodes})")

    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
    with open(FLAGS.out, "wb") as f:
        pkl.dump(transitions, f)
    succ_total = sum(m["success"] for m in ep_meta)
    sidecar = dict(
        profile=FLAGS.profile, theta=theta.as_dict(),
        filtered=bool(FLAGS.filtered), seed=FLAGS.seed,
        episodes_kept=kept, attempts=attempts,
        attempts_per_kept=attempts / max(kept, 1),
        success_rate=succ_total / max(attempts, 1),
        budget_s=budget, episodes=ep_meta,
    )
    side_path = os.path.splitext(FLAGS.out)[0] + "_groundtruth.json"
    with open(side_path, "w") as f:
        json.dump(sidecar, f, indent=1)
    print(f"\nwrote {len(transitions)} transitions from {kept} episodes "
          f"to {FLAGS.out}")
    print(f"ground truth (theta, s, outcomes) -> {side_path}")
    print(f"attempts per kept demo: {sidecar['attempts_per_kept']:.2f} "
          f"(the human-cost currency, doc S9.1)")
    if render_on:
        cv2.destroyAllWindows()

    # self-describing provenance next to the payload (best-effort)
    try:
        import provenance as _prov
        th = theta.as_dict()
        _prov.document(
            os.path.dirname(os.path.abspath(FLAGS.out)) or ".", "demo",
            quiet=True,
            task={"exp_name": FLAGS.env or "toy_insertion",
                  "peg_shape": os.environ.get("SIM_SYN_PEG_SHAPE"),
                  **({"rpy_box": os.environ["SIM_RPY_BOX"]}    # sim orientation box
                     if os.environ.get("SIM_RPY_BOX") else {})},
            operator={"operator_id": th.get("operator_id", FLAGS.profile),
                      "tier": th.get("tier"), "s_exec": th.get("s_exec"),
                      "s_mon": th.get("s_mon"),
                      **{k: th.get(k) for k in ("sigma0", "v_max", "sigma_perc",
                                                "delta_in", "p_miss", "rho_fa",
                                                "l_react")},
                      "theta_path": os.path.basename(side_path)},
            seeds={"seed": FLAGS.seed},
            demos={"n": kept, "kept": kept, "attempts": attempts,
                   "filtered": bool(FLAGS.filtered),
                   "file": os.path.basename(FLAGS.out),
                   "ground_truth": os.path.basename(side_path)})
    except Exception:
        pass


if __name__ == "__main__":
    app.run(main)
