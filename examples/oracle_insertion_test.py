#!/usr/bin/env python3
"""Can the synthetic operator insert the peg on its own? A training-free check
of the geometric operator in the real MuJoCo task, with and without rotation
control (operator_model.SyntheticOperator, --synthetic_rotation_control).

Two scenarios per (peg shape, tier, rotation control):
  solo    the operator drives every step from a training reset, as when it
          records demonstrations;
  rescue  random actions first (like an untrained policy, which also twists
          the tool), then the operator takes over (begin_takeover) and keeps
          control until success or the time limit - a latched takeover.

Reported: success rate, median steps to success, and the tool's orientation
error (deg) at takeover. No images and no learner, so it runs on a CPU node:

    python examples/oracle_insertion_test.py --episodes 30 --out oracle_test.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "serl_launcher"))

from experiments.sim_peg_insertion.wrapper import make_synthetic_sim_env  # noqa: E402
from serl_launcher.aqi.operator_model import (SyntheticOperator,  # noqa: E402
                                              make_operator_theta)


def run_episode(env, op, scenario, ep_seed, rescue_steps, rng, max_steps):
    obs, info = env.reset(seed=ep_seed)
    op.reset()
    orc = env.phase_oracle
    twist = None
    steps = 0
    if scenario == "rescue":
        for _ in range(rescue_steps):
            a = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, _r, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                return dict(success=bool(info.get("succeed", False)), steps=steps,
                            op_steps=0, twist_deg=None, already_done=True)
        op.begin_takeover(obs)
    err = orc.orientation_error(obs)
    twist = float(np.degrees(np.linalg.norm(err))) if err is not None else None
    op_steps = 0
    done = False
    while not done and steps < max_steps:
        a = np.asarray(op.act(obs), dtype=np.float32)
        obs, _r, term, trunc, info = env.step(a)
        steps += 1
        op_steps += 1
        done = bool(term or trunc)
    return dict(success=bool(info.get("succeed", False)), steps=steps, op_steps=op_steps,
                twist_deg=twist, already_done=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="cuboid,cylinder")
    ap.add_argument("--tiers", default="expert,intermediate,novice")
    ap.add_argument("--rotation", default="off,on", help="rotation control settings to test")
    ap.add_argument("--scenarios", default="solo,rescue")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--rescue_steps", type=int, default=60,
                    help="random steps before the takeover in the rescue scenario")
    ap.add_argument("--max_steps", type=int, default=300, help="episode limit (training: 300)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for shape in [s for s in args.shapes.split(",") if s]:
        env = make_synthetic_sim_env(peg_shape=shape, image_obs=False, max_episode_length=args.max_steps)
        for tier in [t for t in args.tiers.split(",") if t]:
            theta = make_operator_theta(tier, rng=np.random.default_rng(args.seed))
            for rot in [r for r in args.rotation.split(",") if r]:
                for scen in [s for s in args.scenarios.split(",") if s]:
                    op = SyntheticOperator(env.phase_oracle, theta,
                                           rng=np.random.default_rng(args.seed + 17),
                                           action_scale=env.operator_action_scale,
                                           rotation_control=(rot == "on"))
                    rng = np.random.default_rng(args.seed + 31)
                    eps = [run_episode(env, op, scen, 1000 * args.seed + i, args.rescue_steps, rng,
                                       args.max_steps) for i in range(args.episodes)]
                    ok = [e for e in eps if e["success"]]
                    tw = [e["twist_deg"] for e in eps if e["twist_deg"] is not None]
                    row = dict(shape=shape, tier=tier, rotation=rot, scenario=scen, episodes=len(eps),
                               success=len(ok) / len(eps),
                               median_steps=float(np.median([e["op_steps"] for e in ok])) if ok else None,
                               twist_deg_mean=float(np.mean(tw)) if tw else None,
                               twist_deg_max=float(np.max(tw)) if tw else None)
                    rows.append(row)
                    print("%-8s %-12s rot %-3s %-6s  success %5.1f %%  median operator steps %-5s  "
                          "twist at takeover %5.1f deg (max %5.1f)   [%.0f s]"
                          % (shape, tier, rot, scen, 100 * row["success"],
                             "-" if row["median_steps"] is None else "%.0f" % row["median_steps"],
                             row["twist_deg_mean"] or 0.0, row["twist_deg_max"] or 0.0, time.time() - t0),
                          flush=True)
        env.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), rows=rows), f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
