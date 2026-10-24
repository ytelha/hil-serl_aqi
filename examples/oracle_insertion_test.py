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

Tolerance sweep (--sweep tilt,yaw): every episode starts centred over the bore
with the tool tilted (random direction) or yawed by a set angle, and the
operator inserts from there. Success against the angle with rotation control
OFF is the task's orientation tolerance: how far the tool may be off and still
go in when nobody turns it. That sets the limits of the monitor's orientation
channel (--synthetic_orientation_tilt_rad / _yaw_rad). With rotation control
ON it shows whether the operator recovers. Runs without the env's orientation
box:

    python examples/oracle_insertion_test.py --sweep tilt,yaw --episodes 10 --out tolerance.json
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

from scipy.spatial.transform import Rotation  # noqa: E402

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


def run_twist_episode(env, op, axis, angle_rad, ep_seed, rng, max_steps):
    """Start centred over the bore, tilted (random direction) or yawed by
    `angle_rad`, then let the operator insert."""
    base = env.unwrapped
    base.set_eval_reset_pose(0.0, angle_rad if axis == "yaw" else 0.0)
    try:
        obs, info = env.reset(seed=ep_seed)
        if axis == "tilt" and angle_rad > 0.0:
            az = rng.uniform(0.0, 2.0 * np.pi)
            tw = Rotation.from_rotvec(angle_rad * np.array([np.cos(az), np.sin(az), 0.0]))
            base._des_quat = (tw * Rotation.from_quat(base._des_quat)).as_quat()
            base._run_controller(settle_steps=400)
        # one held step so every wrapper (RelativeFrame) sees the new pose
        obs, _r, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
    finally:
        base.set_eval_reset_pose(None, None)
    tilt, yaw = env.phase_oracle.tilt_yaw_error(obs)
    op.reset()
    steps, done = 1, bool(term or trunc)
    while not done and steps < max_steps:
        a = np.asarray(op.act(obs), dtype=np.float32)
        obs, _r, term, trunc, info = env.step(a)
        steps += 1
        done = bool(term or trunc)
    return dict(success=bool(info.get("succeed", False)), steps=steps,
                tilt_deg=float(np.degrees(tilt)), yaw_deg=float(np.degrees(yaw)))


def tolerance(rows, frac=0.8):
    """Largest swept angle (deg) up to which success stays >= frac x the
    success at 0 deg (None without a 0 deg row or without success there)."""
    rows = sorted(rows, key=lambda r: r["angle_deg"])
    if not rows or rows[0]["angle_deg"] != 0.0 or rows[0]["success"] <= 0.0:
        return None
    tol = 0.0
    for r in rows:
        if r["success"] < frac * rows[0]["success"]:
            break
        tol = r["angle_deg"]
    return tol


def sweep(args):
    angles = [float(a) for a in args.angles_deg.split(",") if a.strip()]
    rows, t0 = [], time.time()
    for shape in [s for s in args.shapes.split(",") if s]:
        # the raw task: no orientation box, whatever $SIM_RPY_BOX says
        env = make_synthetic_sim_env(peg_shape=shape, image_obs=False,
                                     max_episode_length=args.max_steps, rpy_box=None)
        for axis in [a for a in args.sweep.split(",") if a]:
            if axis == "yaw" and env.phase_oracle.symmetry is None:
                continue                                   # round bore: yaw is free
            for tier in [t for t in args.tiers.split(",") if t]:
                theta = make_operator_theta(tier, rng=np.random.default_rng(args.seed))
                for rot in [r for r in args.rotation.split(",") if r]:
                    op = SyntheticOperator(env.phase_oracle, theta,
                                           rng=np.random.default_rng(args.seed + 17),
                                           action_scale=env.operator_action_scale,
                                           rotation_control=(rot == "on"))
                    rng = np.random.default_rng(args.seed + 31)
                    for ang in angles:
                        eps = [run_twist_episode(env, op, axis, np.radians(ang),
                                                 1000 * args.seed + i, rng, args.max_steps)
                               for i in range(args.episodes)]
                        ok = [e for e in eps if e["success"]]
                        got = [e["tilt_deg" if axis == "tilt" else "yaw_deg"] for e in eps]
                        row = dict(shape=shape, axis=axis, tier=tier, rotation=rot,
                                   angle_deg=ang, episodes=len(eps),
                                   success=len(ok) / len(eps),
                                   median_steps=float(np.median([e["steps"] for e in ok])) if ok else None,
                                   achieved_deg=float(np.mean(got)))
                        rows.append(row)
                        print("%-8s %-4s %-12s rot %-3s %5.1f deg (achieved %5.1f)  success %5.1f %%  "
                              "median steps %-5s [%.0f s]"
                              % (shape, axis, tier, rot, ang, row["achieved_deg"], 100 * row["success"],
                                 "-" if row["median_steps"] is None else "%.0f" % row["median_steps"],
                                 time.time() - t0), flush=True)
        env.close()
    print("\ntolerance (largest angle with success >= 80 % of the success at 0 deg, rotation off):")
    summary = []
    for key in sorted({(r["shape"], r["axis"], r["tier"]) for r in rows}):
        sub = [r for r in rows if (r["shape"], r["axis"], r["tier"]) == key
               and r["rotation"] == "off"]
        tol = tolerance(sub)
        summary.append(dict(shape=key[0], axis=key[1], tier=key[2], tolerance_deg=tol))
        print("  %-8s %-4s %-12s %s" % (key[0], key[1], key[2],
              "-" if tol is None else "%.1f deg = %.3f rad" % (tol, np.radians(tol))))
    return rows, summary


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
    ap.add_argument("--sweep", default="",
                    help="tilt,yaw: the orientation tolerance sweep instead of solo/rescue")
    ap.add_argument("--angles_deg", default="0,2,4,6,8,10,12,15,18,22",
                    help="sweep: initial tilt / yaw angles (deg); include 0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.sweep:
        rows, summary = sweep(args)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(dict(args=vars(args), rows=rows, tolerance=summary), f, indent=2)
            print("wrote", args.out)
        return

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
