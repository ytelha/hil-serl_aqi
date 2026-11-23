"""Run instrumentation of the fixed-budget protocol (extension study, phase 3; docs/extension_phase3.md).

Pure Python and numpy, so it is tested without JAX (tests/test_run_guard.py). Used by examples/train_aqi_rlpd.py.
Everything here is off unless its flag is set, so every earlier campaign runs exactly as before.

Learner side:
  EnvCheckpoints   (--checkpoint_env_every E) checkpoints on environment steps: the k-th is saved once the learner
                   has done U * k * E online updates under the fixed update ratio U, i.e. after it has received at
                   least k * E environment steps. The warm start does not shift them, and all of them are kept.
                   At the stop the learner first catches up with the steps it has received, then saves the last one.
                   Every saved one is a line of env_checkpoints.jsonl.
  DivergenceGuard  (--divergence_guard_every N) every N learner steps: are the core learner metrics (losses,
                   temperature, mean Q values, entropy; CORE_METRICS) finite? On the first non-finite one it writes
                   DIVERGED.json (the step, the phase, the non-finite metrics, every metric of that step, the last
                   finite ones) and the learner exits without a checkpoint.
                   The actor sees the marker at its next episode end and stops; the campaign runner marks the run
                   as crashed, so it is retrained once.
  MetricsLog       (--metrics_log_every M) every M learner steps one line of scalar learner metrics in
                   learner_metrics.jsonl, whatever W&B does (E2HiL's keep fractions per source included).

Actor side:
  CarelessCounter  per episode, with the inconsistent operator: the careless takeovers and their steps, added to
                   episodes.jsonl in every arm (also plain HIL-SERL and E2HiL, which write no takeover log).
  e2hil_label      the source label of a transition for E2HiL's diagnostics: 1 robot step, 2 takeover step, 3 step of
                   a careless takeover. Labels never enter a loss.
  WOULD_STOP.json  (--converge_mode observe) the mini-evaluation keeps running and does not stop the run; its first
                   would-be stop is written there.
"""

import datetime
import json
import math
import os
import time
from collections.abc import Mapping

import numpy as np

DIVERGED_FILE = "DIVERGED.json"
WOULD_STOP_FILE = "WOULD_STOP.json"
ENV_CKPT_FILE = "env_checkpoints.jsonl"
METRICS_FILE = "learner_metrics.jsonl"
CARELESS = "inattentive"          # the inconsistent operator's careless condition (operator_model.py)


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def flatten_metrics(info, prefix=""):
    """Nested mapping of scalars / arrays -> {"a/b": float}. A leaf with more than one element becomes its mean, so a
    NaN or inf anywhere in it makes the value non-finite. Leaves that are not numbers are skipped."""
    out = {}
    if isinstance(info, Mapping):
        for k, v in info.items():
            out.update(flatten_metrics(v, f"{prefix}{k}/"))
        return out
    try:
        a = np.asarray(info, dtype=np.float64)
    except (TypeError, ValueError):
        return out
    if a.size == 0:
        return out
    key = prefix[:-1] if prefix.endswith("/") else prefix
    out[key or "value"] = float(a.reshape(-1)[0]) if a.size == 1 else float(np.mean(a))
    return out


# the learner metrics whose non-finite value means the run diverged (the networks themselves are broken); other
# diagnostics are recorded but never trip the guard, so a diagnostic cannot end a healthy run
CORE_METRICS = ("critic_loss", "actor_loss", "temperature", "temperature_loss", "predicted_qs", "target_qs",
                "entropy", "aqi_bc_loss")


def nonfinite_keys(flat):
    return sorted(k for k, v in flat.items() if not math.isfinite(v))


def core_nonfinite(flat):
    """The non-finite CORE_METRICS of a flattened metrics dict (matched by their own name, whatever the group)."""
    return [k for k in nonfinite_keys(flat) if k.split("/")[-1] in CORE_METRICS]


def json_safe(flat):
    """{k: float}, with 'nan' / 'inf' / '-inf' as strings so the JSON stays strict."""
    if flat is None:
        return None
    return {k: (v if math.isfinite(v) else str(v)) for k, v in flat.items()}


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def diverged(ckpt_dir):
    """Has this run's learner written DIVERGED.json?"""
    return bool(ckpt_dir) and os.path.exists(os.path.join(os.path.abspath(ckpt_dir), DIVERGED_FILE))


# ---------------------------------------------------------------------------- learner
class DivergenceGuard:
    def __init__(self, ckpt_dir, every):
        self.every = max(int(every or 0), 0)
        self.path = os.path.join(os.path.abspath(ckpt_dir), DIVERGED_FILE) if ckpt_dir else None
        self.last_finite = None
        self.last_finite_step = None
        self.tripped = False

    @property
    def on(self):
        return self.every > 0 and self.path is not None

    def reset(self):
        """Drop a marker an earlier run left (a resumed run starts unmarked)."""
        if self.on:
            for p in (self.path, self.path + ".tmp"):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def due(self, step):
        return self.on and int(step) % self.every == 0

    def check(self, step, info, phase, extra=None):
        """True if a core metric (CORE_METRICS) is non-finite (and DIVERGED.json is written), else False."""
        flat = flatten_metrics(info)
        bad = core_nonfinite(flat)
        if not bad:
            self.last_finite, self.last_finite_step = flat, int(step)
            return False
        payload = {"step": int(step), "phase": phase, "nonfinite": bad,
                   "nonfinite_all": nonfinite_keys(flat), "metrics": json_safe(flat),
                   "last_finite_step": self.last_finite_step, "last_finite_metrics": json_safe(self.last_finite),
                   "checked_every": self.every, "when": _now()}
        payload.update(extra or {})
        write_json(self.path, payload)
        self.tripped = True
        return True


class MetricsLog:
    def __init__(self, ckpt_dir, every):
        self.every = max(int(every or 0), 0)
        self.path = os.path.join(os.path.abspath(ckpt_dir), METRICS_FILE) if ckpt_dir else None

    @property
    def on(self):
        return self.every > 0 and self.path is not None

    def due(self, step):
        return self.on and int(step) % self.every == 0

    def write(self, step, info, phase, extra=None):
        row = {"step": int(step), "phase": phase, "t": round(time.time(), 2)}
        row.update(extra or {})
        row["metrics"] = json_safe(flatten_metrics(info))
        append_jsonl(self.path, row)
        return row


class EnvCheckpoints:
    """Checkpoints on environment steps under a fixed update ratio U (see the module docstring)."""

    def __init__(self, ckpt_dir, every, ratio):
        self.every = max(int(every or 0), 0)
        self.ratio = float(ratio or 0.0)
        self.path = os.path.join(os.path.abspath(ckpt_dir), ENV_CKPT_FILE) if ckpt_dir else None
        self.k = 1
        if self.every > 0 and self.ratio <= 0.0:
            raise ValueError("--checkpoint_env_every needs a fixed update ratio (--utd_ratio > 0)")

    @property
    def on(self):
        return self.every > 0 and self.path is not None

    def reset(self):
        if self.on:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def next_updates(self):
        return int(round(self.ratio * self.k * self.every))

    def due(self, n_online):
        return self.on and int(n_online) >= self.next_updates()

    def record(self, learner_step, n_online, received=None, final=False):
        env = (int(round(int(n_online) / self.ratio)) if final else self.k * self.every)
        row = {"learner_step": int(learner_step), "online_updates": int(n_online), "env_steps": env,
               "env_received": (int(received) if received is not None and math.isfinite(received) else None),
               "final": bool(final), "t": round(time.time(), 2)}
        if self.on:
            append_jsonl(self.path, row)
        if not final:
            self.k += 1
        return row

    def catch_up(self, n_online, received, extra_done, limit=2000):
        """At the stop: go on training while fewer than U * received online updates are done (at most `limit`)."""
        return (self.on and received is not None and math.isfinite(received)
                and int(n_online) < self.ratio * received and int(extra_done) < int(limit))


# ---------------------------------------------------------------------------- actor
class CarelessCounter:
    """Careless takeovers and their steps in the current episode (inconsistent operator only)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.takeovers = 0
        self.steps = 0

    def step(self, intervening, onset, condition):
        if intervening and condition == CARELESS:
            self.steps += 1
            if onset:
                self.takeovers += 1

    def record(self):
        return {"careless_takeovers": int(self.takeovers), "careless_steps": int(self.steps)}


def e2hil_label(intervening, condition=None):
    """E2HiL's source label of an online transition: 1 robot step, 2 takeover step, 3 careless takeover step."""
    if not intervening:
        return 1
    return 3 if condition == CARELESS else 2


def would_stop_payload(step, episodes, sr, threshold, consecutive, every, history, poses):
    return {"step": int(step), "episodes": int(episodes), "mode": "observe", "mini_eval_sr": float(sr),
            "mini_eval_threshold": float(threshold), "consecutive": int(consecutive), "every": int(every),
            "mini_eval_history": history, "mini_eval_poses": poses, "when": _now()}
