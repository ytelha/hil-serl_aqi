"""Intervention and trigger logging (proposal S7 "Logging extension").

Everything the offline analysis (RQ1 operator variance, RQ2 what predicts a
useful intervention, trigger calibration) needs is written as JSON-lines so
runs can be concatenated, grepped, and loaded with pandas.

Layout, under <log_dir>/<run_name>/:
    steps.jsonl      one record per control step (trigger timeline; only if
                     cfg.log_every_step)
    segments.jsonl   one record per intervention segment: operator, timing,
                     a_hat / q / w statistics, Option-A features, exclusion
    episodes.jsonl   one record per episode: outcome, return, forces,
                     intervention counts, request counts
    meta.json        run metadata (mode, operator, config, start time)
    gate_state.json  final z-statistics / P_fail weights (for resuming or
                     inspecting calibration)

Numpy types are converted so every record is plain JSON.
"""

import json
import os
import time
from typing import Optional

import numpy as np


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class AQILogger:
    def __init__(self, log_dir: str, run_name: str, operator_id: str,
                 mode: str, config_dict: Optional[dict] = None,
                 log_every_step: bool = True):
        self.dir = os.path.join(log_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.log_every_step = log_every_step
        self._steps_f = open(os.path.join(self.dir, "steps.jsonl"), "a",
                             buffering=1)
        self._segs_f = open(os.path.join(self.dir, "segments.jsonl"), "a",
                            buffering=1)
        self._eps_f = open(os.path.join(self.dir, "episodes.jsonl"), "a",
                           buffering=1)
        meta = dict(
            run_name=run_name,
            operator_id=operator_id,
            mode=mode,
            start_time=time.time(),
            config=_to_jsonable(config_dict or {}),
        )
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    def log_step(self, record: dict) -> None:
        if not self.log_every_step:
            return
        self._steps_f.write(json.dumps(_to_jsonable(record)) + "\n")

    def log_segment(self, record: dict) -> None:
        self._segs_f.write(json.dumps(_to_jsonable(record)) + "\n")

    def log_episode(self, record: dict) -> None:
        self._eps_f.write(json.dumps(_to_jsonable(record)) + "\n")

    def save_state(self, gate_state: dict, fail_state: dict,
                   extra: Optional[dict] = None) -> None:
        state = dict(gate=gate_state, failure_model=fail_state,
                     extra=_to_jsonable(extra or {}))
        with open(os.path.join(self.dir, "gate_state.json"), "w") as f:
            json.dump(_to_jsonable(state), f, indent=2)

    def close(self) -> None:
        for f in (self._steps_f, self._segs_f, self._eps_f):
            try:
                f.close()
            except Exception:
                pass
