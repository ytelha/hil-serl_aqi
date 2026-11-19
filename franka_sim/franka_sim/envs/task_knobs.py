"""Task difficulty knobs of the peg-insertion sim, read per process from the environment (pure Python, no MuJoCo).

examples/run_campaign.py sets the variables for every process of a campaign (demonstrations, learner, actor, ruler),
always, empty when a knob is off, so a value exported in a shell cannot leak into a campaign. Unset or empty = the
env's own default, so every earlier campaign is unchanged (docs/extension_study.md):

    $SIM_SUCCESS_LATERAL  lateral success tolerance (m)   -> success_lateral   (env default 0.012)
    $SIM_RANDOM_XY        start-position range +/- (m)    -> random_xy_range   (env default 0.015)
    $SIM_RANDOM_RZ        start-yaw range +/- (rad)       -> random_rz_range   (env default 0.08)
"""
import os

TASK_ENV = (("success_lateral", "SIM_SUCCESS_LATERAL"),
            ("random_xy_range", "SIM_RANDOM_XY"),
            ("random_rz_range", "SIM_RANDOM_RZ"))


def task_overrides_from_env(environ=None):
    """{constructor keyword: value} for every knob set to a non-empty value. Raises ValueError on a value that is
    not a positive number, so a typo cannot silently fall back to the default."""
    env = os.environ if environ is None else environ
    out = {}
    for key, var in TASK_ENV:
        raw = str(env.get(var, "") or "").strip()
        if not raw:
            continue
        try:
            val = float(raw)
        except ValueError:
            raise ValueError(f"${var}={raw!r}: not a number")
        if not val > 0.0:
            raise ValueError(f"${var}={raw!r}: must be a positive number")
        out[key] = val
    return out
