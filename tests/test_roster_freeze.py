"""Tests for the operator roster: the jitter must not depend on Python's
per-process string hashing, a campaign's roster is frozen on first write,
and examples/roster_audit.py finds cells trained against another operator.

Run from the repo root with
    python -m pytest tests/test_roster_freeze.py -q
    python tests/test_roster_freeze.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

CONFIG = REPO / "examples" / "campaigns" / "rq5_quality_score_ext.yaml"
OPERATOR_MODEL = (REPO / "serl_launcher" / "serl_launcher" / "aqi"
                  / "operator_model.py")


def _run_campaign(*args):
    import io
    from contextlib import redirect_stdout
    import run_campaign as rc
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = rc.main(list(args))
    return code, buf.getvalue()


def test_jitter_is_independent_of_the_hash_seed():
    # operator_model is loaded by path: it needs numpy only
    code = f"""
import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("om", r"{OPERATOR_MODEL}")
om = importlib.util.module_from_spec(spec); sys.modules["om"] = om
spec.loader.exec_module(om)
rng = np.random.default_rng(0)
for tier in ("expert", "intermediate", "novice"):
    th = om.jitter_theta(om.make_operator_theta(tier), rng, strength=0.15)
    print(repr(sorted((k, getattr(th, k)) for k in om._SCALE_JITTERED)))
"""
    outs = set()
    for seed in ("1", "2", "3", "4"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1, "the jitter changed with PYTHONHASHSEED"


def test_roster_is_frozen_on_first_write():
    with tempfile.TemporaryDirectory() as tmp:
        base = ("--config", str(CONFIG), "--campaign", tmp, "--stages",
                "sample")
        code, _ = _run_campaign(*base)
        ops = sorted((Path(tmp) / "operators").glob("*.json"))
        assert code == 0 and len(ops) > 1
        p = ops[0]
        first = json.loads(p.read_text())
        # an older draw on disk (a campaign sampled before the fix) must
        # survive a later run instead of being re-sampled and overwritten
        old = dict(first, sigma0=first["sigma0"] * 1.3)
        p.write_text(json.dumps(old))
        code, out = _run_campaign(*base)
        assert code == 0
        assert json.loads(p.read_text())["sigma0"] == old["sigma0"]
        assert "[roster] keeping the frozen" in out
        csv = (Path(tmp) / "operators.csv").read_text()
        assert f"{old['sigma0']}" in csv
        # an untouched roster is reused silently
        p.write_text(json.dumps(first))
        code, out = _run_campaign(*base)
        assert code == 0 and "[roster]" not in out
        assert json.loads(p.read_text()) == first


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_audit_flags_cells_trained_against_another_operator():
    import roster_audit
    th = {"operator_id": "int00", "tier": "intermediate", "sigma0": 2.31,
          "rho_fa": 0.62, "sigma_perc": 0.76}
    other = dict(th, sigma0=2.76, rho_fa=0.57)
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp)
        _write(c / "operators" / "int00.json", other)       # rewritten later
        _write(c / "demos" / "int00_s0" / "demos_groundtruth.json",
               {"profile": "intermediate", "theta": th})
        _write(c / "checkpoints" / "a_int00_s0" / "manifest.json",
               {"operator": dict(th, theta_path="x")})
        assert roster_audit.main([str(c)]) == 0             # file only
        _write(c / "checkpoints" / "b_int00_s0" / "manifest.json",
               {"operator": dict(other, theta_path="x")})
        assert roster_audit.main([str(c)]) == 1             # a bad cell
        # --restore puts the demos' operator back and keeps the old file
        assert roster_audit.main(["--restore", str(c)]) == 1
        f = c / "operators" / "int00.json"
        assert json.loads(f.read_text()) == th
        assert json.loads(Path(str(f) + ".bak").read_text()) == other


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
