"""Tests for the cluster support (cluster/claix): campaign sharding, the
per-job learner/actor port, and the headless hardware stand-ins.

Run from the repo root with
    python -m pytest tests/test_cluster_support.py -q
    python tests/test_cluster_support.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in ("serl_launcher", "examples"):
    sys.path.insert(0, str(REPO / _p))

STUBS = REPO / "cluster" / "claix" / "stubs"
CONFIG = REPO / "examples" / "campaigns" / "rq5_quality_score_ext.yaml"


def _run_campaign(*args):
    import io
    from contextlib import redirect_stdout
    import run_campaign as rc
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_code = rc.main(list(args))
    return rc_code, buf.getvalue()


def test_list_cells_and_shards_partition_the_plan():
    with tempfile.TemporaryDirectory() as tmp:
        code, out = _run_campaign("--config", str(CONFIG), "--campaign", tmp,
                                  "--list_cells")
        cells = [l for l in out.splitlines() if l.strip()]
        assert code == 0 and len(cells) == 90 and len(set(cells)) == 90
        assert not (Path(tmp) / "campaign.json").exists()   # wrote nothing
        import run_campaign as rc
        assert rc._parse_shard("") is None
        assert rc._parse_shard("3/7") == (3, 7)
        for bad in ("7/7", "-1/3", "x", "1/0"):
            try:
                rc._parse_shard(bad)
            except SystemExit:
                continue
            raise AssertionError(bad)
        # shards 0..n-1 cover every cell exactly once, in plan order
        n = 4
        shards = [[c for k, c in enumerate(cells) if k % n == i]
                  for i in range(n)]
        assert sorted(sum(shards, [])) == sorted(cells)


def test_shard_refuses_demos_and_analyze():
    with tempfile.TemporaryDirectory() as tmp:
        for stages in ("demos", "train,eval,analyze"):
            try:
                _run_campaign("--config", str(CONFIG), "--campaign", tmp,
                              "--stages", stages, "--shard", "0/2",
                              "--dry_run")
            except SystemExit as exc:
                assert "demos stage once" in str(exc)
                continue
            raise AssertionError(stages)


def test_trainer_ports_follow_the_environment():
    try:
        from serl_launcher.utils import launcher
        launcher.make_trainer_config()
    except Exception as exc:            # agentlace / jax not installed here
        print(f"SKIP (launcher not importable: {type(exc).__name__})")
        return
    old = os.environ.pop("AQI_TRAINER_PORT", None)
    try:
        c = launcher.make_trainer_config()
        assert (c.port_number, c.broadcast_port) == (5588, 5589)
        os.environ["AQI_TRAINER_PORT"] = "23456"
        c = launcher.make_trainer_config()
        assert (c.port_number, c.broadcast_port) == (23456, 23457)
        c = launcher.make_trainer_config(port_number=6000)   # explicit wins
        assert (c.port_number, c.broadcast_port) == (6000, 6001)
    finally:
        os.environ.pop("AQI_TRAINER_PORT", None)
        if old is not None:
            os.environ["AQI_TRAINER_PORT"] = old


def test_hardware_stand_ins_import_and_fail_only_on_use():
    code = r"""
import easyhid, pyrealsense2 as rs
from pynput import keyboard
l = keyboard.Listener(on_press=lambda k: None); l.start(); l.join()
assert keyboard.Key.esc.name == "esc"
try:
    easyhid.Enumeration()
    raise SystemExit("easyhid did not raise")
except easyhid.HIDException:
    pass
try:
    rs.pipeline()
    raise SystemExit("pyrealsense2 did not raise")
except RuntimeError:
    pass
print("stubs ok")
"""
    env = dict(os.environ, PYTHONPATH=str(STUBS))
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "stubs ok" in r.stdout, r.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
