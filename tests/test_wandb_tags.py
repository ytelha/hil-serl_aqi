"""Tests for the W&B run tags: a description string must become ONE tag, not
one tag per character (the bug: make_wandb_logger passed "tag": description
and wandb.init(tags="sim_peg_...") iterated the string).

The WandBLogger part runs anywhere wandb is installed (absl / ml_collections
are stubbed when missing); the make_wandb_logger part needs the training
environment (jax) and is skipped otherwise. Run from the repo root with
    python -m pytest tests/test_wandb_tags.py -q
    python tests/test_wandb_tags.py
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serl_launcher"))

DESC = "sim_peg_insertion_logged_op0"


def _stub_missing(name, **attrs):
    if importlib.util.find_spec(name.split(".")[0]) is None and name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod
    return None


# common/wandb.py imports absl.flags and ml_collections at module level
_absl_flags = types.SimpleNamespace(FLAGS=types.SimpleNamespace(is_parsed=lambda: False))
if _stub_missing("absl") is not None:
    sys.modules["absl"].flags = _absl_flags
    sys.modules["absl.flags"] = _absl_flags
_ml = _stub_missing("ml_collections")
if _ml is not None:
    _ml.ConfigDict = dict
    _ml.config_dict = types.SimpleNamespace(FieldReference=lambda *a, **k: None)

import serl_launcher.common.wandb as cw  # noqa: E402


class _FakeRun:
    def __init__(self):
        self.config = types.SimpleNamespace(update=lambda *a, **k: None)


def _capture_init(fn, env_tags=None):
    """Run fn with wandb.init replaced by a recorder (and AQI_WANDB_TAGS set to
    env_tags, or unset); return the kwargs."""
    seen = {}
    real = cw.wandb.init
    old_env = os.environ.pop("AQI_WANDB_TAGS", None)
    if env_tags is not None:
        os.environ["AQI_WANDB_TAGS"] = env_tags

    def fake_init(**kwargs):
        seen.update(kwargs)
        return _FakeRun()

    cw.wandb.init = fake_init
    try:
        fn()
    finally:
        cw.wandb.init = real
        os.environ.pop("AQI_WANDB_TAGS", None)
        if old_env is not None:
            os.environ["AQI_WANDB_TAGS"] = old_env
    return seen


class _Skip(Exception):
    pass


def _skip(msg):
    try:
        import pytest
    except ImportError:
        raise _Skip(msg)
    pytest.skip(msg)


def test_as_tags():
    assert cw._as_tags(DESC) == [DESC]
    assert cw._as_tags("") is None
    assert cw._as_tags(None) is None
    assert cw._as_tags((DESC, "D2")) == [DESC, "D2"]
    assert cw._as_tags(["a", 1]) == ["a", "1"]


def _logger_cfg(tag):
    return types.SimpleNamespace(unique_identifier="", exp_descriptor=DESC, project="hil-serl",
                                 entity=None, group=None, tag=tag)


def test_logger_string_tag_is_one_tag():
    # old call style (a bare string) must no longer explode into characters
    seen = _capture_init(lambda: cw.WandBLogger(_logger_cfg(DESC), {}, tempfile.mkdtemp(), debug=True))
    assert seen["tags"] == [DESC]


def test_logger_list_tag_unchanged():
    seen = _capture_init(lambda: cw.WandBLogger(_logger_cfg([DESC, "extra"]), {}, tempfile.mkdtemp(), debug=True))
    assert seen["tags"] == [DESC, "extra"]


def test_logger_without_tag():
    cfg = _logger_cfg(None)
    del cfg.tag
    seen = _capture_init(lambda: cw.WandBLogger(cfg, {}, tempfile.mkdtemp(), debug=True))
    assert seen["tags"] is None


def test_env_tags_are_added_after_the_description():
    # run_campaign.py sets AQI_WANDB_TAGS per cell; duplicates and blanks drop out
    seen = _capture_init(lambda: cw.WandBLogger(_logger_cfg(DESC), {}, tempfile.mkdtemp(), debug=True),
                         env_tags="active_effect, active ,novice,nov01,seed1,,active")
    assert seen["tags"] == [DESC, "active_effect", "active", "novice", "nov01", "seed1"]


def test_env_tags_alone_and_cut_to_64_chars():
    cfg = _logger_cfg(None)
    del cfg.tag
    seen = _capture_init(lambda: cw.WandBLogger(cfg, {}, tempfile.mkdtemp(), debug=True),
                         env_tags="x" * 80 + ",D3")
    assert seen["tags"] == ["x" * 64, "D3"]


def test_make_wandb_logger_passes_one_tag():
    if importlib.util.find_spec("jax") is None or _ml is not None:
        _skip("needs the training environment (jax, ml_collections)")
    from serl_launcher.utils.launcher import make_wandb_logger
    seen = _capture_init(lambda: make_wandb_logger(description=DESC, debug=True, output_dir=tempfile.mkdtemp()))
    assert seen["tags"] == [DESC]
    assert seen["name"].startswith(DESC + "_")   # the run name is unchanged


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for t in tests:
        try:
            t()
        except _Skip as e:
            skipped += 1
            print(f"SKIP {t.__name__}: {e}")
            continue
        passed += 1
        print(f"PASS {t.__name__}")
    print(f"{passed} passed, {skipped} skipped")
