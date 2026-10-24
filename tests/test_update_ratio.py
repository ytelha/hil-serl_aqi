"""Tests for the fixed learner-updates-per-env-step lock (--utd_ratio,
serl_launcher/utils/update_ratio.py): the actor collects one episode ahead of
the learner, the learner waits for data, neither side can hang, and nothing
waits while the learner is still filling its buffer. A fake clock replaces
time, so the tests run instantly and need no JAX.

Run from the repo root with
    python -m pytest tests/test_update_ratio.py -q
    python tests/test_update_ratio.py
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serl_launcher"))

from serl_launcher.utils.update_ratio import UpdateRatio  # noqa: E402


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _pair(ratio=1.0, timeout_s=600.0, on_sleep=None):
    """A learner-side and an actor-side UpdateRatio on one folder; the actor's
    sleep advances the fake clock and lets `on_sleep` play the learner."""
    d = tempfile.mkdtemp()
    clock, warnings = _Clock(), []

    def sleep(s):
        clock.t += s
        if on_sleep is not None:
            on_sleep()
    learner = UpdateRatio(d, ratio, clock=clock, sleep=sleep)
    actor = UpdateRatio(d, ratio, timeout_s=timeout_s, poll_s=1.0,
                        clock=clock, sleep=sleep, warn=warnings.append)
    return learner, actor, clock, warnings


def test_off_never_waits():
    learner, actor, clock, _ = _pair(ratio=0.0)
    actor.actor_note_sent(100)
    actor.actor_note_sent(200)
    actor.actor_wait()
    learner.learner_wait(10 ** 6, lambda: 0, lambda: False)
    assert clock.t == 0.0 and not actor.on


def test_first_two_episodes_and_the_fill_phase_do_not_wait():
    learner, actor, clock, _ = _pair()
    learner.learner_reset()
    actor.actor_wait()                      # nothing sent yet
    actor.actor_note_sent(100)
    actor.actor_wait()                      # one episode: it is being trained on
    actor.actor_note_sent(200)
    actor.actor_wait()                      # no progress file: learner still filling
    assert clock.t == 0.0 and not actor.disabled


def test_actor_waits_for_all_but_the_newest_episode():
    state = {"n": 0}

    def learner_runs():                     # the learner does 10 updates per poll
        state["n"] += 10
        learner.learner_report(state["n"])
    learner, actor, clock, warnings = _pair(ratio=0.5, on_sleep=learner_runs)
    learner.learner_report(0)
    actor.actor_note_sent(100)
    actor.actor_note_sent(220)
    actor.actor_wait()                      # target 0.5 * 100 = 50 updates
    assert learner.read_updates() == 50
    assert clock.t == 5.0 and actor.waited_s == 5.0 and not warnings
    actor.actor_wait()                      # reached: returns at once
    assert clock.t == 5.0


def test_reached_target_is_not_read_again():
    learner, actor, clock, _ = _pair()
    learner.learner_report(500)
    actor.actor_note_sent(100)
    actor.actor_note_sent(200)
    actor.actor_wait()
    reads = []
    actor.read_updates = lambda: reads.append(1) or 500
    for _ in range(20):                     # every env step of the episode
        actor.actor_wait()
    assert not reads and clock.t == 0.0


def test_silent_learner_disables_the_lock_instead_of_hanging():
    learner, actor, clock, warnings = _pair(timeout_s=30.0)
    learner.learner_report(3)
    actor.actor_note_sent(100)
    actor.actor_note_sent(200)
    actor.actor_wait()
    assert actor.disabled and not actor.on
    assert 30.0 <= clock.t <= 31.0 and len(warnings) == 1
    actor.actor_note_sent(300)
    actor.actor_wait()                      # disabled: no further waiting
    assert clock.t <= 31.0


def test_learner_restart_mid_wait_releases_the_actor():
    def learner_restarts():
        learner.learner_reset()
    learner, actor, clock, warnings = _pair(on_sleep=learner_restarts)
    learner.learner_report(3)
    actor.actor_note_sent(100)
    actor.actor_note_sent(200)
    actor.actor_wait()
    assert clock.t == 1.0 and not actor.disabled and not warnings


def test_learner_waits_for_data_and_for_stop():
    received = {"n": 100}

    def data_arrives():
        received["n"] += 50
    learner, _a, clock, _ = _pair(ratio=2.0, on_sleep=data_arrives)
    learner.poll_s = 1.0
    learner.learner_wait(199, lambda: received["n"], lambda: False)
    assert clock.t == 0.0                   # 199 < 2 * 100: go on
    learner.learner_wait(300, lambda: received["n"], lambda: False)
    assert received["n"] == 200 and clock.t == 2.0   # 300 < 2 * 200
    stop = {"now": False}
    learner2, _a2, clock2, _ = _pair(ratio=2.0,
                                     on_sleep=lambda: stop.update(now=True))
    learner2.poll_s = 1.0
    learner2.learner_wait(10 ** 6, lambda: 0, lambda: stop["now"])
    assert clock2.t == 1.0                  # a stop request ends the wait


def test_reset_removes_a_stale_file():
    learner, actor, _c, _ = _pair()
    learner.learner_report(12345)
    assert actor.read_updates() == 12345
    learner.learner_reset()
    assert actor.read_updates() is None
    learner.learner_reset()                 # idempotent


def test_ratio_holds_over_a_simulated_cell():
    """Interleave a fake learner and actor (the learner only runs while the
    actor waits, the slowest learner possible): after every episode it has
    done U updates per env step of all but the two newest episodes, and it
    never runs ahead of the data."""
    ratio, ep_len = 0.75, 40
    d = tempfile.mkdtemp()
    clock = _Clock()
    lrn = {"n": 0, "received": 0}
    learner = UpdateRatio(d, ratio, clock=clock, sleep=lambda s: None)

    def learner_tick(_s=None):
        clock.t += 0.01
        if lrn["n"] < ratio * lrn["received"]:
            lrn["n"] += 1
            learner.learner_report(lrn["n"])
    actor = UpdateRatio(d, ratio, poll_s=0.01, clock=clock, sleep=learner_tick)
    learner.learner_report(0)
    sent = 0
    for _ep in range(10):
        for _t in range(ep_len):
            actor.actor_wait()
            sent += 1
        lrn["received"] = sent
        actor.actor_note_sent(sent)
        if len(actor.sent) >= 3:
            assert lrn["n"] >= ratio * actor.sent[-3]
    for _t in range(ep_len):
        actor.actor_wait()
    assert lrn["n"] >= ratio * actor.sent[-2]
    assert lrn["n"] <= ratio * sent
    assert not actor.disabled


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
