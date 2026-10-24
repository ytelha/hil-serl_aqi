"""Fixed learner updates per environment step (--utd_ratio).

By default the actor runs at 10 Hz and the learner as fast as its GPU allows,
so the updates per env step depend on the hardware (finding F3: ~0.45 on the
training machine, ~2.6 on an H100). With a ratio U > 0 the two processes of a
cell hold it fixed, through a small progress file in the checkpoint folder
(both processes of a cell run on the same machine):

  learner  removes the file when it starts, writes its online update count
           after every learner step once training began, and waits while it
           already has U updates per env step it has received;
  actor    waits before an env step until the learner has done U updates per
           step of every episode except the one it is collecting, so one
           episode is collected while the previous one is trained on.

A learner step is one learner-loop iteration (cta_ratio - 1 critic updates
plus one critic + actor update), the unit of the W&B step and of finding F3.
Counts are per process (since this process started), so a resumed cell keeps
both sides consistent. While the file is missing the learner is still filling
its buffer (or not up yet), so the actor does not wait: that phase needs its
data. If the learner stops advancing for `timeout_s` while the actor waits,
the actor stops waiting (warning) instead of hanging.
"""

import json
import os
import time

PROGRESS_FILE = "learner_progress.json"


class UpdateRatio:
    def __init__(self, checkpoint_dir, ratio, timeout_s=600.0, poll_s=0.02,
                 clock=time.monotonic, sleep=time.sleep, warn=print):
        self.ratio = float(ratio)
        self.path = os.path.join(os.path.abspath(checkpoint_dir), PROGRESS_FILE)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self._clock, self._sleep, self._warn = clock, sleep, warn
        self.sent = []               # actor: env steps shipped, cumulative
        self.disabled = False        # actor: gave up waiting (learner silent)
        self.waited_s = 0.0          # actor: total time spent waiting
        self._met = 0.0              # actor: largest target already reached

    @property
    def on(self):
        return self.ratio > 0.0 and not self.disabled

    # ---- learner side -------------------------------------------------------
    def learner_reset(self):
        """Drop a file left by an earlier run of this cell."""
        for p in (self.path, self.path + ".tmp"):
            try:
                os.remove(p)
            except OSError:
                pass

    def learner_report(self, online_updates):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"online_updates": int(online_updates), "t": time.time()}, f)
        os.replace(tmp, self.path)

    def learner_wait(self, online_updates, received_fn, should_stop):
        """Block while the learner already did U updates per received step."""
        while (self.ratio > 0.0 and not should_stop()
               and online_updates >= self.ratio * received_fn()):
            self._sleep(self.poll_s)

    # ---- actor side ---------------------------------------------------------
    def actor_note_sent(self, steps_sent):
        self.sent.append(int(steps_sent))

    def target(self):
        return self.ratio * self.sent[-2] if len(self.sent) >= 2 else 0.0

    def read_updates(self):
        try:
            with open(self.path) as f:
                return int(json.load(f)["online_updates"])
        except (OSError, ValueError, KeyError):
            return None

    def actor_wait(self):
        """Block until the learner has caught up with all but the newest episode."""
        if not self.on:
            return
        target = self.target()
        if target <= self._met:      # reached before; no file read per step
            return
        last = self.read_updates()
        if last is None:             # learner still filling its buffer
            return
        t0 = last_change = self._clock()
        while last < target:
            self._sleep(self.poll_s)
            now = self._clock()
            cur = self.read_updates()
            if cur is None:          # learner restarted: it is filling again
                break
            if cur != last:
                last, last_change = cur, now
            elif now - last_change >= self.timeout_s:
                self.disabled = True
                self._warn(f"[utd] learner silent for {self.timeout_s:.0f} s at "
                           f"{last} of {target:.0f} updates; the actor stops "
                           f"waiting (this cell's update ratio is no longer fixed)")
                break
        else:
            self._met = target
        self.waited_s += self._clock() - t0
