"""Role 2 - synthetic interventions (methodology doc S6).

Three pieces:

  MonitoringSignal    the operator's internal risk estimate z_t in
                      [0, 1] (doc S6.1): task-error components from the
                      PhaseOracle (corridor distance, stagnation,
                      timeout fraction), a contact/force component, and
                      optional learner-derived hooks (value drop,
                      sigma_Q) - the same definitions double as
                      classifier features (skill.py).

  SyntheticInterventionGate
                      the takeover decision as signal detection (doc
                      S6.2): zhat = EWMA(z) + N(0, sigma_mon); trigger
                      when zhat > delta_in for m_in consecutive steps,
                      react after l_react more, miss whole events with
                      p_miss, add spontaneous false alarms at Poisson
                      rate rho_fa; hand back when zhat < delta_out for
                      k_out steps or after d_max seconds (hysteresis +
                      over-assistance both expressible, doc S6.3/S6.4).

  SyntheticOperatorWrapper
                      a gym wrapper at EXACTLY the SpaceMouse
                      integration point (requirement R5): when the gate
                      holds control the executed action is the
                      operator's and info["intervene_action"] carries
                      it - the actor loop, the AQI runtime, the buffers
                      and the learner cannot distinguish it from a
                      human. info["synthetic_operator"] additionally
                      carries ground truth (z, zhat, event flags, theta
                      id) for validation and the skill pipeline (R2).

Determinism (R6): all randomness flows through the RNGs handed in.
"""

from collections import Counter, deque
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

from serl_launcher.aqi.operator_model import (CONTROL_HZ, OperatorTheta,
                                              PhaseOracle,
                                              SyntheticOperator)


# ===========================================================================
class MonitoringSignal:
    """z_t in [0, 1] (doc S6.1). Components (each normalized, combined
    by max - trouble on ANY channel is trouble):
      * corridor: |lateral error| / corridor_mm, only below
        approach height (misalignment high up is routine);
            * hover corridor: optional wider no-penalty lateral radius while
                above approach; if exceeded, risk grows with distance so the
                robot cannot drift arbitrarily far with zero signal;
      * stagnation: no descent progress for `stall_s` while not yet
        inserted;
      * timeout: fraction of the episode budget consumed (weight 0.5 -
        pressure, not panic);
      * contact: normalized force magnitude when available;
      * value_fn / sigma_fn: optional learner hooks returning already-
        normalized [0, 1] scores (doc note: these make the synthetic
        supervisor's competence depend on training progress - realistic
        but confounding; off by default)."""

    def __init__(self, oracle: Optional[PhaseOracle] = None,
                 corridor_mm: float = 6.0,
                 stall_s: float = 1.2, force_norm: float = 4.0,
                 budget_steps: int = 150, dt: float = 1.0 / CONTROL_HZ,
                 value_fn=None, sigma_fn=None, force_fn=None,
                 corridor_height_mm=None, hover_corridor_mm=None):
        self.oracle = oracle
        self.corridor = corridor_mm
        self.hover_corridor = hover_corridor_mm
        self.stall_steps = max(1, int(round(stall_s / dt)))
        self.force_norm = force_norm
        self.budget = budget_steps
        self.value_fn, self.sigma_fn = value_fn, sigma_fn
        self.force_fn = force_fn
        # Height below which lateral misalignment / stagnation count as risk
        # (i.e. "near the surface, about to jam"). Defaults to 1.5x the
        # approach height for the toy env; sim sets it explicitly because its
        # approach height ~= its reset height, which would fire at reset.
        self.corridor_height = corridor_height_mm
        self.reset()

    def reset(self):
        self._best_h = np.inf
        self._stall = 0
        self._t = 0
        self.last_channel = "time"

    def __call__(self, obs, info=None) -> float:
        """z_t = max over the channels; `last_channel` names the largest one
        (logging only: time, corridor, stagnation, hover, off_bore, force,
        force_excess, value, sigma)."""
        orc = self.oracle
        self._t += 1
        parts = [("time", min(1.0, self._t / self.budget) * 0.5)]
        if orc is not None:
            h = orc.height(obs)
            err = float(np.linalg.norm(orc.lateral_error(obs)))
            if h < self._best_h - 0.3:
                self._best_h, self._stall = h, 0
            else:
                self._stall += 1
            near = (self.corridor_height if self.corridor_height is not None
                    else orc.approach_height() * 1.5)
            if 0.0 < h < near:
                parts.append(("corridor", min(1.0, err / self.corridor)))
                parts.append(("stagnation",
                              min(1.0, self._stall / self.stall_steps)))
            elif h >= near and self.hover_corridor is not None \
                    and self.hover_corridor > 0.0 and err > self.hover_corridor:
                # Above approach we allow wider roaming than near-contact,
                # but not unlimited drift away from the hole.
                parts.append(("hover", min(1.0, (err - self.hover_corridor)
                                           / self.hover_corridor)))
            elif h <= 0.0:
                # below the opening but off-axis: peg missed the bore entirely
                if err > self.corridor:
                    self.last_channel = "off_bore"
                    return 1.0  # immediate intervention — no gate delay needed
        f = None
        if info is not None and "f_z" in info:
            f = abs(float(info["f_z"])) + abs(float(info.get("f_lat", 0.0)))
        if f is not None:
            parts.append(("force", min(1.0, f / self.force_norm)))
        if self.force_fn is not None:
            parts.append(("force_excess",
                          float(np.clip(self.force_fn(obs), 0.0, 1.0))))
        if self.value_fn is not None:
            parts.append(("value", float(np.clip(self.value_fn(obs), 0.0, 1.0))))
        if self.sigma_fn is not None:
            parts.append(("sigma", float(np.clip(self.sigma_fn(obs), 0.0, 1.0))))
        name, z = max(parts, key=lambda p: p[1])
        self.last_channel = name
        return float(z)


# ===========================================================================
class SyntheticInterventionGate:
    """The SDT takeover/handback machine (doc S6.2-S6.3). Consumes the
    TRUE z_t each step, observes it through EWMA + monitor noise, and
    exposes `taking_over`; `update()` returns event strings for logs."""

    def __init__(self, theta: OperatorTheta, rng: np.random.Generator,
                 dt: float = 1.0 / CONTROL_HZ, ewma_lambda: float = 0.4):
        self.th = theta
        self.rng = rng
        self.dt = dt
        self.lam = ewma_lambda
        self.reset()

    def reset(self):
        self.zhat = 0.0
        self.taking_over = False
        self._above = 0
        self._below = 0
        self._react_left = None
        self._event_live = False     # a true event currently in progress
        self._event_missed = False
        self._hold_steps = 0
        self.last_z = 0.0

    # ------------------------------------------------------------------
    def update(self, z: float) -> str:
        """One control step BEFORE choosing the action. Returns one of
        '', 'trigger', 'trigger_fa', 'handback', 'handback_dmax'."""
        th = self.th
        self.last_z = float(z)
        self.zhat = ((1 - self.lam) * self.zhat + self.lam * z
                     + self.rng.normal(0.0, th.sigma_mon))
        event = ""

        if self.taking_over:
            self._hold_steps += 1
            if self.zhat < th.delta_out:
                self._below += 1
            else:
                self._below = 0
            if self._below >= th.k_out:
                self.taking_over = False
                event = "handback"
            elif self._hold_steps * self.dt >= th.d_max:
                self.taking_over = False
                event = "handback_dmax"
            if not self.taking_over:
                self._above = 0
                self._react_left = None
            return event

        # ---- not in control: detection ---------------------------------
        if self.zhat > th.delta_in:
            self._above += 1
        else:
            self._above = 0
            if self.zhat < th.delta_in * 0.7:
                self._event_live = False      # event over; re-arm p_miss
                self._event_missed = False
            if self._react_left is not None and self.zhat < th.delta_in:
                self._react_left = None       # detection lapsed

        if self._react_left is None and self._above >= th.m_in:
            if not self._event_live:
                self._event_live = True
                self._event_missed = self.rng.random() < th.p_miss
            if not self._event_missed:
                self._react_left = th.l_react   # detected; now reacting

        if self._react_left is not None:
            self._react_left -= 1
            if self._react_left <= 0:
                self._start_takeover()
                return "trigger"

        # ---- spontaneous false alarms (Poisson rho_fa per minute) ------
        if self.rng.random() < th.rho_fa * self.dt / 60.0:
            self._start_takeover()
            return "trigger_fa"
        return event

    def _start_takeover(self):
        self.taking_over = True
        self._hold_steps = 0
        self._below = 0
        self._react_left = None


# ===========================================================================
class SyntheticOperatorWrapper(gym.Wrapper):
    """The gamepad-equivalent wrapper (R5). Wrap the task env with it in
    the actor when --synthetic_operator is set; everything downstream
    (train_aqi_rlpd's intervention routing, AQIRuntime, the buffers) is
    untouched because the contract is byte-compatible with the
    SpaceMouse path: during a takeover the EXECUTED action is the
    operator's and info["intervene_action"] carries it.

    `operator` supplies both competences: its monitoring theta drives
    the gate, its execution theta drives the corrective actions
    (begin_takeover -> recovery macro)."""

    def __init__(self, env, operator: SyntheticOperator,
                 monitor: MonitoringSignal,
                 rng: np.random.Generator = None,
                 dt: float = 1.0 / CONTROL_HZ,
                 respond_to_request: bool = False,
                 action_conditioner=None,
                 intervention_mode: str = "signal",
                 latch_until_success: bool = False,
                 log_ground_truth: bool = False,
                 log_takeover_channels: bool = False):
        super().__init__(env)
        self.op = operator
        self.monitor = monitor
        # diagnostics only: add the oracle's simulator ground truth
        # (gt_lat_err_mm, gt_height_mm, gt_jammed) to info each step
        self.log_ground_truth = bool(log_ground_truth)
        # diagnostics only: name the monitor channel behind each takeover and
        # how it ended (info['synthetic_operator'] takeover_channel /
        # handback_kind); off = info exactly as before
        self.log_takeover_channels = bool(log_takeover_channels)
        self._chan_hist = deque(maxlen=64)
        self.gate = SyntheticInterventionGate(
            operator.th, rng or np.random.default_rng(0), dt=dt)
        # active/aqi: the operator RESPONDS to the robot's active-learning
        # request (fed each step via notify_request) instead of watching the
        # task itself -- so every tier reacts to a request, modulated by its
        # gate (reaction delay l_react, miss p_miss, false alarms rho_fa). In
        # baseline/quality/logged (no active gate) it stays False and the
        # operator acts solely on its own geometric monitor.
        self.respond_to_request = bool(respond_to_request)
        self.action_conditioner = action_conditioner
        mode = str(intervention_mode or "signal").strip().lower()
        valid = {"signal", "always_latch", "latch_until_autonomous_success"}
        if mode not in valid:
            raise ValueError(
                f"unknown intervention_mode '{intervention_mode}' "
                f"(expected one of {sorted(valid)})")
        # Backward compatibility: old boolean flag still maps to always_latch.
        if bool(latch_until_success):
            mode = "always_latch"
        self.intervention_mode = mode
        self.policy_proven = False
        self._ext_request = 0.0
        self._ext_a_score = 0.0
        self._last_obs = None
        self._last_info = {}
        self.episode_events = []
        self._episode_had_takeover = False

    def _latch_active(self) -> bool:
        if self.intervention_mode == "always_latch":
            return True
        if self.intervention_mode == "latch_until_autonomous_success":
            return not self.policy_proven
        return False

    def notify_request(self, request, a_score: float = 0.0):
        """Called by the actor each step BEFORE step() with the robot's active
        gate decision (A(s_t) > tau_A). Used as the operator's stimulus in
        active/aqi mode; ignored (stored only) otherwise."""
        self._ext_request = 1.0 if request else 0.0
        self._ext_a_score = float(a_score)

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.op.reset()
        self.monitor.reset()
        self.gate.reset()
        if self.action_conditioner is not None and \
                hasattr(self.action_conditioner, "reset"):
            self.action_conditioner.reset()
        self.episode_events = []
        self._ext_request = 0.0
        self._ext_a_score = 0.0
        self._episode_had_takeover = False
        self._chan_hist.clear()
        self._last_obs, self._last_info = obs, info
        return obs, info

    def step(self, action):
        was_over = self.gate.taking_over
        latch_active = self._latch_active()
        # Advance the task monitor every step (keeps its clock and is logged).
        # In active/aqi the gate instead responds to the robot's request; in
        # baseline/quality it responds to this task-derived risk.
        z_task = self.monitor(self._last_obs, self._last_info)
        z = self._ext_request if self.respond_to_request else z_task
        if self.log_takeover_channels:
            self._chan_hist.append(getattr(self.monitor, "last_channel", ""))
        if latch_active and was_over:
            # Keep intervening continuously once takeover starts; handback
            # happens only after a successful insertion.
            event = "hold_latch"
        else:
            event = self.gate.update(z)
        if event.startswith("trigger"):
            self.op.begin_takeover(self._last_obs)
            self.episode_events.append(
                dict(t=self.monitor._t, event=event, z=z,
                     zhat=self.gate.zhat))
        elif event.startswith("handback"):
            self.episode_events.append(
                dict(t=self.monitor._t, event=event, z=z,
                     zhat=self.gate.zhat))

        used_operator_action = self.gate.taking_over
        self._episode_had_takeover = self._episode_had_takeover or \
            used_operator_action
        if used_operator_action:
            a_exec = np.asarray(
                self.op.act(self._last_obs,
                            contact=bool(self._last_info.get("contact"))),
                dtype=np.float32)
            if self.action_conditioner is not None:
                a_exec = np.asarray(self.action_conditioner(a_exec),
                                    dtype=np.float32)
        else:
            a_exec = np.asarray(action, dtype=np.float32)

        obs, rew, terminated, truncated, info = self.env.step(a_exec)
        latch_released = False
        if self.gate.taking_over and latch_active and \
                bool(info.get("succeed", False)):
            self.gate.taking_over = False
            latch_released = True
            self.episode_events.append(
                dict(t=self.monitor._t, event="handback_latch_success",
                     z=z, zhat=self.gate.zhat))
        ended = bool(terminated or truncated)
        success = bool(info.get("succeed", False))
        if (ended and success
                and self.intervention_mode == "latch_until_autonomous_success"
                and not self.policy_proven
                and not self._episode_had_takeover):
            self.policy_proven = True
            self.episode_events.append(
                dict(t=self.monitor._t, event="policy_proven", z=z,
                     zhat=self.gate.zhat))
        if used_operator_action:
            info["intervene_action"] = a_exec          # the human path
        info["synthetic_operator"] = dict(
            operator_id=self.op.th.operator_id, z=z,
            z_task=float(z_task), request=float(self._ext_request),
            responds_to_request=bool(self.respond_to_request),
            intervention_mode=self.intervention_mode,
            latch_active=bool(latch_active),
            policy_proven=bool(self.policy_proven),
            zhat=float(self.gate.zhat),
            taking_over=bool(self.gate.taking_over),
            takeover_started=bool(event.startswith("trigger")),
            handback=bool(event.startswith("handback")),
            false_alarm=bool(event == "trigger_fa"),
        )
        if self.log_takeover_channels:
            extra = {}
            if event.startswith("trigger"):
                extra["takeover_channel"] = self._takeover_cause(event)
            if event in ("handback", "handback_dmax"):
                extra["handback_kind"] = ("signal" if event == "handback"
                                         else "time_cap")
            elif latch_released:
                extra["handback_kind"] = "latch_success"
            elif ended and self.gate.taking_over:
                extra["handback_kind"] = "episode_end"
            info["synthetic_operator"].update(extra)
        if self.log_ground_truth and self.op.phase_based:
            try:        # state AFTER this step, read from the simulator
                orc = self.op.oracle
                info["synthetic_operator"].update(
                    gt_lat_err_mm=float(np.linalg.norm(orc.lateral_error(obs))),
                    gt_height_mm=float(orc.height(obs)),
                    gt_jammed=bool(orc.jammed(obs)))
            except Exception:
                pass
        self._last_obs, self._last_info = obs, info
        return obs, rew, terminated, truncated, info

    def _takeover_cause(self, event):
        """The channel behind a takeover: a spontaneous false alarm, the
        robot's request (active modes), or the monitor channel that dominated
        z most often over the detection + reaction window before it."""
        if event == "trigger_fa":
            return "false_alarm"
        if self.respond_to_request:
            return "request"
        th = self.op.th
        win = list(self._chan_hist)[-(int(th.m_in) + int(th.l_react) + 1):]
        return Counter(win).most_common(1)[0][0] if win else ""
