"""HiRIL hyperparameters (no JAX import; used by the agent, the pipeline,
the eval ruler and the campaign runner).

Source of each default:
    [A1] paper Table A1   [A2] paper Table A2   [eq] paper equation
    [TD3] standard TD3 value (paper: "we employ TD3 as the baseline RL
          model" but does not list it)
    [ours] NOT given in the paper: chosen for this repo's sim task
           (sparse 0/1 reward, 10 Hz, ~300-step episodes). Ablate these.
Infrastructure-level settings (batch size, discount, encoder, image
augmentation, replay capacity) are NOT here: train_hiril.py takes them from
the experiment config, exactly like every other method in the campaign, so
the comparison differs only in the learning method.
"""

from dataclasses import asdict, dataclass, field
from typing import Tuple

DU_MODES = ("dual", "aleatoric", "epistemic")


@dataclass
class HiRILConfig:
    # ---- BIQN critic (Sec. 4) -------------------------------------------
    k_tau: int = 8                 # [A1] K_tau quantile samples per update
    k_tau_target: int = 8          # [A1] N' target quantile samples
    k_theta: int = 8               # [A1] K_theta Bayesian weight samples
    kappa: float = 1.0             # [A1] Huber-loss threshold
    critic_hidden_dims: Tuple[int, ...] = (256, 256)   # [ours] = SAC critic
    n_cos: int = 64                # [ours] IQN cosine embedding size
    rho_init: float = -5.0         # [ours] softplus(rho) = initial weight std
    kl_weight: float = 0.0         # [eq] 0: the paper drops the KL term
                                   # (Eq. 10-11). >0 adds KL(q || N(0, prior))
    prior_std: float = 1.0         # [ours] prior std, used only if kl_weight>0

    # ---- dual-mode uncertainty DU(s, a) (Eq. 16-20) ----------------------
    du_k_tau: int = 8              # [ours] M' quantile samples for DU
    du_k_theta: int = 8            # [ours] M weight samples for DU
    du_mode: str = "dual"          # "dual" = Var over (theta, tau) (paper);
                                   # "aleatoric" = w/o epistemic ablation;
                                   # "epistemic" = w/o aleatoric ablation
    sigma_z: float = 0.0           # [eq] output-noise variance added to DU
                                   # (constant: only shifts tau_risk)

    # ---- risk-triggered intervention (Eq. 22-23) ------------------------
    tau_risk: float = 0.0          # >0: absolute DU threshold (paper).
                                   # <=0 [ours]: calibrate ONCE -- tau_risk =
                                   # calib_quantile of DU(s, a_RL) over
                                   # calib_steps actor steps, then frozen.
    calib_quantile: float = 0.8    # [ours] ~20% of steps flagged at calib
    calib_steps: int = 1000        # [ours]
    calib_delay_steps: int = 500   # [ours] after random_steps, before calib
    recalib_period: int = 0        # [ours, F2 fix] >0: RE-calibrate tau_risk
                                   # every recalib_period actor steps over the
                                   # last recalib_period DU(s,a_RL) values, so
                                   # the request rate stays ~1-calib_quantile as
                                   # DU decays with learning (relative trigger,
                                   # like AQI's z-gate). 0 = calibrate once.

    # ---- reward shaping (Eq. 25) ----------------------------------------
    intervention_penalty: float = -0.1   # [ours] r_p, added once per takeover
                                         # onset (paper gives no value)

    # ---- TDRD prioritized replay (Eq. 26-28, Table A2) ------------------
    per_alpha: float = 1.0         # [A2] priority factor
    per_beta: float = 1.0          # [A2] sample factor (IS exponent)
    per_eps: float = 1e-3          # [A2] offset epsilon
    use_rd: bool = True            # [eq] False = the paper's "w/o RD"
                                   # ablation (Fig. 8): no risk-difference
                                   # bonus, plain TD-error PER
    rd_clip: float = 10.0          # [ours] clip of DU_RL - DU_H inside exp()
    demo_bonus: float = 1.0        # [ours] RD term for offline demos, whose
                                   # a_RL was never observed: exp(0) = 1

    # ---- actor / optimisation (TD3, Eq. 21, 32) --------------------------
    lambda_bc: float = 1.0         # [ours] lambda in Eq. 32 (not given)
    actor_hidden_dims: Tuple[int, ...] = (256, 256)    # [ours] = SAC actor
    critic_lr: float = 5e-4        # [A1] value network learning rate
    actor_lr: float = 2e-4         # [A1] policy network learning rate
    policy_delay: int = 2          # [A1] policy update delay frequency
    target_update_rate: float = 0.005   # [TD3] Polyak rate
    policy_noise: float = 0.2      # [TD3] target policy smoothing std
    noise_clip: float = 0.5        # [TD3]
    explore_init: float = 1.0      # [A1] initial exploration (noise std)
    explore_final: float = 0.05    # [A1] final exploration
    explore_decay_steps: int = 20000    # [ours] linear decay horizon in env
                                        # steps (paper: over 400 episodes)

    def __post_init__(self):
        self.critic_hidden_dims = tuple(int(x) for x in self.critic_hidden_dims)
        self.actor_hidden_dims = tuple(int(x) for x in self.actor_hidden_dims)
        assert self.du_mode in DU_MODES, f"du_mode must be one of {DU_MODES}"
        for name in ("k_tau", "k_tau_target", "k_theta", "du_k_tau",
                     "du_k_theta", "policy_delay", "n_cos"):
            assert int(getattr(self, name)) >= 1, f"{name} must be >= 1"
        assert self.du_k_tau >= 2 or self.du_k_theta >= 2, (
            "DU needs >= 2 samples along theta or tau")
        assert 0.0 < self.calib_quantile < 1.0
        assert self.kappa > 0 and self.per_eps > 0
        assert int(self.recalib_period) >= 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["critic_hidden_dims"] = list(self.critic_hidden_dims)
        d["actor_hidden_dims"] = list(self.actor_hidden_dims)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HiRILConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# Every field that can be set from the command line / a campaign YAML.
# (name -> caster). train_hiril.py defines one --hiril_<name> flag per entry;
# bool fields are absl BOOLEAN flags (--hiril_x / --nohiril_x): never pass
# "--hiril_x False", absl would read it as True plus a stray argument.
# Ablations of the paper (Fig. 6 / 8) are all expressible here:
#   w/o epistemic: du_mode=aleatoric      w/o aleatoric: du_mode=epistemic
#   w/o RD: use_rd=False   w/o reward shaping: intervention_penalty=0
#   w/o imitation: lambda_bc=0
CLI_FIELDS = {
    "tau_risk": float, "calib_quantile": float, "calib_steps": int,
    "calib_delay_steps": int, "recalib_period": int, "du_mode": str,
    "sigma_z": float,    "k_tau": int, "k_tau_target": int, "k_theta": int, "du_k_tau": int,
    "du_k_theta": int, "kappa": float, "kl_weight": float,
    "intervention_penalty": float, "lambda_bc": float,
    "per_alpha": float, "per_beta": float, "per_eps": float, "use_rd": bool,
    "critic_lr": float, "actor_lr": float, "policy_delay": int,
    "explore_init": float, "explore_final": float,
    "explore_decay_steps": int,
}
