"""HiRIL agent: TD3-style deterministic actor + BIQN critic (JAX / flax).

Paper: Zhou et al., "Human-in-the-loop reinforcement learning with
risk-aware intervention and imitation", ESWA 308 (2026) 131118.

BIQN (Sec. 4.1) = IQN (implicit quantile network, Dabney et al. 2018) whose
weights are Bayesian, theta ~ q(theta) = N(mu, Sigma) (mean-field Gaussian,
reparameterised theta = mu + softplus(rho) * eps). A return sample is
    Z_{theta,tau}(s, a) = f_tau(s, a; theta),  tau ~ U(0,1), theta ~ q (12)
and one forward pass here evaluates M weight samples x N quantile levels:
output shape (M, B, N).

    Q_hat(s, a) = mean over M x N samples                              (14)
    L_critic    = E_D1[rho] + E_D2[rho],  rho = quantile Huber loss
                  (1/M)(1/N) sum_m sum_ij rho_kappa(delta_ij(theta_m))  (15, 29)
    L_actor     = E_D1[-Q_hat(s, pi(s))] + lambda E_D2[||pi(s) - a_H||^2] (32)
    DU(s, a)    = Var[Z_{theta,tau}(s, a)] + sigma_z
                = E[Z^2] - E[Z]^2 over the M x M' samples           (16-20)
D1 = RL experience, D2 = human experience (batch["hiril_human"] = 0 / 1).
By the law of total variance DU splits exactly into
    aleatoric = mean_theta Var_tau Z   and   epistemic = Var_theta mean_tau Z,
which gives the paper's "w/o epistemic" / "w/o aleatoric" ablations
(HiRILConfig.du_mode).

Implementation choices (documented in docs/hiril_baseline.md):
  * Only the critic HEAD is Bayesian (all BayesianDense layers after the
    image encoder); the pixel encoder is the same deterministic, shared,
    pretrained-ResNet encoder every other method in this repo uses.
  * The paper drops the KL term (Eq. 10-11); kl_weight = 0 reproduces
    that. Without KL the weight std is shaped only by the TD loss.
  * TD3: target policy smoothing on the target actor, delayed actor AND
    target updates (every policy_delay critic updates).
"""

from functools import partial
from typing import Callable, Dict, FrozenSet, Iterable, Optional, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import (JaxRLTrainState, ModuleDict,
                                         default_init, nonpytree_field)
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.hiril.config import HiRILConfig
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


# ===========================================================================
# networks
# ===========================================================================
class BayesianDense(nn.Module):
    """Dense layer with mean-field Gaussian weights q(W) = N(mu, softplus(rho)^2).

    Input  x: (M, ..., d_in) -- leading axis = weight-sample index.
    Output  : (M, ..., features). With ``sample`` each of the M slices uses
    its own weight draw (rng stream "weights"); without it, all use mu.
    """
    features: int
    rho_init: float = -5.0

    @nn.compact
    def __call__(self, x, sample: bool):
        m, d_in = x.shape[0], x.shape[-1]
        k_mu = self.param("kernel_mu", default_init(), (d_in, self.features))
        k_rho = self.param("kernel_rho", nn.initializers.constant(self.rho_init),
                           (d_in, self.features))
        b_mu = self.param("bias_mu", nn.initializers.zeros, (self.features,))
        b_rho = self.param("bias_rho", nn.initializers.constant(self.rho_init),
                           (self.features,))
        if sample:
            k_key, b_key = jax.random.split(self.make_rng("weights"))
            kernel = k_mu + nn.softplus(k_rho) * jax.random.normal(
                k_key, (m, d_in, self.features))
            bias = b_mu + nn.softplus(b_rho) * jax.random.normal(
                b_key, (m, self.features))
        else:
            kernel = jnp.broadcast_to(k_mu, (m, d_in, self.features))
            bias = jnp.broadcast_to(b_mu, (m, self.features))
        y = jnp.einsum("m...i,mio->m...o", x, kernel)
        return y + bias.reshape((m,) + (1,) * (x.ndim - 2) + (self.features,))


class BIQNHead(nn.Module):
    """IQN head with Bayesian layers: psi(s,a) * phi(tau) -> MLP -> Z.

    features (B, F), taus (B, N) -> Z (M, B, N) for M = n_samples."""
    hidden_dims: Sequence[int] = (256, 256)
    n_cos: int = 64
    use_layer_norm: bool = True
    activations: Callable = nn.tanh
    rho_init: float = -5.0

    @nn.compact
    def __call__(self, features, taus, n_samples: int, sample: bool):
        x = jnp.broadcast_to(features, (n_samples,) + features.shape)
        x = BayesianDense(self.hidden_dims[0], self.rho_init)(x, sample)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        psi = self.activations(x)                                  # (M, B, H)
        i = jnp.arange(self.n_cos, dtype=jnp.float32)
        cos = jnp.cos(jnp.pi * i * taus[..., None])                # (B, N, C)
        phi = nn.relu(nn.Dense(self.hidden_dims[0],
                               kernel_init=default_init())(cos))   # (B, N, H)
        h = psi[:, :, None, :] * phi[None]                         # (M, B, N, H)
        for size in self.hidden_dims[1:]:
            h = BayesianDense(size, self.rho_init)(h, sample)
            if self.use_layer_norm:
                h = nn.LayerNorm()(h)
            h = self.activations(h)
        return BayesianDense(1, self.rho_init)(h, sample)[..., 0]  # (M, B, N)


class BIQNCritic(nn.Module):
    """Z(s, a) samples. observations are BATCHED (leading axis B)."""
    encoder: Optional[nn.Module]
    head: nn.Module

    @nn.compact
    def __call__(self, observations, actions, taus, n_samples: int,
                 sample: bool = True):
        enc = (observations if self.encoder is None
               else self.encoder(observations))
        return self.head(jnp.concatenate([enc, actions], axis=-1), taus,
                         n_samples, sample)


class DeterministicPolicy(nn.Module):
    """TD3 actor pi(s) = tanh(MLP(enc(s))). As in the repo's SAC Policy the
    shared encoder is used with stop_gradient (trained by the critic)."""
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int

    @nn.compact
    def __call__(self, observations, train: bool = False):
        enc = (observations if self.encoder is None else
               self.encoder(observations, train=train, stop_gradient=True))
        x = self.network(enc, train=train)
        return jnp.tanh(nn.Dense(self.action_dim,
                                 kernel_init=default_init())(x))


# ===========================================================================
# losses / statistics (pure functions, unit-tested)
# ===========================================================================
def quantile_huber_loss(z, target, taus, kappa):
    """Per-sample IQN loss (Eq. 15). z: (M, B, N) at quantile levels taus
    (B, N); target: (M, B, N'). delta_ij = target_j - z_i. Returns (B,):
    sum over i, mean over j and over the M weight samples."""
    delta = target[:, :, None, :] - z[:, :, :, None]               # (M,B,N,N')
    abs_d = jnp.abs(delta)
    huber = jnp.where(abs_d <= kappa, 0.5 * delta ** 2,
                      kappa * (abs_d - 0.5 * kappa))
    weight = jnp.abs(taus[None, :, :, None] - (delta < 0).astype(delta.dtype))
    return (weight * huber / kappa).sum(axis=2).mean(axis=(0, 2))


def dual_uncertainty_stats(z, sigma_z=0.0):
    """Eq. 16-20 from samples z: (M, B, N) (M weights x N quantile levels).
    total = E[Z^2] - E[Z]^2 over all samples = aleatoric + epistemic."""
    mean = z.mean(axis=(0, 2))
    total = (z ** 2).mean(axis=(0, 2)) - mean ** 2
    aleatoric = z.var(axis=2).mean(axis=0)
    epistemic = z.mean(axis=2).var(axis=0)
    return {"dual": total + sigma_z, "aleatoric": aleatoric + sigma_z,
            "epistemic": epistemic + sigma_z, "q": mean}


def split_mean(values, human):
    """(mean over RL samples, mean over human samples); 0 for an empty part.
    Eq. 29 / 32 take the two expectations separately, so each experience
    type carries its own weight regardless of how many are in the batch."""
    rl = 1.0 - human
    return ((values * rl).sum() / jnp.maximum(rl.sum(), 1.0),
            (values * human).sum() / jnp.maximum(human.sum(), 1.0))


def gaussian_kl(params, prior_std):
    """Mean KL(N(mu, softplus(rho)^2) || N(0, prior_std^2)) over all
    Bayesian weights in a param tree (pairs *_mu / *_rho)."""
    total, count = 0.0, 0
    flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
    for path, mu in flat.items():
        name = path[-1]
        if not name.endswith("_mu"):
            continue
        rho = flat[path[:-1] + (name[:-3] + "_rho",)]
        sd = nn.softplus(rho)
        kl = (jnp.log(prior_std / sd) + (sd ** 2 + mu ** 2)
              / (2.0 * prior_std ** 2) - 0.5)
        total, count = total + kl.sum(), count + kl.size
    return total / max(count, 1)


# ===========================================================================
# agent
# ===========================================================================
class HiRILAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    # ---- forwards ------------------------------------------------------
    def forward_policy(self, observations, *, grad_params=None, target=False,
                       train=False):
        params = grad_params or (self.state.target_params if target
                                 else self.state.params)
        return self.state.apply_fn({"params": params}, observations,
                                   name="actor", train=train)

    def forward_critic(self, observations, actions, taus, rng, *, n_samples,
                       grad_params=None, target=False, sample=True):
        params = grad_params or (self.state.target_params if target
                                 else self.state.params)
        return self.state.apply_fn(
            {"params": params}, observations, actions, taus, n_samples, sample,
            name="critic", rngs={"weights": rng} if sample else {})

    # ---- losses --------------------------------------------------------
    def critic_loss_fn(self, batch, params, rng):
        cfg = self.config
        b = batch["rewards"].shape[0]
        k_noise, k_tn, k_wn, k_tc, k_wc = jax.random.split(rng, 5)
        # TD3 target: target actor + clipped smoothing noise
        next_a = self.forward_policy(batch["next_observations"], target=True)
        noise = jnp.clip(cfg["policy_noise"]
                         * jax.random.normal(k_noise, next_a.shape),
                         -cfg["noise_clip"], cfg["noise_clip"])
        next_a = jnp.clip(next_a + noise, -1.0, 1.0)
        tau_next = jax.random.uniform(k_tn, (b, cfg["k_tau_target"]))
        z_next = self.forward_critic(batch["next_observations"], next_a,
                                     tau_next, k_wn, n_samples=cfg["k_theta"],
                                     target=True)                  # (M, B, N')
        target = jax.lax.stop_gradient(
            batch["rewards"][None, :, None]
            + cfg["discount"] * batch["masks"][None, :, None] * z_next)
        taus = jax.random.uniform(k_tc, (b, cfg["k_tau"]))
        z = self.forward_critic(batch["observations"], batch["actions"], taus,
                                k_wc, n_samples=cfg["k_theta"],
                                grad_params=params)                # (M, B, N)
        per_sample = quantile_huber_loss(z, target, taus, cfg["kappa"])
        human = batch.get("hiril_human", jnp.zeros((b,), jnp.float32))
        isw = batch.get("is_weight", jnp.ones((b,), jnp.float32))
        loss_rl, loss_h = split_mean(per_sample * isw, human)
        loss = loss_rl + loss_h
        info = {"critic_loss": loss, "critic_loss_rl": loss_rl,
                "critic_loss_human": loss_h,
                "q": z.mean(), "target_q": target.mean(),
                "human_frac_batch": human.mean(), "is_weight_mean": isw.mean()}
        if cfg["kl_weight"] > 0.0:
            kl = gaussian_kl(params["modules_critic"], cfg["prior_std"])
            loss = loss + cfg["kl_weight"] * kl
            info["kl"] = kl
        # |delta_TD| of the mean Q (Eq. 27) -> replay priorities
        info["td_abs"] = jax.lax.stop_gradient(
            jnp.abs(target.mean(axis=(0, 2)) - z.mean(axis=(0, 2))))
        return loss, info

    def actor_loss_fn(self, batch, params, rng):
        cfg = self.config
        b = batch["rewards"].shape[0]
        k_t, k_w = jax.random.split(rng)
        a_pi = self.forward_policy(batch["observations"], grad_params=params)
        taus = jax.random.uniform(k_t, (b, cfg["k_tau"]))
        q = self.forward_critic(batch["observations"], a_pi, taus, k_w,
                                n_samples=cfg["k_theta"]).mean(axis=(0, 2))
        human = batch.get("hiril_human", jnp.zeros((b,), jnp.float32))
        q_rl, _ = split_mean(q, human)
        bc = ((a_pi - batch["actions"]) ** 2).sum(axis=-1)
        _, bc_h = split_mean(bc, human)
        loss = -q_rl + cfg["lambda_bc"] * bc_h
        return loss, {"actor_loss": loss, "actor_q": q_rl, "bc_loss": bc_h}

    # ---- update --------------------------------------------------------
    @partial(jax.jit, static_argnames=("networks_to_update",))
    def update(self, batch, *, networks_to_update: FrozenSet[str] =
               frozenset({"actor", "critic"})):
        """One gradient step. The learner calls it policy_delay-1 times with
        {"critic"} and then once with {"critic", "actor"} (TD3 delay); the
        target networks move only on the actor step."""
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if self.config.get("augmentation_function") is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)
        batch = batch.copy(add_or_replace={
            "rewards": batch["rewards"] + self.config["reward_bias"]})
        loss_fns = {"actor": partial(self.actor_loss_fn, batch),
                    "critic": partial(self.critic_loss_fn, batch)}
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})
        new_state, info = self.state.apply_loss_fns(loss_fns, has_aux=True)
        if "actor" in networks_to_update:
            new_state = new_state.target_update(
                self.config["soft_target_update_rate"])
        return self.replace(state=new_state.replace(rng=rng)), info

    # ---- acting / risk ---------------------------------------------------
    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(self, observations, *, seed=None, argmax=False,
                       noise_std=0.1, **kwargs):
        """argmax -> deterministic pi(s) (evaluation); else Eq. 21
        a_RL = pi(s) + noise_std * N(0, I), clipped to the action box."""
        a = self.forward_policy(observations)
        if argmax:
            return a
        return jnp.clip(a + noise_std * jax.random.normal(seed, a.shape),
                        -1.0, 1.0)

    @jax.jit
    def dual_uncertainty(self, observations, actions, seed):
        """DU(s, a) for BATCHED observations (B, ...) and actions (B, A).
        Returns dict of (B,) arrays: du (per config du_mode), dual,
        aleatoric, epistemic, q."""
        cfg = self.config
        b = actions.shape[0]
        k_t, k_w = jax.random.split(seed)
        taus = jax.random.uniform(k_t, (b, cfg["du_k_tau"]))
        z = self.forward_critic(observations, actions, taus, k_w,
                                n_samples=cfg["du_k_theta"])
        stats = dual_uncertainty_stats(z, cfg["sigma_z"])
        stats["du"] = stats[cfg["du_mode"]]
        return stats

    # ---- construction ----------------------------------------------------
    @classmethod
    def create(cls, rng, observations, actions, *, actor_def, critic_def,
               hiril: HiRILConfig, discount: float, image_keys: Iterable[str],
               augmentation_function=None, reward_bias: float = 0.0):
        """observations / actions: ONE unbatched example (as the env gives
        it); a batch axis is added for initialisation."""
        model_def = ModuleDict({"actor": actor_def, "critic": critic_def})
        txs = {"actor": make_optimizer(learning_rate=hiril.actor_lr),
               "critic": make_optimizer(learning_rate=hiril.critic_lr)}
        obs_b = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None],
                                       observations)
        act_b = jnp.asarray(actions)[None]
        taus = jnp.full((1, hiril.k_tau), 0.5)
        rng, init_rng, create_rng = jax.random.split(rng, 3)
        # actor FIRST: the shared encoder's params then live under
        # "modules_actor", where load_resnet10_params expects them
        params = model_def.init(
            init_rng, actor=[obs_b],
            critic=[obs_b, act_b, taus, 1, False])["params"]
        state = JaxRLTrainState.create(apply_fn=model_def.apply, params=params,
                                       txs=txs, target_params=params,
                                       rng=create_rng)
        config = dict(
            discount=float(discount),
            soft_target_update_rate=float(hiril.target_update_rate),
            k_tau=int(hiril.k_tau), k_tau_target=int(hiril.k_tau_target),
            k_theta=int(hiril.k_theta), kappa=float(hiril.kappa),
            du_k_tau=int(hiril.du_k_tau), du_k_theta=int(hiril.du_k_theta),
            du_mode=str(hiril.du_mode), sigma_z=float(hiril.sigma_z),
            policy_noise=float(hiril.policy_noise),
            noise_clip=float(hiril.noise_clip),
            lambda_bc=float(hiril.lambda_bc),
            kl_weight=float(hiril.kl_weight), prior_std=float(hiril.prior_std),
            image_keys=tuple(image_keys), reward_bias=float(reward_bias),
            augmentation_function=augmentation_function)
        return cls(state=state, config=config)


def make_pixel_encoder(encoder_type: str, image_keys: Sequence[str]):
    """The same encoder SACAgent.create_pixels builds (so HiRIL sees the
    images exactly as the other methods do)."""
    if encoder_type == "resnet":
        from serl_launcher.vision.resnet_v1 import resnetv1_configs
        encoders = {k: resnetv1_configs["resnetv1-10"](
            pooling_method="spatial_learned_embeddings", num_spatial_blocks=8,
            bottleneck_dim=256, name=f"encoder_{k}") for k in image_keys}
    elif encoder_type == "resnet-pretrained":
        from serl_launcher.vision.resnet_v1 import (PreTrainedResNetEncoder,
                                                    resnetv1_configs)
        pretrained = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True, name="pretrained_encoder")
        encoders = {k: PreTrainedResNetEncoder(
            pooling_method="spatial_learned_embeddings", num_spatial_blocks=8,
            bottleneck_dim=256, pretrained_encoder=pretrained,
            name=f"encoder_{k}") for k in image_keys}
    else:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}")
    return EncodingWrapper(encoder=encoders, use_proprio=True,
                           enable_stacking=True, image_keys=image_keys)


def make_hiril_defs(action_dim: int, hiril: HiRILConfig, encoder=None):
    """(actor_def, critic_def) sharing ``encoder`` (None = raw features)."""
    actor_def = DeterministicPolicy(
        encoder=encoder,
        network=MLP(hidden_dims=hiril.actor_hidden_dims, activations=nn.tanh,
                    use_layer_norm=True, activate_final=True),
        action_dim=action_dim, name="actor")
    critic_def = BIQNCritic(
        encoder=encoder,
        head=BIQNHead(hidden_dims=hiril.critic_hidden_dims, n_cos=hiril.n_cos,
                      use_layer_norm=True, activations=nn.tanh,
                      rho_init=hiril.rho_init),
        name="critic")
    return actor_def, critic_def
