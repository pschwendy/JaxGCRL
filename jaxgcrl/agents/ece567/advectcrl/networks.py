"""Networks for AdvectCRL.

Four components:

  φ_sa : S × A → R^repr_dim      contrastive state-action encoder (raw state)
  ψ    : G → R^repr_dim           goal encoder (shared for critic + CNF)
  v_θ  : R^(2·repr_dim+1) → R^repr_dim  conditional flow vector field
  π    : S × R^repr_dim → A       goal-conditioned actor (latent goal target)
"""

import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


def _make_activation(use_relu: bool):
    return nn.relu if use_relu else nn.swish


def _make_normalize(use_ln: bool):
    if use_ln:
        return lambda x: nn.LayerNorm()(x)
    return lambda x: x


class MLP(nn.Module):
    """Generic MLP with optional skip connections and layer norm."""

    output_dim: int
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        activation = _make_activation(self.use_relu)
        normalize = _make_normalize(self.use_ln)

        skip = None
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            x = normalize(x)
            x = activation(x)
            if self.skip_connections:
                if i == 0:
                    skip = x
                elif i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        return nn.Dense(self.output_dim, kernel_init=lecun, bias_init=zeros)(x)


class SAEncoder(nn.Module):
    """φ_sa: cat(s, a) → repr ∈ R^repr_dim (L2-normalised).

    Takes the raw state concatenated with the action — no separate state
    encoder so gradients from the contrastive loss train state representations
    directly without a decoupled dynamics objective.  L2 normalisation keeps
    the dot-product Q-value in [-1, 1].
    """

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, sa: jnp.ndarray) -> jnp.ndarray:
        logging.info("SAEncoder input shape: %s", sa.shape)
        out = MLP(
            output_dim=self.repr_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(sa)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)


class GoalEncoder(nn.Module):
    """ψ: g → repr ∈ R^repr_dim (L2-normalised).

    Used both by the contrastive critic and as the embedding space for the
    CNF trajectory learning.  L2 normalisation ensures Q = φ_sa · ψ ∈ [-1,1],
    preventing the unbounded growth that otherwise occurs when ψ receives
    gradients from the actor loss (which maximises Q).
    """

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        out = MLP(
            output_dim=self.repr_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(g)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)


class VectorField(nn.Module):
    """v_θ: cat(z, z_goal, t) → R^repr_dim — conditional flow vector field.

    Trained with flow-matching loss to predict the instantaneous velocity
    along the straight-line interpolation between any z_0 and z_1.  At
    inference, Euler integration of v_θ(·, z_goal, ·) advects the current
    state latent toward the goal latent along the learned manifold.

    Input layout: [z (repr_dim) | z_goal (repr_dim) | t (1)]
    """

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, z_goal_t: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            output_dim=self.repr_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(z_goal_t)


class Actor(nn.Module):
    """π: cat(s, z_target) → (mean, log_std).

    Takes the raw state s concatenated with a latent target z_target ∈ R^repr_dim.
    During training z_target = sg(ψ(s_{t+k}[goal_indices])).
    During inference z_target is produced by CNF advection.
    """

    action_size: int
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, s_z: jnp.ndarray):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        activation = _make_activation(self.use_relu)
        normalize = _make_normalize(self.use_ln)

        logging.info("Actor input shape: %s", s_z.shape)
        x = s_z
        skip = None
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            x = normalize(x)
            x = activation(x)
            if self.skip_connections:
                if i == 0:
                    skip = x
                elif i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(self.action_size, kernel_init=lecun, bias_init=zeros)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun, bias_init=zeros)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std
