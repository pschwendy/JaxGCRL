"""Networks for PlanCRL.

Five components, all mapping into the same repr_dim space so that
f(z, a) · ψ(g) is a well-defined planning reward:

  φ_s  : S → R^repr_dim          state encoder
  f    : R^repr_dim × A → R^repr_dim  dynamics model (predicts next latent)
  φ_sa : R^repr_dim × A → R^repr_dim  contrastive (latent-state, action) encoder
  ψ    : G → R^repr_dim          goal encoder
  π    : R^repr_dim → A          goal-agnostic actor (planner provides goal-direction)
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


class StateEncoder(nn.Module):
    """φ_s: s → z ∈ R^repr_dim"""

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        logging.info("StateEncoder input shape: %s", s.shape)
        return MLP(
            output_dim=self.repr_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(s)


class DynamicsModel(nn.Module):
    """f: (z, a) → z_next ∈ R^repr_dim

    Trained to predict φ_s(s_{t+1}) from (φ_s(s_t), a_t).
    Also used as the per-step planning reward: f(z, a) · ψ(g).
    """

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, z_and_a: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            output_dim=self.repr_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(z_and_a)


class SAEncoder(nn.Module):
    """φ_sa: (z, a) → repr ∈ R^repr_dim  (contrastive Q-function encoder)"""

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, z_and_a: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            output_dim=self.repr_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(z_and_a)


class GoalEncoder(nn.Module):
    """ψ: g → repr ∈ R^repr_dim"""

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            output_dim=self.repr_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(g)


class Actor(nn.Module):
    """π: [z, g] → (mean, log_std)

    Goal-conditioned: receives the concatenation of the latent state z = φ_s(s)
    and the raw goal g.  During planning, g is the goal from the current obs.
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
    def __call__(self, z: jnp.ndarray):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        activation = _make_activation(self.use_relu)
        normalize = _make_normalize(self.use_ln)

        logging.info("Actor input shape: %s", z.shape)
        x = z
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
