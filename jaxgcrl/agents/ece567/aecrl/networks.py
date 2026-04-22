"""Networks for AE-CRL.

AE-CRL adds three components on top of the standard CRL encoders:

  sa_encoder : φ(s,a) → repr_dim   (InfoNCE contrastive critic, identical to CRL)
  g_encoder  : ψ(g)   → repr_dim   (shared: InfoNCE + Bellman TD)
  s_encoder  : φ_v(s) → repr_dim   (Bellman TD, state-only)
  energy_head: (s_repr, g_repr) → scalar ≥ 0  (learned asymmetric energy)
  actor      : π(s,g) → (mean, log_std)       (identical to CRL)

EnergyHead is the key AE-CRL addition: it is a small MLP that maps the pair
(s_repr, g_repr) to a non-negative scalar E_θ(s,g), giving a fully learnable
and asymmetric energy function.  The induced value V_θ(s,g) = exp(-E_θ(s,g))
lies in (0, 1] and satisfies Bellman consistency during training.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


def _activation(use_relu: bool):
    return nn.relu if use_relu else nn.swish


class Encoder(nn.Module):
    """Generic MLP encoder with optional skip connections and layer norm.

    Used for sa_encoder, g_encoder, and s_encoder.
    """

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        act = _activation(self.use_relu)

        skip = None
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = act(x)
            if self.skip_connections:
                if i == 0:
                    skip = x
                elif i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        return nn.Dense(self.repr_dim, kernel_init=lecun, bias_init=zeros)(x)


class EnergyHead(nn.Module):
    """Learned asymmetric energy E_θ(s,g): concat(s_repr, g_repr) → scalar ≥ 0.

    Asymmetric because the concatenation order is (s_repr, g_repr), so swapping
    s and g yields a different input and (in general) a different energy.
    The softplus output guarantees E ≥ 0, so V = exp(-E) ∈ (0, 1].
    """

    hidden_dim: int = 64

    @nn.compact
    def __call__(self, s_repr: jnp.ndarray, g_repr: jnp.ndarray) -> jnp.ndarray:
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        x = jnp.concatenate([s_repr, g_repr], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=lecun, bias_init=zeros)(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=lecun, bias_init=zeros)(x)
        # softplus: always ≥ 0, so V = exp(-E) ∈ (0, 1]
        return jax.nn.softplus(x[..., 0])


class Actor(nn.Module):
    """π: cat(s, g) → (mean, log_std)."""

    action_size: int
    network_width: int = 256
    network_depth: int = 2
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        act = _activation(self.use_relu)

        x = obs
        skip = None
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = act(x)
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
