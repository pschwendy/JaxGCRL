"""Networks for AWCR.

AWCR is CRL with advantage-weighted HER goal sampling.
Networks are identical to CRL: two encoders + actor.
"""

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


def _activation(use_relu: bool):
    return nn.relu if use_relu else nn.swish


class Encoder(nn.Module):
    """φ_sa / ψ: generic MLP encoder with optional skip connections and layer norm."""

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
