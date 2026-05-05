"""Networks for SCC-RL v8: latent-space subgoal generation.

The CVAE now generates subgoals directly in the critic's representation space
(ℝ^repr_dim) instead of goal observation space (ℝ^goal_size).

  q_θ(h | s, ψ(z), ψ(g)) : (state_size + 2*repr_dim) → (latent_dim, latent_dim)
  p_ξ(ψ̃ | h, s, ψ(g))   : (latent_dim + state_size + repr_dim) → repr_dim

The actor is conditioned on [state, ψ̃] with input dim = state_size + repr_dim.
"""

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling

from jaxgcrl.agents.ece567.networks import Actor, Encoder  # noqa: F401


class CVAEEncoder(nn.Module):
    """q_θ(h | s, ψ(z), ψ(g)) — recognition network.

    Takes state plus goal-encoder embeddings of the subgoal and far goal;
    outputs (μ_h, log σ²_h) for the reparameterisation trick.
    """

    latent_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,        # (B, state_size)
        psi_subgoal: jnp.ndarray,  # (B, repr_dim)
        psi_goal: jnp.ndarray,     # (B, repr_dim)
    ):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish
        normalize = (lambda x: nn.LayerNorm()(x)) if self.use_ln else (lambda x: x)

        x = jnp.concatenate([state, psi_subgoal, psi_goal], axis=-1)
        for _ in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            x = normalize(x)
            x = activation(x)

        mu = nn.Dense(self.latent_dim, kernel_init=lecun, bias_init=zeros)(x)
        log_var = nn.Dense(self.latent_dim, kernel_init=lecun, bias_init=zeros)(x)
        return mu, log_var


class CVAEDecoder(nn.Module):
    """p_ξ(ψ̃ | h, s, ψ(g)) — generative network.

    Takes a latent code h, the current state, and the goal embedding ψ(g);
    outputs a predicted latent subgoal ψ̃ ∈ ℝ^repr_dim.
    """

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,          # (B, latent_dim)
        state: jnp.ndarray,       # (B, state_size)
        psi_goal: jnp.ndarray,    # (B, repr_dim)
    ):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish
        normalize = (lambda x: nn.LayerNorm()(x)) if self.use_ln else (lambda x: x)

        x = jnp.concatenate([h, state, psi_goal], axis=-1)
        for _ in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            x = normalize(x)
            x = activation(x)

        return nn.Dense(self.repr_dim, kernel_init=lecun, bias_init=zeros)(x)
