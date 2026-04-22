"""Networks for SCC-RL (Subgoal-Conditioned Contrastive RL).

Adds a Geometric Subgoal Sampler f(z | s, g) implemented as a CVAE on top of
the standard CRL (s,a)-encoder φ and goal-encoder ψ:

  φ(s, a)  : (state_size + action_size) → repr_dim     [contrastive SA encoder]
  ψ(g)     : goal_size → repr_dim                       [contrastive goal encoder]
  q_θ(h|s,z,g) : (state_size + 2*goal_size) → (latent_dim, latent_dim)  [CVAE encoder]
  p_ξ(z|h,s,g) : (latent_dim + state_size + goal_size) → goal_size       [CVAE decoder]

The CVAE encoder and decoder are trained with a ELBO objective.  During
actor training and inference the CVAE is used in *generation* mode:
  h ~ N(0, I),  z = p_ξ(h, s, g)
"""

import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling

# Re-export the CRL base networks so callers only need one import.
from jaxgcrl.agents.ece567.networks import Actor, Encoder  # noqa: F401


class CVAEEncoder(nn.Module):
    """q_θ(h | s_t, z, g)  — recognition network for training.

    Takes the current state, the intermediate subgoal z (in goal space),
    and the far goal g; outputs (μ_h, log σ²_h) for the reparameterisation
    trick.

    Input: concat(s_t, z, g)  with sizes (state_size, goal_size, goal_size).
    Output: (mu, log_var) each of shape (latent_dim,).
    """

    latent_dim: int = 64
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,    # (B, state_size)
        subgoal: jnp.ndarray,  # (B, goal_size)
        goal: jnp.ndarray,     # (B, goal_size)
    ):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish
        normalize = (lambda x: nn.LayerNorm()(x)) if self.use_ln else (lambda x: x)

        x = jnp.concatenate([state, subgoal, goal], axis=-1)
        logging.info("CVAEEncoder input shape: %s", x.shape)

        for _ in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            x = normalize(x)
            x = activation(x)

        mu = nn.Dense(self.latent_dim, kernel_init=lecun, bias_init=zeros)(x)
        log_var = nn.Dense(self.latent_dim, kernel_init=lecun, bias_init=zeros)(x)
        return mu, log_var


class CVAEDecoder(nn.Module):
    """p_ξ(z | h, s_t, g)  — generative network.

    Takes a latent code h (sampled from either the posterior during training
    or N(0,I) during inference), the current state, and the far goal; outputs
    the predicted intermediate subgoal z in *goal* space.

    Input: concat(h, s_t, g)  with sizes (latent_dim, state_size, goal_size).
    Output: z_pred of shape (goal_size,).
    """

    goal_size: int
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,       # (B, latent_dim)
        state: jnp.ndarray,   # (B, state_size)
        goal: jnp.ndarray,    # (B, goal_size)
    ):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish
        normalize = (lambda x: nn.LayerNorm()(x)) if self.use_ln else (lambda x: x)

        x = jnp.concatenate([h, state, goal], axis=-1)
        logging.info("CVAEDecoder input shape: %s", x.shape)

        for _ in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=zeros)(x)
            x = normalize(x)
            x = activation(x)

        return nn.Dense(self.goal_size, kernel_init=lecun, bias_init=zeros)(x)
