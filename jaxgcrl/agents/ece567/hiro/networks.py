"""Networks for HIRO: Actor (deterministic) and DoubleCritic (TD3-style)."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


def _make_init():
    return nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")


class Actor(nn.Module):
    """Deterministic MLP actor for TD3.

    Outputs tanh(x) * action_scale so actions are in [-action_scale, action_scale].
    Input is (state, goal) concatenated for the low-level actor, or state-only
    for the high-level actor.
    """

    action_size: int
    action_scale: float
    h_dim: int = 300
    n_hidden: int = 2
    use_ln: bool = False

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_hidden):
            x = nn.Dense(self.h_dim, kernel_init=_make_init())(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_size, kernel_init=_make_init())(x)
        return jnp.tanh(x) * self.action_scale


class DoubleCritic(nn.Module):
    """Two parallel MLP Q-functions (TD3 double critic).

    Input: concatenation of state (+ goal) and action.
    Output: (Q1, Q2) scalar values.
    """

    h_dim: int = 300
    n_hidden: int = 2
    use_ln: bool = False

    @nn.compact
    def __call__(self, x):
        # Q1
        q1 = x
        for _ in range(self.n_hidden):
            q1 = nn.Dense(self.h_dim, kernel_init=_make_init())(q1)
            if self.use_ln:
                q1 = nn.LayerNorm()(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1, kernel_init=_make_init())(q1)
        q1 = jnp.squeeze(q1, axis=-1)

        # Q2
        q2 = x
        for _ in range(self.n_hidden):
            q2 = nn.Dense(self.h_dim, kernel_init=_make_init())(q2)
            if self.use_ln:
                q2 = nn.LayerNorm()(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1, kernel_init=_make_init())(q2)
        q2 = jnp.squeeze(q2, axis=-1)

        return q1, q2
