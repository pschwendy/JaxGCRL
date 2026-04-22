"""AWCR losses — identical to CRL losses.

AWCR's only innovation is in flatten_batch (advantage-weighted HER sampling).
The critic and actor losses are unchanged from CRL.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


def _energy(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def _infonce(name, logits):
    if name == "fwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        return -jnp.mean(
            2 * jnp.diag(logits)
            - jax.nn.logsumexp(logits, axis=1)
            - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        return -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss: {name}")


def update_critic(config, networks, transitions, training_state, key):
    """InfoNCE loss — identical to CRL."""

    def critic_loss_fn(critic_params):
        state = transitions.observation[:, :config["state_size"]]
        action = transitions.action
        g_repr = networks["g_encoder"].apply(
            critic_params["g_encoder"],
            transitions.observation[:, config["state_size"]:],
        )
        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )

        logits = _energy(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss = _infonce(config["contrastive_loss_fn"], logits)

        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp ** 2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return loss, (logsumexp, correct, logits_pos, logits_neg)

    (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(training_state.critic_state.params)

    training_state = training_state.replace(
        critic_state=training_state.critic_state.apply_gradients(grads=grad)
    )
    return training_state, {
        "critic_loss": loss,
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
    }


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC actor + automatic entropy tuning — identical to CRL."""

    def actor_loss_fn(actor_params, log_alpha):
        obs = transitions.observation
        state = obs[:, :config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_t = means + stds * jax.random.normal(key, means.shape, dtype=means.dtype)
        action = nn.tanh(x_t)

        log_prob = jax.scipy.stats.norm.logpdf(x_t, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # (B,)

        critic_params = jax.lax.stop_gradient(training_state.critic_state.params)
        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal)
        q = _energy(config["energy_fn"], sa_repr, g_repr)  # (B,)

        loss = jnp.mean(jnp.exp(log_alpha) * log_prob - q)
        return loss, log_prob

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(
        training_state.actor_state.params,
        training_state.alpha_state.params["log_alpha"],
    )

    def alpha_loss_fn(alpha_params):
        alpha = jnp.exp(alpha_params["log_alpha"])
        return alpha * jnp.mean(
            jax.lax.stop_gradient(-log_prob - config["target_entropy"])
        )

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        training_state.alpha_state.params
    )

    training_state = training_state.replace(
        actor_state=training_state.actor_state.apply_gradients(grads=actor_grad),
        alpha_state=training_state.alpha_state.apply_gradients(grads=alpha_grad),
    )
    return training_state, {
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "entropy": -log_prob.mean(),
    }
