"""SCC-RL v7 losses.

Same as v6 except no CVAE — random subgoals (ablation).
Subgoals are random linear interpolations between the current state's
goal-space coordinates and the true goal: sg = s_g + t*(g - s_g), t~U[0,1].
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


def energy_fn(name, x, y):
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


def contrastive_loss_fn(name, logits):
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
        raise ValueError(f"Unknown contrastive loss function: {name}")


# ---------------------------------------------------------------------------
# 1. Actor + alpha update (random subgoal instead of CVAE)
# ---------------------------------------------------------------------------

def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC actor with alpha_subgoal-mixed goal conditioning.

    Subgoal = random linear interpolation between state's goal-space coords
    and the true goal.  No CVAE involved.
    """
    key, mix_key, interp_key, actor_key = jax.random.split(key, 4)

    state     = transitions.observation[:, : config["state_size"]]
    true_goal = transitions.extras["future_state"][:, config["goal_indices"]]

    # Random subgoal: sg = s_g + t * (g - s_g), t ~ U[0, 1]
    state_goal_coords = state[:, jnp.array(config["goal_indices"])]
    t = jax.random.uniform(interp_key, (state.shape[0], 1))
    z_subgoal = jax.lax.stop_gradient(state_goal_coords + t * (true_goal - state_goal_coords))

    use_subgoal = jax.random.bernoulli(mix_key, p=config["alpha_subgoal"], shape=(state.shape[0],))
    goal_tilde  = jnp.where(use_subgoal[:, None], z_subgoal, true_goal)

    def actor_loss(actor_params, critic_params, log_alpha, key):
        observation = jnp.concatenate([state, goal_tilde], axis=1)
        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds  = jnp.exp(log_stds)
        x_ts  = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)

        log_prob  = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob  = log_prob.sum(-1)

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"], jnp.concatenate([state, action], axis=-1)
        )
        g_repr  = networks["g_encoder"].apply(critic_params["g_encoder"], goal_tilde)
        qf_pi   = energy_fn(config["energy_fn"], sa_repr, g_repr)

        return jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi), log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - config["target_entropy"]))

    (actor_loss_val, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        actor_key,
    )
    training_state = training_state.replace(
        actor_state=training_state.actor_state.apply_gradients(grads=actor_grad)
    )

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss)(
        training_state.alpha_state.params, log_prob
    )
    training_state = training_state.replace(
        alpha_state=training_state.alpha_state.apply_gradients(grads=alpha_grad)
    )

    return training_state, {
        "entropy": -log_prob.mean(),
        "actor_loss": actor_loss_val,
        "alpha_loss": alpha_loss_val,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "subgoal_fraction": use_subgoal.mean(),
    }


# ---------------------------------------------------------------------------
# 2. Critic update (same as v6)
# ---------------------------------------------------------------------------

def update_critic(config, networks, transitions, training_state, key):
    """InfoNCE critic calibrated on the same mixed goal distribution as the actor."""

    def critic_loss(critic_params, transitions, key):
        mix_key, _ = jax.random.split(key)

        state     = transitions.observation[:, : config["state_size"]]
        action    = transitions.action
        true_goal = transitions.observation[:, config["state_size"]:]
        subgoal   = transitions.extras["subgoal"]

        use_subgoal = jax.random.bernoulli(
            mix_key, p=config["alpha_subgoal"], shape=(state.shape[0],)
        )
        goal_tilde = jnp.where(use_subgoal[:, None], subgoal, true_goal)

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"], jnp.concatenate([state, action], axis=-1)
        )
        g_repr  = networks["g_encoder"].apply(critic_params["g_encoder"], goal_tilde)

        logits      = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss        = contrastive_loss_fn(config["contrastive_loss_fn"], logits)
        logsumexp   = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss       += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp ** 2)

        I           = jnp.eye(logits.shape[0])
        correct     = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos  = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg  = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        return loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    training_state = training_state.replace(
        critic_state=training_state.critic_state.apply_gradients(grads=grad)
    )
    return training_state, {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }
