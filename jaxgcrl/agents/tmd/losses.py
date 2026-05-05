import flax.linen as nn
import jax
import jax.numpy as jnp


def quasimetric_distance(x, y):
    """Asymmetric L1 quasimetric: d(x,y) = mean(relu(x - y)).

    Satisfies d(x,x)=0, d>=0, and the triangle inequality, but NOT symmetry.
    Intuitively, d(x,y) measures how much y "falls short of" x in each dimension.
    """
    return jnp.mean(jax.nn.relu(x - y), axis=-1)


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    def actor_loss_fn(actor_params, critic_params, log_alpha, transitions, key):
        state = transitions.extras["state"]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal)

        # Actor minimizes quasimetric distance to goal + entropy regularization
        dist = quasimetric_distance(sa_repr, g_repr)
        loss = jnp.mean(jnp.exp(log_alpha) * log_prob + dist)

        return loss, log_prob

    def alpha_loss_fn(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - config["target_entropy"]))

    (loss, log_prob), actor_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        training_state.alpha_state.params, log_prob
    )
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        actor_state=new_actor_state, alpha_state=new_alpha_state
    )

    return training_state, {
        "entropy": -log_prob,
        "actor_loss": loss,
        "alpha_loss": alpha_loss_val,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss_fn(critic_params, transitions):
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        state = transitions.extras["state"]
        action = transitions.action
        next_state = transitions.extras["next_state"]
        # Relabeled goal: goal features of a discounted-future state (from flatten_batch)
        goal = transitions.observation[:, config["state_size"]:]

        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1)
        )
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)

        # ψ(s') — next state encoded as a goal, used as the Bellman bootstrap target
        next_s_repr = networks["g_encoder"].apply(
            g_encoder_params, next_state[:, config["goal_indices"]]
        )
        # ψ(s) — current state encoded as a goal, used for I-invariance
        s_repr = networks["g_encoder"].apply(
            g_encoder_params, state[:, config["goal_indices"]]
        )

        # T-invariance: Bellman regression in quasimetric space.
        # Encourages: exp(-d(φ(s,a), ψ(g))) ≈ γ · exp(-d(ψ(s'), ψ(g)))
        pred = jnp.exp(-quasimetric_distance(sa_repr, g_repr))
        target = jax.lax.stop_gradient(
            config["discounting"] * jnp.exp(-quasimetric_distance(next_s_repr, g_repr))
        )
        t_loss = jnp.mean((pred - target) ** 2)

        # I-invariance: offline state-action pairs should have near-zero immediate cost.
        # Encourages: d(ψ(s), φ(s, a_offline)) ≈ 0 — the dataset actions are "free."
        i_loss = jnp.mean(quasimetric_distance(s_repr, sa_repr))

        total_loss = t_loss + config["i_invariance_coeff"] * i_loss

        return total_loss, (t_loss, i_loss)

    (loss, (t_loss, i_loss)), grad = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        training_state.critic_state.params, transitions
    )
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    return training_state, {
        "critic_loss": loss,
        "t_invariance_loss": t_loss,
        "i_invariance_loss": i_loss,
    }
