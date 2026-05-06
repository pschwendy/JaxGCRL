"""Loss functions and update steps for HIRO (TD3-based hierarchical RL)."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Off-policy correction
# ---------------------------------------------------------------------------

def off_policy_correction(
    lo_actor,
    lo_actor_params,
    s_seq,
    a_seq,
    s0,
    s_c,
    original_g,
    key,
    goal_indices,
    hi_action_scale,
):
    """Re-label the high-level goal to maximise lo-policy log-probability.

    Args:
        lo_actor: Flax module for the low-level actor.
        lo_actor_params: Parameters of the low-level actor.
        s_seq: [c, state_size]  — states observed during the window.
        a_seq: [c, action_size] — actions taken during the window.
        s0:   [state_size]      — state at the start of the window.
        s_c:  [state_size]      — state at the end of the window.
        original_g: [goal_size] — goal actually used at collection time.
        key: JAX PRNG key.
        goal_indices: tuple/array of indices into state for goal dimensions.
        hi_action_scale: scalar clip range for high-level goals.

    Returns:
        best_g: [goal_size] — re-labelled goal.
    """
    goal_indices = jnp.array(goal_indices)
    diff = s_c[goal_indices] - s0[goal_indices]           # [goal_size]
    goal_size = original_g.shape[0]

    # 8 Gaussian samples centred at diff with std = 0.5 * hi_action_scale
    gaussian_samples = (
        diff[None, :]
        + 0.5 * hi_action_scale * jax.random.normal(key, shape=(8, goal_size))
    )  # [8, goal_size]

    # Stack 10 candidates: [original_g, diff, 8 Gaussian samples]
    candidates = jnp.concatenate(
        [original_g[None, :], diff[None, :], gaussian_samples], axis=0
    )  # [10, goal_size]
    candidates = jnp.clip(candidates, -hi_action_scale, hi_action_scale)

    def log_prob_for_g0(g0):
        """Compute Σ_t -0.5 * ||a_t - lo_actor(s_t, g_t)||² for a candidate g0."""

        # Goal transition: g_{t+1} = s_t[goal_idx] + g_t - s_{t+1}[goal_idx]
        def propagate_goal(g_t, t):
            s_t = s_seq[t]
            s_t1 = s_seq[jnp.minimum(t + 1, s_seq.shape[0] - 1)]
            g_next = s_t[goal_indices] + g_t - s_t1[goal_indices]
            g_next = jnp.clip(g_next, -hi_action_scale, hi_action_scale)
            return g_next, g_t  # carry next, output current

        c_steps = s_seq.shape[0]
        _, g_seq = jax.lax.scan(
            propagate_goal, g0, jnp.arange(c_steps)
        )  # g_seq: [c, goal_size]

        # lo_actor predictions for every (s_t, g_t) pair
        def actor_pred(args):
            s_t, g_t = args
            inp = jnp.concatenate([s_t, g_t], axis=-1)
            return lo_actor.apply(lo_actor_params, inp)  # [action_size]

        preds = jax.vmap(actor_pred)((s_seq, g_seq))  # [c, action_size]

        diff_a = a_seq - preds  # [c, action_size]
        log_prob = -0.5 * jnp.sum(diff_a ** 2)
        return log_prob

    log_probs = jax.vmap(log_prob_for_g0)(candidates)  # [10]
    best_idx = jnp.argmax(log_probs)
    return candidates[best_idx]  # [goal_size]


# ---------------------------------------------------------------------------
# Soft target update helper
# ---------------------------------------------------------------------------

def _soft_update(params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1.0 - tau) * tp, params, target_params
    )


# ---------------------------------------------------------------------------
# Low-level (worker) TD3 update
# ---------------------------------------------------------------------------

def update_lo(config, lo_networks, lo_transitions, training_state, key):
    """TD3 update for the low-level actor and critic.

    Always updates the critic. Updates the actor and target networks only
    when gradient_steps % policy_delay == 0.

    Args:
        config: dict with hyperparameters (discounting, tau, lo_smoothing_noise,
                noise_clip, policy_delay, lo_reward_scale).
        lo_networks: dict with 'lo_actor' and 'lo_critic' Flax modules.
        lo_transitions: LoTransition namedtuple.
        training_state: full TrainingState.
        key: JAX PRNG key.

    Returns:
        (updated_training_state, metrics_dict)
    """
    lo_actor = lo_networks["lo_actor"]
    lo_critic = lo_networks["lo_critic"]

    gamma = config["discounting"]
    tau = config["tau"]
    smoothing = config["lo_smoothing_noise"]
    noise_clip = config["noise_clip"]
    policy_delay = config["policy_delay"]

    key, noise_key = jax.random.split(key)

    # Unpack transitions
    state = lo_transitions.state        # [B, state_size]
    goal = lo_transitions.goal          # [B, goal_size]
    action = lo_transitions.action      # [B, action_size]
    reward = lo_transitions.reward      # [B]
    next_state = lo_transitions.next_state
    next_goal = lo_transitions.next_goal
    discount = lo_transitions.discount  # [B]

    # ---- Critic update ----
    def critic_loss_fn(critic_params):
        # Target policy smoothing
        a_next = lo_actor.apply(
            training_state.lo_actor_target_params,
            jnp.concatenate([next_state, next_goal], axis=-1),
        )
        noise = jnp.clip(
            smoothing * jax.random.normal(noise_key, shape=a_next.shape),
            -noise_clip,
            noise_clip,
        )
        a_next = jnp.clip(a_next + noise, -1.0, 1.0)

        # Target Q value
        q1_t, q2_t = lo_critic.apply(
            training_state.lo_critic_target_params,
            jnp.concatenate([next_state, next_goal, a_next], axis=-1),
        )
        q_target = jax.lax.stop_gradient(
            reward + gamma * discount * jnp.minimum(q1_t, q2_t)
        )

        # Online Q prediction
        q1, q2 = lo_critic.apply(
            critic_params,
            jnp.concatenate([state, goal, action], axis=-1),
        )
        loss = jnp.mean((q1 - q_target) ** 2 + (q2 - q_target) ** 2)
        return loss, {"lo_critic_loss": loss}

    (c_loss, c_metrics), c_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        training_state.lo_critic_state.params
    )
    new_lo_critic_state = training_state.lo_critic_state.apply_gradients(grads=c_grads)

    # ---- Actor update (every policy_delay steps) ----
    def actor_loss_fn(actor_params):
        a = lo_actor.apply(
            actor_params,
            jnp.concatenate([state, goal], axis=-1),
        )
        q1, _ = lo_critic.apply(
            new_lo_critic_state.params,
            jnp.concatenate([state, goal, a], axis=-1),
        )
        loss = -jnp.mean(q1)
        return loss, {"lo_actor_loss": loss}

    (a_loss, a_metrics), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        training_state.lo_actor_state.params
    )

    # Conditionally apply actor update + soft target update
    do_update = (training_state.gradient_steps % policy_delay) == 0

    new_lo_actor_state = jax.lax.cond(
        do_update,
        lambda: training_state.lo_actor_state.apply_gradients(grads=a_grads),
        lambda: training_state.lo_actor_state,
    )
    new_lo_actor_target = jax.lax.cond(
        do_update,
        lambda: _soft_update(new_lo_actor_state.params, training_state.lo_actor_target_params, tau),
        lambda: training_state.lo_actor_target_params,
    )
    new_lo_critic_target = jax.lax.cond(
        do_update,
        lambda: _soft_update(new_lo_critic_state.params, training_state.lo_critic_target_params, tau),
        lambda: training_state.lo_critic_target_params,
    )

    new_ts = training_state.replace(
        lo_actor_state=new_lo_actor_state,
        lo_actor_target_params=new_lo_actor_target,
        lo_critic_state=new_lo_critic_state,
        lo_critic_target_params=new_lo_critic_target,
    )
    metrics = {**c_metrics, **a_metrics}
    return new_ts, metrics


# ---------------------------------------------------------------------------
# High-level (manager) TD3 update
# ---------------------------------------------------------------------------

def update_hi(config, hi_networks, hi_transitions, training_state, key):
    """TD3 update for the high-level actor and critic.

    Args:
        config: dict with hyperparameters.
        hi_networks: dict with 'hi_actor' and 'hi_critic' Flax modules.
        hi_transitions: HiTransition namedtuple.
        training_state: full TrainingState.
        key: JAX PRNG key.

    Returns:
        (updated_training_state, metrics_dict)
    """
    hi_actor = hi_networks["hi_actor"]
    hi_critic = hi_networks["hi_critic"]

    gamma = config["discounting"]
    tau = config["tau"]
    smoothing = config["lo_smoothing_noise"]  # reuse smoothing noise for hi
    noise_clip = config["noise_clip"]
    policy_delay = config["policy_delay"]
    hi_action_scale = config["hi_action_scale"]

    key, noise_key = jax.random.split(key)

    state = hi_transitions.state
    goal = hi_transitions.goal
    reward = hi_transitions.reward
    next_state = hi_transitions.next_state
    discount = hi_transitions.discount

    # ---- Critic update ----
    def critic_loss_fn(critic_params):
        # Target policy smoothing for hi-level
        g_next = hi_actor.apply(training_state.hi_actor_target_params, next_state)
        noise = jnp.clip(
            smoothing * jax.random.normal(noise_key, shape=g_next.shape),
            -noise_clip,
            noise_clip,
        )
        g_next = jnp.clip(g_next + noise, -hi_action_scale, hi_action_scale)

        # Target Q
        q1_t, q2_t = hi_critic.apply(
            training_state.hi_critic_target_params,
            jnp.concatenate([next_state, g_next], axis=-1),
        )
        q_target = jax.lax.stop_gradient(
            reward + gamma * discount * jnp.minimum(q1_t, q2_t)
        )

        q1, q2 = hi_critic.apply(
            critic_params,
            jnp.concatenate([state, goal], axis=-1),
        )
        loss = jnp.mean((q1 - q_target) ** 2 + (q2 - q_target) ** 2)
        return loss, {"hi_critic_loss": loss}

    (c_loss, c_metrics), c_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        training_state.hi_critic_state.params
    )
    new_hi_critic_state = training_state.hi_critic_state.apply_gradients(grads=c_grads)

    # ---- Actor update ----
    def actor_loss_fn(actor_params):
        g = hi_actor.apply(actor_params, state)
        q1, _ = hi_critic.apply(
            new_hi_critic_state.params,
            jnp.concatenate([state, g], axis=-1),
        )
        loss = -jnp.mean(q1)
        return loss, {"hi_actor_loss": loss}

    (a_loss, a_metrics), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        training_state.hi_actor_state.params
    )

    do_update = (training_state.gradient_steps % policy_delay) == 0

    new_hi_actor_state = jax.lax.cond(
        do_update,
        lambda: training_state.hi_actor_state.apply_gradients(grads=a_grads),
        lambda: training_state.hi_actor_state,
    )
    new_hi_actor_target = jax.lax.cond(
        do_update,
        lambda: _soft_update(new_hi_actor_state.params, training_state.hi_actor_target_params, tau),
        lambda: training_state.hi_actor_target_params,
    )
    new_hi_critic_target = jax.lax.cond(
        do_update,
        lambda: _soft_update(new_hi_critic_state.params, training_state.hi_critic_target_params, tau),
        lambda: training_state.hi_critic_target_params,
    )

    new_ts = training_state.replace(
        hi_actor_state=new_hi_actor_state,
        hi_actor_target_params=new_hi_actor_target,
        hi_critic_state=new_hi_critic_state,
        hi_critic_target_params=new_hi_critic_target,
    )
    metrics = {**c_metrics, **a_metrics}
    return new_ts, metrics
