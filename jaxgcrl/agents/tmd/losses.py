import flax.linen as nn
import jax
import jax.numpy as jnp


def mrn_distance(x, y, K):
    """MRN (Metric Residual Network) quasimetric distance.

    Splits repr into K components. Each component uses max(relu(.)) on the first
    half of dims and L2 on the second half. Returns the mean over K components.

    This is a valid quasimetric: d(x,x)=0, d>=0, triangle inequality holds,
    but d(x,y) != d(y,x) in general.
    """
    def component(xk, yk):
        d = xk.shape[-1]
        mask = jnp.arange(d) < d // 2
        max_part = jnp.max(jax.nn.relu((xk - yk) * mask), axis=-1)
        l2_part = jnp.sqrt(jnp.sum(jnp.square((xk - yk) * (1 - mask)), axis=-1) + 1e-6)
        return max_part + l2_part

    # [..., repr_dim] -> [..., repr_dim//K, K]
    x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
    y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
    dists = jax.vmap(component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)
    return dists.mean(axis=-1)


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    K = config["mrn_components"]

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

        dist = mrn_distance(sa_repr, g_repr, K)
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
    K = config["mrn_components"]

    def critic_loss_fn(critic_params, transitions):
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        state = transitions.extras["state"]
        action = transitions.action
        next_state = transitions.extras["next_state"]
        goal = transitions.observation[:, config["state_size"]:]  # relabeled future goal

        phi = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1)
        )
        psi_s = networks["g_encoder"].apply(
            g_encoder_params, state[:, config["goal_indices"]]
        )
        psi_next = networks["g_encoder"].apply(
            g_encoder_params, next_state[:, config["goal_indices"]]
        )
        psi_g = networks["g_encoder"].apply(g_encoder_params, goal)

        # ── Contrastive (symmetric InfoNCE) loss ──────────────────────────────
        # dist_matrix[i,j] = MRN_distance(phi_i, psi_g_j)
        dist_matrix = mrn_distance(phi[:, None, :], psi_g[None, :, :], K)  # [B, B]
        logits = -dist_matrix / jnp.sqrt(phi.shape[-1])
        contrastive_loss = -jnp.mean(
            2 * jnp.diag(logits)
            - jax.nn.logsumexp(logits, axis=1)
            - jax.nn.logsumexp(logits, axis=0)
        )

        # ── Action invariance: d(ψ(s), φ(s,a)) → 0 ───────────────────────────
        # Diagonal only: each (s_i, a_i) pair from the offline dataset.
        inv_loss = jnp.mean(mrn_distance(psi_s, phi, K))

        # ── Temporal backup (LINEX divergence) ────────────────────────────────
        # Encourages: dist[i,j] ≈ dist_next[i,j] + |log(gamma)|
        # = d(phi(s_i,a_i), psi_g_j) ≈ d(psi(s'_i), psi_g_j) + |log(gamma)|
        dist_next_matrix = jax.lax.stop_gradient(
            mrn_distance(psi_next[:, None, :], psi_g[None, :, :], K)  # [B, B]
        )
        gamma = config["discounting"]
        t = config["linex_t"]
        delta = dist_matrix - dist_next_matrix
        mask = delta > t
        delta_clipped = jnp.where(mask, t, delta)
        # Linear for large delta (prevents gradient explosion), LINEX for small delta.
        # Optimum is at delta = -log(gamma) = |log(gamma)|.
        divergence = jnp.where(mask, delta, gamma * jnp.exp(delta_clipped) - dist_matrix)

        # Blend full-matrix backup with diagonal-only backup (diag_backup in [0,1]).
        dw = config["diag_backup"]
        divergence = (1 - dw) * divergence + dw * jnp.diag(divergence)[:, None]
        backup_loss = jnp.mean(divergence)

        zeta = config["linex_zeta"]
        total_loss = contrastive_loss + zeta * (inv_loss + backup_loss)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return total_loss, (contrastive_loss, inv_loss, backup_loss, correct, logits_pos, logits_neg)

    (loss, (contrastive_loss, inv_loss, backup_loss, correct, logits_pos, logits_neg)), grad = (
        jax.value_and_grad(critic_loss_fn, has_aux=True)(
            training_state.critic_state.params, transitions
        )
    )
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    return training_state, {
        "critic_loss": loss,
        "contrastive_loss": contrastive_loss,
        "inv_loss": inv_loss,
        "backup_loss": backup_loss,
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
    }
