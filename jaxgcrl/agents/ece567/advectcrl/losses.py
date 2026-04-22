"""AdvectCRL losses.

Three independent update steps per gradient step:

1. update_contrastive_critic
   Loss: InfoNCE(φ_sa(cat(s_t, a_t)),  ψ(g))
   Updates: sa_state (φ_sa) and g_enc_state (ψ) jointly.

2. update_cnf
   Loss: flow-matching  ||v_θ(z_τ, z_1, τ) - (z_1 - z_0)||²
         where z_0 = sg(ψ(s_t[goal_indices])),
               z_1 = sg(ψ(s_future[goal_indices]))
   Updates: cnf_state (v_θ only; ψ frozen here).

3. update_actor_and_alpha
   Loss: E[α log π(a|s,z_target) - φ_sa(cat(s, a)) · z_target]
         where z_target = ψ(s_future[goal_indices])  (NO stop_gradient on ψ)
   Updates: actor_state (π), g_enc_state (ψ via actor gradient), alpha_state.
   φ_sa is read-only (stop_gradient on sa_state.params).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Contrastive helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Contrastive critic
# ---------------------------------------------------------------------------

def update_contrastive_critic(config, networks, transitions, training_state):
    """InfoNCE loss updating φ_sa (sa_state) and ψ (g_enc_state) jointly."""

    def critic_loss_fn(sa_params, g_enc_params):
        state_t = transitions.extras["state"]
        action_t = transitions.action
        goal = transitions.observation[:, config["state_size"]:]

        sa_repr = networks["sa_encoder"].apply(
            sa_params,
            jnp.concatenate([state_t, action_t], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(g_enc_params, goal)

        logits = _energy(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss = _infonce(config["contrastive_loss_fn"], logits)

        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp ** 2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return loss, (logsumexp, correct, logits_pos, logits_neg)

    (loss, (logsumexp, correct, logits_pos, logits_neg)), (sa_grad, g_enc_grad) = (
        jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
            training_state.sa_state.params,
            training_state.g_enc_state.params,
        )
    )

    training_state = training_state.replace(
        sa_state=training_state.sa_state.apply_gradients(grads=sa_grad),
        g_enc_state=training_state.g_enc_state.apply_gradients(grads=g_enc_grad),
    )
    return training_state, {
        "critic_loss": loss,
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
    }


# ---------------------------------------------------------------------------
# 2. CNF — conditional flow matching
# ---------------------------------------------------------------------------

def update_cnf(config, networks, transitions, training_state, key):
    """Flow-matching loss for v_θ.  ψ is frozen here (stop_gradient)."""
    goal_indices = list(config["goal_indices"])

    def cnf_loss_fn(cnf_params):
        state_t = transitions.extras["state"]
        future_state = transitions.extras["future_state"]

        g_enc_params = jax.lax.stop_gradient(training_state.g_enc_state.params)
        z_0 = networks["g_encoder"].apply(g_enc_params, state_t[:, goal_indices])
        z_1 = networks["g_encoder"].apply(g_enc_params, future_state[:, goal_indices])

        tau = jax.random.uniform(key, (z_0.shape[0],))
        tau_bc = tau[:, None]

        z_tau = (1.0 - tau_bc) * z_0 + tau_bc * z_1
        u = z_1 - z_0

        cnf_input = jnp.concatenate([z_tau, z_1, tau_bc], axis=-1)
        v_pred = networks["cnf"].apply(cnf_params["cnf"], cnf_input)

        return jnp.mean(jnp.sum((v_pred - u) ** 2, axis=-1))

    loss, grad = jax.value_and_grad(cnf_loss_fn)(training_state.cnf_state.params)
    training_state = training_state.replace(
        cnf_state=training_state.cnf_state.apply_gradients(grads=grad)
    )
    return training_state, {"cnf_loss": loss}


# ---------------------------------------------------------------------------
# 3. Actor + alpha — entropy-regularised, no sg on ψ
# ---------------------------------------------------------------------------

def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC-style actor update.

    z_target = ψ(s_future[goal_indices]) — gradients flow back into ψ.
    φ_sa is read-only (stop_gradient on sa_state.params).
    α is updated separately to maintain target entropy.
    """
    goal_indices = list(config["goal_indices"])

    def actor_loss_fn(actor_params, g_enc_params, log_alpha):
        state_t = transitions.extras["state"]
        future_state = transitions.extras["future_state"]

        # z_target via ψ — no stop_gradient, gradients flow into g_enc_params
        z_target = networks["g_encoder"].apply(
            g_enc_params, future_state[:, goal_indices]
        )  # (B, repr_dim)

        # Sample action with reparameterisation
        actor_input = jnp.concatenate([state_t, z_target], axis=-1)
        means, log_stds = networks["actor"].apply(actor_params, actor_input)
        stds = jnp.exp(log_stds)
        x_t = means + stds * jax.random.normal(key, means.shape, dtype=means.dtype)
        action = nn.tanh(x_t)

        # Tanh-corrected log probability
        log_prob = jax.scipy.stats.norm.logpdf(x_t, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # (B,)

        # Q-value — φ_sa is frozen
        sa_params = jax.lax.stop_gradient(training_state.sa_state.params)
        sa_repr = networks["sa_encoder"].apply(
            sa_params,
            jnp.concatenate([state_t, action], axis=-1),
        )
        q = jnp.sum(sa_repr * z_target, axis=-1)  # (B,)

        alpha = jnp.exp(log_alpha)
        loss = jnp.mean(alpha * log_prob - q)
        return loss, log_prob

    (actor_loss, log_prob), (actor_grad, g_enc_grad) = jax.value_and_grad(
        actor_loss_fn, argnums=(0, 1), has_aux=True
    )(
        training_state.actor_state.params,
        training_state.g_enc_state.params,
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
        g_enc_state=training_state.g_enc_state.apply_gradients(grads=g_enc_grad),
        alpha_state=training_state.alpha_state.apply_gradients(grads=alpha_grad),
    )
    return training_state, {
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "entropy": -log_prob.mean(),
    }
