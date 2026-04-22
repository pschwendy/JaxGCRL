"""PlanCRL losses.

Three independent update steps per gradient step:

1. update_dynamics
   Loss: MSE  |f(φ_s(s_t), a_t) - sg(φ_s(s_{t+1}))|²
   Updates: encoder_dynamics_state  (φ_s and f jointly)

2. update_contrastive_critic
   Loss: InfoNCE(φ_sa(sg(z_t), a_t),  ψ(g))
   Updates: critic_state  (φ_sa and ψ)
   Note: z_t = sg(φ_s(s_t)) — encoder is frozen here to decouple losses.

3. update_actor_and_alpha
   Loss: E[α log π(a|z) - φ_sa(z, a)·ψ(g)]  (SAC-style)
   α update: standard automatic entropy tuning
   Updates: actor_state, alpha_state
   Note: z = sg(φ_s(s)), critic params are read-only.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Contrastive helpers (same API as ece567/losses.py)
# ---------------------------------------------------------------------------

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
# 1. Dynamics + encoder update
# ---------------------------------------------------------------------------

def update_dynamics(config, networks, transitions, training_state):
    """Train φ_s and f with one-step latent-space prediction loss."""

    def dynamics_loss_fn(enc_dyn_params):
        state_t = transitions.extras["state"]          # (B, state_size)
        next_state_t = transitions.extras["next_state"]  # (B, state_size)
        action_t = transitions.action                   # (B, action_size)

        z_t = networks["state_encoder"].apply(enc_dyn_params["state_encoder"], state_t)
        z_next_pred = networks["dynamics"].apply(
            enc_dyn_params["dynamics"],
            jnp.concatenate([z_t, action_t], axis=-1),
        )

        # Stop-gradient on target so only the prediction side moves
        z_next_target = jax.lax.stop_gradient(
            networks["state_encoder"].apply(enc_dyn_params["state_encoder"], next_state_t)
        )

        loss = jnp.mean(jnp.sum((z_next_pred - z_next_target) ** 2, axis=-1))
        return loss

    loss, grad = jax.value_and_grad(dynamics_loss_fn)(
        training_state.encoder_dynamics_state.params
    )
    new_enc_dyn_state = training_state.encoder_dynamics_state.apply_gradients(grads=grad)
    training_state = training_state.replace(encoder_dynamics_state=new_enc_dyn_state)

    return training_state, {"dynamics_loss": loss}


# ---------------------------------------------------------------------------
# 2. Contrastive critic update
# ---------------------------------------------------------------------------

def update_contrastive_critic(config, networks, transitions, training_state):
    """Train φ_sa and ψ with InfoNCE.  The state encoder φ_s is frozen."""

    def critic_loss_fn(critic_params):
        state_t = transitions.extras["state"]   # (B, state_size)
        action_t = transitions.action            # (B, action_size)
        # Goal comes from the goal portion of the observation (state_size : end)
        goal = transitions.observation[:, config["state_size"]:]  # (B, goal_size)

        # Freeze state encoder – gradients do NOT flow back to φ_s from here
        z_t = jax.lax.stop_gradient(
            networks["state_encoder"].apply(
                training_state.encoder_dynamics_state.params["state_encoder"], state_t
            )
        )

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([z_t, action_t], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal)

        # Pairwise energy matrix for InfoNCE
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # Logsumexp regularisation
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

    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    return training_state, {
        "critic_loss": loss,
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
    }


# ---------------------------------------------------------------------------
# 3. Actor + alpha update
# ---------------------------------------------------------------------------

def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC-style actor update using contrastive Q-values.

    The actor sees only the latent state z = sg(φ_s(s)).
    Q(s, a, g) = φ_sa(z, a) · ψ(g).
    """

    def actor_loss_fn(actor_params, log_alpha, key):
        state_t = transitions.extras["state"]       # (B, state_size)
        future_state = transitions.extras["future_state"]  # (B, state_size)
        goal = future_state[:, config["goal_indices"]]     # (B, goal_size)

        # Latent state — frozen encoder, no gradients back to φ_s
        z_t = jax.lax.stop_gradient(
            networks["state_encoder"].apply(
                training_state.encoder_dynamics_state.params["state_encoder"], state_t
            )
        )

        # Goal-conditioned actor input: [z(s), g]
        actor_input = jnp.concatenate([z_t, goal], axis=-1)

        # Sample action from policy
        means, log_stds = networks["actor"].apply(actor_params, actor_input)
        stds = jnp.exp(log_stds)
        x_t = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_t)

        # Tanh-corrected log probability
        log_prob = jax.scipy.stats.norm.logpdf(x_t, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # (B,)

        # Contrastive Q-value (read-only critic params)
        critic_params = jax.lax.stop_gradient(training_state.critic_state.params)
        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([z_t, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal)
        q = jnp.sum(sa_repr * g_repr, axis=-1)  # (B,)  dot-product Q-value

        alpha = jnp.exp(log_alpha)
        loss = jnp.mean(alpha * log_prob - q)
        return loss, log_prob

    def alpha_loss_fn(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(loss)

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        training_state.actor_state.params,
        training_state.alpha_state.params["log_alpha"],
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        training_state.alpha_state.params, log_prob
    )
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        actor_state=new_actor_state, alpha_state=new_alpha_state
    )

    return training_state, {
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "entropy": -log_prob.mean(),
    }
