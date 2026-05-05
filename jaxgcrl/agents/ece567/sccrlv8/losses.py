"""SCC-RL v8 losses: latent-space subgoal generation.

Key differences from v5:
  - update_cvae: reconstruction target is ψ(subgoal) in repr space (not subgoal_obs).
    CVAE encoder receives (state, ψ(subgoal), ψ(goal)); decoder receives (h, state, ψ(goal)).
    Alignment term uses ψ̃_prior directly — no extra g_encoder pass needed.
  - update_actor_and_alpha: goal_tilde is ψ̃ or ψ(true_goal), both in repr_dim space.
    g_repr in the actor Q-value is goal_tilde directly (no second g_encoder call).
  - update_critic: unchanged — trains on obs-space goals passed through g_encoder.
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
# 1. CVAE update — reconstruction in latent repr space
# ---------------------------------------------------------------------------

def update_cvae(config, networks, transitions, training_state, key):
    """ELBO + alignment loss with latent-space reconstruction target.

    The CVAE reconstructs ψ(subgoal) ∈ ℝ^repr_dim rather than subgoal_obs.
    Improvement filter and warm-up logic identical to v5.
    """

    def cvae_loss_fn(cvae_params, key):
        elbo_key, align_key = jax.random.split(key)

        encoder_params = cvae_params["cvae_encoder"]
        decoder_params = cvae_params["cvae_decoder"]

        state   = transitions.extras["state"]
        subgoal = transitions.extras["subgoal"]   # subgoal_obs in goal space
        goal    = transitions.observation[:, config["state_size"]:]
        action  = transitions.action

        # Freeze critic encoders for improvement filter and reconstruction targets
        g_enc_frozen  = jax.lax.stop_gradient(training_state.critic_state.params["g_encoder"])
        sa_enc_frozen = jax.lax.stop_gradient(training_state.critic_state.params["sa_encoder"])

        psi_g   = jax.lax.stop_gradient(networks["g_encoder"].apply(g_enc_frozen, goal))
        psi_sub = jax.lax.stop_gradient(networks["g_encoder"].apply(g_enc_frozen, subgoal))
        phi_sa  = jax.lax.stop_gradient(
            networks["sa_encoder"].apply(sa_enc_frozen, jnp.concatenate([state, action], axis=-1))
        )

        # Improvement filter: subgoal embedding must improve over current (s,a)
        q_subgoal = jnp.sum(psi_sub * psi_g, axis=-1)
        q_current = jnp.sum(phi_sa  * psi_g, axis=-1)
        improvement_mask = (
            q_subgoal > q_current + config["cvae_improvement_margin"]
        ).astype(jnp.float32)

        # Bypass filter during warm-up
        improvement_mask = jnp.where(
            training_state.gradient_steps < config["cvae_warmup_steps"],
            jnp.ones_like(improvement_mask),
            improvement_mask,
        )
        denom = jnp.sum(improvement_mask) + 1e-8

        # CVAE encode (inputs now in repr space) → reparameterise → decode
        mu_h, log_var_h = networks["cvae_encoder"].apply(encoder_params, state, psi_sub, psi_g)
        log_var_h = jnp.clip(log_var_h, -10.0, 10.0)
        eps    = jax.random.normal(elbo_key, shape=mu_h.shape)
        h      = mu_h + jnp.exp(0.5 * log_var_h) * eps
        psi_pred = networks["cvae_decoder"].apply(decoder_params, h, state, psi_g)

        # Reconstruction in repr space: ||ψ̃_pred - ψ(subgoal)||²
        recon_per  = jnp.sum((psi_pred - psi_sub) ** 2, axis=-1)
        recon_loss = jnp.sum(improvement_mask * recon_per) / denom

        kl_per  = 0.5 * jnp.sum(mu_h ** 2 + jnp.exp(log_var_h) - 1.0 - log_var_h, axis=-1)
        kl_loss = jnp.sum(improvement_mask * kl_per) / denom

        elbo = recon_loss + config["cvae_beta"] * kl_loss

        # Alignment: ψ̃_prior already in repr space — dot directly with ψ(goal)
        h_prior   = jax.random.normal(align_key, shape=mu_h.shape)
        psi_prior = networks["cvae_decoder"].apply(decoder_params, h_prior, state, psi_g)
        alignment = jnp.sum(improvement_mask * jnp.sum(psi_prior * psi_g, axis=-1)) / denom

        total_loss = elbo - config["cvae_alignment_coeff"] * alignment
        return total_loss, (recon_loss, kl_loss, elbo, alignment, jnp.mean(improvement_mask))

    (loss, (recon_loss, kl_loss, elbo, alignment, mask_frac)), grad = jax.value_and_grad(
        cvae_loss_fn, has_aux=True
    )(training_state.cvae_state.params, key)
    training_state = training_state.replace(
        cvae_state=training_state.cvae_state.apply_gradients(grads=grad)
    )
    return training_state, {
        "cvae_loss": loss,
        "cvae_elbo": elbo,
        "cvae_recon_loss": recon_loss,
        "cvae_kl_loss": kl_loss,
        "cvae_alignment": alignment,
        "cvae_improvement_frac": mask_frac,
    }


# ---------------------------------------------------------------------------
# 2. Actor + alpha update — goal_tilde lives in repr space
# ---------------------------------------------------------------------------

def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC actor conditioned on latent subgoal ψ̃ or ψ(true_goal).

    goal_tilde is in repr_dim space; it is used directly as g_repr for the
    Q-value — no second g_encoder call needed.
    """
    key, cvae_key, mix_key, actor_key = jax.random.split(key, 4)

    state        = transitions.observation[:, : config["state_size"]]
    true_goal_obs = transitions.extras["future_state"][:, config["goal_indices"]]

    # Encode true goal into repr space (frozen critic)
    g_enc_frozen = jax.lax.stop_gradient(training_state.critic_state.params["g_encoder"])
    psi_goal = jax.lax.stop_gradient(
        networks["g_encoder"].apply(g_enc_frozen, true_goal_obs)
    )

    # Generate latent subgoal from CVAE prior
    h = jax.random.normal(cvae_key, (state.shape[0], config["cvae_latent_dim"]))
    psi_tilde = jax.lax.stop_gradient(
        networks["cvae_decoder"].apply(
            training_state.cvae_state.params["cvae_decoder"], h, state, psi_goal
        )
    )

    use_subgoal = jax.random.bernoulli(mix_key, p=config["alpha_subgoal"], shape=(state.shape[0],))
    goal_tilde  = jnp.where(use_subgoal[:, None], psi_tilde, psi_goal)  # both in repr_dim

    def actor_loss(actor_params, critic_params, log_alpha, key):
        observation = jnp.concatenate([state, goal_tilde], axis=1)
        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds   = jnp.exp(log_stds)
        x_ts   = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)

        log_prob  = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob  = log_prob.sum(-1)

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"], jnp.concatenate([state, action], axis=-1)
        )
        # goal_tilde is already in repr space — use directly as g_repr
        qf_pi = energy_fn(config["energy_fn"], sa_repr, goal_tilde)

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
# 3. Critic update — unchanged from v5
# ---------------------------------------------------------------------------

def update_critic(config, networks, transitions, training_state, key):
    """InfoNCE critic calibrated on the mixed goal distribution (Fix 1 from v5).

    Trains g_encoder on obs-space goals; unchanged from v5.
    """

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

        I          = jnp.eye(logits.shape[0])
        correct    = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
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
