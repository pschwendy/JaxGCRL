"""SCC-RL losses.

Three independent update steps per gradient step:

1. update_cvae
   ELBO loss for the Geometric Subgoal Sampler f(z | s, g):
     L_f = E[ MSE(p_ξ(h, s, g), z) + β * KL(q_θ(h|s,z,g) || N(0,I)) ]
   Updates: cvae_state  (CVAEEncoder + CVAEDecoder jointly)

2. update_actor_and_alpha
   SAC-style maximum-entropy actor, using a *mixed* goal:
     g̃ = z ~ f(z|s,g)  with probability alpha_subgoal
     g̃ = g             otherwise
   J(π) = E[ φ(s,a)^T ψ(g̃) − α log π(a|s,g̃) ]
   Updates: actor_state, alpha_state

3. update_critic
   Forward InfoNCE contrastive loss (identical to CRL):
     L_critic = InfoNCE(φ(s,a), ψ(g))
   Updates: critic_state
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared helpers (same as CRL ece567/losses.py)
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
# 1. CVAE update (Geometric Subgoal Sampler)
# ---------------------------------------------------------------------------

def update_cvae(config, networks, transitions, training_state, key):
    """Train the CVAE (Geometric Subgoal Sampler) with the full objective:

        L_f_total = L_CVAE  −  λ · E_{z ~ p_ξ}[ ψ(z)^T ψ(g) ]

    where L_CVAE is the standard ELBO (reconstruction + β·KL) and the second
    term maximises the alignment of the generated subgoal z with the far goal g
    in the contrastive critic's latent space.  Gradients from the alignment term
    flow only into the CVAE decoder; the goal encoder ψ is frozen via
    stop_gradient so the critic is not inadvertently updated here.

    Samples are filtered to only train on subgoals that improve on the current
    (s, a) pair in the critic's latent space:
        ψ(subgoal)^T ψ(g) > φ(s_t, a_t)^T ψ(g) + margin
    The ELBO is computed as a weighted mean (weight=1 if improving, 0 otherwise).
    """

    def cvae_loss_fn(cvae_params, key):
        elbo_key, align_key = jax.random.split(key)

        encoder_params = cvae_params["cvae_encoder"]
        decoder_params = cvae_params["cvae_decoder"]

        state = transitions.extras["state"]      # (B, state_size)
        subgoal = transitions.extras["subgoal"]  # (B, goal_size)  — intermediate
        goal = transitions.observation[:, config["state_size"]:]  # (B, goal_size)
        action = transitions.action              # (B, action_size)

        # --- Improvement filter (frozen critic, no gradients here) ---
        g_enc_frozen = jax.lax.stop_gradient(training_state.critic_state.params["g_encoder"])
        sa_enc_frozen = jax.lax.stop_gradient(training_state.critic_state.params["sa_encoder"])

        psi_g = jax.lax.stop_gradient(
            networks["g_encoder"].apply(g_enc_frozen, goal)         # (B, repr_dim)
        )
        psi_sub = jax.lax.stop_gradient(
            networks["g_encoder"].apply(g_enc_frozen, subgoal)      # (B, repr_dim)
        )
        phi_sa = jax.lax.stop_gradient(
            networks["sa_encoder"].apply(
                sa_enc_frozen, jnp.concatenate([state, action], axis=-1)
            )                                                        # (B, repr_dim)
        )

        q_subgoal = jnp.sum(psi_sub * psi_g, axis=-1)               # (B,)
        q_current = jnp.sum(phi_sa  * psi_g, axis=-1)               # (B,)
        improvement_mask = (
            q_subgoal > q_current + config["cvae_improvement_margin"]
        ).astype(jnp.float32)                                        # (B,)
        denom = jnp.sum(improvement_mask) + 1e-8

        # --- Encode: q_θ(h | s_t, z, g) ---
        mu_h, log_var_h = networks["cvae_encoder"].apply(
            encoder_params, state, subgoal, goal
        )

        # --- Reparameterise ---
        eps = jax.random.normal(elbo_key, shape=mu_h.shape)
        h = mu_h + jnp.exp(0.5 * log_var_h) * eps

        # --- Decode: p_ξ(z_pred | h, s_t, g) ---
        z_pred = networks["cvae_decoder"].apply(decoder_params, h, state, goal)

        # Reconstruction: weighted MSE (only improving subgoals)
        recon_per = jnp.sum((z_pred - subgoal) ** 2, axis=-1)       # (B,)
        recon_loss = jnp.sum(improvement_mask * recon_per) / denom

        # KL divergence: KL(N(μ, σ²) || N(0, 1)), weighted
        kl_per = 0.5 * jnp.sum(mu_h ** 2 + jnp.exp(log_var_h) - 1.0 - log_var_h, axis=-1)
        kl_loss = jnp.sum(improvement_mask * kl_per) / denom

        elbo = recon_loss + config["cvae_beta"] * kl_loss

        # --- Alignment: E_{h~N(0,I)}[ ψ(z)^T ψ(g) ] ---
        # Sample from the prior to evaluate how well the *generative* distribution
        # aligns with the goal, independently of the ELBO sample above.
        # --- Alignment: E_{h~N(0,I)}[ ψ(z)^T ψ(g) ] (reuse frozen psi_g from above) ---
        h_prior = jax.random.normal(align_key, shape=mu_h.shape)
        z_prior = networks["cvae_decoder"].apply(decoder_params, h_prior, state, goal)
        # psi_g already computed and frozen above; gradients flow into decoder only
        psi_z = networks["g_encoder"].apply(g_enc_frozen, z_prior)           # (B, repr_dim)
        alignment = jnp.mean(jnp.sum(psi_z * psi_g, axis=-1))               # scalar

        total_loss = elbo - config["cvae_alignment_coeff"] * alignment
        mask_frac = jnp.mean(improvement_mask)
        return total_loss, (recon_loss, kl_loss, elbo, alignment, mask_frac)

    (loss, (recon_loss, kl_loss, elbo, alignment, mask_frac)), grad = jax.value_and_grad(
        cvae_loss_fn, has_aux=True
    )(training_state.cvae_state.params, key)
    new_cvae_state = training_state.cvae_state.apply_gradients(grads=grad)
    training_state = training_state.replace(cvae_state=new_cvae_state)

    return training_state, {
        "cvae_loss": loss,
        "cvae_elbo": elbo,
        "cvae_recon_loss": recon_loss,
        "cvae_kl_loss": kl_loss,
        "cvae_alignment": alignment,
        "cvae_improvement_frac": mask_frac,
    }


# ---------------------------------------------------------------------------
# 2. Actor + alpha update (subgoal-conditioned)
# ---------------------------------------------------------------------------

def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC-style actor update with mixed true-goal / CVAE-subgoal conditioning.

    With probability alpha_subgoal we replace the far goal g with a subgoal
    z sampled from the CVAE prior: h ~ N(0,I), z = p_ξ(h, s, g).
    Gradients do NOT flow back to the CVAE through this path.
    """
    key, cvae_key, mix_key, actor_key = jax.random.split(key, 4)

    state = transitions.observation[:, : config["state_size"]]  # (B, state_size)
    future_state = transitions.extras["future_state"]            # (B, state_size)
    true_goal = future_state[:, config["goal_indices"]]          # (B, goal_size)

    # Sample subgoal z from CVAE prior (frozen — no gradient back to CVAE)
    h = jax.random.normal(cvae_key, (state.shape[0], config["cvae_latent_dim"]))
    z_subgoal = jax.lax.stop_gradient(
        networks["cvae_decoder"].apply(
            training_state.cvae_state.params["cvae_decoder"], h, state, true_goal
        )
    )  # (B, goal_size)

    # Bernoulli mask: use CVAE subgoal with probability alpha_subgoal
    use_subgoal = jax.random.bernoulli(
        mix_key, p=config["alpha_subgoal"], shape=(state.shape[0],)
    )
    goal_tilde = jnp.where(use_subgoal[:, None], z_subgoal, true_goal)  # (B, goal_size)

    def actor_loss(actor_params, critic_params, log_alpha, key):
        observation = jnp.concatenate([state, goal_tilde], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)

        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # (B,)

        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal_tilde)
        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
        return loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        return jnp.mean(
            alpha * jax.lax.stop_gradient(-log_prob - config["target_entropy"])
        )

    (actor_loss_val, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        actor_key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss)(
        training_state.alpha_state.params, log_prob
    )
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        actor_state=new_actor_state, alpha_state=new_alpha_state
    )

    return training_state, {
        "entropy": -log_prob.mean(),
        "actor_loss": actor_loss_val,
        "alpha_loss": alpha_loss_val,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "subgoal_fraction": use_subgoal.mean(),
    }


# ---------------------------------------------------------------------------
# 3. Critic update (InfoNCE — identical to CRL)
# ---------------------------------------------------------------------------

def update_critic(config, networks, transitions, training_state, key):
    """Contrastive critic update with forward InfoNCE, identical to CRL."""

    def critic_loss(critic_params, transitions, key):
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1)
        )
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, config["state_size"]:]
        )

        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp ** 2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)

    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    return training_state, {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }
