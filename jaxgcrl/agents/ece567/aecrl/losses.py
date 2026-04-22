"""AE-CRL losses.

AE-CRL = CRL InfoNCE critic  +  Bellman TD loss on a learned asymmetric energy.

The key addition over CRL is a Bellman consistency loss:

    L_TD = E[ (V_θ(s_t, g) - γ · V_θ̄(s_{t+1}, g))² ]

where:
    E_θ(s, g)  = energy_head(s_encoder(s), g_encoder(g))   [learned, asymmetric, ≥ 0]
    V_θ(s, g)  = exp(-E_θ(s, g))                           [∈ (0, 1]]

energy_head is a small MLP: concat(s_repr, g_repr) → softplus(scalar).
It is asymmetric because the input order matters and the MLP has no symmetry constraint.
It is LEARNED — all parameters (s_encoder, g_encoder, energy_head) are jointly optimised.

The target network θ̄ is a slow EMA copy of (s_encoder, g_encoder, energy_head).
The actor loss is identical to CRL (uses sa_encoder for Q gradient).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers (mirror CRL)
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
# Critic update
# ---------------------------------------------------------------------------

def update_critic(config, networks, transitions, training_state, key):
    """InfoNCE + λ_td · Bellman TD loss.

    Critic params: {sa_encoder, g_encoder, s_encoder, energy_head}.
    Target params (EMA, stop-grad): {s_encoder, g_encoder, energy_head}.
    """

    def critic_loss_fn(critic_params):
        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action
        goal = transitions.observation[:, config["state_size"] :]  # HER goal
        next_state = transitions.extras["next_state"]              # s_{t+1}

        # ------------------------------------------------------------------
        # 1. InfoNCE loss (identical to CRL)
        # ------------------------------------------------------------------
        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal)

        logits = _energy(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        infonce_loss = _infonce(config["contrastive_loss_fn"], logits)

        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        infonce_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp ** 2)

        # ------------------------------------------------------------------
        # 2. Bellman TD loss on V_θ(s, g) = exp(-E_θ(s, g))
        #
        # E_θ(s, g) = energy_head(s_encoder(s), g_encoder(g))
        #           = softplus(MLP(concat(s_repr, g_repr)))  ≥ 0
        #
        # This is a LEARNED, ASYMMETRIC energy — not just L2 distance:
        #   • Learned: energy_head, s_encoder, g_encoder all have trainable params.
        #   • Asymmetric: concat(s_repr, g_repr) ≠ concat(g_repr, s_repr) in general.
        # ------------------------------------------------------------------
        s_repr = networks["s_encoder"].apply(critic_params["s_encoder"], state)
        # g_repr reused from InfoNCE (same g_encoder, same goal)
        E_current = networks["energy_head"].apply(
            critic_params["energy_head"], s_repr, g_repr
        )  # (B,), ≥ 0 via softplus
        V_current = jnp.exp(-E_current)  # (B,), in (0, 1]

        # Target network: stop gradient so it acts as a fixed bootstrap target
        target_params = jax.lax.stop_gradient(training_state.target_params)
        s_repr_next = networks["s_encoder"].apply(target_params["s_encoder"], next_state)
        g_repr_target = networks["g_encoder"].apply(target_params["g_encoder"], goal)
        E_next = networks["energy_head"].apply(
            target_params["energy_head"], s_repr_next, g_repr_target
        )  # (B,), ≥ 0
        V_next = jnp.exp(-E_next)  # (B,)

        # γ · (1 - done) · V_target(s_{t+1}, g)
        # transitions.discount = 1 - done, so this correctly zeroes out at episode end
        V_td_target = config["discounting"] * transitions.discount * V_next
        td_loss = jnp.mean((V_current - V_td_target) ** 2)

        total_loss = infonce_loss + config["lambda_td"] * td_loss

        # Diagnostic metrics
        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return total_loss, (logsumexp, correct, logits_pos, logits_neg, td_loss, V_current.mean())

    (loss, (logsumexp, correct, logits_pos, logits_neg, td_loss, v_mean)), grad = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(training_state.critic_state.params)

    training_state = training_state.replace(
        critic_state=training_state.critic_state.apply_gradients(grads=grad)
    )
    return training_state, {
        "critic_loss": loss,
        "td_loss": td_loss,
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "V_mean": v_mean,
    }


# ---------------------------------------------------------------------------
# Actor + alpha update  (identical to CRL)
# ---------------------------------------------------------------------------

def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC actor + automatic entropy tuning — identical to CRL."""

    def actor_loss_fn(actor_params, log_alpha):
        obs = transitions.observation
        state = obs[:, : config["state_size"]]
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
