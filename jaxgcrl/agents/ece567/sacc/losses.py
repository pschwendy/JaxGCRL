"""SACC losses: contrastive critic (CRL-style) + entropy-regularized actor (SAC-style).

The critic is trained with InfoNCE contrastive loss, exactly as in CRL.
The actor always maximises expected contrastive Q-value minus entropy:

    L_actor = E[α log π(a|s,g) - Q_contrastive(s, a, g)]

where Q_contrastive(s, a, g) = energy_fn(φ(s,a), ψ(g)).

The entropy coefficient α is adapted automatically via:

    L_alpha = α * E[-log π(a|s,g) - H_target]
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
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        return -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """SAC-style maximum-entropy actor update using contrastive Q-values."""

    def actor_loss_fn(actor_params, critic_params, log_alpha, transitions, key):
        state = transitions.observation[:, : config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        # Sample action from current policy
        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)

        # Compute log probability (with tanh correction)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # shape: (B,)

        # Contrastive Q-value: energy between (s, a) and g representations
        sa_repr = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = networks["g_encoder"].apply(critic_params["g_encoder"], goal)
        q_contrastive = energy_fn(config["energy_fn"], sa_repr, g_repr)

        # SAC actor loss: minimise α * log π(a|s,g) - Q(s,a,g)
        alpha = jnp.exp(log_alpha)
        loss = jnp.mean(alpha * log_prob - q_contrastive)

        return loss, log_prob

    def alpha_loss_fn(alpha_params, log_prob):
        """Automatic entropy tuning: push entropy towards target."""
        alpha = jnp.exp(alpha_params["log_alpha"])
        loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(loss)

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
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

    metrics = {
        "entropy": -log_prob.mean(),
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }
    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    """Contrastive critic update (InfoNCE), identical to CRL."""

    def critic_loss_fn(critic_params, transitions, key):
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(
            sa_encoder_params, jnp.concatenate([state, action], axis=-1)
        )
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, config["state_size"] :]
        )

        # Pairwise energy matrix: logits[i,j] = energy(φ(s_i,a_i), ψ(g_j))
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # Logsumexp regularisation (penalises large off-diagonal energies)
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp ** 2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return loss, (logsumexp, correct, logits_pos, logits_neg)

    (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(training_state.critic_state.params, transitions, key)

    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }
    return training_state, metrics
