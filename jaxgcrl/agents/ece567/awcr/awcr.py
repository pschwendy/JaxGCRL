"""Advantage-Weighted Contrastive Relabeling (AWCR).

CRL with one targeted change: the HER positive goal is sampled from a
Boltzmann distribution weighted by Monte Carlo returns G_t, rather than
discount-weighted uniform.

  P(f=k | t) ∝ γ^(k-t) · exp(τ · G_k / max_G)

where G_k = Σ_{l≥0} γ^l · r_{k+l} is computed from stored rewards
(no value network needed).

Key properties
--------------
- At τ=0 the weights are all 1 → identical to CRL.  τ is linearly
  annealed from 0 to adv_temp_final over the full training run so the
  agent starts as CRL and gradually prefers breakthrough states.
- When all rewards are 0 (no goal reached this trajectory), max_G≈0
  and all weights are 1 → pure discount sampling, same as CRL. ✓
- Weights are bounded in [1, exp(adv_temp_final)] so Boltzmann never
  collapses regardless of reward scale.
- Networks and losses are identical to CRL — AWCR is a drop-in
  replacement that differs only in flatten_batch.
"""

import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import update_actor_and_alpha, update_critic
from .networks import Actor, Encoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Identical to CRL's TrainingState."""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        return pickle.loads(fin.read())


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class AWCR:
    """Advantage-Weighted Contrastive Relabeling agent.

    Identical to CRL except that HER goal selection is biased toward
    high-return (breakthrough) future states via a Boltzmann distribution
    over Monte Carlo returns.
    """

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62

    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = False

    repr_dim: int = 64

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # Boltzmann temperature for advantage-weighted HER.
    # τ is annealed from 0 → adv_temp_final over the full training run.
    # At τ=0 sampling is identical to CRL (pure discount-weighted).
    adv_temp_final: float = 2.0

    # target entropy = -target_entropy_coeff * action_size
    target_entropy_coeff: float = 0.5

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = int(np.ceil(
            (config.total_env_steps - num_prefill_env_steps)
            / (config.num_evals * env_steps_per_actor_step)
        ))
        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_key, env_key = jax.random.split(key, 4)
        key, actor_key, sa_key, g_key = jax.random.split(key, 4)

        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        goal_indices = train_env.goal_indices
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size

        target_entropy = -self.target_entropy_coeff * action_size

        # ---- Networks (identical to CRL) ----
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )

        # ---- Train states (identical to CRL) ----
        sa_params = sa_encoder.init(sa_key, np.ones([1, state_size + action_size]))
        g_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_params, "g_encoder": g_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        # ---- Replay buffer (identical to CRL) ----
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        # ---- Environment interaction (identical to CRL) ----
        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, env, env_state, key, extra_fields):
            means, log_stds = actor.apply(actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(
                means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        @jax.jit
        def get_experience(actor_state, env_state, buffer_state, key):
            def f(carry, _):
                env_state, k = carry
                k, step_key = jax.random.split(k)
                env_state, transition = actor_step(
                    actor_state, train_env, env_state, step_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, step_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            def f(carry, _):
                ts, es, bs, k = carry
                k, step_key = jax.random.split(k)
                es, bs = get_experience(ts.actor_state, es, bs, step_key)
                ts = ts.replace(env_steps=ts.env_steps + env_steps_per_actor_step)
                return (ts, es, bs, k), ()

            return jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps
            )[0]

        # ---- Advantage-weighted flatten_batch ----
        # The ONLY difference from CRL: Boltzmann weights based on MC returns.
        #
        # G_t = Σ_{l≥0} γ^l r_{t+l}  (computed from stored rewards, no V needed)
        # P(f=k | t) ∝ γ^(k-t) · exp(τ · G_k / (max_G + ε))
        #
        # τ is annealed 0 → adv_temp_final over the training run via env_steps.
        # At τ=0: weights=1 → identical to CRL.
        # When all r=0: max_G≈0 → G/max_G=0 → weights=1 → identical to CRL.
        # Weights bounded in [1, exp(adv_temp_final)] — cannot collapse.

        total_env_steps = float(config.total_env_steps)
        adv_temp_final = self.adv_temp_final
        discounting = self.discounting

        @functools.partial(jax.jit, static_argnames=("buffer_config",))
        def flatten_batch(buffer_config, env_steps, transition, sample_key):
            gamma, state_size_b, goal_indices_b = buffer_config

            seq_len = transition.observation.shape[0]
            arrangement = jnp.arange(seq_len)

            # CRL discount matrix (proximity bias)
            discount = gamma ** jnp.array(
                arrangement[None] - arrangement[:, None], dtype=jnp.float32
            )  # (seq_len, seq_len)

            # Same-trajectory future mask (identical to CRL)
            is_future_mask = jnp.array(
                arrangement[:, None] < arrangement[None], dtype=jnp.float32
            )
            single_trajectories = jnp.concatenate(
                [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
                axis=0,
            )
            same_traj = jnp.equal(single_trajectories, single_trajectories.T)

            # Monte Carlo returns G_t = r_t + γ·r_{t+1} + ...
            # Backward scan: G_t = r_t + γ·discount_t·G_{t+1}
            rewards = transition.reward    # (seq_len,)
            dones = transition.discount    # (seq_len,) = 1 - done

            def return_step(carry, x):
                r, d = x
                ret = r + gamma * d * carry
                return ret, ret

            _, returns = jax.lax.scan(
                return_step,
                jnp.zeros(()),
                jnp.stack([rewards, dones], axis=-1)[::-1],
            )
            returns = returns[::-1]  # (seq_len,)  G_t

            # Normalize by trajectory max so weights stay in [1, exp(adv_temp_final)].
            # When max_G≈0 (no reward this trajectory), returns_norm≈0 → weights=1.
            max_return = jnp.max(returns) + 1e-6
            returns_norm = returns / max_return  # ∈ [0, 1]

            # Anneal τ based on env_steps fraction of total training.
            # τ=0 at start (CRL-equivalent) → adv_temp_final at end.
            tau = adv_temp_final * jnp.clip(env_steps / total_env_steps, 0.0, 1.0)

            # Boltzmann advantage weights indexed over future states (columns)
            adv_weights = jnp.exp(tau * returns_norm)  # (seq_len,)

            # Final probs: CRL discount × Boltzmann advantage × future-only × same-traj
            probs = (
                is_future_mask * discount * adv_weights[None, :] * same_traj
                + jnp.eye(seq_len) * 1e-5
            )

            goal_index = jax.random.categorical(sample_key, jnp.log(probs))

            future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
            future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
            goal = future_state[:, list(goal_indices_b)]
            future_state = future_state[:, :state_size_b]
            state = transition.observation[:-1, :state_size_b]
            new_obs = jnp.concatenate([state, goal], axis=1)

            extras = {
                "policy_extras": {},
                "state_extras": {
                    "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
                    "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
                },
                "state": state,
                "future_state": future_state,
                "future_action": future_action,
            }

            return transition._replace(
                observation=jnp.squeeze(new_obs),
                action=jnp.squeeze(transition.action[:-1]),
                reward=jnp.squeeze(transition.reward[:-1]),
                discount=jnp.squeeze(transition.discount[:-1]),
                extras=extras,
            )

        # ---- Network updates (identical to CRL) ----
        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=goal_indices,
                target_entropy=target_entropy,
            )
            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
            )

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)
            return (training_state, key), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            env_state, buffer_state = get_experience(
                training_state.actor_state, env_state, buffer_state, experience_key1
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )

            buffer_state, transitions = replay_buffer.sample(buffer_state)

            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, None, 0, 0))(
                (self.discounting, state_size, tuple(goal_indices)),
                training_state.env_steps,
                transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, training_key), transitions
            )
            return (training_state, env_state, buffer_state), metrics

        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key):
            def f(carry, _):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (), length=num_training_steps_per_epoch
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        # ---- Init ----
        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_key,
        )

        # ---- Training loop (identical to CRL) ----
        training_walltime = 0
        logging.info("starting AWCR training...")
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps
        logging.info("total steps: %s", total_steps)

        params = (
            training_state.alpha_state.params,
            training_state.actor_state.params,
            training_state.critic_state.params,
        )
        return make_policy, params, metrics
