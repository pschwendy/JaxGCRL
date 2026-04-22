"""SCC-RL v4: Subgoal-Conditioned Contrastive RL with four architectural fixes.

Fixes over v2:
  1. Critic Calibration   — InfoNCE trains on the same mixed goal distribution
                            (subgoal / true-goal) as the actor, so φ and ψ are
                            calibrated for both short- and long-horizon targets.
  2. Temporal Commitment  — During eval, a CVAE subgoal is held for
                            `commitment_horizon` steps before resampling, instead
                            of a per-step coin flip. Avoids zig-zag paths.
  3. Active Exploration   — During rollout collection the actor is conditioned on
                            a CVAE-generated subgoal with prob alpha_subgoal,
                            populating the replay buffer with subgoal-directed
                            behaviour so the CVAE trains on its own policy.
  4. Improvement Filter Warm-up — The CVAE improvement filter is disabled for
                            the first `cvae_warmup_steps` gradient steps while
                            critic embeddings are still noisy.
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
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import update_actor_and_alpha, update_critic, update_cvae
from .networks import Actor, CVAEDecoder, CVAEEncoder, Encoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    cvae_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def _padded_geom_probs(gamma, is_future_mask, same_traj_mask, arrangement, seq_len):
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount * same_traj_mask + jnp.eye(seq_len) * 1e-5
    valid_future = is_future_mask * same_traj_mask
    has_future = jnp.any(valid_future > 0, axis=1)
    last_j = jnp.max(jnp.where(valid_future > 0, arrangement[None, :], 0), axis=1)
    dist_to_last = last_j - arrangement
    tail_mass = jnp.where(has_future, gamma ** (dist_to_last + 1) / (1.0 - gamma), 0.0)
    tail_add = jnp.zeros((seq_len, seq_len)).at[jnp.arange(seq_len), last_j].add(tail_mass)
    return probs + tail_add


@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_batch(buffer_config, transition, sample_key):
    """Sample future goals (discounting) and subgoals (subgoal_discounting) independently."""
    gamma, subgoal_gamma, state_size, goal_indices = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
    )
    same_traj_mask = jnp.equal(single_trajectories, single_trajectories.T)

    probs_goal = _padded_geom_probs(gamma, is_future_mask, same_traj_mask, arrangement, seq_len)
    probs_sub  = _padded_geom_probs(subgoal_gamma, is_future_mask, same_traj_mask, arrangement, seq_len)

    sample_key, subgoal_key = jax.random.split(sample_key)
    idx_a = jax.random.categorical(sample_key, jnp.log(probs_goal))
    idx_b = jax.random.categorical(subgoal_key, jnp.log(probs_sub))

    goal_index    = jnp.maximum(idx_a, idx_b)
    subgoal_index = jnp.minimum(idx_a, idx_b)

    future_state  = jnp.take(transition.observation, goal_index[:-1], axis=0)
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal          = future_state[:, goal_indices]
    future_state  = future_state[:, :state_size]

    subgoal_obs = jnp.take(transition.observation, subgoal_index[:-1], axis=0)
    subgoal     = subgoal_obs[:, goal_indices]

    state   = transition.observation[:-1, :state_size]
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id":    jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state":        state,
        "future_state": future_state,
        "future_action": future_action,
        "subgoal":      subgoal,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        return pickle.loads(fin.read())


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class SCCRLV4:
    """SCC-RL v4: four architectural fixes over v2."""

    # Learning rates
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr:  float = 3e-4
    cvae_lr:   float = 3e-4

    batch_size: int = 256
    discounting:         float = 0.99
    subgoal_discounting: float = 0.9   # shorter horizon for CVAE subgoal targets

    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier:   int   = 1
    disable_entropy_actor:   bool  = False
    target_entropy_coeff:    float = 0.5

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length:   int = 62
    h_dim:           int = 256
    n_hidden:        int = 2
    skip_connections: int = 4
    use_relu:  bool = False
    repr_dim:  int  = 64
    use_ln:    bool = False

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # CVAE hyperparameters
    cvae_latent_dim:         int   = 64
    cvae_beta:               float = 1.0
    cvae_hidden_depth:       int   = 2
    cvae_alignment_coeff:    float = 0.1
    cvae_improvement_margin: float = 0.0

    # Fix 4: gradient steps before improvement filter is applied
    cvae_warmup_steps: int = 10000

    # Mixing probability (actor training, critic training, rollout collection)
    alpha_subgoal: float = 0.5

    # Fix 2: how many eval steps to hold a subgoal before resampling
    commitment_horizon: int = 10

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0

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
            train_env, episode_length=config.episode_length, action_repeat=config.action_repeat
        )
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env, episode_length=config.episode_length, action_repeat=config.action_repeat
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps    = self.min_replay_size * config.num_envs
        num_prefill_actor_steps  = np.ceil(self.min_replay_size / self.unroll_length)
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
        (
            key, buffer_key, eval_env_key, env_key, actor_key,
            sa_key, g_key, cvae_enc_key, cvae_dec_key,
        ) = jax.random.split(key, 9)

        env_keys  = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        action_size = train_env.action_size
        state_size  = train_env.state_dim
        goal_size   = len(train_env.goal_indices)
        obs_size    = state_size + goal_size
        assert obs_size == train_env.observation_size

        # ------------------------------------------------------------------ #
        # Networks                                                             #
        # ------------------------------------------------------------------ #

        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

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
        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder.init(sa_key, np.ones([1, state_size + action_size])),
                "g_encoder":  g_encoder.init(g_key, np.ones([1, goal_size])),
            },
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        cvae_encoder = CVAEEncoder(
            latent_dim=self.cvae_latent_dim,
            network_width=self.h_dim,
            network_depth=self.cvae_hidden_depth,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        cvae_decoder = CVAEDecoder(
            goal_size=goal_size,
            network_width=self.h_dim,
            network_depth=self.cvae_hidden_depth,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        cvae_state = TrainState.create(
            apply_fn=None,
            params={
                "cvae_encoder": cvae_encoder.init(
                    cvae_enc_key, np.ones([1, state_size]), np.ones([1, goal_size]), np.ones([1, goal_size])
                ),
                "cvae_decoder": cvae_decoder.init(
                    cvae_dec_key, np.ones([1, self.cvae_latent_dim]), np.ones([1, state_size]), np.ones([1, goal_size])
                ),
            },
            tx=optax.adam(learning_rate=self.cvae_lr),
        )

        target_entropy = -self.target_entropy_coeff * action_size
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
            cvae_state=cvae_state,
        )

        # ------------------------------------------------------------------ #
        # Replay buffer                                                        #
        # ------------------------------------------------------------------ #

        dummy_transition = Transition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
        )

        def jit_wrap(buf):
            buf.insert_internal = jax.jit(buf.insert_internal)
            buf.sample_internal  = jax.jit(buf.sample_internal)
            return buf

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

        # ------------------------------------------------------------------ #
        # Environment interaction                                              #
        # ------------------------------------------------------------------ #

        # Fix 3: inject CVAE into data collection — actor is conditioned on a
        # CVAE-generated subgoal with prob alpha_subgoal, otherwise true goal.
        def actor_step(training_state, env, env_state, key, extra_fields):
            key, cvae_key, mix_key, noise_key = jax.random.split(key, 4)
            obs   = env_state.obs
            state = obs[:, :state_size]
            goal  = obs[:, state_size:]

            h       = jax.random.normal(cvae_key, (state.shape[0], self.cvae_latent_dim))
            subgoal = jax.lax.stop_gradient(
                cvae_decoder.apply(training_state.cvae_state.params["cvae_decoder"], h, state, goal)
            )
            obs_sub = jnp.concatenate([state, subgoal], axis=-1)

            use_sub  = jax.random.bernoulli(mix_key, p=self.alpha_subgoal, shape=(state.shape[0],))
            obs_mixed = jnp.where(use_sub[:, None], obs_sub, obs)

            means, log_stds = actor.apply(training_state.actor_state.params, obs_mixed)
            stds    = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(noise_key, shape=means.shape))

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
        def get_experience(training_state, env_state, buffer_state, key):
            def f(carry, _):
                env_state, k = carry
                k, next_k = jax.random.split(k)
                env_state, transition = actor_step(
                    training_state, train_env, env_state, k,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_k), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            def f(carry, _):
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state, env_state, buffer_state, key
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps
            )[0]

        # ------------------------------------------------------------------ #
        # Training                                                             #
        # ------------------------------------------------------------------ #

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, cvae_key, critic_key, actor_key = jax.random.split(key, 4)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )
            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
                cvae_encoder=cvae_encoder,
                cvae_decoder=cvae_decoder,
            )

            training_state, cvae_metrics   = update_cvae(context, networks, transitions, training_state, cvae_key)
            training_state, actor_metrics  = update_actor_and_alpha(context, networks, transitions, training_state, actor_key)
            training_state, critic_metrics = update_critic(context, networks, transitions, training_state, critic_key)
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(cvae_metrics)
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)
            return (training_state, key), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            env_state, buffer_state = get_experience(
                training_state, env_state, buffer_state, experience_key1
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )

            buffer_state, transitions = replay_buffer.sample(buffer_state)
            batch_keys  = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (self.discounting, self.subgoal_discounting, state_size, tuple(train_env.goal_indices)),
                transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]), transitions
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

        # ------------------------------------------------------------------ #
        # Prefill + main loop                                                  #
        # ------------------------------------------------------------------ #

        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        # Fix 2: temporal commitment eval loop.
        # A CVAE subgoal is held for commitment_horizon steps (resampled when
        # the horizon elapses or an episode ends), eliminating per-step jitter.
        eval_env_wrapped = envs.training.EvalWrapper(eval_env)

        @jax.jit
        def run_eval(training_state, key):
            reset_keys = jax.random.split(key, config.num_eval_envs)
            eval_state = eval_env_wrapped.reset(reset_keys)

            # Initialise subgoal from episode-start observation
            init_obs   = eval_state.obs
            init_state = init_obs[:, :state_size]
            init_goal  = init_obs[:, state_size:]
            h0 = jnp.zeros((config.num_eval_envs, self.cvae_latent_dim))
            current_subgoal = cvae_decoder.apply(
                training_state.cvae_state.params["cvae_decoder"], h0, init_state, init_goal
            )
            # Start at commitment_horizon to force a resample on the first step
            steps_held = jnp.full((config.num_eval_envs,), self.commitment_horizon, dtype=jnp.int32)

            def step(carry, _):
                env_state, current_subgoal, steps_held, step_key = carry
                step_key, resample_key = jax.random.split(step_key)

                obs       = env_state.obs
                state_obs = obs[:, :state_size]
                goal      = obs[:, state_size:]

                # Resample when commitment horizon elapsed
                needs_resample  = steps_held >= self.commitment_horizon
                h_new           = jax.random.normal(resample_key, (config.num_eval_envs, self.cvae_latent_dim))
                new_subgoal     = cvae_decoder.apply(
                    training_state.cvae_state.params["cvae_decoder"], h_new, state_obs, goal
                )
                current_subgoal = jnp.where(needs_resample[:, None], new_subgoal, current_subgoal)
                steps_held      = jnp.where(needs_resample, 0, steps_held + 1)

                obs_subgoal = jnp.concatenate([state_obs, current_subgoal], axis=-1)
                means, _    = actor.apply(training_state.actor_state.params, obs_subgoal)
                actions     = nn.tanh(means)

                nstate = eval_env_wrapped.step(env_state, actions)

                # When an episode ends, schedule an immediate resample from fresh obs
                steps_held = jnp.where(nstate.done > 0, self.commitment_horizon, steps_held)

                return (nstate, current_subgoal, steps_held, step_key), (current_subgoal, goal, state_obs[:, :2], nstate.done)

            (final_state, _, _, _), (subgoals, goals, state_xys, dones) = jax.lax.scan(
                step,
                (eval_state, current_subgoal, steps_held, key),
                (),
                length=config.episode_length,
            )
            return final_state, subgoals, goals, state_xys, dones

        eval_walltime    = 0.0
        training_walltime = 0
        logging.info("starting SCC-RL v4 training...")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime  += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            key, eval_key = jax.random.split(key)
            t_eval = time.time()
            final_eval_state, subgoals, goals, state_xys, dones = run_eval(training_state, eval_key)
            eval_metrics = final_eval_state.info["eval_metrics"]
            eval_metrics.active_episodes.block_until_ready()
            eval_walltime += time.time() - t_eval

            for name in ("reward", "success", "success_easy", "success_super_easy", "dist", "distance_from_origin"):
                metrics[f"eval/episode_{name}"] = float(np.mean(eval_metrics.episode_metrics[name]))
            if "success" in eval_metrics.episode_metrics:
                metrics["eval/episode_success_any"] = float(
                    np.mean(eval_metrics.episode_metrics["success"] > 0.0)
                )
            metrics["eval/avg_episode_length"] = float(np.mean(eval_metrics.episode_steps))
            metrics["eval/walltime"] = eval_walltime

            sg  = np.array(subgoals)
            gl  = np.array(goals)
            sxy = np.array(state_xys)
            metrics["eval/subgoal_dist_to_goal"]  = float(np.mean(np.linalg.norm(sg - gl,  axis=-1)))
            metrics["eval/subgoal_dist_to_state"] = float(np.mean(np.linalg.norm(sg - sxy, axis=-1)))

            logging.info("step: %d", current_step)

            do_render  = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            ep_success   = np.array(eval_metrics.episode_metrics["success"]) > 0
            failed_mask  = ~ep_success
            success_mask = ep_success
            n_failed     = int(np.sum(failed_mask))
            n_success    = int(np.sum(success_mask))
            if n_failed > 0:
                metrics["eval/failed_subgoal_dist_to_goal"]  = float(np.mean(np.linalg.norm(sg[:, failed_mask,  :] - gl[:,  failed_mask,  :], axis=-1)))
                metrics["eval/failed_subgoal_dist_to_state"] = float(np.mean(np.linalg.norm(sg[:, failed_mask,  :] - sxy[:, failed_mask,  :], axis=-1)))
            if n_success > 0:
                metrics["eval/success_subgoal_dist_to_goal"]  = float(np.mean(np.linalg.norm(sg[:, success_mask, :] - gl[:,  success_mask, :], axis=-1)))
                metrics["eval/success_subgoal_dist_to_state"] = float(np.mean(np.linalg.norm(sg[:, success_mask, :] - sxy[:, success_mask, :], axis=-1)))

            if n_failed > 0 and ne % config.visualization_interval == 0:
                fail_env  = int(np.argmax(failed_mask))
                traj_sg   = sg[:,  fail_env, :]
                traj_gl   = gl[:,  fail_env, :]
                traj_sxy  = sxy[:, fail_env, :]

                # Truncate to first episode only
                dones_env = np.array(dones)[:, fail_env]
                done_idx  = np.where(dones_env > 0)[0]
                ep_end    = int(done_idx[0]) + 1 if len(done_idx) > 0 else len(traj_sg)
                traj_sg   = traj_sg[:ep_end]
                traj_gl   = traj_gl[:ep_end]
                traj_sxy  = traj_sxy[:ep_end]

                d_st_gl  = np.linalg.norm(traj_sxy - traj_gl,  axis=-1)
                d_sg_gl  = np.linalg.norm(traj_sg  - traj_gl,  axis=-1)
                d_sg_st  = np.linalg.norm(traj_sg  - traj_sxy, axis=-1)

                goal_xy = traj_gl[0]
                step_sz = max(1, ep_end // 10)
                logging.info(
                    "Failed traj (env %d, ep_len=%d, goal=[%.2f, %.2f]):\n"
                    "  %4s  %-18s  %-18s  %-18s  %-14s  %-14s  %-14s",
                    fail_env, ep_end, goal_xy[0], goal_xy[1],
                    "t", "state (x,y)", "subgoal (x,y)", "goal (x,y)",
                    "d(state→goal)", "d(sg→goal)", "d(sg→state)",
                )
                for t in range(0, ep_end, step_sz):
                    logging.info(
                        "  %4d  (%7.2f,%7.2f)   (%7.2f,%7.2f)   (%7.2f,%7.2f)   %8.3f       %8.3f       %8.3f",
                        t, traj_sxy[t, 0], traj_sxy[t, 1],
                        traj_sg[t, 0],  traj_sg[t, 1],
                        traj_gl[t, 0],  traj_gl[t, 1],
                        d_st_gl[t], d_sg_gl[t], d_sg_st[t],
                    )

                if config.log_wandb:
                    import wandb as _wandb
                    fail_table = _wandb.Table(
                        data=[
                            [t,
                             float(traj_sxy[t, 0]), float(traj_sxy[t, 1]),
                             float(traj_sg[t, 0]),  float(traj_sg[t, 1]),
                             float(traj_gl[t, 0]),  float(traj_gl[t, 1]),
                             float(d_st_gl[t]), float(d_sg_gl[t]), float(d_sg_st[t])]
                            for t in range(ep_end)
                        ],
                        columns=["t", "state_x", "state_y", "subgoal_x", "subgoal_y",
                                 "goal_x", "goal_y", "state_dist_to_goal",
                                 "subgoal_dist_to_goal", "subgoal_dist_to_state"],
                    )
                    _wandb.log({"eval/failed_trajectory": fail_table}, step=current_step)

            if config.checkpoint_logdir:
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                    training_state.cvae_state.params,
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
            training_state.cvae_state.params,
        )
        return make_policy, params, metrics
