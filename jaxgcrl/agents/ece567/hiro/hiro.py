"""HIRO: Hierarchical RL with Off-policy Correction (Nachum et al. 2018).

Two-level TD3 hierarchy:
  - High-level (manager): observes state, outputs relative goal every c steps.
  - Low-level (worker): observes (state, goal), outputs action every step.

Goal transition: h(s_t, g_t, s_{t+1}) = s_t[goal_idx] + g_t - s_{t+1}[goal_idx]
Intrinsic reward: r_intr = -||s_t[goal_idx] + g_t - s_{t+1}[goal_idx]||_2
"""

import logging
import random
import time
from typing import Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.training.replay_buffers import UniformSamplingQueue
from brax.v1 import envs as envs_v1
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.utils.evaluator import ActorEvaluator, generate_unroll

from .losses import off_policy_correction, update_hi, update_lo
from .networks import Actor, DoubleCritic

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


# ---------------------------------------------------------------------------
# Transition containers
# ---------------------------------------------------------------------------

class LoTransition(NamedTuple):
    """Low-level (worker) transition tuple."""
    state: jnp.ndarray       # [state_size]
    goal: jnp.ndarray        # [goal_size]
    action: jnp.ndarray      # [action_size]
    reward: jnp.ndarray      # scalar
    next_state: jnp.ndarray  # [state_size]
    next_goal: jnp.ndarray   # [goal_size]
    discount: jnp.ndarray    # scalar (1 - done)


class HiTransition(NamedTuple):
    """High-level (manager) transition tuple."""
    state: jnp.ndarray       # [state_size]
    goal: jnp.ndarray        # [goal_size]  (re-labelled by off-policy correction)
    reward: jnp.ndarray      # scalar       (sum of env rewards over c steps)
    next_state: jnp.ndarray  # [state_size]
    discount: jnp.ndarray    # scalar


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Contains all training state for HIRO."""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray

    # Low-level networks
    lo_actor_state: TrainState
    lo_actor_target_params: any
    lo_critic_state: TrainState
    lo_critic_target_params: any

    # High-level networks
    hi_actor_state: TrainState
    hi_actor_target_params: any
    hi_critic_state: TrainState
    hi_critic_target_params: any

    # Carried during evaluation: current hi-level goal per eval env and step counter
    eval_hi_goal: jnp.ndarray  # [num_eval_envs, goal_size]
    eval_step_count: jnp.ndarray  # scalar int


# ---------------------------------------------------------------------------
# HIRO agent
# ---------------------------------------------------------------------------

@dataclass
class HIRO:
    """HIRO agent (Nachum et al. 2018, Data-Efficient Hierarchical RL)."""

    # Hierarchy
    manager_period: int = 10          # c: hi-level acts every c lo-level steps

    # Optimisation
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    discounting: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_delay: int = 2

    # Exploration / smoothing noise
    lo_exploration_noise: float = 1.0
    hi_exploration_noise: float = 1.0
    lo_smoothing_noise: float = 0.2
    noise_clip: float = 0.5

    # Goal scale
    hi_action_scale: float = 10.0

    # Reward scales
    lo_reward_scale: float = 1.0
    hi_reward_scale: float = 0.1

    # Replay buffers
    lo_max_replay_size: int = 200000
    hi_max_replay_size: int = 200000
    min_replay_size: int = 1000

    # Architecture
    h_dim: int = 300
    n_hidden: int = 2
    use_ln: bool = False

    # Data collection
    unroll_length: int = 10  # number of hi-level steps per epoch iteration

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
        # ------------------------------------------------------------------
        # Wrap environments
        # ------------------------------------------------------------------
        unwrapped_env = train_env
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        # ------------------------------------------------------------------
        # Dimensions
        # ------------------------------------------------------------------
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_indices = train_env.goal_indices          # tuple of ints
        goal_size = len(goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # ------------------------------------------------------------------
        # Step bookkeeping
        # ------------------------------------------------------------------
        # Each "actor step" = unroll_length hi-level steps = unroll_length * manager_period lo-steps
        lo_steps_per_actor_step = self.unroll_length * self.manager_period
        env_steps_per_actor_step = config.num_envs * lo_steps_per_actor_step
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / lo_steps_per_actor_step))
        num_training_steps_per_epoch = int(np.ceil(
            (config.total_env_steps - self.min_replay_size * config.num_envs)
            / (config.num_evals * env_steps_per_actor_step)
        ))
        num_training_steps_per_epoch = max(1, num_training_steps_per_epoch)

        logging.info("env_steps_per_actor_step: %d", env_steps_per_actor_step)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        # ------------------------------------------------------------------
        # Random seeds
        # ------------------------------------------------------------------
        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        (
            key,
            lo_buf_key,
            hi_buf_key,
            eval_env_key,
            env_key,
            lo_actor_key,
            lo_critic_key,
            hi_actor_key,
            hi_critic_key,
        ) = jax.random.split(key, 9)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # ------------------------------------------------------------------
        # Networks
        # ------------------------------------------------------------------
        # Low-level actor: input = (state, goal), output = action in [-1, 1]
        lo_actor = Actor(
            action_size=action_size,
            action_scale=1.0,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            use_ln=self.use_ln,
        )
        lo_actor_params = lo_actor.init(lo_actor_key, np.ones([1, state_size + goal_size]))
        lo_actor_state = TrainState.create(
            apply_fn=lo_actor.apply,
            params=lo_actor_params,
            tx=optax.adam(learning_rate=self.actor_lr),
        )

        # Low-level critic: input = (state, goal, action)
        lo_critic = DoubleCritic(h_dim=self.h_dim, n_hidden=self.n_hidden, use_ln=self.use_ln)
        lo_critic_params = lo_critic.init(
            lo_critic_key, np.ones([1, state_size + goal_size + action_size])
        )
        lo_critic_state = TrainState.create(
            apply_fn=lo_critic.apply,
            params=lo_critic_params,
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # High-level actor: input = state, output = relative goal in [-hi_action_scale, hi_action_scale]
        hi_actor = Actor(
            action_size=goal_size,
            action_scale=self.hi_action_scale,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            use_ln=self.use_ln,
        )
        hi_actor_params = hi_actor.init(hi_actor_key, np.ones([1, state_size]))
        hi_actor_state = TrainState.create(
            apply_fn=hi_actor.apply,
            params=hi_actor_params,
            tx=optax.adam(learning_rate=self.actor_lr),
        )

        # High-level critic: input = (state, hi_goal)
        hi_critic = DoubleCritic(h_dim=self.h_dim, n_hidden=self.n_hidden, use_ln=self.use_ln)
        hi_critic_params = hi_critic.init(
            hi_critic_key, np.ones([1, state_size + goal_size])
        )
        hi_critic_state = TrainState.create(
            apply_fn=hi_critic.apply,
            params=hi_critic_params,
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # ------------------------------------------------------------------
        # Training state
        # ------------------------------------------------------------------
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            lo_actor_state=lo_actor_state,
            lo_actor_target_params=lo_actor_params,
            lo_critic_state=lo_critic_state,
            lo_critic_target_params=lo_critic_params,
            hi_actor_state=hi_actor_state,
            hi_actor_target_params=hi_actor_params,
            hi_critic_state=hi_critic_state,
            hi_critic_target_params=hi_critic_params,
            eval_hi_goal=jnp.zeros((config.num_eval_envs, goal_size)),
            eval_step_count=jnp.zeros((), dtype=jnp.int32),
        )

        # ------------------------------------------------------------------
        # Replay buffers
        # ------------------------------------------------------------------
        dummy_lo = LoTransition(
            state=jnp.zeros((state_size,)),
            goal=jnp.zeros((goal_size,)),
            action=jnp.zeros((action_size,)),
            reward=jnp.zeros(()),
            next_state=jnp.zeros((state_size,)),
            next_goal=jnp.zeros((goal_size,)),
            discount=jnp.zeros(()),
        )
        dummy_hi = HiTransition(
            state=jnp.zeros((state_size,)),
            goal=jnp.zeros((goal_size,)),
            reward=jnp.zeros(()),
            next_state=jnp.zeros((state_size,)),
            discount=jnp.zeros(()),
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        lo_buffer = jit_wrap(
            UniformSamplingQueue(
                max_replay_size=self.lo_max_replay_size,
                dummy_data_sample=dummy_lo,
                sample_batch_size=self.batch_size,
            )
        )
        hi_buffer = jit_wrap(
            UniformSamplingQueue(
                max_replay_size=self.hi_max_replay_size,
                dummy_data_sample=dummy_hi,
                sample_batch_size=self.batch_size,
            )
        )
        lo_buffer_state = jax.jit(lo_buffer.init)(lo_buf_key)
        hi_buffer_state = jax.jit(hi_buffer.init)(hi_buf_key)

        # ------------------------------------------------------------------
        # Config dict for loss functions
        # ------------------------------------------------------------------
        loss_config = dict(
            discounting=self.discounting,
            tau=self.tau,
            lo_smoothing_noise=self.lo_smoothing_noise,
            noise_clip=self.noise_clip,
            policy_delay=self.policy_delay,
            hi_action_scale=self.hi_action_scale,
        )
        lo_networks = dict(lo_actor=lo_actor, lo_critic=lo_critic)
        hi_networks = dict(hi_actor=hi_actor, hi_critic=hi_critic)

        # ------------------------------------------------------------------
        # Goal indices as jnp array (used inside JIT-compiled functions)
        # ------------------------------------------------------------------
        goal_indices_arr = jnp.array(goal_indices)

        # ------------------------------------------------------------------
        # Data collection
        # ------------------------------------------------------------------

        @jax.jit
        def get_experience(training_state, env_state, lo_buffer_state, hi_buffer_state, key):
            """Collect unroll_length hi-level steps (each = manager_period lo steps).

            Returns updated (env_state, lo_buffer_state, hi_buffer_state).
            """

            def outer_step(carry, unused):
                """One hi-level step: sample hi_goal, run manager_period lo steps."""
                env_state, lo_bs, hi_bs, key = carry
                key, hi_key, inner_key = jax.random.split(key, 3)

                # Observe state at start of window
                s0 = env_state.obs[:, :state_size]  # [num_envs, state_size]

                # Sample hi-level goal with exploration noise
                hi_goal_raw = hi_actor.apply(training_state.hi_actor_state.params, s0)
                hi_noise = self.hi_exploration_noise * jax.random.normal(
                    hi_key, shape=hi_goal_raw.shape
                )
                hi_goal = jnp.clip(hi_goal_raw + hi_noise, -self.hi_action_scale, self.hi_action_scale)
                # hi_goal: [num_envs, goal_size] — relative goal

                # Inner loop: manager_period lo-level steps
                def inner_step(carry, t):
                    env_state, lo_goal, sum_r, min_disc, key = carry
                    key, lo_key = jax.random.split(key)

                    s_t = env_state.obs[:, :state_size]  # [num_envs, state_size]

                    # Low-level action with exploration noise
                    lo_inp = jnp.concatenate([s_t, lo_goal], axis=-1)  # [num_envs, s+g]
                    lo_act_raw = lo_actor.apply(training_state.lo_actor_state.params, lo_inp)
                    lo_noise = self.lo_exploration_noise * jax.random.normal(
                        lo_key, shape=lo_act_raw.shape
                    )
                    lo_act = jnp.clip(lo_act_raw + lo_noise, -1.0, 1.0)

                    # Environment step
                    new_env_state = train_env.step(env_state, lo_act)
                    s_t1 = new_env_state.obs[:, :state_size]  # [num_envs, state_size]
                    r_env = new_env_state.reward               # [num_envs]
                    done = new_env_state.done                  # [num_envs]
                    disc = 1.0 - done                          # [num_envs]

                    # Intrinsic reward: -||s_t[goal_idx] + g_t - s_{t+1}[goal_idx]||_2
                    s_t_goal = s_t[:, goal_indices_arr]    # [num_envs, goal_size]
                    s_t1_goal = s_t1[:, goal_indices_arr]  # [num_envs, goal_size]
                    r_intr = -jnp.linalg.norm(s_t_goal + lo_goal - s_t1_goal, axis=-1)  # [num_envs]
                    r_intr = r_intr * self.lo_reward_scale

                    # Goal transition for next lo step
                    lo_goal_next = s_t_goal + lo_goal - s_t1_goal
                    lo_goal_next = jnp.clip(lo_goal_next, -self.hi_action_scale, self.hi_action_scale)

                    # Accumulate hi-level reward and discount
                    sum_r = sum_r + disc * r_env
                    min_disc = min_disc * disc

                    return (new_env_state, lo_goal_next, sum_r, min_disc, key), (
                        s_t, lo_goal, lo_act, r_intr, s_t1, lo_goal_next, disc
                    )

                # Initial lo_goal is the hi_goal clipped to valid range
                # (hi_goal already in [-hi_action_scale, hi_action_scale])
                lo_goal_init = hi_goal  # [num_envs, goal_size]
                sum_r_init = jnp.zeros((config.num_envs,))
                min_disc_init = jnp.ones((config.num_envs,))

                (env_state_c, _, sum_r, min_disc, inner_key), inner_data = jax.lax.scan(
                    inner_step,
                    (env_state, lo_goal_init, sum_r_init, min_disc_init, inner_key),
                    jnp.arange(self.manager_period),
                )
                # inner_data: each field has shape [manager_period, num_envs, ...]

                # State at end of window
                s_c = env_state_c.obs[:, :state_size]  # [num_envs, state_size]

                # ---- Insert lo-transitions ----
                # Reshape from [manager_period, num_envs, ...] to [manager_period * num_envs, ...]
                lo_batch = LoTransition(
                    state=inner_data[0].reshape(self.manager_period * config.num_envs, state_size),
                    goal=inner_data[1].reshape(self.manager_period * config.num_envs, goal_size),
                    action=inner_data[2].reshape(self.manager_period * config.num_envs, action_size),
                    reward=inner_data[3].reshape(self.manager_period * config.num_envs),
                    next_state=inner_data[4].reshape(self.manager_period * config.num_envs, state_size),
                    next_goal=inner_data[5].reshape(self.manager_period * config.num_envs, goal_size),
                    discount=inner_data[6].reshape(self.manager_period * config.num_envs),
                )
                lo_bs = lo_buffer.insert(lo_bs, lo_batch)

                # ---- Off-policy correction (vmapped over envs) ----
                # s_seq: [manager_period, state_size] per env → [num_envs, manager_period, state_size]
                s_seq_all = jnp.transpose(inner_data[0], (1, 0, 2))  # [num_envs, c, state_size]
                a_seq_all = jnp.transpose(inner_data[2], (1, 0, 2))  # [num_envs, c, action_size]

                key, opc_key = jax.random.split(key)
                opc_keys = jax.random.split(opc_key, config.num_envs)

                g_tilde = jax.vmap(
                    lambda s_seq, a_seq, s0_e, s_c_e, orig_g, k: off_policy_correction(
                        lo_actor,
                        training_state.lo_actor_state.params,
                        s_seq,
                        a_seq,
                        s0_e,
                        s_c_e,
                        orig_g,
                        k,
                        goal_indices_arr,
                        self.hi_action_scale,
                    )
                )(s_seq_all, a_seq_all, s0, s_c, hi_goal, opc_keys)
                # g_tilde: [num_envs, goal_size]

                # ---- Insert hi-transition ----
                hi_batch = HiTransition(
                    state=s0,
                    goal=g_tilde,
                    reward=sum_r * self.hi_reward_scale,
                    next_state=s_c,
                    discount=min_disc,
                )
                hi_bs = hi_buffer.insert(hi_bs, hi_batch)

                return (env_state_c, lo_bs, hi_bs, key), ()

            (env_state, lo_buffer_state, hi_buffer_state, _), _ = jax.lax.scan(
                outer_step,
                (env_state, lo_buffer_state, hi_buffer_state, key),
                (),
                length=self.unroll_length,
            )
            return env_state, lo_buffer_state, hi_buffer_state

        # ------------------------------------------------------------------
        # Prefill
        # ------------------------------------------------------------------

        def prefill_replay_buffer(
            training_state, env_state, lo_buffer_state, hi_buffer_state, key
        ):
            @jax.jit
            def f(carry, unused):
                training_state, env_state, lo_bs, hi_bs, key = carry
                key, step_key = jax.random.split(key)
                env_state, lo_bs, hi_bs = get_experience(
                    training_state, env_state, lo_bs, hi_bs, step_key
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step
                )
                return (training_state, env_state, lo_bs, hi_bs, key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, lo_buffer_state, hi_buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ------------------------------------------------------------------
        # Network updates
        # ------------------------------------------------------------------

        @jax.jit
        def update_networks(carry, unused):
            training_state, lo_bs, hi_bs, key = carry
            key, lo_sample_key, hi_sample_key, lo_update_key, hi_update_key = jax.random.split(key, 5)

            # Sample from buffers
            lo_bs, lo_batch = lo_buffer.sample(lo_bs)
            hi_bs, hi_batch = hi_buffer.sample(hi_bs)

            # Update lo manager_period times using a scan
            def lo_update_step(carry, unused):
                ts, k = carry
                k, update_key = jax.random.split(k)
                ts, lo_metrics = update_lo(loss_config, lo_networks, lo_batch, ts, update_key)
                ts = ts.replace(gradient_steps=ts.gradient_steps + 1)
                return (ts, k), lo_metrics

            (training_state, lo_update_key), lo_metrics = jax.lax.scan(
                lo_update_step,
                (training_state, lo_update_key),
                (),
                length=self.manager_period,
            )
            lo_metrics = jax.tree_util.tree_map(jnp.mean, lo_metrics)

            # Update hi once
            training_state, hi_metrics = update_hi(
                loss_config, hi_networks, hi_batch, training_state, hi_update_key
            )

            metrics = {**lo_metrics, **hi_metrics}
            return (training_state, lo_bs, hi_bs, key), metrics

        # ------------------------------------------------------------------
        # Training step = collect + update
        # ------------------------------------------------------------------

        @jax.jit
        def training_step(training_state, env_state, lo_bs, hi_bs, key):
            key, collect_key, update_key = jax.random.split(key, 3)

            env_state, lo_bs, hi_bs = get_experience(
                training_state, env_state, lo_bs, hi_bs, collect_key
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )

            (training_state, lo_bs, hi_bs, _), metrics = update_networks(
                (training_state, lo_bs, hi_bs, update_key), None
            )

            return (training_state, env_state, lo_bs, hi_bs), metrics

        # ------------------------------------------------------------------
        # Training epoch
        # ------------------------------------------------------------------

        @jax.jit
        def training_epoch(training_state, env_state, lo_bs, hi_bs, key):
            def f(carry, unused_t):
                ts, es, lo_bs, hi_bs, k = carry
                k, step_key = jax.random.split(k)
                (ts, es, lo_bs, hi_bs), metrics = training_step(ts, es, lo_bs, hi_bs, step_key)
                return (ts, es, lo_bs, hi_bs, k), metrics

            (training_state, env_state, lo_bs, hi_bs, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, lo_bs, hi_bs, key),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics["lo_buffer_size"] = lo_buffer.size(lo_bs)
            metrics["hi_buffer_size"] = hi_buffer.size(hi_bs)
            return training_state, env_state, lo_bs, hi_bs, metrics

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        # We implement a custom eval actor_step that:
        #   - Resamples hi_goal when eval_step_count % manager_period == 0
        #   - Applies goal transition h otherwise
        #   - Feeds (state, hi_goal) to the lo_actor for the action

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            obs = env_state.obs                    # [num_eval_envs, obs_size]
            s_t = obs[:, :state_size]              # [num_eval_envs, state_size]

            step_count = training_state.eval_step_count
            current_hi_goal = training_state.eval_hi_goal  # [num_eval_envs, goal_size]

            # Decide whether to query hi actor
            new_hi_goal_from_actor = hi_actor.apply(
                training_state.hi_actor_state.params, s_t
            )  # [num_eval_envs, goal_size]

            is_new_window = (step_count % self.manager_period) == 0
            # We need eval_hi_goal initialised; on first step we always query
            hi_goal_to_use = jax.lax.cond(
                is_new_window,
                lambda: new_hi_goal_from_actor,
                lambda: current_hi_goal,
            )

            # Compute lo action
            lo_inp = jnp.concatenate([s_t, hi_goal_to_use], axis=-1)
            lo_act = lo_actor.apply(training_state.lo_actor_state.params, lo_inp)

            nstate = env.step(env_state, lo_act)
            s_t1 = nstate.obs[:, :state_size]

            # Update hi_goal via h for next step
            hi_goal_next = (
                s_t[:, goal_indices_arr] + hi_goal_to_use - s_t1[:, goal_indices_arr]
            )
            hi_goal_next = jnp.clip(hi_goal_next, -self.hi_action_scale, self.hi_action_scale)

            # On new window, keep the hi_goal fixed for the whole window
            # After this step we advance goal via h; reset is handled by is_new_window above
            hi_goal_carried = jax.lax.cond(
                is_new_window,
                lambda: hi_goal_next,
                lambda: hi_goal_next,
            )

            new_training_state = training_state.replace(
                eval_hi_goal=hi_goal_carried,
                eval_step_count=step_count + 1,
            )

            # Build a dummy Transition-like object compatible with ActorEvaluator
            # We return the env state extras that EvalWrapper expects
            state_extras = {x: nstate.info[x] for x in extra_fields}

            from brax.training import types as brax_types
            # Use a simple NamedTuple the generate_unroll loop can handle
            class EvalTransition(NamedTuple):
                observation: jnp.ndarray
                action: jnp.ndarray
                reward: jnp.ndarray
                discount: jnp.ndarray
                extras: dict

            return nstate, EvalTransition(
                observation=obs,
                action=lo_act,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            ), new_training_state

        # Wrap deterministic_actor_step to match ActorEvaluator's expected signature:
        # actor_step(training_state, env, env_state, extra_fields) -> (nstate, transition)
        # We track hi_goal in training_state (eval_hi_goal, eval_step_count).
        # generate_unroll passes fixed training_state, but we need to thread the
        # mutable hi_goal through the scan carry. We implement a custom eval loop.

        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, config.num_eval_envs)
            eval_env_wrapped = envs.training.EvalWrapper(eval_env)
            eval_first_state = eval_env_wrapped.reset(reset_keys)

            # Reset eval counters
            ts = training_state.replace(
                eval_hi_goal=jnp.zeros((config.num_eval_envs, goal_size)),
                eval_step_count=jnp.zeros((), dtype=jnp.int32),
            )

            def step_fn(carry, unused_t):
                ts, env_state = carry
                obs = env_state.obs
                s_t = obs[:, :state_size]

                step_count = ts.eval_step_count
                current_hi_goal = ts.eval_hi_goal

                new_hi_goal_from_actor = hi_actor.apply(
                    ts.hi_actor_state.params, s_t
                )

                is_new_window = (step_count % self.manager_period) == 0
                hi_goal_to_use = jax.lax.cond(
                    is_new_window,
                    lambda: new_hi_goal_from_actor,
                    lambda: current_hi_goal,
                )

                lo_inp = jnp.concatenate([s_t, hi_goal_to_use], axis=-1)
                lo_act = lo_actor.apply(ts.lo_actor_state.params, lo_inp)

                nstate = eval_env_wrapped.step(env_state, lo_act)
                s_t1 = nstate.obs[:, :state_size]

                hi_goal_next = (
                    s_t[:, goal_indices_arr] + hi_goal_to_use - s_t1[:, goal_indices_arr]
                )
                hi_goal_next = jnp.clip(hi_goal_next, -self.hi_action_scale, self.hi_action_scale)

                new_ts = ts.replace(
                    eval_hi_goal=hi_goal_next,
                    eval_step_count=step_count + 1,
                )
                return (new_ts, nstate), None

            (final_ts, final_state), _ = jax.lax.scan(
                step_fn,
                (ts, eval_first_state),
                (),
                length=config.episode_length,
            )
            return final_state

        generate_eval_unroll_jit = jax.jit(generate_eval_unroll)

        # Custom evaluator using our generate_eval_unroll
        class HiroEvaluator:
            def __init__(self, key):
                self._key = key
                self._eval_walltime = 0.0
                self._steps_per_unroll = config.episode_length * config.num_eval_envs

            def run_evaluation(self, training_state, training_metrics):
                self._key, unroll_key = jax.random.split(self._key)
                t = time.time()
                eval_state = generate_eval_unroll_jit(training_state, unroll_key)
                eval_metrics = eval_state.info["eval_metrics"]
                eval_metrics.active_episodes.block_until_ready()
                epoch_eval_time = time.time() - t

                metrics = {}
                for name in ["reward", "success", "success_easy", "success_super_easy", "dist", "distance_from_origin"]:
                    metrics[f"eval/episode_{name}"] = np.mean(
                        eval_metrics.episode_metrics[name]
                    )

                if "success" in eval_metrics.episode_metrics:
                    metrics["eval/episode_success_any"] = np.mean(
                        eval_metrics.episode_metrics["success"] > 0.0
                    )

                metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
                metrics["eval/epoch_eval_time"] = epoch_eval_time
                metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
                self._eval_walltime += epoch_eval_time
                metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}
                return metrics

        evaluator = HiroEvaluator(eval_env_key)

        # ------------------------------------------------------------------
        # Prefill the buffers
        # ------------------------------------------------------------------
        key, prefill_key = jax.random.split(key)
        training_state, env_state, lo_buffer_state, hi_buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, lo_buffer_state, hi_buffer_state, prefill_key
        )

        # ------------------------------------------------------------------
        # Main training loop
        # ------------------------------------------------------------------
        training_walltime = 0.0
        logging.info("Starting HIRO training...")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, lo_buffer_state, hi_buffer_state, metrics = training_epoch(
                training_state, env_state, lo_buffer_state, hi_buffer_state, epoch_key
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

            make_policy = lambda param: lambda obs, rng: lo_actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.lo_actor_state.params,
                unwrapped_env,
                do_render=(ne % config.visualization_interval == 0),
            )

        total_steps = current_step
        logging.info("total steps: %s", total_steps)

        params = (
            training_state.lo_actor_state.params,
            training_state.hi_actor_state.params,
            training_state.lo_critic_state.params,
            training_state.hi_critic_state.params,
        )
        return make_policy, params, metrics
