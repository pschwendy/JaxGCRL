"""AdvectCRL: Contrastive RL with Conditional Flow Matching and Latent Advection.

Training:
  1. Contrastive critic (φ_sa, ψ) trained jointly with InfoNCE.
  2. CNF vector field v_θ trained with flow-matching in ψ-space (ψ frozen).
  3. Actor π(s, z_target) trained with entropy-regularised direct Q-gradient:
       L = E[α log π - φ_sa(cat(s, a)) · z_target]
     z_target = ψ(s_future) — no stop_gradient on ψ, so the actor loss also
     refines the goal encoder toward action-useful representations.

Evaluation / Inference:
  For each environment step:
    1. z_current = ψ(s_t[goal_indices])
    2. z_goal    = ψ(goal)
    3. Euler-integrate v_θ(·, z_goal, ·) for `advect_lookahead` steps
       to produce z_target ahead of z_current on the learned path.
    4. Execute a_t = π(s_t, z_target) (deterministic mean).
"""

import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import update_actor_and_alpha, update_cnf, update_contrastive_critic
from .networks import Actor, GoalEncoder, SAEncoder, VectorField

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    sa_state: TrainState      # φ_sa — updated by contrastive loss
    g_enc_state: TrainState   # ψ  — updated by contrastive loss + actor loss
    cnf_state: TrainState     # v_θ — updated by flow-matching loss
    actor_state: TrainState   # π  — updated by actor loss
    alpha_state: TrainState   # α  — updated by entropy loss


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


# ---------------------------------------------------------------------------
# Batch pre-processing
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_batch(buffer_config, transition, sample_key):
    """Sample future goals and build (state, future_state) pairs."""
    gamma, state_size, goal_indices = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    probs = (
        probs * jnp.equal(single_trajectories, single_trajectories.T)
        + jnp.eye(seq_len) * 1e-5
    )

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]

    state = transition.observation[:-1, :state_size]
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


# ---------------------------------------------------------------------------
# Param I/O
# ---------------------------------------------------------------------------

def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        return pickle.loads(fin.read())


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


# ---------------------------------------------------------------------------
# Advection evaluator
# ---------------------------------------------------------------------------

class AdvectionEvaluator:
    """Evaluator using CNF target advection at each step."""

    def __init__(self, advection_actor_step, eval_env, num_eval_envs, episode_length, key):
        self._key = key
        self._eval_walltime = 0.0
        self._steps_per_unroll = episode_length * num_eval_envs

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            reset_key, rollout_key = jax.random.split(key)
            env_state = eval_env.reset(jax.random.split(reset_key, num_eval_envs))

            def step(carry, _):
                env_state, step_key = carry
                step_key, next_key = jax.random.split(step_key)
                env_state, _ = advection_actor_step(
                    training_state, eval_env, env_state, step_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), ()

            (final_state, _), _ = jax.lax.scan(
                step, (env_state, rollout_key), (), length=episode_length
            )
            return final_state

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)

    def run_evaluation(self, training_state, training_metrics, aggregate_episodes=True):
        import time as _time
        import numpy as _np

        self._key, unroll_key = jax.random.split(self._key)
        t = _time.time()
        eval_state = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = _time.time() - t

        metrics = {}
        for name in ["reward", "success", "success_easy", "dist", "distance_from_origin"]:
            if name in eval_metrics.episode_metrics:
                metrics[f"eval/episode_{name}"] = (
                    _np.mean(eval_metrics.episode_metrics[name])
                    if aggregate_episodes
                    else eval_metrics.episode_metrics[name]
                )

        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = _np.mean(
                eval_metrics.episode_metrics["success"] > 0.0
            )

        metrics["eval/avg_episode_length"] = _np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime += epoch_eval_time
        return {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AdvectCRL:
    """AdvectCRL agent.

    Contrastive critic + conditional flow matching in goal latent space.
    Actor trained with SAC entropy regularisation, conditioning on a latent
    target produced by CNF advection at inference.
    ψ receives gradients from both the contrastive loss and the actor loss.
    """

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    cnf_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1

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

    # Advection at eval: Euler-integrate v_θ for advect_lookahead steps (dt = 1/advect_total_steps)
    advect_lookahead: int = 5
    advect_total_steps: int = 20

    # target entropy = -target_entropy_coeff * action_size
    target_entropy_coeff: float = 0.5

    # Gradient norm clipping applied to every optimizer
    max_grad_norm: float = 1.0

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0

    def train_fn(
        self,
        config: "RunConfig",
        train_env,
        eval_env=None,
        randomization_fn=None,
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
        key, actor_key, sa_key, g_key, cnf_key = jax.random.split(key, 5)

        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size

        target_entropy = -self.target_entropy_coeff * action_size

        # ---- Networks ----
        sa_encoder = SAEncoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder = GoalEncoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        cnf = VectorField(
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

        def make_optimizer(lr):
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(learning_rate=lr),
            )

        # ---- Train states ----
        sa_state = TrainState.create(
            apply_fn=None,
            params=sa_encoder.init(sa_key, np.ones([1, state_size + action_size])),
            tx=make_optimizer(self.critic_lr),
        )
        g_enc_state = TrainState.create(
            apply_fn=None,
            params=g_encoder.init(g_key, np.ones([1, goal_size])),
            tx=make_optimizer(self.critic_lr),
        )
        cnf_state = TrainState.create(
            apply_fn=None,
            params={"cnf": cnf.init(cnf_key, np.ones([1, 2 * self.repr_dim + 1]))},
            tx=make_optimizer(self.cnf_lr),
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, state_size + self.repr_dim])),
            tx=make_optimizer(self.policy_lr),
        )
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
            tx=make_optimizer(self.alpha_lr),
        )

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            sa_state=sa_state,
            g_enc_state=g_enc_state,
            cnf_state=cnf_state,
            actor_state=actor_state,
            alpha_state=alpha_state,
        )

        # ---- Replay buffer ----
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

        # ---- Environment interaction (training — stochastic) ----
        def actor_step(g_enc_params, actor_params, env, env_state, key, extra_fields):
            """Training step: z_target = ψ(goal), action ~ π(s, z_target)."""
            state = env_state.obs[:, :state_size]
            goal = env_state.obs[:, state_size:]
            z_target = g_encoder.apply(g_enc_params, goal)
            actor_input = jnp.concatenate([state, z_target], axis=-1)
            means, log_stds = actor.apply(actor_params, actor_input)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(
                means + stds * jax.random.normal(key, means.shape, dtype=means.dtype)
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
        def get_experience(training_state, env_state, buffer_state, key):
            def f(carry, _):
                env_state, k = carry
                k, step_key = jax.random.split(k)
                env_state, transition = actor_step(
                    training_state.g_enc_state.params,
                    training_state.actor_state.params,
                    train_env, env_state, step_key,
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
                es, bs = get_experience(ts, es, bs, step_key)
                ts = ts.replace(env_steps=ts.env_steps + env_steps_per_actor_step)
                return (ts, es, bs, k), ()

            return jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps
            )[0]

        # ---- Network updates ----
        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, cnf_key, actor_key = jax.random.split(key, 3)

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
            nets = dict(sa_encoder=sa_encoder, g_encoder=g_encoder, cnf=cnf, actor=actor)

            training_state, crit_metrics = update_contrastive_critic(context, nets, transitions, training_state)
            training_state, cnf_metrics = update_cnf(context, nets, transitions, training_state, cnf_key)
            training_state, actor_metrics = update_actor_and_alpha(context, nets, transitions, training_state, actor_key)
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            return (training_state, key), {**crit_metrics, **cnf_metrics, **actor_metrics}

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            exp_key1, exp_key2, sample_key, train_key = jax.random.split(key, 4)

            env_state, buffer_state = get_experience(training_state, env_state, buffer_state, exp_key1)
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )

            buffer_state, transitions = replay_buffer.sample(buffer_state)
            batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (self.discounting, state_size, tuple(train_env.goal_indices)),
                transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            permutation = jax.random.permutation(exp_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]), transitions
            )

            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, train_key), transitions
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

        # ---- Advection actor step (eval only) ----
        goal_indices = list(train_env.goal_indices)
        advect_lookahead = self.advect_lookahead
        advect_total_steps = self.advect_total_steps

        def advection_actor_step(training_state, env, env_state, key, extra_fields):
            """Eval step: advect z_current → z_goal via CNF, execute π(s, z_target)."""
            state = env_state.obs[:, :state_size]
            goal = env_state.obs[:, state_size:]

            g_enc_params = training_state.g_enc_state.params
            cnf_params = training_state.cnf_state.params
            actor_params = training_state.actor_state.params

            z_current = g_encoder.apply(g_enc_params, state[:, goal_indices])
            z_goal = g_encoder.apply(g_enc_params, goal)

            dt = 1.0 / advect_total_steps

            def advect_single(z_c, z_g):
                def euler_step(z, t_scalar):
                    t_exp = jnp.expand_dims(t_scalar, axis=-1)
                    v = cnf.apply(
                        cnf_params["cnf"],
                        jnp.concatenate([z, z_g, t_exp], axis=-1),
                    )
                    return z + dt * v, None

                ts = jnp.arange(advect_lookahead, dtype=jnp.float32) * dt
                z_target, _ = jax.lax.scan(euler_step, z_c, ts)
                return z_target

            z_target = jax.vmap(advect_single)(z_current, z_goal)

            # Deterministic mean at eval
            means, _ = actor.apply(actor_params, jnp.concatenate([state, z_target], axis=-1))
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

        # ---- Init ----
        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        evaluator = AdvectionEvaluator(
            advection_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_key,
        )

        # ---- Training loop ----
        training_walltime = 0
        logging.info("starting AdvectCRL training...")
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
            make_policy = lambda param: lambda obs, rng: actor.apply(
                param["actor"],
                jnp.concatenate([
                    obs[:, :state_size],
                    g_encoder.apply(param["g_encoder"], obs[:, state_size:]),
                ], axis=-1),
            )

            progress_fn(
                current_step,
                metrics,
                make_policy,
                {
                    "actor": training_state.actor_state.params,
                    "g_encoder": training_state.g_enc_state.params,
                },
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                params = (
                    training_state.actor_state.params,
                    training_state.sa_state.params,
                    training_state.g_enc_state.params,
                    training_state.cnf_state.params,
                    training_state.alpha_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps
        logging.info("total steps: %s", total_steps)

        params = (
            training_state.actor_state.params,
            training_state.sa_state.params,
            training_state.g_enc_state.params,
            training_state.cnf_state.params,
            training_state.alpha_state.params,
        )
        return make_policy, params, metrics
