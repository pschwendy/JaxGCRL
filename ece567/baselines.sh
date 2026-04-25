#!/bin/bash
# baselines.sh — PPO/SAC/TD3 on ant_u_maze, ant_big_maze, ant_push;
#                SAC+HER / TD3+HER on all 7 benchmark envs.
# Builds on exp_paper_hparams.sh hyperparameters (same agent args).
#
# New envs (not in exp_paper_hparams.sh): ant_u_maze, ant_big_maze, ant_push
# HER envs: ant, humanoid, reacher, pusher_hard, ant_u_maze, ant_big_maze, ant_push
# Seeds: 0-4 (5 per config)
#
# Batching: slow (ant_big_maze, ant_push, pusher_hard) together, fast separate.
# 9 GPUs (0-8): 5+4 fills a batch; overflow seeds start the next batch.
# humanoid: --num_envs 512; PPO: --num_envs 4096; all others: --num_envs 1024

SAC_ARGS="sac --discounting 0.99 --unroll_length 62"
TD3_ARGS="td3 --discounting 0.99 --unroll_length 62"
PPO_ARGS="ppo --discounting 0.97 --batch_size 256 --num_minibatches 16"
SAC_HER_ARGS="sac --discounting 0.99 --unroll_length 62 --use_her"
TD3_HER_ARGS="td3 --discounting 0.99 --unroll_length 62 --use_her"

# ── Section A: new env baselines (no HER) ────────────────────────────────────

# Slow batch 1/4: SAC ant_big_maze s0-4 + SAC ant_push s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_big_maze --num_envs 1024 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_big_maze --num_envs 1024 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_big_maze --num_envs 1024 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_big_maze --num_envs 1024 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_big_maze --num_envs 1024 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_push --num_envs 1024 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_push --num_envs 1024 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_push --num_envs 1024 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_push --num_envs 1024 --seed 3 $SAC_ARGS &
wait

# Slow batch 2/4: SAC ant_push s4 + TD3 ant_big_maze s0-4 + TD3 ant_push s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_push --num_envs 1024 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_big_maze --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_big_maze --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_big_maze --num_envs 1024 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_big_maze --num_envs 1024 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_big_maze --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_push --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_push --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_push --num_envs 1024 --seed 2 $TD3_ARGS &
wait

# Slow batch 3/4: TD3 ant_push s3-4 + PPO ant_big_maze s0-4 + PPO ant_push s0-1
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_push --num_envs 1024 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_push --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_big_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_big_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_big_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_big_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_big_maze --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_push --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_push --num_envs 4096 --seed 1 $PPO_ARGS &
wait

# Slow batch 4/4: PPO ant_push s2-4
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_push --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_push --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_push --num_envs 4096 --seed 4 $PPO_ARGS &
wait

# Fast batch 1/2: SAC ant_u_maze s0-4 + TD3 ant_u_maze s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_u_maze --num_envs 1024 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_u_maze --num_envs 1024 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_u_maze --num_envs 1024 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_u_maze --num_envs 1024 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_u_maze --num_envs 1024 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_u_maze --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_u_maze --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_u_maze --num_envs 1024 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_u_maze --num_envs 1024 --seed 3 $TD3_ARGS &
wait

# Fast batch 2/2: TD3 ant_u_maze s4 + PPO ant_u_maze s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_u_maze --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_u_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_u_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_u_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_u_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_u_maze --num_envs 4096 --seed 4 $PPO_ARGS &
wait

# ── Section B: HER baselines ──────────────────────────────────────────────────

# HER slow batch 1/4: SAC+HER ant_big_maze s0-4 + SAC+HER ant_push s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_big_maze --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_big_maze --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_big_maze --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_big_maze --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_big_maze --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_push --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_push --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_push --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_push --num_envs 1024 --seed 3 $SAC_HER_ARGS &
wait

# HER slow batch 2/4: SAC+HER ant_push s4 + TD3+HER ant_big_maze s0-4 + TD3+HER ant_push s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_push --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_big_maze --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_big_maze --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_big_maze --num_envs 1024 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_big_maze --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_big_maze --num_envs 1024 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_push --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_push --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_push --num_envs 1024 --seed 2 $TD3_HER_ARGS &
wait

# HER slow batch 3/4: TD3+HER ant_push s3-4 + SAC+HER pusher_hard s0-4 + TD3+HER pusher_hard s0-1
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_push --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_push --num_envs 1024 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env pusher_hard --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env pusher_hard --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env pusher_hard --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env pusher_hard --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env pusher_hard --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env pusher_hard --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env pusher_hard --num_envs 1024 --seed 1 $TD3_HER_ARGS &
wait

# HER slow batch 4/4: TD3+HER pusher_hard s2-4
CUDA_VISIBLE_DEVICES=0 python run.py --env pusher_hard --num_envs 1024 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env pusher_hard --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env pusher_hard --num_envs 1024 --seed 4 $TD3_HER_ARGS &
wait

# HER med/fast batch 1/5: SAC+HER humanoid s0-4 + TD3+HER humanoid s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid --num_envs 512 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid --num_envs 512 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid --num_envs 512 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid --num_envs 512 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid --num_envs 512 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid --num_envs 512 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid --num_envs 512 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid --num_envs 512 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid --num_envs 512 --seed 3 $TD3_HER_ARGS &
wait

# HER med/fast batch 2/5: TD3+HER humanoid s4 + SAC+HER ant s0-4 + TD3+HER ant s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid --num_envs 512 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant --num_envs 1024 --seed 2 $TD3_HER_ARGS &
wait

# HER fast batch 3/5: TD3+HER ant s3-4 + SAC+HER ant_u_maze s0-4 + TD3+HER ant_u_maze s0-1
CUDA_VISIBLE_DEVICES=0 python run.py --env ant --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant --num_envs 1024 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_u_maze --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_u_maze --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_u_maze --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_u_maze --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_u_maze --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_u_maze --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_u_maze --num_envs 1024 --seed 1 $TD3_HER_ARGS &
wait

# HER fast batch 4/5: TD3+HER ant_u_maze s2-4 + SAC+HER reacher s0-4 + TD3+HER reacher s0
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_u_maze --num_envs 1024 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_u_maze --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_u_maze --num_envs 1024 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env reacher --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env reacher --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env reacher --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env reacher --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env reacher --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env reacher --num_envs 1024 --seed 0 $TD3_HER_ARGS &
wait

# HER fast batch 5/5: TD3+HER reacher s1-4
CUDA_VISIBLE_DEVICES=0 python run.py --env reacher --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env reacher --num_envs 1024 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env reacher --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env reacher --num_envs 1024 --seed 4 $TD3_HER_ARGS &
wait
