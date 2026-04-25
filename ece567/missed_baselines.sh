#!/bin/bash
# missed_baselines.sh — ant_ball baselines omitted from baselines.sh.
# Covers: SAC, TD3, PPO, SAC+HER, TD3+HER on ant_ball × 5 seeds.
# Speed tier: medium-fast (same as ant / ant_u_maze).
# 9 GPUs (0-8): 5+4 cascade batching.

SAC_ARGS="sac --discounting 0.99 --unroll_length 62"
TD3_ARGS="td3 --discounting 0.99 --unroll_length 62"
PPO_ARGS="ppo --discounting 0.97 --batch_size 256 --num_minibatches 16"
SAC_HER_ARGS="sac --discounting 0.99 --unroll_length 62 --use_her"
TD3_HER_ARGS="td3 --discounting 0.99 --unroll_length 62 --use_her"

# Batch 1/3: SAC ant_ball s0-4 + TD3 ant_ball s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_ball --num_envs 1024 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_ball --num_envs 1024 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_ball --num_envs 1024 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_ball --num_envs 1024 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_ball --num_envs 1024 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_ball --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_ball --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_ball --num_envs 1024 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_ball --num_envs 1024 --seed 3 $TD3_ARGS &
wait

# Batch 2/3: TD3 ant_ball s4 + SAC+HER ant_ball s0-4 + TD3+HER ant_ball s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_ball --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_ball --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_ball --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_ball --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_ball --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_ball --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_ball --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_ball --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_ball --num_envs 1024 --seed 2 $TD3_HER_ARGS &
wait

# Batch 3/3: TD3+HER ant_ball s3-4 + PPO ant_ball s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_ball --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_ball --num_envs 1024 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_ball --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_ball --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_ball --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_ball --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_ball --num_envs 4096 --seed 4 $PPO_ARGS &
wait
