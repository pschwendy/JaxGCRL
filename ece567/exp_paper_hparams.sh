#!/bin/bash
# Reproduce paper hyperparameters for all agents × 4 envs × 5 seeds (80 runs).
#
# Fixes vs defaults:
#   CRL : policy_lr 3e-4→6e-4, fwd_infonce→sym_infonce, norm→l2
#   SAC : discounting 0.9→0.99, unroll_length 50→62 (UTD 1:16)
#   TD3 : discounting 0.9→0.99, unroll_length 50→62 (UTD 1:16)
#   PPO : discounting 0.9→0.97, num_envs 256→4096
#   ALL : num_envs 256→1024  (humanoid: 512, PPO: 4096)
#
# 8 batches of 10 (one per GPU). Runtime: 8 × (time per run).

CRL_ARGS="crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln"
SAC_ARGS="sac --discounting 0.99 --unroll_length 62"
TD3_ARGS="td3 --discounting 0.99 --unroll_length 62"
PPO_ARGS="ppo --discounting 0.97 --batch_size 256 --num_minibatches 16"

# # ── Ant  (1024 envs, PPO 4096) ─────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES=0 jaxgcrl --env ant --num_envs 1024 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 jaxgcrl --env ant --num_envs 1024 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 jaxgcrl --env ant --num_envs 1024 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 jaxgcrl --env ant --num_envs 1024 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 jaxgcrl --env ant --num_envs 1024 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 jaxgcrl --env ant --num_envs 1024 --seed 0 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=6 jaxgcrl --env ant --num_envs 1024 --seed 1 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=7 jaxgcrl --env ant --num_envs 1024 --seed 2 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=8 jaxgcrl --env ant --num_envs 1024 --seed 3 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=9 jaxgcrl --env ant --num_envs 1024 --seed 4 $SAC_ARGS &
# wait

# CUDA_VISIBLE_DEVICES=0 jaxgcrl --env ant --num_envs 1024 --seed 0 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=1 jaxgcrl --env ant --num_envs 1024 --seed 1 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=2 jaxgcrl --env ant --num_envs 1024 --seed 2 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=3 jaxgcrl --env ant --num_envs 1024 --seed 3 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=4 jaxgcrl --env ant --num_envs 1024 --seed 4 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=5 jaxgcrl --env ant --num_envs 4096 --seed 0 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=6 jaxgcrl --env ant --num_envs 4096 --seed 1 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=7 jaxgcrl --env ant --num_envs 4096 --seed 2 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=8 jaxgcrl --env ant --num_envs 4096 --seed 3 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=9 jaxgcrl --env ant --num_envs 4096 --seed 4 $PPO_ARGS &
# wait

# # ── Humanoid  (CRL/SAC/TD3: 512 envs, PPO: 4096) ──────────────────────────
# CUDA_VISIBLE_DEVICES=0 jaxgcrl --env humanoid --num_envs 512 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 jaxgcrl --env humanoid --num_envs 512 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 jaxgcrl --env humanoid --num_envs 512 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 jaxgcrl --env humanoid --num_envs 512 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 jaxgcrl --env humanoid --num_envs 512 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 jaxgcrl --env humanoid --num_envs 512 --seed 0 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=6 jaxgcrl --env humanoid --num_envs 512 --seed 1 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=7 jaxgcrl --env humanoid --num_envs 512 --seed 2 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=8 jaxgcrl --env humanoid --num_envs 512 --seed 3 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=9 jaxgcrl --env humanoid --num_envs 512 --seed 4 $SAC_ARGS &
# wait

# CUDA_VISIBLE_DEVICES=0 jaxgcrl --env humanoid --num_envs 512 --seed 0 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=1 jaxgcrl --env humanoid --num_envs 512 --seed 1 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=2 jaxgcrl --env humanoid --num_envs 512 --seed 2 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=3 jaxgcrl --env humanoid --num_envs 512 --seed 3 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=4 jaxgcrl --env humanoid --num_envs 512 --seed 4 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=5 jaxgcrl --env humanoid --num_envs 4096 --seed 0 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=6 jaxgcrl --env humanoid --num_envs 4096 --seed 1 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=7 jaxgcrl --env humanoid --num_envs 4096 --seed 2 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=8 jaxgcrl --env humanoid --num_envs 4096 --seed 3 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=9 jaxgcrl --env humanoid --num_envs 4096 --seed 4 $PPO_ARGS &
# wait

# ── Pusher Hard  (1024 envs, PPO: 4096) ───────────────────────────────────
# CUDA_VISIBLE_DEVICES=0 jaxgcrl --env pusher_hard --num_envs 1024 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 jaxgcrl --env pusher_hard --num_envs 1024 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 jaxgcrl --env pusher_hard --num_envs 1024 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 jaxgcrl --env pusher_hard --num_envs 1024 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 jaxgcrl --env pusher_hard --num_envs 1024 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 jaxgcrl --env pusher_hard --num_envs 1024 --seed 0 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=6 jaxgcrl --env pusher_hard --num_envs 1024 --seed 1 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=7 jaxgcrl --env pusher_hard --num_envs 1024 --seed 2 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=8 jaxgcrl --env pusher_hard --num_envs 1024 --seed 3 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=9 jaxgcrl --env pusher_hard --num_envs 1024 --seed 4 $SAC_ARGS &
# wait

CUDA_VISIBLE_DEVICES=0 jaxgcrl --env pusher_hard --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 jaxgcrl --env pusher_hard --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 jaxgcrl --env pusher_hard --num_envs 1024 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=3 jaxgcrl --env pusher_hard --num_envs 1024 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=4 jaxgcrl --env pusher_hard --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=5 jaxgcrl --env pusher_hard --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=6 jaxgcrl --env pusher_hard --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=7 jaxgcrl --env pusher_hard --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=8 jaxgcrl --env pusher_hard --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=9 jaxgcrl --env pusher_hard --num_envs 4096 --seed 4 $PPO_ARGS &
wait

# # ── Reacher  (1024 envs, PPO: 4096) ───────────────────────────────────────
# CUDA_VISIBLE_DEVICES=0 jaxgcrl --env reacher --num_envs 1024 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 jaxgcrl --env reacher --num_envs 1024 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 jaxgcrl --env reacher --num_envs 1024 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 jaxgcrl --env reacher --num_envs 1024 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 jaxgcrl --env reacher --num_envs 1024 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 jaxgcrl --env reacher --num_envs 1024 --seed 0 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=6 jaxgcrl --env reacher --num_envs 1024 --seed 1 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=7 jaxgcrl --env reacher --num_envs 1024 --seed 2 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=8 jaxgcrl --env reacher --num_envs 1024 --seed 3 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=9 jaxgcrl --env reacher --num_envs 1024 --seed 4 $SAC_ARGS &
# wait

# CUDA_VISIBLE_DEVICES=0 jaxgcrl --env reacher --num_envs 1024 --seed 0 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=1 jaxgcrl --env reacher --num_envs 1024 --seed 1 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=2 jaxgcrl --env reacher --num_envs 1024 --seed 2 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=3 jaxgcrl --env reacher --num_envs 1024 --seed 3 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=4 jaxgcrl --env reacher --num_envs 1024 --seed 4 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=5 jaxgcrl --env reacher --num_envs 4096 --seed 0 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=6 jaxgcrl --env reacher --num_envs 4096 --seed 1 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=7 jaxgcrl --env reacher --num_envs 4096 --seed 2 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=8 jaxgcrl --env reacher --num_envs 4096 --seed 3 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=9 jaxgcrl --env reacher --num_envs 4096 --seed 4 $PPO_ARGS &
# wait
