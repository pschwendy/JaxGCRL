#!/bin/bash
# new_envs.sh — New environment sweep: ant_hardest_maze, arm_binpick_hard,
#   humanoid_u_maze, humanoid_big_maze, humanoid_hardest_maze.
# Priority order: SCCRLv5 → CRL → baselines (SAC/TD3/PPO/SAC+HER/TD3+HER).
# Seeds: 0-4 (5 per config). 9 GPUs (0-8).
#
# num_envs: ant_hardest / binpick  → 1024 (PPO: 4096)
#           humanoid_*             → 512  (PPO: 4096)
#
# Batching note: SCCRL/CRL sections mix slow+medium tiers (25 runs each → 3 batches).
#   Baselines split by tier: slow (50 runs → 6 batches), humanoid (75 runs → 9 batches).

SCCRL_ARGS="sccrlv5 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0"
CRL_ARGS="crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln"
SAC_ARGS="sac --discounting 0.99 --unroll_length 62"
TD3_ARGS="td3 --discounting 0.99 --unroll_length 62"
PPO_ARGS="ppo --discounting 0.97 --batch_size 256 --num_minibatches 16"
SAC_HER_ARGS="sac --discounting 0.99 --unroll_length 62 --use_her"
TD3_HER_ARGS="td3 --discounting 0.99 --unroll_length 62 --use_her"

# ── Section 1: SCCRLv5 (25 runs → 3 batches) ─────────────────────────────────

# SCCRL batch 1/3: ant_hardest s0-4 + binpick s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $SCCRL_ARGS &
wait

# SCCRL batch 2/3: binpick s4 + hum_u s0-4 + hum_big s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $SCCRL_ARGS &
wait

# SCCRL batch 3/3: hum_big s3-4 + hum_hardest s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 4 $SCCRL_ARGS &
wait

# ── Section 2: CRL (25 runs → 3 batches) ─────────────────────────────────────

# CRL batch 1/3: ant_hardest s0-4 + binpick s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $CRL_ARGS &
wait

# CRL batch 2/3: binpick s4 + hum_u s0-4 + hum_big s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $CRL_ARGS &
wait

# CRL batch 3/3: hum_big s3-4 + hum_hardest s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 0 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 2 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 3 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 4 $CRL_ARGS &
wait

# ── Section 3: Baselines — slow envs (ant_hardest_maze, arm_binpick_hard) ─────
# 5 agents × 2 envs × 5 seeds = 50 runs → 6 batches

# Slow batch 1/6: SAC ant_hardest s0-4 + SAC binpick s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $SAC_ARGS &
wait

# Slow batch 2/6: SAC binpick s4 + TD3 ant_hardest s0-4 + TD3 binpick s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $TD3_ARGS &
wait

# Slow batch 3/6: TD3 binpick s3-4 + PPO ant_hardest s0-4 + PPO binpick s0-1
CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_hardest_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_hardest_maze --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 4096 --seed 1 $PPO_ARGS &
wait

# Slow batch 4/6: PPO binpick s2-4 + SAC+HER ant_hardest s0-4 + SAC+HER binpick s0
CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $SAC_HER_ARGS &
wait

# Slow batch 5/6: SAC+HER binpick s1-4 + TD3+HER ant_hardest s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $TD3_HER_ARGS &
wait

# Slow batch 6/6: TD3+HER binpick s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $TD3_HER_ARGS &
wait

# ── Section 4: Baselines — humanoid mazes ─────────────────────────────────────
# 5 agents × 3 envs × 5 seeds = 75 runs → 9 batches
# num_envs: 512 (SAC/TD3), 4096 (PPO)

# Humanoid batch 1/9: SAC hum_u s0-4 + SAC hum_big s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $SAC_ARGS &
wait

# Humanoid batch 2/9: SAC hum_big s4 + SAC hum_hardest s0-4 + TD3 hum_u s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $TD3_ARGS &
wait

# Humanoid batch 3/9: TD3 hum_u s3-4 + TD3 hum_big s0-4 + TD3 hum_hardest s0-1
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 1 $TD3_ARGS &
wait

# Humanoid batch 4/9: TD3 hum_hardest s2-4 + PPO hum_u s0-4 + PPO hum_big s0
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_u_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_u_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_u_maze --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 4096 --seed 0 $PPO_ARGS &
wait

# Humanoid batch 5/9: PPO hum_big s1-4 + PPO hum_hardest s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_big_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_big_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_big_maze --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_hardest_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_hardest_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_hardest_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_hardest_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_hardest_maze --num_envs 4096 --seed 4 $PPO_ARGS &
wait

# Humanoid batch 6/9: SAC+HER hum_u s0-4 + SAC+HER hum_big s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $SAC_HER_ARGS &
wait

# Humanoid batch 7/9: SAC+HER hum_big s4 + SAC+HER hum_hardest s0-4 + TD3+HER hum_u s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $TD3_HER_ARGS &
wait

# Humanoid batch 8/9: TD3+HER hum_u s3-4 + TD3+HER hum_big s0-4 + TD3+HER hum_hardest s0-1
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 1 $TD3_HER_ARGS &
wait

# Humanoid batch 9/9: TD3+HER hum_hardest s2-4
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_hardest_maze --num_envs 512 --seed 4 $TD3_HER_ARGS &
wait
