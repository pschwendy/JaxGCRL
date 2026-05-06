#!/bin/bash
# new_envs.sh — Sweep: ant_hardest_maze, arm_binpick_hard,
#   humanoid_u_maze, humanoid_big_maze.
# Priority order: [SCCRLv5 — done, commented out] → CRL → baselines.
# Seeds: 0-4 (5 per config). 9 GPUs (0-8). 14 batches total.
#
# num_envs: ant_hardest / binpick      → 1024 (PPO: 4096)
#           humanoid_* CRL             → 512  (271-dim obs OOMs at 512)
#           humanoid_* SAC/TD3/HER     → 512  (PPO: 4096)
#
# Batching: partial batches at section boundaries are merged with the
# next section to keep all 9 GPUs busy. 120 runs → 13×9 + 1×3 = min achievable.

SCCRL_ARGS="sccrlv5 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0"
CRL_ARGS="crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln"
SAC_ARGS="sac --discounting 0.99 --unroll_length 62"
TD3_ARGS="td3 --discounting 0.99 --unroll_length 62"
PPO_ARGS="ppo --discounting 0.97 --batch_size 256 --num_minibatches 16"
SAC_HER_ARGS="sac --discounting 0.99 --unroll_length 62 --use_her"
TD3_HER_ARGS="td3 --discounting 0.99 --unroll_length 62 --use_her"

# ── Section 1: SCCRLv5 — DONE (commented out) ────────────────────────────────

# # SCCRL batch 1: ant_hardest s0-4 + binpick s0-3
# CUDA_VISIBLE_DEVICES=0 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $SCCRL_ARGS &
# wait

# # SCCRL batch 2: binpick s4 + hum_u s0-4 + hum_big s0-2
# CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $SCCRL_ARGS &
# wait

# # SCCRL batch 3: hum_big s3-4
# CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $SCCRL_ARGS &
# wait

# ── Section 2: CRL (20 runs) ──────────────────────────────────────────────────
# Batches 1-2 are full; the 2-run CRL overflow merges with SAC slow start (batch 3).

# CRL batch 1: ant_hardest s0-4 + binpick s0-3
# CUDA_VISIBLE_DEVICES=0 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $CRL_ARGS &
# wait

# # CRL batch 2: binpick s4 + hum_u s0-4 + hum_big s0-2
# CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $CRL_ARGS &
# wait

# CRL overflow + SAC slow start: hum_big s3-4 (CRL) + SAC ant_hardest s0-4 + SAC binpick s0-1
# CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=7 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $SAC_ARGS &
# wait

# ── Section 3: Baselines — slow envs (ant_hardest_maze, arm_binpick_hard) ─────
# 5 agents × 2 envs × 5 seeds = 50 runs; SAC ant_hardest s0-4 + SAC binpick s0-1 already done above.

# SAC binpick s2-4 + TD3 ant_hardest s0-4 + TD3 binpick s0
# CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_hardest_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_hardest_maze --num_envs 4096 --seed 4 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $TD3_ARGS &
wait

# TD3 binpick s1-4 + PPO ant_hardest s0-4
# CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $TD3_ARGS &

# PPO binpick s0-4 + SAC+HER ant_hardest s0-3
# CUDA_VISIBLE_DEVICES=0 python run.py --env arm_binpick_hard --num_envs 4096 --seed 0 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 4096 --seed 1 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 4096 --seed 2 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env arm_binpick_hard --num_envs 4096 --seed 3 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env arm_binpick_hard --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env ant_hardest_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_hardest_maze --num_envs 1024 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_hardest_maze --num_envs 1024 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant_hardest_maze --num_envs 1024 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $SAC_HER_ARGS &
wait

# SAC+HER ant_hardest s4 + SAC+HER binpick s0-4 + TD3+HER ant_hardest s0-2

# CUDA_VISIBLE_DEVICES=1 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $SAC_HER_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $SAC_HER_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $SAC_HER_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $SAC_HER_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $SAC_HER_ARGS &

# wait

# Slow → humanoid bridge: TD3+HER ant_hardest s3-4 + TD3+HER binpick s0-4 + SAC hum_u s0-1

# CUDA_VISIBLE_DEVICES=2 python run.py --env arm_binpick_hard --num_envs 1024 --seed 0 $TD3_HER_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env arm_binpick_hard --num_envs 1024 --seed 1 $TD3_HER_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env arm_binpick_hard --num_envs 1024 --seed 2 $TD3_HER_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env arm_binpick_hard --num_envs 1024 --seed 3 $TD3_HER_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env arm_binpick_hard --num_envs 1024 --seed 4 $TD3_HER_ARGS &

# wait

# ── Section 4: Baselines — humanoid mazes (hum_u, hum_big) ───────────────────
# 5 agents × 2 envs × 5 seeds = 50 runs; SAC hum_u s0-1 already done above.

# SAC hum_u s2-4 + SAC hum_big s0-4 + TD3 hum_u s0
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $TD3_ARGS &
wait

# TD3 hum_u s1-4 + TD3 hum_big s0-4
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $TD3_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $TD3_ARGS &
wait

# PPO hum_u s0-4 + PPO hum_big s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_u_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 4096 --seed 3 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_big_maze --num_envs 4096 --seed 0 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 4096 --seed 1 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 4096 --seed 2 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 4096 --seed 3 $PPO_ARGS &
wait

# PPO hum_big s4 + SAC+HER hum_u s0-4 + SAC+HER hum_big s0-2
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 4096 --seed 4 $PPO_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $SAC_HER_ARGS &
wait

# SAC+HER hum_big s3-4 + TD3+HER hum_u s0-4 + TD3+HER hum_big s0-1
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $SAC_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env humanoid_u_maze --num_envs 512 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=5 python run.py --env humanoid_u_maze --num_envs 512 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=6 python run.py --env humanoid_u_maze --num_envs 512 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_big_maze --num_envs 512 --seed 0 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_big_maze --num_envs 512 --seed 1 $TD3_HER_ARGS &
wait

# TD3+HER hum_big s2-4
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid_big_maze --num_envs 512 --seed 2 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid_big_maze --num_envs 512 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid_big_maze --num_envs 512 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env ant_hardest_maze --num_envs 1024 --seed 3 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=4 python run.py --env ant_hardest_maze --num_envs 1024 --seed 4 $TD3_HER_ARGS &
CUDA_VISIBLE_DEVICES=7 python run.py --env humanoid_u_maze --num_envs 512 --seed 0 $SAC_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env humanoid_u_maze --num_envs 512 --seed 1 $SAC_ARGS &
wait
