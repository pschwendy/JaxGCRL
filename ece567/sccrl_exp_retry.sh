#!/bin/bash
# sccrl_exp_retry.sh — retry SCCRLv5 humanoid seeds 1-4 and all reacher baselines.
# Uses run.py format (exp_paper_hparams.sh used jaxgcrl for the same runs).
#
# Humanoid seed 0 already ran in sccrl_exp.sh batch 1.
# Reacher CRL/SAC/TD3/PPO were commented out in exp_paper_hparams.sh.

SCCRL_ARGS="sccrlv5 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0"
SCCRLV7_ARGS="sccrlv7 --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0"
CRL_ARGS="crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln"
SAC_ARGS="sac --discounting 0.99 --unroll_length 62"
TD3_ARGS="td3 --discounting 0.99 --unroll_length 62"
PPO_ARGS="ppo --discounting 0.97 --batch_size 256 --num_minibatches 16"
SAC_HER_ARGS="sac --discounting 0.99 --unroll_length 62 --use_her"
TD3_HER_ARGS="td3 --discounting 0.99 --unroll_length 62 --use_her"

# Batch 1: SCCRLv5 humanoid s1-4 + CRL reacher s0-4 (humanoid is bottleneck)
CUDA_VISIBLE_DEVICES=0 python run.py --env ant_big_maze --num_envs 1024 $SCCRLV7_ARGS &
CUDA_VISIBLE_DEVICES=1 python run.py --env humanoid --num_envs 512 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=2 python run.py --env humanoid --num_envs 512 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=3 python run.py --env humanoid --num_envs 512 --seed 4 $SCCRL_ARGS &
# ablations: sccrlv6 (no alignment loss) + sccrlv7 (random subgoals)
CUDA_VISIBLE_DEVICES=4 python run.py --env ant --num_envs 1024 sccrlv6 --policy_lr 6e-4 --cvae_alignment_coeff 0.0 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
CUDA_VISIBLE_DEVICES=5 python run.py --env ant_u_maze --num_envs 1024 sccrlv6 --policy_lr 6e-4 --cvae_alignment_coeff 0.0 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
CUDA_VISIBLE_DEVICES=6 python run.py --env ant_big_maze --num_envs 1024 sccrlv6 --policy_lr 6e-4 --cvae_alignment_coeff 0.0 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
CUDA_VISIBLE_DEVICES=7 python run.py --env ant --num_envs 1024 $SCCRLV7_ARGS &
CUDA_VISIBLE_DEVICES=8 python run.py --env ant_u_maze --num_envs 1024 $SCCRLV7_ARGS &

wait

# Batch 1/3: SAC ant_ball s0-4 + TD3 ant_ball s0-3
CUDA_VISIBLE_DEVICES=0 python run.py --env humanoid --num_envs 512 --seed 1 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES=0 python run.py --env ant_ball --num_envs 1024 --seed 0 $SAC_ARGS &
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

# CUDA_VISIBLE_DEVICES=4 python run.py --env reacher --num_envs 1024 --seed 0 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env reacher --num_envs 1024 --seed 1 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env reacher --num_envs 1024 --seed 2 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=7 python run.py --env reacher --num_envs 1024 --seed 3 $CRL_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env reacher --num_envs 1024 --seed 4 $CRL_ARGS &
# wait

# # Batch 2: SAC reacher s0-4 + TD3 reacher s0-3
# CUDA_VISIBLE_DEVICES=0 python run.py --env reacher --num_envs 1024 --seed 0 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env reacher --num_envs 1024 --seed 1 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env reacher --num_envs 1024 --seed 2 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env reacher --num_envs 1024 --seed 3 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env reacher --num_envs 1024 --seed 4 $SAC_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env reacher --num_envs 1024 --seed 0 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=6 python run.py --env reacher --num_envs 1024 --seed 1 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=7 python run.py --env reacher --num_envs 1024 --seed 2 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=8 python run.py --env reacher --num_envs 1024 --seed 3 $TD3_ARGS &
# wait

# # Batch 3: TD3 reacher s4 + PPO reacher s0-4
# CUDA_VISIBLE_DEVICES=0 python run.py --env reacher --num_envs 1024 --seed 4 $TD3_ARGS &
# CUDA_VISIBLE_DEVICES=1 python run.py --env reacher --num_envs 4096 --seed 0 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=2 python run.py --env reacher --num_envs 4096 --seed 1 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=3 python run.py --env reacher --num_envs 4096 --seed 2 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=4 python run.py --env reacher --num_envs 4096 --seed 3 $PPO_ARGS &
# CUDA_VISIBLE_DEVICES=5 python run.py --env reacher --num_envs 4096 --seed 4 $PPO_ARGS &
# wait
