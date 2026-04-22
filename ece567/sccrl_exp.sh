# CUDA_VISIBLE_DEVICES="0" python run.py --env ant --num_envs 1024 sccrlv5 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
# CUDA_VISIBLE_DEVICES="1" python run.py --env ant_u_maze --num_envs 1024 sccrlv5 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
# CUDA_VISIBLE_DEVICES="2" python run.py --env ant_big_maze --num_envs 1024 sccrlv5 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
# CUDA_VISIBLE_DEVICES="3" python run.py --env ant --num_envs 1024 sccrlv6 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
# CUDA_VISIBLE_DEVICES="4" python run.py --env ant_u_maze --num_envs 1024 sccrlv6 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
# CUDA_VISIBLE_DEVICES="5" python run.py --env ant_big_maze --num_envs 1024 sccrlv6 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0 &
# CUDA_VISIBLE_DEVICES="6" python run.py --env ant --num_envs 1024 crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 &
# CUDA_VISIBLE_DEVICES="7" python run.py --env ant_u_maze --num_envs 1024 crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 &
# CUDA_VISIBLE_DEVICES="8" python run.py --env ant_big_maze --num_envs 1024 crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 &

# wait

# SCCRLv5 is the victor!
CRL_ARGS="crl --policy_lr 6e-4 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln"
SCCRL_ARGS="sccrlv5 --policy_lr 6e-4 --cvae_alignment_coeff 0.1 --contrastive_loss_fn sym_infonce --energy_fn l2 --n_hidden 4 --h_dim 1024 --use_ln --discounting 0.999 --subgoal_discounting 0.99 --alpha_subgoal=1.0"
# CUDA_VISIBLE_DEVICES="0" python run.py --env ant_u_maze --num_envs 1024 --seed 1 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="1" python run.py --env ant_u_maze --num_envs 1024 --seed 2 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="2" python run.py --env ant_u_maze --num_envs 1024 --seed 3 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="3" python run.py --env ant_u_maze --num_envs 1024 --seed 4 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="4" python run.py --env ant_big_maze --num_envs 1024 --seed 1 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="5" python run.py --env ant_big_maze --num_envs 1024 --seed 2 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="6" python run.py --env ant_big_maze --num_envs 1024 --seed 3 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="7" python run.py --env ant_big_maze --num_envs 1024 --seed 4 $SCCRL_ARGS &
# CUDA_VISIBLE_DEVICES="8" python run.py --env humanoid --num_envs 512 --seed 0 $SCCRL_ARGS &

# wait

CUDA_VISIBLE_DEVICES="0" python run.py --env humanoid --num_envs 512 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="1" python run.py --env humanoid --num_envs 512 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="2" python run.py --env humanoid --num_envs 512 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="3" python run.py --env humanoid --num_envs 512 --seed 4 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="4" python run.py --env ant_ball --num_envs 1024 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="5" python run.py --env ant_ball --num_envs 1024 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="6" python run.py --env ant_ball --num_envs 1024 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="7" python run.py --env ant_ball --num_envs 1024 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="8" python run.py --env ant_ball --num_envs 1024 --seed 4 $SCCRL_ARGS &

wait

CUDA_VISIBLE_DEVICES="0" python run.py --env pusher_hard --num_envs 1024 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="1" python run.py --env pusher_hard --num_envs 1024 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="2" python run.py --env pusher_hard --num_envs 1024 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="3" python run.py --env pusher_hard --num_envs 1024 --seed 4 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="4" python run.py --env ant_push --num_envs 1024 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="5" python run.py --env ant_push --num_envs 1024 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="6" python run.py --env ant_push --num_envs 1024 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="7" python run.py --env ant_push --num_envs 1024 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="8" python run.py --env ant_push --num_envs 1024 --seed 4 $SCCRL_ARGS &

wait

CUDA_VISIBLE_DEVICES="0" python run.py --env pusher_hard --num_envs 1024 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="1" python run.py --env humanoid --num_envs 512 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES="2" python run.py --env ant_u_maze --num_envs 1024 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="3" python run.py --env ant_u_maze --num_envs 1024 --seed 2 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="4" python run.py --env ant_u_maze --num_envs 1024 --seed 3 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="5" python run.py --env ant_u_maze --num_envs 1024 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="6" python run.py --env ant_big_maze --num_envs 1024 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="7" python run.py --env ant_big_maze --num_envs 1024 --seed 2 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="8" python run.py --env ant_big_maze --num_envs 1024 --seed 3 $CRL_ARGS &

wait

CUDA_VISIBLE_DEVICES="0" python run.py --env ant_big_maze --num_envs 1024 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="1" python run.py --env ant_ball --num_envs 1024 --seed 0 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="2" python run.py --env ant_ball --num_envs 1024 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="3" python run.py --env ant_ball --num_envs 1024 --seed 2 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="4" python run.py --env ant_ball --num_envs 1024 --seed 3 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="5" python run.py --env ant_ball --num_envs 1024 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="6" python run.py --env ant_push --num_envs 1024 --seed 0 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="7" python run.py --env ant_push --num_envs 1024 --seed 1 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="8" python run.py --env ant_push --num_envs 1024 --seed 2 $CRL_ARGS &

wait 

CUDA_VISIBLE_DEVICES="0" python run.py --env ant_push --num_envs 1024 --seed 3 $CRL_ARGS &
CUDA_VISIBLE_DEVICES="1" python run.py --env ant_push --num_envs 1024 --seed 4 $CRL_ARGS &
CUDA_VISIBLE_DEVICES=2 jaxgcrl --env reacher --num_envs 1024 --seed 0 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=3 jaxgcrl --env reacher --num_envs 1024 --seed 1 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=4 jaxgcrl --env reacher --num_envs 1024 --seed 2 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=5 jaxgcrl --env reacher --num_envs 1024 --seed 3 $SCCRL_ARGS &
CUDA_VISIBLE_DEVICES=6 jaxgcrl --env reacher --num_envs 1024 --seed 4 $SCCRL_ARGS &

wait



