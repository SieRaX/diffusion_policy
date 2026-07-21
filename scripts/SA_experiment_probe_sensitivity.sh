#!/usr/bin/env bash
# Experiment 2 — probe-based observation-sensitivity S(o) on already-trained flow-matching
# checkpoints. Recipe file: run from the repo root and copy the block you need.
# Task / obs variant / dims are DERIVED FROM THE CHECKPOINT (no task= override).
#
# Probe pool = the FULL dataset; query set = eval_episodes only. One binary (run_exp2)
# builds/loads the full-dataset similarity index, computes the cheap velocity estimator
# S_vel on all query states and the expensive endpoint estimator S_end on a stratified
# subset, and saves exp2.npz. Ablation cells are the SAME binary with a single override
# (probe_type=… / similarity_space=… / directions=…); cells that share
# checkpoint+similarity_space+stride REUSE the cached index (index_reused=True).
#
# HYDRA + '=' in the checkpoint name: checkpoints are named epoch=NNNN.ckpt, so both the
# checkpoint value AND the output_dir (which contains the checkpoint stem) contain '=',
# which Hydra's override parser rejects. Wrap BOTH in SINGLE quotes -> checkpoint="'$CK'"
# so Hydra treats them as literal strings.

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa
PY=/home/cspark/anaconda3/envs/adp/bin/python

# 0) unit tests (verify implementation)
$PY -m pytest diffusion_policy/experiments/spatial_attention_exp2/tests/ -q

# 1) low_dim MAIN run  (default probe_type=real_neighbor_diff, similarity_space=latent,
#    directions=config). Builds+caches the full-dataset index next to the checkpoint.
LOWDIM_CKPT=outputs_HDD4/tool_hang_lowdim_ph_abs_reproduction/train_by_seed_flow_matching/seed0_2026.07.20-11.23.32_train_flow_matching_unet_lowdim_tool_hang_lowdim_cnn_48/checkpoints/epoch=0150-test_mean_score=0.640.ckpt
BASE="${LOWDIM_CKPT%.ckpt}/exp2_tool_hang_abs"
$PY -m diffusion_policy.experiments.spatial_attention_exp2.run_exp2 \
  checkpoint="'$LOWDIM_CKPT'" output_dir="'$BASE/real_neighbor_diff'" eval_episodes='[0]' device=cuda:0
$PY -m diffusion_policy.experiments.spatial_attention_exp2.analysis.run_analysis \
  npz="'$BASE/real_neighbor_diff/exp2.npz'" output_dir="'$BASE/real_neighbor_diff'" \
  compare_prelim_npz="'${LOWDIM_CKPT%.ckpt}/prelim_perturb/episode_0/obs_noise/perturb.npz'"

# 2) probe-type ablation  (single override each; REUSES the index cache from cell 1)
$PY -m diffusion_policy.experiments.spatial_attention_exp2.run_exp2 \
  checkpoint="'$LOWDIM_CKPT'" output_dir="'$BASE/diag_std_noise'" probe_type=diag_std_noise device=cuda:0
$PY -m diffusion_policy.experiments.spatial_attention_exp2.run_exp2 \
  checkpoint="'$LOWDIM_CKPT'" output_dir="'$BASE/fullcov_noise'" probe_type=fullcov_noise device=cuda:0
for cell in diag_std_noise fullcov_noise; do
  $PY -m diffusion_policy.experiments.spatial_attention_exp2.analysis.run_analysis \
    npz="'$BASE/$cell/exp2.npz'" output_dir="'$BASE/$cell'"
done

# 3) directions ablation  (temporal | both; config is the default in cell 1)
$PY -m diffusion_policy.experiments.spatial_attention_exp2.run_exp2 \
  checkpoint="'$LOWDIM_CKPT'" output_dir="'$BASE/dir_both'" directions=both device=cuda:0
$PY -m diffusion_policy.experiments.spatial_attention_exp2.analysis.run_analysis \
  npz="'$BASE/dir_both/exp2.npz'" output_dir="'$BASE/dir_both'"

# 4) similarity-space ablation  (IMAGE checkpoint — pixel vs latent; each builds its own
#    one-time latent/pixel index cache). Reduce eval_episodes / stride before N or M.
IMAGE_CKPT=outputs_HDD4/can_image_ph_reproduction/train_by_seed_flow_matching/seed0_2026.07.20-00.21.07_train_flow_matching_unet_hybrid_can_image_cnn_32/checkpoints/epoch=0300-test_mean_score=0.960.ckpt
IMG_BASE="${IMAGE_CKPT%.ckpt}/exp2_can_image"
for space in latent pixel; do
  $PY -m diffusion_policy.experiments.spatial_attention_exp2.run_exp2 \
    checkpoint="'$IMAGE_CKPT'" output_dir="'$IMG_BASE/$space'" similarity_space=$space \
    eval_episodes='[0]' stride=5 device=cuda:0
  $PY -m diffusion_policy.experiments.spatial_attention_exp2.analysis.run_analysis \
    npz="'$IMG_BASE/$space/exp2.npz'" output_dir="'$IMG_BASE/$space'"
done

# Other checkpoints are simply additional cell-1 invocations with a different checkpoint path.
# Tune measurement params by appending e.g.  K=32 N=8 num_tz_draws=32 stride=1 include_gripper=false
# to a run_exp2 line. figures.timeline_episode must be one of eval_episodes.
