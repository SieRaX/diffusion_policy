#!/usr/bin/env bash
# +SA VISION (paper method) estimator pipeline — recipe file, run blocks as needed.
#
# The paper computes Spatial Attention for image observations in a VAE latent
# space (Eq. 18). Producing a vision estimator for a task needs, in order:
#   1) VAE on that task's images            (notebook/vae_<task>_latent_dim_64/)
#   2) image CGE score models in VAE latent (train_cge.py, hybrid_VAE workspace)
#   3) SA labels on the demo data           (gather_spatial_attention_CGE_image.py)
#   4) seq2seq estimator, horizon 48        (train_seq2seq_attention_transformer_image_version_2_VAE.py)
#
# STATUS (2026-08-26), for the three vision tasks we train:
#   tool_hang : READY — estimator + VAE copied to
#               data/sa_artifacts/tool_hang_vision/attention_estimator_horizon_48/
#               (abs actions, horizon 48; load verified)
#   square    : VAE ready (notebook/vae_square_latent_dim_64), but cspark's image
#               estimator is for REL actions — steps 2-4 must be redone on abs
#   transport : nothing exists — needs all of 1-4
#
# GPU note: both machines are booked with the vision policy trainings until
# ~Aug 29. Run these afterwards, or on whichever machine frees up first.

set -e
P=/home/jsha/anaconda3/envs/adp
export PATH=/home/jsha/anaconda3/bin:$P/bin:$PATH LD_LIBRARY_PATH=$P/lib MUJOCO_GL=osmesa HYDRA_FULL_ERROR=1
ulimit -n 65536

# =============================================================================
# SQUARE  (VAE exists; steps 2-4)
# =============================================================================
VAE_SQUARE=notebook/vae_square_latent_dim_64/vae_square_latent_dim_64_checkpoint_100.pth

$P/bin/python train_cge.py \
  --config-name=train_conditional_gradient_hybrid_workspace_VAE \
  task=square_image_abs \
  wrapper_dataset.vae_checkpoint=$VAE_SQUARE \
  wrapper_dataset.vae_latent_dim=64 wrapper_dataset.vae_input_size=84 \
  device=cuda:0 \
  hydra.run.dir='outputs/cge_vision/${task.name}_abs_${task.dataset_type}/seed0_${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}'

# SA labels (evaluation_diffusion_time 0.1 matches cspark's *_attention_time_0.1 runs)
CGE_SQ=$(ls -td outputs/cge_vision/square_image_abs_ph/seed0_* | head -1)
$P/bin/python gather_spatial_attention_CGE_image.py \
  -c "$CGE_SQ/checkpoints/latest.ckpt" \
  -o "$CGE_SQ/dataset_with_spatial_attention_attention_time_0.1" \
  -t 0.1 -d cuda:0

# estimator (horizon 48 = our prediction horizon)
$P/bin/python train_seq2seq_attention_transformer_image_version_2_VAE.py \
  -p "$CGE_SQ/dataset_with_spatial_attention_attention_time_0.1/low_dim_abs_with_attention.hdf5" \
  -o data/sa_artifacts/square_vision/attention_estimator_horizon_48 \
  -h 48 -d cuda:0

# =============================================================================
# TRANSPORT  (needs a VAE first — notebook/vae_*.ipynb in cspark's repo trains them;
# transport is dual-arm with 4 cameras, so confirm which view(s) the VAE encodes
# before running steps 2-4 with the same commands as SQUARE.)
# =============================================================================
echo "transport: train a VAE (latent 64) on its images first, then repeat the square block"
