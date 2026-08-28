#!/usr/bin/env bash
# MoH (Mixture-of-Horizons, Jing et al. ICML'26) baseline pilot on Diffusion Policy.
# Tasks: square + tool_hang (lowdim, abs actions). Sampler: DDIM 16 steps.
# Recipe file: run from the repo root ON THE TRAINING MACHINE (needs the
# robomimic datasets under data/robomimic/datasets/<task>/ph/low_dim_abs.hdf5
# and a full env: robomimic/robosuite/hydra/zarr/diffusers).
#
# MoH-on-CNN caveats (documented in the policy docstring): candidate horizons
# are multiples of 4 (UNet downsampling), each horizon is a separate UNet
# forward (6 horizons -> ~6x train step cost vs plain DP), gate head sits on
# the UNet's final per-step feature.

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa
# adp env on this machine (recreated 2026-08-21 from my_env_4.yaml; conda env
# vars already set MUJOCO_GL + LD_LIBRARY_PATH on activation)
PY=/home/jsha/anaconda3/envs/adp/bin/python
export LD_LIBRARY_PATH=/home/jsha/anaconda3/envs/adp/lib:$LD_LIBRARY_PATH

# 0) unit tests (verify implementation; no simulator needed)
$PY -m pytest tests/test_moh_policy.py -q

# 1) debug run — 2 epochs x 3 steps including one rollout, verifies the whole
#    train->rollout->checkpoint loop end-to-end (~minutes)
$PY train.py --config-name=train_moh_unet_lowdim_workspace \
  task=square_lowdim_abs training.debug=True logging.mode=offline \
  training.device=cuda:0 \
  hydra.run.dir='outputs_HDD4/debug/moh_pilot_${task.name}_abs'

# 2) pilot training (seed 42; horizon 48, candidates {8,...,48}, DDIM 16)
$PY train.py --config-name=train_moh_unet_lowdim_workspace \
  task=square_lowdim_abs \
  training.seed=42 training.device=cuda:0 \
  logging.group='${task.name}_${task.dataset_type}_moh_ddim' \
  logging.name='seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_moh_ddim_cnn_${horizon}' \
  hydra.run.dir='outputs_HDD4/${task.name}_abs_${task.dataset_type}_reproduction/train_by_seed_moh_ddim/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

$PY train.py --config-name=train_moh_unet_lowdim_workspace \
  task=tool_hang_lowdim_abs \
  training.seed=42 training.device=cuda:1 \
  logging.group='${task.name}_${task.dataset_type}_moh_ddim' \
  logging.name='seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_moh_ddim_cnn_${horizon}' \
  hydra.run.dir='outputs_HDD4/${task.name}_abs_${task.dataset_type}_reproduction/train_by_seed_moh_ddim/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

# Training-time rollouts use the FIXED prefix n_action_steps=8 (test/mean_score
# in wandb is therefore the fixed-horizon MoH score). Dynamic (consensus)
# evaluation and the scale_ratio sweep to match +SA's average Ta come next as a
# dedicated eval runner (phase 2) — policy.predict_action(use_dynamic=True) is
# already implemented and unit-tested.
#
# Useful overrides:
#   horizons='[16,32,48]'   fewer horizon branches (cheaper, MoH d=10-style)
#   training.num_epochs=300 shorter pilot
#   dataloader.batch_size=64 if VRAM-bound (6 UNet forwards per step)
