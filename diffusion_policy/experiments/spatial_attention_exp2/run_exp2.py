"""Hydra entry point (ONE binary) for Experiment 2: builds/loads the full-dataset
similarity index, computes S_vel on all query states + S_end on a stratified subset, and
saves exp2.npz. Ablation cells are the same binary with a single override
(probe_type=… / similarity_space=… / directions=…).

Usage (from repo root):
    MUJOCO_GL=osmesa python -m \
      diffusion_policy.experiments.spatial_attention_exp2.run_exp2 \
      checkpoint=<ckpt.ckpt> output_dir=<dir> device=cuda:0
"""
import pathlib

import hydra
from omegaconf import OmegaConf

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.experiments.spatial_attention_exp2 import runner


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config')),
    config_name='exp2')
def main(cfg):
    OmegaConf.resolve(cfg)
    runner.run(cfg)


if __name__ == '__main__':
    main()
