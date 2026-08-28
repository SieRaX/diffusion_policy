# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The research codebase for the paper **"Spatial Attention: Adapting Execution Horizons for Diffusion Policies via Observation Sensitivity"** (Park, Ha, Fu, Park — `spatial attention.pdf` at the repo root), built as a fork of Columbia's Diffusion Policy (Cheng Chi et al.).

**Spatial Attention (SA)** is defined as `Att(o) = E_{a~π(·|o)}[‖∇_o log π(a|o)‖²]` — the expected squared norm of the gradient of the action log-likelihood w.r.t. the observation, i.e. how sensitive the policy's action distribution is to observation perturbations. The paper shows the execution horizon `Ta` that minimizes cumulative likelihood drop under disturbances should shrink where SA is high, giving an adaptive-execution-horizon method (+SA) that beats fixed-horizon DDPM/DDIM/Consistency-Policy baselines on Robomimic (Lift/Can/Square/Tool Hang), on disturbed variants, and on a real Franka robot.

How the paper's pipeline maps to code:
1. **Score estimation via Bayes rule**: `∇_o log π(a|o) = ∇_o log π(o|a) − ∇_o log π(o)`; two NCSN score models are trained by **CGE** (`conditional_gradient_estimator/`, `train_cge.py`). `gather_spatial_attention_CGE.py` evaluates SA per timestep on demonstration data and writes it into the dataset hdf5 as `obs/spatial_attention`. For vision input, SA is computed in a VAE latent space (`conditional_gradient_estimator/VAE/`).
2. **SA forecasting**: `train_seq2seq_attention_transformer.py` trains a Seq2SeqTransformer (`diffusion_policy/model/transformer.py`) that forecasts the SA sequence for a sampled action chunk from the current observation — this avoids generating future observations at inference.
3. **Adaptive execution horizon (AHC runners)**: at rollout, `Ta = min{k | Σ Att^(1/(2γ+1)) > C_att}` (paper Eq. 20) — the chunk is cut where cumulative forecasted SA crosses the threshold `c_att`. Implemented in the `robomimic_lowdim_AHC_*`/ADP runners and driven by the `eval_AHC*.py` entrypoints; `*_by_seed` runners sweep `c_att` to hit target average horizons for fair comparison. Disturbance experiments (moving the target object) live in `env_runner/disturbance_generator/` and the `*disturbance*` runner variants.
4. **Follow-up experiments**: `diffusion_policy/experiments/spatial_attention_exp1` (per-state MSE convergence of flow-matching policies) and `spatial_attention_prelim_perturb` (perturbation sensitivity) extend this line of work; flow-matching (rectified-flow) policies were added for these.
5. **MoH baseline** (comparison target, paper `MoH.pdf` + clone in `MixtureOfHorizons/`): Mixture-of-Horizons ported onto Diffusion Policy — `policy/moh_diffusion_unet_lowdim_policy.py` + `model/diffusion/moh_conditional_unet1d.py` + `workspace/train_moh_unet_lowdim_workspace.py` + `config/train_moh_unet_lowdim_workspace.yaml`. CNN adaptation: horizons are multiples of 4 run as separate UNet forwards, gate head on final per-step features, DDIM-space consensus for dynamic inference. Tests: `tests/test_moh_policy.py`; recipe: `scripts/train_moh_pilot.sh`.

## Environment & commands

Conda env: upstream uses `robodiff` (`conda_environment.yaml`); this fork's own work runs in the `adp` env (`my_env_*.yaml`) — scripts hardcode `PY=/home/cspark/anaconda3/envs/adp/bin/python`.

Common env vars for any run that touches mujoco/robosuite:
```bash
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa
```

### Training (Hydra entrypoint)
```bash
# Config from diffusion_policy/config/
python train.py --config-name=train_flow_matching_unet_lowdim_workspace.yaml \
    task=can_lowdim_abs training.seed=42 training.device=cuda:0 \
    hydra.run.dir='outputs/...'

# Repo-root reproduction configs (low_dim_*.yaml / image_*.yaml) need --config-dir=.
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml ...

# Experiment-local configs
python train.py --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
    --config-name train_flow_matching_lowdim_exp1 task=lift_lowdim_abs ...
```
`task=<name>` swaps in `config/task/<name>.yaml`. `_abs` task variants use absolute actions; never compare abs vs rel results (different action spaces). When changing `horizon`, also override `policy.horizon`, `task.dataset.horizon`, and `task.dataset.pad_after=$((horizon-1))`.

Multi-seed training uses ray: `ray start --head --num-gpus=N` then `python ray_train_multirun.py --config-dir=. --config-name=<cfg> --seeds=42,43,44 --monitor_key=test/mean_score -- <overrides>`. CGE training: `python train_cge.py --config-name=train_conditional_gradient_lowdim_workspace.yaml` (configs in `conditional_gradient_estimator/config/`).

### Evaluation
```bash
python eval.py --checkpoint <run>/checkpoints/<ckpt>.ckpt --output_dir <dir> --device cuda:0
```
`eval_AHC*.py`, `eval_likelihood*.py` are variants that load a checkpoint then rewire `cfg.task.env_runner._target_` (and the noise scheduler to `diffusion_policy/schedulers/scheduling_ddpm.py`) before running.

### Tests (pytest)
```bash
python -m pytest tests/test_normalizer.py                                          # upstream unit tests
python -m pytest diffusion_policy/experiments/spatial_attention_exp1/tests/ -q     # experiment tests
python -m pytest diffusion_policy/experiments/spatial_attention_prelim_perturb/tests/ -q
```
Some `tests/` files need simulator assets/hardware (realsense, robomimic datasets); run the specific file you care about, not the whole dir.

### Shell scripts are recipes
Files in `scripts/` (e.g. `SA_experiment.sh`, `train.sh`) are recipe files: copy/run individual blocks from the repo root, do NOT execute top-to-bottom (the training blocks run for days). They document the exact override sets and output-dir conventions for each experiment.

## Architecture

### Upstream task/method split (unchanged)
Tasks and methods are independent; adding N tasks + M methods costs O(N+M) code at the price of deliberate copy-paste between implementations.

- **Task side**: `Dataset` (`diffusion_policy/dataset/`), `EnvRunner` (`diffusion_policy/env_runner/`), `config/task/<task>.yaml`, optional gym `Env` (`diffusion_policy/env/`).
- **Method side**: `Policy` (`diffusion_policy/policy/`), `Workspace` (`diffusion_policy/workspace/`), `config/<workspace>.yaml`.
- **Interface**: policies take an obs dict of `(B, To, ...)` tensors and return `{"action": (B, Ta, Da)}`. Terminology: `To=n_obs_steps`, `Ta=n_action_steps`, `T=horizon`. Datasets provide `get_normalizer()` → `LinearNormalizer`; normalization happens inside the policy on GPU and the normalizer is saved in the checkpoint.
- **Workspace** owns the whole train/eval lifecycle; the Hydra config fully specifies the experiment; checkpoints are saved at workspace level (all attributes) — the checkpoint payload embeds `cfg`, which is how every eval script reconstructs the policy.
- **Data**: `ReplayBuffer` (`diffusion_policy/common/replay_buffer.py`) stores episodes as zarr (dir or `.zarr.zip`); `SequenceSampler` (`diffusion_policy/common/sampler.py`) handles To/Ta episode-boundary padding — read it before writing any custom sampling.
- **Vectorized eval** uses a modified `AsyncVectorEnv` with fork subprocesses: environments that init OpenGL in the constructor (robosuite) need a `dummy_env_fn` or they segfault in child processes.
- `diffusion_policy/real_world/` + `shared_memory/` are the UR5/RealSense real-robot stack (not used in the simulation experiments).

### Fork-specific additions
- **Flow matching**: `policy/flow_matching_unet_{lowdim,hybrid_image}_policy.py` — rectified flow (data `x0` ↔ noise `x1`, velocity regression, Euler sampling) as a drop-in replacement for the DDPM scheduler; workspaces/configs `train_flow_matching_unet_*_workspace`. Run dirs and wandb groups are tagged abs/rel.
- **Disturbance/robustness runners**: the many `robomimic_lowdim_{AHC,ADP,likelihood}*_runner.py` variants in `env_runner/` (several dated filenames) inject disturbances (see `env_runner/disturbance_generator/`) during rollout and are selected by the matching `eval_*.py` entrypoint at eval time, not via training configs. Multiple files define the same class name (e.g. `RobomimicLowdimAHCRunner`) — the `_target_` path chosen in the eval script decides which is used.
- **`diffusion_policy/experiments/`** — self-contained experiment packages, each with own `config/`, `tests/`, and Hydra `python -m ...` entrypoints:
  - `spatial_attention_exp1`: per-state MSE convergence of flow-matching policies. 3-stage pipeline: train → `...exp1.dense_eval.run_dense_eval` (reads a run dir, writes `dense_eval.npz`) → `...exp1.analysis.run_analysis` (figures + summary). Recipe: `scripts/SA_experiment.sh`.
  - `spatial_attention_prelim_perturb`: perturbation-sensitivity on trained checkpoints via `...prelim_perturb.run_perturb` then its `analysis.run_analysis`. Recipe: `scripts/SA_experiment_raw_perturbation.sh`.
- **`conditional_gradient_estimator/`** — separate parallel package (own workspace/model/dataset/config trees) trained via `train_cge.py`; spatial-attention rendering via `gather_spatial_attention_CGE*.py`.

### Output conventions
Training runs land in `outputs/`, `outputs_HDD4/`, or `data/outputs/` with pattern `<task>_<abs|rel>_<dataset_type>_reproduction/train_by_seed_<method>/seed<seed>_<timestamp>_<name>...`; scripts locate the latest run with `ls -td .../seed* | head -1`. Experiment outputs (npz, figures) are written next to the checkpoint/run they came from.

## Gotchas

- **Hydra + `=` in checkpoint paths**: checkpoints are named `epoch=NNNN-....ckpt`; Hydra's override parser rejects the inner `=` unless the value is wrapped in single quotes: `checkpoint="'$CKPT'"`.
- **zsh word-splitting**: don't collect multi-word Hydra flags into an unquoted variable (`COMMON="--config-dir x"`) — zsh passes it as one argument and Hydra fails. Write flags out literally.
- Normalization is the most common source of bugs; when debugging, print the `scale`/`bias` of each `LinearNormalizer` key.
