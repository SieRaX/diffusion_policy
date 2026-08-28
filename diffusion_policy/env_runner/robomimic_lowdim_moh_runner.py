# Sequential single-env evaluation runner mirroring the episode/seed protocol
# of robomimic_lowdim_AHC_runner_by_seed (no disturbance). Supports:
#   mode='fixed'   — execute a fixed n_action_steps prefix per chunk
#   mode='dynamic' — MoH cross-horizon-consensus prefix (policy.predict_action
#                    with use_dynamic=True); prefix length varies per chunk
#   mode='sa'      — Spatial Attention adaptive prefix on a plain policy:
#                    a seq2seq attention estimator forecasts SA for the
#                    sampled chunk; the prefix is cut where cumulative
#                    SA^exponent exceeds c_att (mirrors
#                    robomimic_lowdim_ADP_jumpying_disturbance_runner logic)
import os
import numpy as np
import torch
import tqdm
import json

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper_Gymnasium
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from mujoco_py.builder import MujocoException
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    return env


class RobomimicLowdimMoHRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            dataset_path,
            obs_keys,
            n_test=50,
            test_start_seed=100000,
            max_steps=400,
            n_obs_steps=2,
            n_latency_steps=0,
            abs_action=False,
            tqdm_interval_sec=5.0,
        ):
        super().__init__(output_dir)
        dataset_path = os.path.expanduser(dataset_path)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def make_env():
            robomimic_env = create_env(env_meta=env_meta, obs_keys=obs_keys)
            return MultiStepWrapper_Gymnasium(
                RobomimicLowdimWrapper(
                    env=robomimic_env,
                    obs_keys=obs_keys,
                    init_state=None,
                ),
                n_obs_steps=n_obs_steps + n_latency_steps,
                # step() iterates over whatever sequence it is given;
                # this only caps obs/reward aggregation
                n_action_steps=1,
                max_episode_steps=max_steps,
            )

        self.make_env = make_env
        self.n_test = n_test
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy, mode='dynamic', n_action_steps=None, n_episodes=None,
            sa=None):
        """Returns dict with test/mean_score, avg executed horizon and
        per-episode records. n_episodes overrides self.n_test (quick probes).
        For mode='sa', sa is a dict with keys: estimator, normalizer, c_att,
        and optionally attention_exponent (1.0), min_n_action_steps (2)."""
        assert mode in ('fixed', 'dynamic', 'sa')
        if mode == 'sa':
            assert sa is not None and 'estimator' in sa and 'normalizer' in sa \
                and 'c_att' in sa
        device = policy.device
        n_eps = self.n_test if n_episodes is None else n_episodes

        env = self.make_env()
        rewards, ep_lengths, ep_mean_horizons = [], [], []
        try:
            for ep in tqdm.tqdm(range(n_eps), desc=f"MoH eval ({mode})",
                    mininterval=self.tqdm_interval_sec):
                seed = self.test_start_seed + ep
                np.random.seed(seed)
                torch.manual_seed(seed)
                obs, _ = env.reset(seed=seed)
                policy.reset()

                max_reward = 0.0
                done = False
                steps = 0
                horizon_lengths = []
                try:
                    max_reward, steps = self._episode_loop(
                        env, policy, obs, mode, n_action_steps,
                        horizon_lengths, device, sa=sa)
                except MujocoException as e:
                    # tool_hang occasionally destabilizes the sim (QACC
                    # explosion); count the episode as a failure and rebuild
                    # the env (its sim state is corrupted)
                    print(f"[moh_runner] episode {ep} MujocoException ({e}); "
                          f"counted as failure")
                    env.close()
                    env = self.make_env()

                rewards.append(max_reward)
                ep_lengths.append(steps)
                ep_mean_horizons.append(
                    float(np.mean(horizon_lengths)) if horizon_lengths else 0.0)
        finally:
            env.close()

        log = {
            'test/mean_score': float(np.mean(rewards)),
            'test/avg_exec_horizon': float(np.mean(ep_mean_horizons)),
            'test/avg_episode_steps': float(np.mean(ep_lengths)),
            'episode_rewards': rewards,
            'episode_mean_horizons': ep_mean_horizons,
            'episode_steps': ep_lengths,
        }
        return log

    def _episode_loop(self, env, policy, obs, mode, n_action_steps,
                      horizon_lengths, device, sa=None):
        max_reward = 0.0
        done = False
        steps = 0
        start = policy.n_obs_steps
        if getattr(policy, 'oa_step_convention', True):
            start = policy.n_obs_steps - 1
        max_exec = policy.horizon - start
        while not done and steps < self.max_steps and max_reward < 1 - 1e-4:
            np_obs = {'obs': obs[..., :self.n_obs_steps, :].astype(np.float32)}
            obs_dict = dict_apply(np_obs,
                lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0))
            with torch.no_grad():
                if mode == 'dynamic':
                    result = policy.predict_action(obs_dict, use_dynamic=True)
                    action = result['action'][0].cpu().numpy()
                elif mode == 'sa':
                    # sample the full executable chunk, then cut it where the
                    # forecasted cumulative spatial attention crosses c_att
                    saved = policy.n_action_steps
                    policy.n_action_steps = max_exec
                    result = policy.predict_action(obs_dict)
                    policy.n_action_steps = saved
                    nobs = sa['normalizer']['obs'].normalize(obs_dict['obs'])
                    naction = sa['normalizer']['action'].normalize(
                        result['action_pred'])
                    out = sa['estimator'](
                        nobs.reshape(nobs.shape[0], -1), naction)
                    att = sa['normalizer']['spatial_attention']\
                        .unnormalize(out).detach().cpu().squeeze()
                    att = torch.pow(att, sa.get('attention_exponent', 1.0))
                    cumsum = torch.cumsum(att, dim=-1)
                    over = torch.where(cumsum > sa['c_att'])[0]
                    idx = over[0].item() if len(over) > 0 else att.shape[0]
                    idx += 1
                    idx = max(idx, sa.get('min_n_action_steps', 2))
                    action = result['action'][0, :idx].cpu().numpy()
                else:
                    saved = policy.n_action_steps
                    if n_action_steps is not None:
                        policy.n_action_steps = n_action_steps
                    result = policy.predict_action(obs_dict)
                    policy.n_action_steps = saved
                    action = result['action'][0].cpu().numpy()
            action = action[self.n_latency_steps:]
            if not np.all(np.isfinite(action)):
                raise RuntimeError("Nan or Inf action")

            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)
            horizon_lengths.append(len(env_action))

            # MultiStepWrapper_Gymnasium aggregates reward with 'max'
            # over the whole episode so far, and latches termination
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            max_reward = float(reward)
            steps += len(env_action)
        return max_reward, steps

    def undo_transform_action(self, action):
        # 10-dim (pos3 + rot6d + gripper1) -> 7-dim (pos3 + axis_angle3 + gripper1)
        # dual-arm (e.g. transport): 20-dim -> 14-dim, arms transformed per-arm
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)
        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction
