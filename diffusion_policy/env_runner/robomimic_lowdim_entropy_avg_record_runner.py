import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import pickle
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env_gymnasium import AsyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.entropy_change_recording_wrapper import (
    EntropyChangeRecordingWrapper,
    MultiStepWrapper_EntropyAdaptive,
    compute_average_action_entropy,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    return env


class RobomimicLowdimEntropyAvgRecordRunner(BaseLowdimRunner):
    """
    Evaluation runner that records entropy-based graphs (diffs, avg_entropy,
    h_star) alongside simulation videos.

    Wrapper stack per env:
        MultiStepWrapper_Gymnasium(
            VideoRecordingWrapper(
                EntropyChangeRecordingWrapper(
                    RobomimicLowdimWrapper(...)
                )
            )
        )

    At each decision step the runner samples `n_action_samples` action chunks
    from the policy, computes per-env entropy via
    `compute_average_action_entropy`, and pushes the results into each worker
    through `env.call_each('record_entropy', ...)`.  The
    EntropyChangeRecordingWrapper renders the three graph panels into the
    frames captured by VideoRecordingWrapper.
    """

    def __init__(
        self,
        output_dir,
        dataset_path,
        obs_keys,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        n_latency_steps=0,
        render_hw=(256, 256),
        render_camera_name='agentview',
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        n_action_samples=64,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # ---- env factory --------------------------------------------------
        # Stack: MultiStep( Video( EntropyChange( RobomimicLowdim ) ) )
        # Mirrors the wrapping pattern in
        # robomimic_lowdim_ADP_jumpying_disturbance_runner_by_avg_length.py
        # (L140-L175) but uses EntropyChangeRecordingWrapper instead of
        # AttentionRecordingWrapper, and MultiStepWrapper_Gymnasium instead
        # of SubMultiStepWrapperwithDisturbance.
        # -------------------------------------------------------------------
        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                obs_keys=obs_keys,
            )
            return MultiStepWrapper_EntropyAdaptive(
                VideoRecordingWrapper(
                    EntropyChangeRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name,
                        ),
                        max_timesteps=max_steps,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # ---- train init functions -----------------------------------------
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                            enable_render=enable_render):
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', "train_seed_" + str(train_idx) + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # EntropyChangeRecordingWrapper sits between Video and
                    # RobomimicLowdim.
                    assert isinstance(env.env.env,
                                      EntropyChangeRecordingWrapper)
                    assert isinstance(env.env.env.env,
                                      RobomimicLowdimWrapper)
                    env.env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # ---- test init functions ------------------------------------------
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed,
                        enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', "test_seed_" + str(seed) + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env.env.env,
                                  EntropyChangeRecordingWrapper)
                assert isinstance(env.env.env.env,
                                  RobomimicLowdimWrapper)
                env.env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, autoreset_mode="Disabled")

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.n_action_samples = n_action_samples

    # ======================================================================
    #  run
    # ======================================================================
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        N = self.n_action_samples

        # rot6d -> axis-angle (3D) converter for entropy computation.
        # The dimension-agnostic compute_average_action_entropy lets us
        # swap in a 4D quaternion variant (rot6d_to_quat_canonical) if
        # axis-angle's theta=pi singularity ever becomes a problem.
        rot6d_to_rotvec = None
        if self.abs_action and self.rotation_transformer is not None:
            rot6d_to_rotvec = self.rotation_transformer.inverse

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_entropy_histories = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                          args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs, _ = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # ---- build obs dict ----
                np_obs_dict = {
                    'obs': obs[:, :self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

                # --- sample N action chunks for entropy ----------------
                np_obs_repeated = np.repeat(
                    np_obs_dict['obs'], N, axis=0)
                obs_dict_repeated = dict_apply(
                    {'obs': np_obs_repeated},
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict_all = policy.predict_action(
                        obs_dict_repeated)

                # ---- compute & push entropy ---------------------------
                assert rot6d_to_rotvec is not None, (
                    "Entropy eval requires abs_action=True so the rotation "
                    "converter is constructed; got rot6d_to_rotvec=None. "
                    "Check that the loaded checkpoint's task config sets "
                    "abs_action=True."
                )
                assert 'action' in action_dict_all, (
                    "Policy.predict_action must return 'action' "
                    "(executable chunk) for entropy computation; got keys: "
                    f"{list(action_dict_all.keys())}."
                )

                # Entropy is computed over the EXECUTABLE chunk (post-obs
                # slice), not the full horizon prediction, so h* indexes
                # the same actions that MultiStepWrapper_EntropyAdaptive
                # will execute. Using 'action_pred' here would shift h* by
                # n_obs_steps (the obs-prefix predictions) and the wrapper
                # would truncate the wrong slice.
                action_exec_all = action_dict_all['action']
                exec_H = action_exec_all.shape[1]
                D = action_exec_all.shape[2]
                # (n_envs, N, exec_H, D)
                action_exec_per_env = action_exec_all.reshape(
                    n_envs, N, exec_H, D)

                entropy_args = []
                h_stars = []
                for i in range(n_envs):
                    ent = compute_average_action_entropy(
                        action_exec_per_env[i], rot6d_to_rotvec)
                    h_star_i = int(ent['optimal_chunk_size'])
                    h_stars.append(h_star_i)
                    entropy_args.append((
                        ent['diffs'].cpu().numpy(),
                        ent['average_entropy'].cpu().numpy(),
                        h_star_i,
                        ent['entropy_per_timestep'].cpu().numpy(),
                    ))
                # Pushes h_star into each env's EntropyChangeRecordingWrapper;
                # MultiStepWrapper_EntropyAdaptive.step() will read it and
                # only execute the first h_star actions of the chunk.
                env.call_each('record_entropy',
                              args_list=entropy_args)

                # ---- pick first sample for execution ------------------
                all_actions = action_dict_all['action']  # (n_envs*N, Ha, D)
                first_idx = torch.arange(0, n_envs * N, N)
                np_action = all_actions[first_idx].detach().cpu().numpy()
                action = np_action[:, self.n_latency_steps:]
                
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, _, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # Advance pbar to reflect the actual furthest env timestep.
                # Each MultiStepWrapper_EntropyAdaptive appends one entry to
                # self.reward per inner sim step (and skips entirely once
                # done), so len(reward) is the per-env true step count.
                reward_lists = env.call('get_attr', 'reward')
                furthest = max(len(r) for r in reward_lists)
                furthest = min(furthest, self.max_steps)
                delta = furthest - pbar.n
                if delta > 0:
                    pbar.update(delta)
            pbar.close()

            # ---- collect data -----------------------------------------
            all_video_paths[this_global_slice] = \
                env.render()[this_local_slice]
            all_rewards[this_global_slice] = \
                env.call('get_attr', 'reward')[this_local_slice]

            # save per-env entropy histories
            ent_hists = env.call('get_entropy_history')
            all_entropy_histories[this_global_slice] = \
                ent_hists[this_local_slice]

        # ================================================================
        #  Logging
        # ================================================================
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        # persist entropy histories to disk
        pickle_path = pathlib.Path(self.output_dir).joinpath(
            'entropy_histories.pickle')
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_entropy_histories, f)

        return log_data

    # ======================================================================
    #  helpers
    # ======================================================================
    def undo_transform_action(self, action):
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
