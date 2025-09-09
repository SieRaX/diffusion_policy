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
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env_gymnasium import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper, SubMultiStepWrapperwithDisturbance
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.attention_recording_wrapper import AttentionRecordingWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.transformer import Seq2SeqTransformer
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.env_runner.disturbance_generator.jumping_disturbance_generator import BaseDisturbanceGenerator

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


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
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
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            disturbance_generator=BaseDisturbanceGenerator(),
            max_attention=700.0,
            uniform_horizon=False,
            min_n_action_steps=2,
            attention_exponent=1.0,
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        
        if env_meta['env_name'] in ['Lift', 'PickPlaceCan']:
            def graped_object(normal_env):
                mujoco_env = normal_env.env.env.env
                # return mujoco_env._check_grasp(gripper=mujoco_env.robots[0].gripper, object_geoms=mujoco_env.cube)
                return False
        elif 'NutAssembly' in env_meta['env_name']:
            def graped_object(normal_env):
                return normal_env.env.env.env.staged_rewards()[1]
        else:
            raise ValueError(f"Unknown environment: {env_meta['env_name']}")
        self.graped_object = graped_object
        
        if env_meta['env_name'] in ['NutAssembly', 'Lift', 'NutAssemblySquare', 'ToolHang']:
            self.slice = slice(10, 17)
        elif env_meta['env_name'] == 'PickPlaceCan':
            self.slice = slice(31, 38)
        else:
            raise ValueError(f"Unknown environment: {env_meta['env_name']}")
        
        disturbance_generator.slice = self.slice

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return SubMultiStepWrapperwithDisturbance(
                    VideoRecordingWrapper(
                        AttentionRecordingWrapper(
                            RobomimicLowdimWrapper(
                                env=robomimic_env,
                                obs_keys=obs_keys,
                                init_state=None,
                                render_hw=render_hw,
                                render_camera_name=render_camera_name
                                ),
                            max_timesteps=max_steps,
                            max_attention=max_attention
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps,
                    disturbance_generator=disturbance_generator
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
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

                    # switch to init_state reset
                    assert isinstance(env.env.env.env, RobomimicLowdimWrapper)
                    env.env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', "test_seed_" + str(seed) + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env.env, RobomimicLowdimWrapper)
                env.env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns, autoreset_mode="Disabled") ## autoreset_mode="Disabled" is important for the env runner to work or else it will reset the env when the episode is done
        # env = SyncVectorEnv(env_fns)

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
        self.disturbance_generator = disturbance_generator
        self.uniform_horizon = uniform_horizon
        self.min_n_action_steps = min_n_action_steps
        self.attention_exponent = attention_exponent
        
    def run(self, policy: BaseLowdimPolicy, attention_estimator: Seq2SeqTransformer, attention_normalizer: LinearNormalizer, seed=100000, init_catt = 50.0, init_dcatt = 200.0):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_c_att_array = [None] * n_inits
        all_dc_att_array = [None] * n_inits
        all_horizon_length_avg = [None] * n_inits

        horizon_length_avg = None
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            this_env_seeds = self.env_seeds[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
                this_env_seeds.extend([this_env_seeds[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            c_att_array = torch.ones(n_envs)*init_catt
            dc_att_array = torch.ones(n_envs)*init_dcatt
            complete = torch.zeros(n_envs, dtype=bool)
            big_step = torch.ones(n_envs, dtype=bool)
            number_of_attempts = torch.zeros(n_envs, dtype=torch.int)
            plus_minus_mask = torch.zeros(n_envs, dtype=torch.int)
            
            env_name = self.env_meta['env_name']
            find_catt_bar = tqdm.tqdm(total=n_envs, 
                                      desc=f"[Task: {env_name}Lowdim | Chunk: {chunk_idx+1}/{n_chunks}]Finding c_att_bar | Average C_att: None | Average length: None", 
                                      leave=False, mininterval=self.tqdm_interval_sec)
            while not torch.all(complete):
                obs, _ = env.reset(seed=this_env_seeds)
                # env.seed(this_env_seeds) # somehow, robomimic lowdim wrapper sets the seed to None after reset. So we give the seed again to the env.
                past_action = None
                policy.reset()

                pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                    leave=False, mininterval=self.tqdm_interval_sec)

                done = False

                while not done:
                    # create obs dict
                    np_obs_dict = {
                        # handle n_latency_steps by discarding the last n_latency_steps
                        'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                    }
                    if self.past_action and (past_action is not None):
                        # TODO: not tested
                        np_obs_dict['past_action'] = past_action[
                            :,-(self.n_obs_steps-1):].astype(np.float32)
                    
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, 
                        lambda x: torch.from_numpy(x).to(
                            device=device))

                    # run policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())

                    with torch.no_grad():
                        # obs_dict_with_dummy = torch.cat([obs_dict['obs'], torch.zeros_like(obs_dict['obs'][..., -1:]).to(device=device)], dim=-1)
                        nobs = attention_normalizer['obs'].normalize(obs_dict['obs']).to(device=device)
                        naction = attention_normalizer['action'].normalize(action_dict['action_pred']).to(device=device)
                        output = attention_estimator(nobs.reshape(nobs.shape[0], -1), naction)
                        attention_pred = attention_normalizer['spatial_attention'].unnormalize(output).detach().cpu().squeeze()
                        attention_pred = torch.pow(attention_pred, self.attention_exponent)
                        attention_pred_cumsum = torch.cumsum(attention_pred, dim=-1)
                        
                        c_att_array_expanded = c_att_array.unsqueeze(1).expand_as(attention_pred_cumsum)
                        
                        # Create a mask where a > b
                        mask = attention_pred_cumsum > c_att_array_expanded

                        # Use torch.argmax on the mask to get the first index where condition is True
                        # But be careful: if no element is > b, argmax will return 0, which may be incorrect
                        # So we mask out invalid rows later
                        horizon_idx = mask.float().cumsum(dim=1).eq(1).float().argmax(dim=1)

                        # Optionally mark rows where the condition was never true
                        valid = mask.any(dim=1)
                        horizon_idx[~valid] = attention_pred.shape[1]  # Or any sentinel value like 32
                        horizon_idx += 1
                        horizon_idx[horizon_idx < self.min_n_action_steps] = self.min_n_action_steps
                        
                        attention_pred = attention_pred.numpy()
                        
                    if self.uniform_horizon:
                        horizon_idx[:] = self.n_action_steps
                        
                    # handle latency_steps, we discard the first n_latency_steps actions
                    # to simulate latency
                    action = np_action_dict['action_pred'][:,self.n_latency_steps:]
                    if not np.all(np.isfinite(action)):
                        print(action)
                        raise RuntimeError("Nan or Inf action")
                    
                    # step env
                    env_action = action
                    if self.abs_action:
                        env_action = self.undo_transform_action(action)
                        
                    env.call_each('register_horizon_idx', args_list=[[idx.item()] for idx in horizon_idx])
                    # env.call_wait()
                    env.call_each('register_attention_pred', args_list=[[attention_pred[i, :horizon_idx[i]]] for i in range(n_envs)])

                    obs, reward, done, _, info = env.step(env_action)
                    done = np.all(done)
                    past_action = action
                    
                    env.call_each('deregister_horizon_idx')
                    # env.call_wait()
                    
                    if done:
                        c_att_array_before = c_att_array.clone()
                        
                        horizon_length_lst = env.call('get_attr', 'horizon_idx_list')

                        # horizon_length_lst = torch.stack(horizon_length_lst, dim=1)
                        horizon_length_avg = torch.tensor([torch.tensor(arr, dtype=torch.float).mean().item() for arr in horizon_length_lst])
                        
                        indices_mask = ((horizon_length_avg < self.n_action_steps-0.5).logical_and(complete.logical_not()))
                        big_step_mask = (indices_mask & (plus_minus_mask < 0))
                        big_step[big_step_mask] = False
                        plus_minus_mask[indices_mask] = 1
                        dc_att_array[indices_mask & big_step.logical_not()] = 0.75 * dc_att_array[indices_mask & big_step.logical_not()]
                        number_of_attempts[indices_mask & big_step.logical_not()] += 1
                        c_att_array[indices_mask] += dc_att_array[indices_mask]
                        
                        indices_mask = ((horizon_length_avg > self.n_action_steps+0.5).logical_and( complete.logical_not()))
                        big_step_mask = (indices_mask & (plus_minus_mask > 0))
                        big_step[big_step_mask] = False
                        plus_minus_mask[indices_mask] = -1
                        dc_att_array[indices_mask & big_step.logical_not()] = 0.75 * dc_att_array[indices_mask & big_step.logical_not()]
                        number_of_attempts[indices_mask & big_step.logical_not()] += 1
                        c_att_array[indices_mask] -= dc_att_array[indices_mask]
                        
                        indices_mask = ((horizon_length_avg > self.n_action_steps-0.5).logical_and(horizon_length_avg < self.n_action_steps+0.5))
                        # complete[:] = False # reset complete... for sanity check
                        env.call_each('register_c_att', args_list=[[c_att_array_before_el] for c_att_array_before_el in c_att_array_before])

                        # saving data finished, now we freeze the env that has satisfied the av horizon condition.
                        complete[indices_mask] = True
                        complete[number_of_attempts > 9] = True
                        
                        env.call_each('set_complete', args_list=[[complete_el] for complete_el in complete])
                        env.call_each('set_invalid_env', args_list=[[is_valid_el] for is_valid_el in (number_of_attempts > 9).tolist()])

                        sanity_check_complete = torch.zeros(n_envs, dtype=bool)
                        sanity_check_complete[indices_mask] = True
                        sanity_check_complete[number_of_attempts > 9] = True

                        if sanity_check_complete.sum().item() != complete.sum().item():
                            print(f"sanity check complete: {sanity_check_complete.sum().item()}, complete: {complete.sum().item()}")
                            raise RuntimeError("Sanity check failed")

                        # update find_catt_bar description
                        find_catt_bar.set_description(f"[Task: {env_name}Lowdim | Chunk: {chunk_idx+1}/{n_chunks}]Finding c_att_bar | Average C_att: {c_att_array_before[~complete].mean().item():.2f} | Average length of Incomplete Env:"+(f"None" if horizon_length_avg is None else f"{horizon_length_avg[~complete].mean().item():.2f}"))
                        

                    # update pbar
                    pbar.refresh()
                    pbar.n = min(env.call('get_attr', 'total_steps'))
                    pbar.refresh()
                    
                    
                pbar.close()
                
                find_catt_bar.refresh()
                find_catt_bar.n = complete.sum().item()
                # find_catt_bar.n = indices_mask.sum().item() # for sanity check
                find_catt_bar.refresh()
                # collect data for this round
                if torch.all(complete):
                    all_video_paths[this_global_slice] = env.render()[this_local_slice]
                    all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

                    all_c_att_array[this_global_slice] = env.call('get_attr', 'c_att')[this_local_slice]
                    all_dc_att_array[this_global_slice] = dc_att_array
                    all_horizon_length_avg[this_global_slice] = horizon_length_avg

                    ## set the complete to False for all envs so we can start new chunk
                    env.call_each('set_complete', args_list=[[False] for _ in range(n_envs)])

                    init_catt = c_att_array.mean().item()
            
            find_catt_bar.close()
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix+f'sim_catt_{seed}'] = all_c_att_array[i].item()
            log_data[prefix+f'sim_horizon_avg_length_{seed}'] = all_horizon_length_avg[i].item()

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value_el in max_rewards.items():
            value_el = np.array(value_el)
            name = prefix+'mean_score'
            value = np.mean(value_el)
            log_data[name] = value
            
            name = prefix+'mean_score_valid'
            value = np.mean(value_el[~(value_el<0)])
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
