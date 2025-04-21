import os
import wandb
import numpy as np
from copy import deepcopy
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.diffusion_unet_lowdim_policy_state_estimator import DiffusionUnetLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.pose_trajectory_interpolator import quat2matrix

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

class RobomimicLowdimAHCRunner(BaseLowdimRunner):
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
            n_envs=None
        ):
        super().__init__(output_dir)

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

        def make_env():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            return MultiStepWrapper(
                    RobomimicLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
                        init_state=None,
                        render_hw=render_hw,
                        render_camera_name=render_camera_name
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        self.env_meta = env_meta
        self.make_env = make_env
        self.n_train = n_train
        self.n_test = n_test
        self.train_start_idx = train_start_idx
        self.test_start_seed = test_start_seed
        self.n_train_vis = n_train_vis
        self.n_test_vis = n_test_vis
        self.dataset_path = dataset_path
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        
        if 'Lift' == env_meta['env_name']:
            def graped_object(normal_env):
                mujoco_env = normal_env.env.env.env
                return mujoco_env._check_grasp(gripper=mujoco_env.robots[0].gripper, object_geoms=mujoco_env.cube)
        elif 'NutAssembly' in env_meta['env_name']:
            def graped_object(normal_env):
                return normal_env.env.env.env.staged_rewards()[1]
        else:
            raise ValueError(f"Unknown environment: {env_meta['env_name']}")
        
        self.graped_object = graped_object

    def run(self, policy: DiffusionUnetLowdimPolicy, c_att=0.01):
        device = policy.device
        dtype = policy.dtype
        
        all_video_paths = []
        all_rewards = []
        all_seeds = []
        all_prefixs = []
        all_action_horizon_average_lengths = []
        # Train episodes
        with h5py.File(self.dataset_path, 'r') as f:
            for i in range(self.n_train):
                train_idx = self.train_start_idx + i
                enable_render = i < self.n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]
                
                env = self.make_env()
                env.env.env.init_state = init_state
                
                max_reward, output_path, action_horizon_average_length = self._run_episode(
                    env=env,
                    policy=policy,
                    device=device,
                    enable_render=enable_render,
                    prefix='train',
                    episode_idx=i,
                    seed=train_idx,
                    c_att=c_att
                )
                all_video_paths.append(output_path)
                all_rewards.append(max_reward)
                all_seeds.append(train_idx)
                all_prefixs.append('train/')
                all_action_horizon_average_lengths.append(action_horizon_average_length)
                
        # Test episodes
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            enable_render = i < self.n_test_vis
            
            env = self.make_env()
            env.env.env.init_state = None
            env.seed(seed)
            print(f"test episode {i+1} with seed {seed}")
            
            max_reward, output_path, action_horizon_average_length = self._run_episode(
                env=env,
                policy=policy,
                device=device,
                enable_render=enable_render,
                prefix='test',
                episode_idx=i,
                seed=seed,
                c_att=c_att
            )
            all_video_paths.append(output_path)
            all_rewards.append(max_reward)
            all_seeds.append(seed)
            all_prefixs.append('test/')
            all_action_horizon_average_lengths.append(action_horizon_average_length)
            
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(len(all_rewards)): # 여기 zip으로 바꾸자
            seed = all_seeds[i]
            prefix = all_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                video_path = str(video_path)
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
                log_data[prefix+f'sim_action_horizon_average_length_{seed}'] = all_action_horizon_average_lengths[i]

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
            

    def _run_episode(self, env, policy:DiffusionUnetLowdimPolicy, device, enable_render, prefix, episode_idx, c_att, seed=42):
        np.random.seed(seed)
        
        obs = env.reset()
        policy.reset()
        
        image_frames = []
        kl_divergence_drop_frame = []
        sample_triggered_lst = []
        rewards = []
        done = False
        reward = 0 # for lift only, for Square, the code might be wrong.
        action_horizon_length_lst = []
        
        step_bar = tqdm.tqdm(
            total=self.max_steps, 
            desc=f"{prefix} Episode {episode_idx+1}", 
            leave=False,
            mininterval=self.tqdm_interval_sec
        )
        past_action = None
        while reward < 1-1e-4 and not done:
            # create obs dict
            np_obs_dict = {
                'obs': obs[...,:self.n_obs_steps, :].astype(np.float32)
            }
            if self.past_action and (past_action is not None):
                assert False, "past_action is not supported"
                np_obs_dict['past_action'] = past_action[
                    :,-(self.n_obs_steps-1):].astype(np.float32)
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0))
            
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            
            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').squeeze(0).numpy())
            
            action = np_action_dict['action'][:,self.n_latency_steps:]
            state = np_action_dict['state'][:,self.n_latency_steps:]
            np_obs_dict_AHC = {'obs': np.stack([state[:-1], state[1:]], axis=1)}
            obs_dict_AHC = dict_apply(np_obs_dict_AHC, lambda x: torch.from_numpy(x).to(device=device))
            
            with torch.no_grad():
                kl_divergence_drop = policy.kl_divergence_drop(obs_dict_AHC)
                kl_divergence_cumsum = torch.cumsum(kl_divergence_drop, dim=-1)
                indices = torch.where(kl_divergence_cumsum > c_att)[0]
                horizon_idx = indices[0].item() if len(indices) > 0 else kl_divergence_cumsum.shape[0]
            
            
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")
            past_action = action
            
            # step env
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action) 
            
            env_action = env_action[:horizon_idx]
            action_horizon_length_lst.append(horizon_idx)
            for i in range(env_action.shape[0]):
                obs, reward, done, info = env.step(env_action[[i]])
                
                if np.random.uniform() < 1.1 and not self.graped_object(env) and np.linalg.norm(env.env.env.get_observation()['object'][7:10]) < 0.05:
                    speed = 0.0017
                    # if np.linalg.norm(env.env.env.get_observation()['object'][7:10]) < 0.10:
                    direction = quat2matrix(env.env.env.get_observation()['robot0_eef_quat'])[0:2, 0]
                    direction = direction/np.linalg.norm(direction)
                    state = env.env.env.get_state()
                    object_pos_quat = state['states'][10:17]
                    # direction = -object_pos_quat[0:2]/np.linalg.norm(object_pos_quat[0:2])
                    new_state = {k:v for k,v in deepcopy(state).items() if k != 'model'} # If model is copied the robot could not open the gripper.
                    new_state['states'][10:17] = object_pos_quat + np.array([*speed*direction, 0.0, 0.0, 0, 0, 0])
                    env.env.env.reset_to(new_state)
                
                np_obs_dict = {
                    'obs': obs[...,:self.n_obs_steps, :].astype(np.float32)
                }
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0))
                rewards.append(reward)
            
                if enable_render:
                    frame = env.render(mode='rgb_array')
                    image_frames.append(frame)
                    sample_triggered_lst.append(i==0)
                    with torch.no_grad():
                        kl_divergence_drop = policy.kl_divergence_drop(obs_dict)
                        kl_divergence_drop_frame.append(kl_divergence_drop.detach().cpu().numpy().item())
                        # kl_divergence_drop_frame.append(0)
            
            step_bar.update(env_action.shape[0])
        step_bar.close()

        # Save video with attention visualization
        if len(image_frames) > 0:
            output_path = self._save_visualization(
                image_frames=image_frames,
                kl_divergence_drop_frame=kl_divergence_drop_frame,
                prefix=prefix,
                episode_idx=episode_idx,
                sample_triggered_lst=sample_triggered_lst
            )
        else:
            output_path = None
            
        return np.max(env.get_attr('reward')), output_path, np.mean(action_horizon_length_lst)

    def _save_visualization(self, image_frames, kl_divergence_drop_frame, prefix, episode_idx, sample_triggered_lst):
        # Create the attention graph
        fig, ax = plt.subplots(figsize=(4, 3))
        kl_divergence_drop_data = np.array(kl_divergence_drop_frame)
        time_steps = np.arange(len(kl_divergence_drop_data))
        ax.plot(time_steps, kl_divergence_drop_data, 'b-', linewidth=1)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('KL Divergence Drop')
        ax.grid(True)
        
        # Get graph dimensions
        fig.canvas.draw()
        graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_height, graph_width = graph_img.shape[:2]
        
        # Create graph frames with moving dot
        graph_bar = tqdm.tqdm(total=len(kl_divergence_drop_data), 
                            desc=f"plotting graphs", 
                            leave=False, 
                            mininterval=self.tqdm_interval_sec, 
                            position=2)
        graph_frames = []
        for t in range(len(kl_divergence_drop_data)):
            ax.scatter(time_steps, kl_divergence_drop_data, color='blue', s=0)
            ax.scatter(t, kl_divergence_drop_data[t], color='red', s=50)
            
            if sample_triggered_lst[t]:
                ax.scatter(t, 0, color='green', s=20, marker='x')
            
            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            graph_frames.append(graph_img)
            graph_bar.update(1)
        graph_bar.close()
        plt.close(fig)
        
        # Prepare combined frames
        video_bar = tqdm.tqdm(total=len(image_frames), 
                            desc=f"preparing video", 
                            leave=False, 
                            mininterval=self.tqdm_interval_sec, 
                            position=2)
        combined_frames = []
        for frame, graph in zip(image_frames, graph_frames):
            # Resize frame to match graph height
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            new_height = graph_height
            new_width = int(new_height * aspect_ratio)
            
            frame_resized = np.array(Image.fromarray(frame).resize((new_width, new_height)))
            
            # Create canvas and center the frame
            canvas = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
            x_offset = (graph_width - new_width) // 2
            
            if x_offset >= 0:
                canvas[:, x_offset:x_offset+new_width] = frame_resized
            else:
                crop_start = (-x_offset) // 2
                canvas = frame_resized[:, crop_start:crop_start+graph_width]
            
            combined_frames.append(np.hstack([canvas, graph]))
            video_bar.update(1)
        video_bar.close()
        # Save video
        output_path = pathlib.Path(self.output_dir) / f'media/{prefix}_episode_{episode_idx}.mp4'
        output_path.parent.mkdir(exist_ok=True)
        
        clip = ImageSequenceClip(combined_frames, fps=self.fps)
        clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio=False,
            preset='medium',
            ffmpeg_params=['-pix_fmt', 'yuv420p'],
            logger=None
        )
        clip.close()
        
        return output_path

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