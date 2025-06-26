import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import json
import dill
import math
import wandb.sdk.data_types.video as wv
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.d4rl.d4rl_lowdim_wrapper import D4RLLowdimWrapper

from minari import MinariDataset

class D4RLLowdimLikelihoodRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
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
        meta_data = json.load(open(os.path.join(dataset_path, 'metadata.json')))
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        # env_meta = FileUtils.get_env_metadata_from_dataset(
        #     dataset_path)
        # rotation_transformer = None
        # if abs_action:
        #     env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        #     rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
            
        def make_env():
            dataset = MinariDataset(os.path.expanduser(dataset_path))
            env = dataset.recover_environment()
            return MultiStepWrapper(
                    D4RLLowdimWrapper(
                        env=env,
                        init_state=None,
                        render_hw=render_hw,
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

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
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        
        all_video_paths = []
        all_rewards = []
        all_seeds = []
        all_prefixs = []
        all_action_horizon_average_lengths = []
        
        # Test episodes # D4RL dataset does not provide states, therefore we cannot recover simulation states that gives us dataset observation
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            enable_render = i < self.n_test_vis
            
            env = self.make_env()
            env.seed(seed)
            print(f"test episode {i+1} with seed {seed}")
            
            max_reward, output_path, action_horizon_average_length = self._run_episode(
                env=env,
                policy=policy,
                device=device,
                enable_render=enable_render,
                prefix='test',
                episode_idx=i,
                seed=seed
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
            log_data[prefix+f'sim_action_horizon_average_length_{seed}'] = all_action_horizon_average_lengths[i]
            
            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                video_path = str(video_path)
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
    
    def _run_episode(self, env, policy: BaseLowdimPolicy, device, enable_render, prefix, episode_idx, seed):
        obs = env.reset()
        policy.reset()
        
        image_frames = []
        kl_divergence_drop_frame = []
        sample_triggered_lst = []
        rewards = []
        done = False
        action_horizon_length_lst = []
        step = 0
        step_bar = tqdm.tqdm(
            total=self.max_steps, 
            desc=f"{prefix} Episode {episode_idx+1}", 
            leave=False,
            mininterval=self.tqdm_interval_sec
        )
        past_action = None
        while not done:
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
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")
            past_action = action
            
            # step env
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action) 
            action_horizon_length_lst.append(env_action.shape[0])
            for i in range(env_action.shape[0]):
                obs, reward, done, info = env.step(env_action[[i]])
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
                        # kl_divergence_drop = policy.predict_kl_sample(obs_dict)
                        kl_divergence_drop_frame.append(kl_divergence_drop.detach().cpu().numpy().item())
            
            step_bar.update(env_action.shape[0])
        step_bar.close()

        # Save video with attention visualization
        if len(image_frames) > 0:
            output_path = self._save_visualization(
                image_frames=image_frames,
                kl_divergence_drop_frame=kl_divergence_drop_frame,
                sample_triggered_lst=sample_triggered_lst,
                prefix=prefix,
                episode_idx=episode_idx
            )
        else:
            output_path = None

        return np.max(rewards), output_path, np.mean(action_horizon_length_lst)

    def _save_visualization(self, image_frames, kl_divergence_drop_frame, sample_triggered_lst, prefix, episode_idx):
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
