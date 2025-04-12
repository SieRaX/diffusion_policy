import cv2
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
import matplotlib.pyplot as plt                
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image  # For image resizing

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class PushTKeypointsLikelihoodRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        env_seeds = list()
        env_prefixs = list()
        env_vis = list()

        # train
        for i in range(n_train):
            seed = train_start_seed + i
            env_seeds.append(seed)
            env_prefixs.append('train_')
            env_vis.append(i < n_train_vis)

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            env_seeds.append(seed)
            env_prefixs.append('test_')
            env_vis.append(i < n_test_vis)
        
        self.env = PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    )
        
        self.n_envs = n_envs
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_vis = env_vis
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        
        self.make_new_env = lambda: MultiStepWrapper(
                        PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                        ),
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    max_episode_steps=max_steps
                    )
    
    def run(self, policy: DiffusionUnetLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        n_envs = self.n_envs
        env_vis = self.env_vis
        env_prefix = self.env_prefixs
        
        pbar = tqdm.tqdm(total=n_envs, desc=f"Eval PushtKeypointsRunner", leave=False, mininterval=self.tqdm_interval_sec)
        for i, vis, prefix in zip(np.arange(n_envs), env_vis, env_prefix):
            
            env = self.make_new_env()
            env.seed(self.env_seeds[i])
            obs = env.reset()
            policy.reset()
            
            image_frame = list()
            loglikelihood_frame = list()
            num_anomaly_frame = list()
            reward_list = list()
            done_lst = list()
            pause_lst = list()
            done = False
            
            step = 0
            step_bar = tqdm.tqdm(total=self.max_steps, desc=f"Episode {i+1}/{n_envs}", leave=False, 
                                mininterval=self.tqdm_interval_sec, position=1)
            while not done:
                Do = obs.shape[-1]//2
                # create obs dict
                np_obs_dict = {
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                # past action not implemented yet
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device).unsqueeze(0))
                
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').squeeze(0).numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                ### No latency for now
                assert self.n_latency_steps == 0
                action = np_action_dict['action'][:,self.n_latency_steps:]
                
                # step env
                env.env.action_seq = action
                env.env.info['action_pred'] = np_action_dict['action_pred']
                for act in action:
                    if len(done_lst) > 0 and done_lst[-1]:
                        # termination
                        break
                    # TODO: Render image here
                    obs, reward, done, info = env.step(act[np.newaxis])
                    reward_list.append(reward)
                    np_obs_dict = {
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                    }
                    obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device).unsqueeze(0))
                    
                    if vis:
                        frame = env.render(mode='rgb_array')
                        image_frame.append(frame)
                        # TODO: save loglikelihood here
                        # _ = policy.predict_kl_grad(obs_dict)
                        with torch.no_grad():
                            loglikelihood, num_anomaly = policy.predict_kl_divergence(obs_dict)
                            difference = policy.kl_divergence_drop(obs_dict)
                            loglikelihood_frame.append(difference.detach().cpu().numpy().item())
                            # loglikelihood_frame.append(loglikelihood.detach().cpu().numpy().item())
                            num_anomaly_frame.append(num_anomaly.detach().cpu().numpy().item())
                    
                    if (self.max_steps is not None) \
                        and (len(reward_list) >= self.max_steps):
                        # truncation
                        done = True
                    done_lst.append(done)
                    pause_lst.append(False)
                    step += 1
                    step_bar.update(1)
                    step_bar.set_description(f"Episode {i+1}/{n_envs} [Step {step}/{self.max_steps}]")
                
                for _ in range(0): ## Neglect pausing for now
                    if vis:
                        pause_lst.append(True)

            # TODO: save run video concat with the graph with red dot...
            if len(image_frame) > 0:
                # Verify lengths match using numpy
                n_pause = len(pause_lst) - np.sum(pause_lst)
                assert len(image_frame) == n_pause, f"Number of frames ({len(image_frame)}) does not match number of pause steps ({n_pause})"

                # Create the loglikelihood graph
                fig, ax1 = plt.subplots(figsize=(4, 3))
                loglikelihood_data = np.array(loglikelihood_frame)
                anomaly_data = np.array(num_anomaly_frame)
                time_steps = np.arange(len(loglikelihood_data))
                
                # Plot loglikelihood on primary y-axis
                color1 = 'tab:blue'
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Dkl', color=color1)
                line1 = ax1.plot(time_steps, loglikelihood_data, color=color1, linewidth=1)
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.grid(True)
                
                # Create secondary y-axis for anomaly data
                ax2 = ax1.twinx()
                color2 = 'tab:red'
                ax2.set_ylabel('Anomaly Count', color=color2)
                line2 = ax2.plot(time_steps, anomaly_data, color=color2, linewidth=1)
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Add legend
                lines = line1 + line2
                labels = ['Dkl', 'Anomaly Count']
                ax1.legend(lines, labels, loc='upper right')
                
                # Prepare the output path
                output_path = pathlib.Path(self.output_dir) / f'media/{prefix}episode_{i}.mp4'
                output_path.parent.mkdir(exist_ok=True)
                
                # First create one graph frame to get dimensions
                fig.canvas.draw()
                graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                graph_height, graph_width = graph_img.shape[:2]
                
                # Create graph frames with moving dot
                graph_frames = []
                loglikelihood_bar = tqdm.tqdm(total=len(loglikelihood_data), 
                                        desc=f"Rendering Log Likelihood", 
                                        leave=False, 
                                        mininterval=self.tqdm_interval_sec, 
                                        position=2)
                
                for t in range(len(loglikelihood_data)):
                    # Clear previous dots
                    ax1.scatter(time_steps, loglikelihood_data, color='blue', s=0)
                    ax2.scatter(time_steps, anomaly_data, color='red', s=0)
                    
                    # Add red dot at current position for loglikelihood
                    ax1.scatter(t, loglikelihood_data[t], color='red', s=50)
                    # Add blue dot at current position for anomaly
                    ax2.scatter(t, anomaly_data[t], color='blue', s=50)
                    
                    # Convert matplotlib figure to numpy array
                    fig.canvas.draw()
                    graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    graph_frames.append(graph_img)
                    loglikelihood_bar.update(1)
                plt.close(fig)
                loglikelihood_bar.close()

                # Prepare combined frames
                combined_frames = []
                render_bar = tqdm.tqdm(total=len(pause_lst), 
                                     desc=f"Preparing Frames", 
                                     leave=False, 
                                     mininterval=self.tqdm_interval_sec, 
                                     position=1)
                
                frame_idx = 0
                current_combined = None
                
                for pause in pause_lst:
                    if not pause:  # If pause is False, use the current frame
                        if frame_idx >= len(image_frame):
                            break
                            
                        frame = image_frame[frame_idx]
                        graph = graph_frames[frame_idx]
                        frame_idx += 1
                        
                        # Calculate new width while maintaining aspect ratio
                        frame_height, frame_width = frame.shape[:2]
                        aspect_ratio = frame_width / frame_height
                        new_height = graph_height
                        new_width = int(new_height * aspect_ratio)
                        
                        # Resize frame maintaining aspect ratio using numpy
                        frame_resized = np.array(Image.fromarray(frame).resize((new_width, new_height)))
                        
                        # Create a black canvas of the same width as graph
                        canvas = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
                        
                        # Calculate position to center the resized frame
                        x_offset = (graph_width - new_width) // 2
                        if x_offset >= 0:
                            # If resized frame is narrower than graph, center it
                            canvas[:, x_offset:x_offset+new_width] = frame_resized
                        else:
                            # If resized frame is wider than graph, crop it centrally
                            crop_start = (-x_offset) // 2
                            canvas = frame_resized[:, crop_start:crop_start+graph_width]
                        
                        # Combine frame and graph horizontally
                        current_combined = np.hstack([canvas, graph])
                    
                    if current_combined is not None:
                        combined_frames.append(current_combined)
                    render_bar.update(1)
                
                render_bar.close()

                # Create video using moviepy
                try:
                    clip = ImageSequenceClip(combined_frames, fps=self.fps)
                    clip.write_videofile(str(output_path), 
                                       codec='libx264', 
                                       audio=False,
                                       preset='medium',
                                       ffmpeg_params=['-pix_fmt', 'yuv420p'], logger=None)  # This ensures compatibility with web browsers
                finally:
                    # Clean up
                    clip.close()
            
            step_bar.close()
            pbar.update(1)
            pbar.set_description(f"Episodes Progress [{i+1}/{n_envs}]")
            
            # Clear frame buffers for next episode
            image_frame = []
            loglikelihood_frame = []
        # TODO: save log_data
        pbar.close()
        return None
