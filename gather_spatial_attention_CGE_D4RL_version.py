"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import shutil
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import numpy as np
import click
import hydra
import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import dill
import h5py
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import time

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-r', '--render', default=False)
## Note that this rendering only renders the attention graph, not the environment since Minari dataset does not provide any state information necessary for rendering
def main(checkpoint, output_dir, device, render):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    start_time = time.time()
    
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.training.device = device
    OmegaConf.set_struct(cfg, False)
    cfg.training.smoothing_loss_weight = 0.0
    # cfg.policy.n_action_steps = n_action_steps
    # cfg.task.env_runner.n_test = n_test
    # cfg.task.env_runner.n_test_vis = n_test_vis
    
    # if max_steps is not None:
    #     cfg.task.env_runner.max_steps = max_steps
    # # cfg.task.env_runner.n_train = 10
    # # cfg.task.env_runner.n_train_vis = 10
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    device = torch.device(device)

    # get model from workspace
    score_model = workspace.model.to(device)
    score_model.eval()

    # get replay buffer from workspace
    # 1. replay buffer (for episode check)
    # 2. low_dim_dataset (for deriving spatial attention)
    # 3. image_dataset (for rendering)
    low_dim_dataset = hydra.utils.instantiate(cfg.task.dataset, val_ratio=0.0)
    low_dim_dataloader = DataLoader(low_dim_dataset, batch_size=1, shuffle=False)
    # Set env for rendering
    minari_dataset = low_dim_dataset.minari_dataset
    env = minari_dataset.recover_environment()
    env.unwrapped.render_mode = "rgb_array"
    env.reset()

    replay_buffer = low_dim_dataset.replay_buffer

    assert len(low_dim_dataset) == replay_buffer.n_steps

    # get normalizer from workspace
    normalizer = workspace.normalizer
    
    pwd = os.path.dirname(os.path.abspath(__file__))

    # bring meta_data.json
    meta_data_path = os.path.expanduser(os.path.join(pwd, f'data/D4RL/{cfg.task.task_name}/{cfg.task.dataset_type}/data/metadata.json'))
    new_meta_data_path = os.path.expanduser(os.path.join(output_dir, f'metadata.json'))
    shutil.copy(meta_data_path, new_meta_data_path)

    # bring h5py file
    dataset_path = os.path.expanduser(os.path.join(pwd, f'data/D4RL/{cfg.task.task_name}/{cfg.task.dataset_type}/data/main_data.hdf5'))
    new_dataset_path = os.path.expanduser(os.path.join(pwd, output_dir, f'main_data.hdf5'))

    # First, copy the entire file to preserve original dataset structure
    shutil.copy(dataset_path, new_dataset_path)
    
    try:
        file = h5py.File(new_dataset_path, 'r+')
        
        # check sanity of datasets
        num_demos = len(file.keys())
        print(f"Number of demonstrations: {num_demos}")
        
        length_of_each_demo = list()
        for i in tqdm(range(num_demos)):
            episode_key = f'episode_{i}'
            episode = file[episode_key]
            length_of_each_demo.append(episode['actions'].shape[0])
        length_of_each_demo = np.array(length_of_each_demo)
        
        low_dim_iterator = iter(low_dim_dataloader)
        for i in range(replay_buffer.n_episodes):
            epi = replay_buffer.get_episode(i)
            minari_episode = minari_dataset[i]
            # set env state
            minari_state_dict = {key: minari_episode.infos["state"][key] for key in env.unwrapped._state_space.keys()}
            
            T = epi['action'].shape[0]
            print(f"episode {i}| T: {T}")
            assert T == length_of_each_demo[i] == minari_episode.actions.shape[0], f"T: {T} != length_of_each_demo[{i}]: {length_of_each_demo[i]} != minari_episode.actions.shape[0]: {minari_episode.actions.shape[0]}"
            
            set_state_dict = {k:v[0] for k,v in minari_state_dict.items()}
            env.unwrapped.set_env_state(set_state_dict)
            frame = env.render()
            H, W, C = frame.shape
            
            imgs = np.zeros((T, H, W, C), dtype=np.uint8)
            imgs[0, :, 0*W:(0+1)*W, :] = frame

            normalized_condition_batch = list()
            normalized_data_batch = list()
            for t in tqdm(range(T), desc=f"Getting Observation Data for episode {i}", leave=False):
                low_dim_sample = next(low_dim_iterator)
                
                condition = low_dim_sample['action'][:, 1, :].to(device)
                normalized_condition = normalizer['condition'].normalize(condition).to(device)
                
                data = low_dim_sample['obs'].reshape(low_dim_sample['obs'].shape[0], -1).to(device)
                normalized_data = normalizer['data'].normalize(data).to(device)

                normalized_condition_batch.append(normalized_condition)
                normalized_data_batch.append(normalized_data)
                
                set_state_dict = {k:v[t] for k,v in minari_state_dict.items()}
                env.unwrapped.set_env_state(set_state_dict)
                frame = env.render()
                imgs[t, :, 0*W:(0+1)*W, :] = frame

            normalized_condition_batch = torch.cat(normalized_condition_batch, dim=0)
            normalized_data_batch = torch.cat(normalized_data_batch, dim=0)

            with torch.no_grad():
                t = 1e-3
                batched_time = torch.ones(normalized_condition_batch.shape[0], device=device) * t

                conditioned_score = score_model(normalized_data_batch, batched_time, normalized_condition_batch)
                unconditioned_score = score_model(normalized_data_batch, batched_time, None)

                conditional_gradient = conditioned_score - unconditioned_score
                
            spatial_attention = conditional_gradient.norm(dim=-1).detach().cpu().numpy()
            conditioned_score_norm = conditioned_score.norm(dim=-1).detach().cpu().numpy()
            unconditioned_score_norm = unconditioned_score.norm(dim=-1).detach().cpu().numpy()

            if render:
                video_dir = os.path.join(output_dir, "media")
                os.makedirs(video_dir, exist_ok=True)
                # get attention graph images with two y-axes
                fig, ax1 = plt.subplots(figsize=(4, 3))
                ax2 = ax1.twinx()
                time_steps = np.arange(len(spatial_attention))
                
                # Plot spatial attention on left y-axis
                ax1.plot(time_steps, spatial_attention, 'b-', linewidth=1, label='Spatial Attention')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Spatial Attention', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                
                # Plot scores on right y-axis
                ax2.plot(time_steps, conditioned_score_norm, 'r-', linewidth=1, label='Conditioned Score')
                ax2.plot(time_steps, unconditioned_score_norm, 'g-', linewidth=1, label='Unconditioned Score')
                ax2.set_ylabel('Score Norm', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                
                # Add legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                ax1.grid(True)
                
                graph_frames = list()
                for t in tqdm(range(len(spatial_attention)), desc="Generating attention visualization frames", leave=False):
                    ax1.scatter(t, spatial_attention[t], color='red', s=30)
                    
                    fig.canvas.draw()
                    graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    graph_frames.append(graph_img)
                graph_height, graph_width = graph_frames[0].shape[:2]
                
                # Prepare combined frames
                combined_frames = list()
                for frame, graph in tqdm(zip(imgs, graph_frames), desc="Combining frames", leave=False):
                    # Resize frame to match graph height
                    frame_height, frame_width = frame.shape[:2]
                    aspect_ratio = frame_width / frame_height
                    new_height = graph_height
                    new_width = int(new_height * aspect_ratio)
                    
                    frame_resized = np.array(Image.fromarray(frame).resize((new_width, new_height)))
                    
                    # Create canvas and center the frame
                    canvas = np.zeros((graph_height, graph_width+new_width, 3), dtype=np.uint8)
                    
                    canvas[:, :new_width, :] = frame_resized
                    canvas[:, new_width:, :] = graph
                    combined_frames.append(canvas)
                    
                video_path = os.path.join(video_dir, f"episode_{i}.mp4")
                
                # Create and write video with moviepy
                clip = ImageSequenceClip(combined_frames, fps=20)
                clip.write_videofile(video_path, codec='libx264')
            
            episode_key = f'episode_{i}'
            episode = file[episode_key]
            # Put spatial attention at info
            episode["infos"]["spatial_attention"] = spatial_attention[:, None]
        
    finally:
        file.close()

    # dump log to json
    json_log = dict()
    json_log['command'] = ' '.join(sys.argv)
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    json_log['elapsed_time'] = f"{hours}h {minutes}min {seconds}sec"
    out_path = os.path.join(output_dir, 'log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
