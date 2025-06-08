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
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    start_time = time.time()
    
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
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

    image_dataset_config_path = os.path.join(f"conditional_gradient_estimator/config/task/{cfg.task.task_name}_image{'_abs' if cfg.task.abs_action else ''}.yaml")
    with open(image_dataset_config_path, 'r') as f:
        image_dataset_config = OmegaConf.load(f)
    image_dataset_config["task"] = {"task_name": cfg.task.task_name, "dataset_type": cfg.task.dataset_type, "abs_action": cfg.task.abs_action}
    image_dataset_config["horizon"] = cfg.horizon
    image_dataset_config["n_obs_steps"] = cfg.n_obs_steps
    image_dataset_config["n_action_steps"] = cfg.n_action_steps
    image_dataset_config["n_latency_steps"] = cfg.n_latency_steps
    image_dataset_config["dataset_obs_steps"] = cfg.n_obs_steps
    image_dataset = hydra.utils.instantiate(image_dataset_config.dataset, val_ratio=0.0)
    image_dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)

    replay_buffer = image_dataset.replay_buffer

    assert len(image_dataset) == len(low_dim_dataset) == replay_buffer.n_steps

    # get normalizer from workspace
    normalizer = workspace.normalizer
    
    image_iterator = iter(image_dataloader)
    low_dim_iterator = iter(low_dim_dataloader)
    for i in range(replay_buffer.n_episodes):
        epi = replay_buffer.get_episode(i)
        
        T, H, W, C = epi['agentview_image'].shape
    
        imgs = np.zeros((T, H, 2*W, C), dtype=epi['agentview_image'].dtype)
        print(f"episode {i}| T: {T}")

        normalized_condition_batch = list()
        data_batch = list()
        for t in tqdm(range(T), desc="Getting Image and Spatial Attention", leave=False):
            image_sample = next(image_iterator)
            low_dim_sample = next(low_dim_iterator)

            # Transpose from (3,84,84) to (84,84,3) format
            imgs[t, :, :W, :] = ((image_sample['obs']['agentview_image'][0, 0].permute(1, 2, 0) * 255).numpy()).astype(np.uint8)
            imgs[t, :, W:, :] = (image_sample['obs']['robot0_eye_in_hand_image'][0, 0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)

            condition = low_dim_sample['action'][:, 1, :].to(device)
            normalized_condition = normalizer['condition'].normalize(condition).to(device)
            
            data = low_dim_sample['obs'].reshape(low_dim_sample['obs'].shape[0], -1).to(device)

            normalized_condition_batch.append(normalized_condition)
            data_batch.append(data)

        normalized_condition_batch = torch.cat(normalized_condition_batch, dim=0)
        data_batch = torch.cat(data_batch, dim=0)

        with torch.no_grad():
            t = 1e-3
            batched_time = torch.ones(normalized_condition_batch.shape[0], device=device) * t

            conditioned_score = score_model(data_batch, batched_time, normalized_condition_batch)
            unconditioned_score = score_model(data_batch, batched_time, None)

            conditional_gradient = conditioned_score - unconditioned_score
            
        spatial_attention = conditional_gradient.norm(dim=-1).detach().cpu().numpy()

        # get attention graph images
        # get attention graph images
        fig, ax = plt.subplots(figsize=(4, 3))
        time_steps = np.arange(len(spatial_attention))
        ax.plot(time_steps, spatial_attention, 'b-', linewidth=1, label='Spatial Attention')
        ax.set_xlabel('Time Step')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True)
        
        graph_frames = list()
        for t in range(len(spatial_attention)):
            ax.scatter(t, spatial_attention[t], color='red', s=30)
            
            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            graph_frames.append(graph_img)
        graph_height, graph_width = graph_frames[0].shape[:2]
        
        # Prepare combined frames
        combined_frames = list()
        for frame, graph in zip(imgs, graph_frames):
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
            
        video_path = os.path.join(output_dir, f"episode_{i}.mp4")
        
        # Create and write video with moviepy
        clip = ImageSequenceClip(combined_frames, fps=20)
        clip.write_videofile(video_path, codec='libx264')

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
