import os
import sys
import click

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from robomimic.utils import file_utils as FileUtils
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.model.transformer import Seq2SeqTransformer
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import normalizer_from_stat
from diffusion_policy.common.normalize_util import array_to_stats

from PIL import Image
import h5py
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt

@click.command()
@click.option('-p', '--lowdim_dataset_path', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(lowdim_dataset_path, device):

    # Extract horizon length from path
    horizon = 32

    lowdim_dataset = RobomimicReplayLowdimDataset(lowdim_dataset_path, horizon=horizon, pad_before=1, pad_after=horizon-1, obs_keys=['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'], abs_action=True, rotation_rep='rotation_6d', use_legacy_normalizer=False, seed=42, val_ratio=0.02, max_train_episodes=None, info_keys=['spatial_attention'])
    val_dataset = lowdim_dataset.get_validation_dataset()

    train_dataloader = DataLoader(lowdim_dataset, batch_size=256, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # display_dataloader = DataLoader(
    #     RobomimicReplayLowdimDataset(lowdim_dataset_path, horizon=horizon, pad_before=1, pad_after=horizon-1, obs_keys=['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'], abs_action=True, rotation_rep='rotation_6d', use_legacy_normalizer=False, seed=42, val_ratio=0.0, max_train_episodes=None, info_keys=['spatial_attention']),
    #     batch_size=1,
    #     shuffle=False
    # )

    normalizer = lowdim_dataset.get_normalizer()
        
    # obs_normalizer = SingleFieldLinearNormalizer()
    # obs_normalizer.fit(lowdim_dataset.replay_buffer['obs'])

    # normalizer['obs'] = obs_normalizer
    
    device = torch.device(device)

    train_data_sample = next(iter(train_dataloader))

    train_data_sample.keys()
    obs_dim = (train_data_sample['obs'].shape[-1])*2
    action_dim = train_data_sample['action'].shape[-1]
    seq_len = train_data_sample['action'].shape[1]

    n_epoch = 200
    steps_per_epoch = len(train_dataloader)

    model = Seq2SeqTransformer(obs_dim=obs_dim, action_dim=action_dim, seq_len=seq_len)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch*steps_per_epoch, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(n_epoch):
        model.train()
        for batch in train_dataloader:
            
            optimizer.zero_grad()
            
            B, T, D = batch['obs'].shape
            nobs = normalizer['obs'].normalize(batch['obs'])
            naction = normalizer['action'].normalize(batch['action'])
            nattention = normalizer['spatial_attention'].normalize(batch['spatial_attention'].unsqueeze(-1))
            
            obs = nobs[:, :2, :].reshape(B, -1).to(device)
            action_seq = naction.to(device)
            attention = nattention.to(device)
            
            output = model(obs, action_seq)
            loss = criterion(output, attention)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        model.eval()    
        for batch in val_dataloader:
            val_loss = 0
            with torch.no_grad():
                B, T, D = batch['obs'].shape
                nobs = normalizer['obs'].normalize(batch['obs'])
                naction = normalizer['action'].normalize(batch['action'])
                nattention = normalizer['spatial_attention'].normalize(batch['spatial_attention'])
                
                obs = nobs[:, :2, :].reshape(B, -1).to(device)
                action_seq = naction.to(device)
                attention = nattention.unsqueeze(-1).to(device)
                
                output = model(obs, action_seq)
                loss = criterion(output, attention)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_state_dict = model.state_dict()
        print(f"Best val loss: {best_val_loss}")
        
    # Now plot both true attention and predicted attention
    # zarr_path = os.path.expanduser('../data/robomimic/datasets/lift/ph/image_abs.hdf5.zarr.zip')
    
    env_meta = FileUtils.get_env_metadata_from_dataset(lowdim_dataset_path)
    task_name = env_meta['env_name']
    
    image_dataset_dict = {"Lift": "lift", "PickPlaceCan": "can", "NutAssemblySquare": "square", "ToolHang": "tool_hang"}
    dataset_path = os.path.expanduser(f'data/robomimic/datasets/{image_dataset_dict[task_name]}/ph/image_abs.hdf5')
    # replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)

    # Define shape metadata
    object_dict = {"Lift": 10, "PickPlaceCan": 14, "NutAssemblySquare": 14, "ToolHang": 44}
    if task_name == "ToolHang":
        image_shape = [3, 240, 240]
        view_key = 'sideview_image'
    else:
        image_shape = [3, 84, 84]
        view_key = 'agentview_image'
    shape_meta = {
        'action': {
            'shape': [7]
        },
        'obs': {
            'object': {
                'shape': [object_dict[task_name]]
            },
            view_key: {
                'shape': image_shape,
                'type': 'rgb'
            },
            'robot0_eef_pos': {
                'shape': [3]
            },
            'robot0_eef_quat': {
                'shape': [4]
            },
            'robot0_eye_in_hand_image': {
                'shape': image_shape,
                'type': 'rgb'
            },
            'robot0_gripper_qpos': {
                'shape': [2]
            }
        }
    }

    # Create dataset
    image_dataset = RobomimicReplayImageDataset(
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        horizon=2,
        pad_before=1,
        pad_after=1,
        rotation_rep='rotation_6d',
        seed=42,
        val_ratio=0.0,
        use_legacy_normalizer=False,
    )

    img_dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)
    img_iterator = iter(img_dataloader)

    display_dataloader = DataLoader(
        RobomimicReplayLowdimDataset(lowdim_dataset_path, horizon=horizon, pad_before=1, pad_after=horizon-1, obs_keys=['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'], abs_action=True, rotation_rep='rotation_6d', use_legacy_normalizer=False, seed=42, val_ratio=0.0, max_train_episodes=None, info_keys=['spatial_attention']),
        batch_size=1,
    )

    assert len(display_dataloader) == len(img_dataloader), f"display_dataloader: {len(display_dataloader)}, img_dataloader: {len(img_dataloader)}"


    new_dataset_path = lowdim_dataset_path
    file = h5py.File(new_dataset_path, 'r')
    num_demos = len(file['data'].keys())

    video_dir = os.path.join(os.path.dirname(lowdim_dataset_path), f"videos_of_inferred_spatial_attention")
    os.makedirs(video_dir, exist_ok=True)

    sample = next(iter(img_dataloader))

    _, _, C, H, W = sample['obs'][view_key].shape

    img_iterator = iter(img_dataloader)
    lowdim_iterator = iter(display_dataloader)
    model.to(device)
    
    for i in tqdm(range(10)):
        demo_key = f'data/demo_{i}'
        demo = file[demo_key]
        num_samples = demo.attrs['num_samples']
        
        imgs = np.zeros((num_samples, H, 2*W, C), dtype=np.uint8)
        
        spatial_attention = list()
        gt_spatial_attention = list()
        for sample_idx in tqdm(range(num_samples), leave=False):        
            sample = next(img_iterator)
            batch = next(lowdim_iterator)
            
            # Transpose from (3,84,84) to (84,84,3) format
            imgs[sample_idx, :, :W, :] = ((sample['obs'][view_key][0, 0].permute(1, 2, 0) * 255).numpy()).astype(np.uint8)
            imgs[sample_idx, :, W:, :] = (sample['obs']['robot0_eye_in_hand_image'][0, 0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            
            assert np.linalg.norm(sample['obs']['object'][0, 1, :].numpy()-demo['obs']['object'][sample_idx]) < 1e-4 , f"sample['obs']['object'][0, 1, :].numpy(): {sample['obs']['object'][0, 1, :].numpy()}, demo['obs']['object'][sample_idx]: {demo['obs']['object'][sample_idx]}" #check if this is sync well with the dataset
            
            B, T, D = batch['obs'].shape
            nobs = normalizer['obs'].normalize(batch['obs']).to(device)
            naction = normalizer['action'].normalize(batch['action']).to(device)
            
            obs = nobs[:, :2, :].reshape(B, -1).to(device)
            action_seq = naction.to(device)
            # attention = nobs[:, :, -1].unsqueeze(-1).to(device)
            
            with torch.no_grad():
                output = model(obs, action_seq).detach()
                attention_pred = normalizer['spatial_attention'].unnormalize(output).detach().cpu().numpy()
            spatial_attention.append(attention_pred[0, 0])
            gt_spatial_attention.append(batch['spatial_attention'][0, 0])
            
        spatial_attention = np.array(spatial_attention)
        next(img_iterator) # For syncing
        next(lowdim_iterator)
                
        # Get attention graph images
        fig, ax = plt.subplots(figsize=(4, 3))
        time_steps = np.arange(len(spatial_attention))
        ax.plot(time_steps, gt_spatial_attention, 'b-', color='tab:blue', linewidth=1, label='Spatial Attention')
        ax.plot(time_steps, spatial_attention, 'b-', color='tab:orange', linewidth=1, label='Spatial Attention')
        ax.set_xlabel('Time Step')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True)
        graph_frames = list()
        for t in range(len(gt_spatial_attention)):
            ax.scatter(t, gt_spatial_attention[t], color='blue', s=30)
            ax.scatter(t, spatial_attention[t], color='orange', s=30)
            
            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            graph_frames.append(graph_img)
        graph_height, graph_width = graph_frames[0].shape[:2]
        # prepare combined frames
        
        # attention_data = get_spatial_attention_from_episode(epi)
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
            
        video_path = os.path.join(video_dir, f"episode_{i}.mp4")
        
        # Create and write video with moviepy
        clip = ImageSequenceClip(combined_frames, fps=20)
        clip.write_videofile(video_path, codec='libx264')
        
    ## Do you like it? Then save it!!
    model.to("cpu")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(lowdim_dataset_path), f"seq2seq_attention_estimator.pth"))
    torch.save(normalizer.state_dict(), os.path.join(os.path.dirname(lowdim_dataset_path), f"normalizer.pth"))
        
if __name__ == '__main__':
    main()
