import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
import numpy as np
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from PIL import Image
from matplotlib import pyplot as plt
import torch
import dill
from torch.utils.data import DataLoader

from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

def main():
    # Load replay buffer
    # Get current working directory
    pwd = os.path.dirname(os.path.abspath(__file__))
    zarr_path = os.path.expanduser(os.path.join(pwd, '../data/robomimic/datasets/lift/ph/image_abs.hdf5.zarr.zip'))
    dataset_path = os.path.expanduser(os.path.join(pwd, '../data/robomimic/datasets/lift/ph/image_abs.hdf5'))
    replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)

    # Define shape metadata
    shape_meta = {
        'action': {
            'shape': [7]
        },
        'obs': {
            'object': {
                'shape': [10]
            },
            'agentview_image': {
                'shape': [3, 84, 84],
                'type': 'rgb'
            },
            'robot0_eef_pos': {
                'shape': [3]
            },
            'robot0_eef_quat': {
                'shape': [4]
            },
            'robot0_eye_in_hand_image': {
                'shape': [3, 84, 84],
                'type': 'rgb'
            },
            'robot0_gripper_qpos': {
                'shape': [2]
            }
        }
    }

    # Create dataset
    dataset = RobomimicReplayImageDataset(
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

    # Load model checkpoint
    checkpoint = os.path.join(pwd, "../data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/checkpoints/epoch=0200-test_mean_score=1.000.ckpt")
    output_dir = os.path.join(pwd, "../data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/dummy")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.policy.noise_scheduler._target_ = 'diffusion_policy.schedulers.scheduling_ddpm.DDPMScheduler'

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Get policy from workspace
    policy = workspace.model

    # Setup device and model
    device = torch.device('cuda:0')
    policy.to(device)
    policy.eval()

    video_dir = os.path.join(pwd, "../data/robomimic/datasets/lift/ph/videos_with_spatial_attention_200")
    os.makedirs(video_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    iterator = iter(dataloader)

    # Process each episode
    for i in tqdm(range(replay_buffer.n_episodes)):
        epi = replay_buffer.get_episode(i)
        T, H, W, C = epi['agentview_image'].shape
        
        imgs = np.zeros((T+2, H, 2*W, C), dtype=epi['agentview_image'].dtype)
        print(f"episode {i}| T: {T}")
        
        if i >= 100:
            for t in tqdm(range(T+1), desc="Passing episode", leave=False):
                sample = next(iterator)
        else:
        
            spatial_attention = list()
            for t in tqdm(range(T+1), desc="Getting Image and Spatial Attention", leave=False):
                sample = next(iterator)
                # Transpose from (3,84,84) to (84,84,3) format
                imgs[t, :, :W, :] = ((sample['obs']['agentview_image'][0, 0].permute(1, 2, 0) * 255).numpy()).astype(np.uint8)
                imgs[t, :, W:, :] = (sample['obs']['robot0_eye_in_hand_image'][0, 0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                n_obs_dict = {
                    'obs': np.concatenate([
                        sample['obs']['object'], 
                        sample['obs']['robot0_eef_pos'], 
                        sample['obs']['robot0_eef_quat'], 
                        sample['obs']['robot0_gripper_qpos']
                    ], axis=-1).astype(np.float32)
                }
                
                # Device transfer
                obs_dict = dict_apply(n_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    spatial_attention.append(policy.kl_divergence_drop(obs_dict).detach().cpu().numpy().item())
            spatial_attention = np.array(spatial_attention)
            
            # Get attention graph images
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
                
            video_path = os.path.join(video_dir, f"episode_{i}.mp4")
            
            # Create and write video with moviepy
            clip = ImageSequenceClip(combined_frames, fps=10)
            clip.write_videofile(video_path, codec='libx264')

if __name__ == "__main__":
    main()