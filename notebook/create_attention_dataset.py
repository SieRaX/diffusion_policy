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
import h5py
import shutil

from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

def main():
    pwd = os.path.dirname(os.path.abspath(__file__))

    check_point_path = os.path.expanduser(os.path.join(pwd, "../outputs/can_ph_lowdim_reproduction/2025.04.24_18.13.10_train_diffusion_unet_lowdim_can_lowdim_cnn_16/checkpoints/epoch=0300-test_mean_score=1.000.ckpt"))
    task_name = torch.load(open(check_point_path, 'rb'), pickle_module=dill)['cfg'].task.task_name
    epoch_name = check_point_path.split('epoch=')[1].split('-')[0]
    checkpoint_dir_path = os.path.dirname(check_point_path)
    # dataset_path = os.path.expanduser(os.path.join(pwd, '../data/robomimic/datasets/lift/ph/image_abs.hdf5'))
    dataset_path = os.path.expanduser(os.path.join(pwd, f'../data/robomimic/datasets/{task_name}/ph/image_abs.hdf5'))

    # Define shape metadata
    shape_meta = {
        'action': {
            'shape': [7]
        },
        'obs': {
            'object': {
                'shape': [14]
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

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    iterator = iter(dataloader)

    # bring h5py file
    dataset_path = os.path.expanduser(os.path.join(pwd, f'../data/robomimic/datasets/{task_name}/ph/low_dim_abs.hdf5'))
    new_dataset_path = os.path.expanduser(os.path.join(pwd, checkpoint_dir_path, f'low_dim_abs_with_attention_epoch={epoch_name}.hdf5'))
    # new_dataset_path = os.path.expanduser(os.path.join(pwd, '../data/robomimic/datasets/lift/ph/low_dim_abs_with_sine_wave_2.hdf5'))

    # First, copy the entire file to preserve all structure
    shutil.copy(dataset_path, new_dataset_path)
    try:
        file = h5py.File(new_dataset_path, 'r+')

        num_demos = len(file['data'].keys())
        print(f"Number of demonstrations: {num_demos}")

        length_of_each_demo = list()
        for i in tqdm(range(num_demos)):
            demo_key = f'data/demo_{i}'
            demo = file[demo_key]
            length_of_each_demo.append(demo.attrs['num_samples'])
        length_of_each_demo = np.array(length_of_each_demo)

        assert (length_of_each_demo+1).sum() == len(dataset)

        # Prepaer model
        # Load model checkpoint
        output_dir = os.path.expanduser(os.path.join(pwd, checkpoint_dir_path, f'dummy'))
        payload = torch.load(open(check_point_path, 'rb'), pickle_module=dill)
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
        policy.to(device);
        policy.eval();


        iterator = iter(dataloader)
        # num_demos = 2
        for i in tqdm(range(num_demos)):
            demo_key = f'data/demo_{i}'
            demo = file[demo_key]
            num_samples = demo.attrs['num_samples']
            
            spatial_attention = list()
            for sample_idx in tqdm(range(num_samples), leave=False):        
                sample = next(iterator)
                
                assert np.linalg.norm(sample['obs']['object'][0, 1, :].numpy()-demo['obs']['object'][sample_idx]) < 1e-4
                
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
                    # spatial_attention.append(0.01*np.sin(np.pi*sample_idx/num_samples)+0.01*np.cos(np.pi*i/num_samples))
            spatial_attention = np.array(spatial_attention).reshape(-1, 1)
            
            next(iterator) # Fro syncing
            
            if 'spatial_attention' not in demo['obs'].keys():
                demo['obs'].create_dataset(
                    'spatial_attention',
                    shape=(num_samples,1),   
                    data=spatial_attention,
                    dtype=np.float32
                )
            else:
                demo['obs']['spatial_attention'][:] = spatial_attention
    finally:
        file.close()
    
if __name__ == "__main__":
    main()
