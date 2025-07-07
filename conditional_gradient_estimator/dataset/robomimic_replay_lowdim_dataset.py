from torch.utils.data import Dataset
import torch
import numpy as np
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.dataset.d4rl_lowdim_dataset import D4RLReplayLowdimDataset

class RobomimicReplayLowdimDatasetWrapper(Dataset):
    def __init__(self, root_dataset, n_obs_steps=2):
        assert isinstance(root_dataset, RobomimicReplayLowdimDataset) or isinstance(root_dataset, D4RLReplayLowdimDataset)
        self.dataset = root_dataset
        self.n_obs_steps = n_obs_steps
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        data = self.dataset[idx]
        # obs = data["obs"][:2, :self.obs_dim] # B, T, O
        # obs = np.concatenate([obs, np.zeros((2,1))], axis=1)
        obs = torch.cat([data["obs"][i, :] for i in range(self.n_obs_steps)], dim=-1) # B, T, O
        # obs = np.concatenate([obs, np.zeros((2,1))], axis=1)
        action = data["action"][1] # B, T, A
        
        # concat_obs = obs[:2].reshape( -1)
        
        return {"data": obs, "condition": action}
    
    def get_normalizer(self, mode='limits', **kwargs):
        
        data = self.dataset.replay_buffer['obs']
        # data = np.stack([data[:-1, :], data[1:, :]], axis=1)
        data = np.stack([data[i: len(data)-self.n_obs_steps+i+1, :] for i in range(self.n_obs_steps)], axis=1)
        # data = data[...,:self.obs_dim]
        # data = np.concatenate([data, np.zeros((data.shape[0], data.shape[1], 1))], axis=2)

        data = {
            'data': data,
            'condition': self.dataset.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-3
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer        