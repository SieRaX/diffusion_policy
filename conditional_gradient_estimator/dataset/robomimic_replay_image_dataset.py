import hydra
from torch.utils.data import Dataset
import torch
import numpy as np
import dill
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.dataset.d4rl_lowdim_dataset import D4RLReplayLowdimDataset
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from omegaconf import OmegaConf, open_dict

class RobomimicReplayImageDatasetWrapper(Dataset):
    def __init__(self, root_dataset, vision_encoder_checkpoint, n_obs_steps=2, device='cuda:0'):
        assert isinstance(root_dataset, RobomimicReplayImageDataset)
        self.dataset = root_dataset
        self.n_obs_steps = n_obs_steps
        
        print(f"Loading vision encoder from {vision_encoder_checkpoint}")
        payload = torch.load(open(vision_encoder_checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        
        '''
        For configs that does not have keys before... erase this when the code is all set up...
        '''
        if "image_feature_dim" not in cfg.policy.keys():
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.policy.image_feature_dim = 5
            OmegaConf.set_struct(cfg, False)

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        
        self.policy = policy
        self.obs_normalizer = policy.normalizer
        self.obs_encoder = policy.obs_encoder
        
        # Pre-compute encoded observations
        self.encoded_obs = self._precompute_encoded_observations(device)
    
    def _precompute_encoded_observations(self, device):
        """Pre-compute encoded observations for the entire dataset"""
        
        
        # Process in batches for efficiency
        batch_size = 1024  # Adjust based on your GPU memory        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        encoded_obs = []
        
        self.policy.to(device=device)
        self.policy.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Encoding observations", leave=False):
                obs_dict = data["obs"]
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))
                nobs_features = self.policy.encode_obs(obs_dict)
                feature_dim = nobs_features.shape[-1]
                encoded_obs.append(nobs_features.reshape(-1, feature_dim*self.n_obs_steps).detach().cpu())
                # obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))
                # nobs = self.obs_normalizer.normalize(obs_dict)
                # nobs_features = self.obs_encoder(nobs)
                # encoded_obs.append(nobs_features.detach().cpu())
        self.policy.to(device='cpu')
        self.policy.train()
        return torch.cat(encoded_obs, dim=0)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        obs = self.encoded_obs[idx].squeeze()
        action = data["action"][1] # B, T, A
        
        # concat_obs = obs[:2].reshape( -1)
        
        return {"data": obs, "condition": action}
    
    def get_normalizer(self, mode='limits', **kwargs):
        
        data = self.encoded_obs
        # data = np.stack([data[:-1, :], data[1:, :]], axis=1)
        # data = np.stack([data[i: len(data)-self.n_obs_steps+i+1, :] for i in range(self.n_obs_steps)], axis=1)
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