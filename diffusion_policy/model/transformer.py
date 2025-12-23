import hydra
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from conditional_gradient_estimator.VAE.vae import VAE
import torch
import torch.nn as nn

# class Seq2SeqTransformer(nn.Module):
#     def __init__(self, obs_dim=38, action_dim=10, seq_len=16, d_model=128, nhead=4, num_layers=2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.obs_proj = nn.Linear(obs_dim, d_model)
#         self.action_proj = nn.Linear(action_dim, d_model)

#         self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))

#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.decoder = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, 1)
#         )
        
        

#     def forward(self, obs, action_seq):
#         # obs: (B, 38), action_seq: (B, 16, 10)
#         B = obs.size(0)

#         # Project inputs
#         obs_emb = self.obs_proj(obs)  # (B, d_model)
#         obs_emb = obs_emb.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, 16, d_model)

#         action_emb = self.action_proj(action_seq)  # (B, 16, d_model)

#         # Combine observation embedding with action sequence embedding
#         x = obs_emb + action_emb + self.pos_embedding.unsqueeze(0)  # (B, 16, d_model)

#         # Encode
#         encoded = self.encoder(x)  # (B, 16, d_model)

#         # Decode to (B, 16, 1)
#         output = self.decoder(encoded)
#         return output
    
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, obs_dim=38, action_dim=10, seq_len=16, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        # For saving configs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.register_buffer("seq_len", torch.tensor(seq_len))
        # self.seq_len = seq_len # uncomment this for running old codes. (The ones that does not have seq_len as register buffer)

    def forward(self, obs, action_seq):
        # obs: (B, 38), action_seq: (B, 16, 10)
        B = obs.size(0)

        # Project inputs
        obs_emb = self.obs_proj(obs)  # (B, d_model)
        obs_emb = obs_emb.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, 16, d_model)

        action_emb = self.action_proj(action_seq)  # (B, 16, d_model)

        # Combine observation embedding with action sequence embedding
        x = obs_emb + action_emb + self.pos_embedding.unsqueeze(0)  # (B, 16, d_model)

        # Encode
        encoded = self.encoder(x)  # (B, 16, d_model)

        # Decode to (B, 16, 1)
        output = self.decoder(encoded)
        return output
    
class Seq2SeqTransformerWithVisionEncoder(nn.Module):
    def __init__(self, vision_encoder_checkpoint=None, obs_dim=38, action_dim=10, seq_len=16, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.seq2seq_transformer = Seq2SeqTransformer(obs_dim, action_dim, seq_len, d_model, nhead, num_layers)
        
        # Store policy config dictionary (can be None if loading from checkpoint)
        self.vision_encoder_checkpoint = vision_encoder_checkpoint
        self.encoder_initialized = False
        
    def forward(self, obs, action_seq):
        return self.seq2seq_transformer(obs, action_seq)
            
    def state_dict(self, *args, **kwargs):
        """
        Override state_dict to include policy_cfg_dict metadata.
        This ensures the policy config dictionary is saved along with model parameters.
        Excludes self.policy parameters from the state_dict.
        """
        assert self.encoder_initialized, f"Encoder not initialized. Please call initialize_policy() before saving the model."
        state_dict = super().state_dict(*args, **kwargs)
        # Remove all parameters from self.policy (keys starting with 'policy.')
        keys_to_remove = [key for key in state_dict.keys() if key.startswith('policy.')]
        for key in keys_to_remove:
            del state_dict[key]
            
        state_dict['_metadata.vision_encoder_checkpoint'] = self.vision_encoder_checkpoint
        state_dict['_metadata.seq2seq_transformer_config'] = {'obs_dim': self.seq2seq_transformer.obs_dim, 'action_dim': self.seq2seq_transformer.action_dim, 'seq_len': self.seq2seq_transformer.seq_len, 'd_model': self.seq2seq_transformer.d_model, 'nhead': self.seq2seq_transformer.nhead, 'num_layers': self.seq2seq_transformer.num_layers}
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to extract and set policy_cfg_dict.
        This automatically restores the policy config dictionary when loading a saved model.
        Filters out policy. keys if present (for backward compatibility).
        """
        # Extract metadata before calling parent's load_state_dict
        metadata_key = '_metadata.vision_encoder_checkpoint'
        if metadata_key in state_dict:
            self.vision_encoder_checkpoint = state_dict.pop(metadata_key)
        elif hasattr(self, 'policy_cfg_dict') and self.policy_cfg_dict is None:
            # If not in checkpoint and not set in __init__, warn user
            import warnings
            warnings.warn(
                "policy_cfg_dict not found in checkpoint and not provided in __init__. "
                "The model may not function correctly without it.",
                UserWarning
            )
            
        metadata_key = '_metadata.seq2seq_transformer_config'
        if metadata_key in state_dict:
            seq2seq_transformer_config = state_dict.pop(metadata_key)
            self.seq2seq_transformer = Seq2SeqTransformer(seq2seq_transformer_config['obs_dim'], seq2seq_transformer_config['action_dim'], seq2seq_transformer_config['seq_len'], seq2seq_transformer_config['d_model'], seq2seq_transformer_config['nhead'], seq2seq_transformer_config['num_layers'])
        
        # # Extract and load seq2seq_transformer state dict (keys with prefix "seq2seq_transformer.")
        # seq2seq_transformer_state_dict = {}
        # keys_to_remove = []
        # prefix = 'seq2seq_transformer.'
        # for key in state_dict.keys():
        #     if key.startswith(prefix):
        #         # Strip the prefix to get the actual parameter name
        #         new_key = key[len(prefix):]
        #         seq2seq_transformer_state_dict[new_key] = state_dict[key]
        #         keys_to_remove.append(key)
        
        # # Load the state dict into seq2seq_transformer
        # if seq2seq_transformer_state_dict:
        #     self.seq2seq_transformer.load_state_dict(seq2seq_transformer_state_dict, strict=strict)
        
        # # Remove seq2seq_transformer keys from main state_dict
        # for key in keys_to_remove:
        #     state_dict.pop(key)
        
        # Remove any policy. keys if present (shouldn't be there, but handle for backward compatibility)
        policy_keys_to_remove = [key for key in state_dict.keys() if key.startswith('policy.')]
        for key in policy_keys_to_remove:
            state_dict.pop(key)
        
        res = super().load_state_dict(state_dict, strict=strict)
        self.initialize_policy()
        return res
    
    def initialize_policy(self):
        assert self.vision_encoder_checkpoint is not None, f"Vision encoder checkpoint is not provided."
        
        payload = torch.load(open(self.vision_encoder_checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.policy = workspace.model
        
        self.encoder_initialized = True
        
    def encode_obs(self, obs):
        assert self.encoder_initialized
        return self.policy.encode_obs(obs)

class Seq2SeqTransformerWithVAE(nn.Module):
    def __init__(self, vae_checkpoint=None, obs_dim=38, action_dim=10, seq_len=16, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.seq2seq_transformer = Seq2SeqTransformer(obs_dim, action_dim, seq_len, d_model, nhead, num_layers)
        
        # Store policy config dictionary (can be None if loading from checkpoint)
        self.vae_checkpoint = vae_checkpoint
        self.encoder_initialized = False
        
    def forward(self, obs, action_seq):
        return self.seq2seq_transformer(obs, action_seq)
            
    def state_dict(self, *args, **kwargs):
        """
        Override state_dict to include policy_cfg_dict metadata.
        This ensures the policy config dictionary is saved along with model parameters.
        Excludes self.policy parameters from the state_dict.
        """
        assert self.encoder_initialized, f"Encoder not initialized. Please call initialize_policy() before saving the model."
        state_dict = super().state_dict(*args, **kwargs)
        # Remove all parameters from self.policy (keys starting with 'policy.')
        keys_to_remove = [key for key in state_dict.keys() if key.startswith('policy.')]
        for key in keys_to_remove:
            del state_dict[key]
            
        state_dict['_metadata.vae_checkpoint'] = self.vae_checkpoint
        state_dict['_metadata.seq2seq_transformer_config'] = {'obs_dim': self.seq2seq_transformer.obs_dim, 'action_dim': self.seq2seq_transformer.action_dim, 'seq_len': self.seq2seq_transformer.seq_len, 'd_model': self.seq2seq_transformer.d_model, 'nhead': self.seq2seq_transformer.nhead, 'num_layers': self.seq2seq_transformer.num_layers}
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to extract and set policy_cfg_dict.
        This automatically restores the policy config dictionary when loading a saved model.
        Filters out vae. and seq2seq_transformer. keys if present (for backward compatibility).
        """
        # Extract metadata before calling parent's load_state_dict
        vae_metadata_key = '_metadata.vae_checkpoint'
        if vae_metadata_key in state_dict:
            self.vae_checkpoint = state_dict.pop(vae_metadata_key)
        seq2seq_transformer_metadata_key = '_metadata.seq2seq_transformer_config'
        if seq2seq_transformer_metadata_key in state_dict:
            seq2seq_transformer_config = state_dict.pop(seq2seq_transformer_metadata_key)
            self.seq2seq_transformer = Seq2SeqTransformer(seq2seq_transformer_config['obs_dim'], seq2seq_transformer_config['action_dim'], seq2seq_transformer_config['seq_len'], seq2seq_transformer_config['d_model'], seq2seq_transformer_config['nhead'], seq2seq_transformer_config['num_layers'])
        
        # # Extract and load seq2seq_transformer state dict (keys with prefix "seq2seq_transformer.")
        # seq2seq_transformer_state_dict = {}
        # keys_to_remove = []
        # prefix = 'seq2seq_transformer.'
        # for key in state_dict.keys():
        #     if key.startswith(prefix):
        #         # Strip the prefix to get the actual parameter name
        #         new_key = key[len(prefix):]
        #         seq2seq_transformer_state_dict[new_key] = state_dict[key]
        #         keys_to_remove.append(key)
        
        # # Load the state dict into seq2seq_transformer
        # if seq2seq_transformer_state_dict:
        #     self.seq2seq_transformer.load_state_dict(seq2seq_transformer_state_dict, strict=strict)
        
        # # Remove seq2seq_transformer keys from main state_dict
        # for key in keys_to_remove:
        #     state_dict.pop(key)
        
        # Remove any policy. keys if present (shouldn't be there, but handle for backward compatibility)
        vae_keys_to_remove = [key for key in state_dict.keys() if key.startswith('vae.')]
        for key in vae_keys_to_remove:
            state_dict.pop(key)
        
        res = super().load_state_dict(state_dict, strict=strict)
        self.initialize_vae()
        return res
    
    def initialize_vae(self):
        assert self.vae_checkpoint is not None, f"VAE checkpoint is not provided."

        if "14" in self.vae_checkpoint:
            latent_dim = 14
            input_size = 84
        elif "7" in self.vae_checkpoint:
            latent_dim = 7
            input_size = 84
        elif "64" in self.vae_checkpoint:
            latent_dim = 64
            input_size = 240
        else:
            raise ValueError(f"VAE checkpoint {self.vae_checkpoint} is not supported.")
        self.vae = VAE(latent_dim=latent_dim, input_size=input_size)
        self.vae.load_state_dict(torch.load(self.vae_checkpoint))
        
        self.encoder_initialized = True
        
    def encode_obs(self, obs_dict):
        assert self.encoder_initialized

        imgs = []
        lowdims = []
        for key in obs_dict.keys():
            if 'image' in key:
                imgs.append(obs_dict[key]) 
            else:
                lowdims.append(obs_dict[key])
        imgs = torch.cat(imgs, dim=2)
        B, T, C, H, W = imgs.shape
        imgs = imgs.reshape(B*T, C, H, W)
        encoded_obs, _ = self.vae.encoder(imgs)
        encoded_obs = encoded_obs.reshape(B, T, -1)

        lowdims = torch.cat(lowdims, dim=2)
        nobs_features = torch.cat([encoded_obs, lowdims], dim=-1)
        feature_dim = nobs_features.shape[-1]
        
        return nobs_features.reshape(B, -1)