#@title Defining a time-dependent score-based model (double click to expand or collapse)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, data_dim=38, condition_dim=2, embed_dim=512, hidden_dim=512):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    self.condition_layer = nn.Linear(condition_dim+1, hidden_dim)
    
    # Encoding layers
    self.fc1 = nn.Linear(data_dim, hidden_dim)
    self.fc2 = nn.Linear(2*hidden_dim+embed_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    self.fc4 = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, data_dim),
    )
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
    
    self.condition_dim = condition_dim
  
  def forward(self, x, t, condition=None):
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))
    
    if condition is not None:
      condition = torch.cat([torch.zeros(condition.shape[0], 1, device=condition.device), condition], dim=-1)
    else:
      condition = torch.cat([torch.ones(x.shape[0], 1, device=x.device), torch.zeros(x.shape[0], self.condition_dim, device=x.device)], dim=-1)
      
    condition = self.act(self.condition_layer(condition))
      
    h1 = self.act(self.fc1(x))
    h2 = self.act(self.fc2(torch.cat([h1, embed, condition], dim=-1)))
    h3 = self.act(self.fc3(h2))
    h = self.fc4(h3)
  
    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None]
    return h
  
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D