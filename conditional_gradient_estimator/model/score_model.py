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
    return self.dense(x)[..., None]


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
  
class UnetBasedScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, condition_dim, input_channels=1, channels=[32, 64, 128, 256, 512, 1024], stride=[1, 2, 2, 2, 2, 2], padding=[1, 1, 1, 1, 1, 1], output_padding=[0, 0, 0, 0, 0, 0], kernel_size=3, embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      stride: The stride for each encoding layer.
      padding: The padding for each encoding layer.
      output_padding: The output padding for each decoding layer.
      kernel_size: The kernel size for convolutions.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    self.condition_layer = nn.Sequential(
      nn.Linear(condition_dim+1, embed_dim),
      nn.SiLU(),
      nn.Linear(embed_dim, embed_dim),
      nn.SiLU(),
      nn.Linear(embed_dim, embed_dim),
    )
    
    # Validate input lists
    assert len(channels) == len(stride) == len(padding) == len(output_padding), f"channels and stride lists must have the same length len(channels): {len(channels)}, len(stride): {len(stride)}, len(padding): {len(padding)}, len(output_padding): {len(output_padding)}"
    self.num_encoding_layers = len(channels)
    
    # Dynamic encoding layers using ModuleList
    self.encoding_convs = nn.ModuleList()
    self.encoding_denses = nn.ModuleList()
    self.encoding_gnorms = nn.ModuleList()
    
    # First layer: input channels = 2
    self.encoding_convs.append(
      nn.Conv1d(input_channels, channels[0], kernel_size=kernel_size, stride=stride[0], bias=False, padding=padding[0])
    )
    self.encoding_denses.append(Dense(2*embed_dim, channels[0]))
    self.encoding_gnorms.append(nn.GroupNorm(4, num_channels=channels[0]))
    
    # Remaining layers
    for i in range(1, self.num_encoding_layers):
      self.encoding_convs.append(
        nn.Conv1d(channels[i-1], channels[i], kernel_size=kernel_size, stride=stride[i], bias=False, padding=padding[i])
      )
      self.encoding_denses.append(Dense(2*embed_dim, channels[i]))
      self.encoding_gnorms.append(nn.GroupNorm(32, num_channels=channels[i]))

    # Dynamic decoding layers using ModuleList
    self.decoding_tconvs = nn.ModuleList()
    self.decoding_denses = nn.ModuleList()
    self.decoding_gnorms = nn.ModuleList()
    
    # First decoding layer (no skip connection)
    self.decoding_tconvs.append(
      nn.ConvTranspose1d(channels[-1], channels[-2], kernel_size=kernel_size, stride=stride[-1], bias=False, output_padding=output_padding[-1], padding=padding[-1])
    )
    self.decoding_denses.append(Dense(2*embed_dim, channels[-2]))
    self.decoding_gnorms.append(nn.GroupNorm(32, num_channels=channels[-2]))
    
    # Middle decoding layers (with skip connections)
    for i in range(self.num_encoding_layers - 2, 0, -1):
      # Input channels = current + skip connection
      tconv_input_channels = channels[i] + channels[i]
      self.decoding_tconvs.append(
        nn.ConvTranspose1d(tconv_input_channels, channels[i-1], kernel_size=kernel_size, stride=stride[i], bias=False, output_padding=output_padding[i], padding=padding[i])
      )
      self.decoding_denses.append(Dense(2*embed_dim, channels[i-1]))
      self.decoding_gnorms.append(nn.GroupNorm(32, num_channels=channels[i-1]))
    
    # Final decoding layer (with skip connection, output to 2 channels)
    tconv_input_channels = channels[0] + channels[0]
    self.decoding_tconvs.append(
      nn.ConvTranspose1d(tconv_input_channels, input_channels, kernel_size=kernel_size, stride=stride[0], output_padding=output_padding[0], padding=padding[0])
    )
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
    self.condition_dim = condition_dim
    self.channels = channels
    self.stride = stride
  
  def forward(self, x, t, condition=None):
    # Obtain the Gaussian random feature embedding for t
    seq_len = x.shape[-1]
    if x.dim() == 2:
      x = x[:, None, :]
    embed = self.act(self.embed(t))
    
    # Padd the input to the nearest multiple of 2**(self.unet_depth-1)
    # x.shape = (batch_size, channels, length)
    assert x.shape[-1]%2 == 0, f"The length of features should be even"
    pad_len = (2**(self.num_encoding_layers) - x.shape[-1]%2**(self.num_encoding_layers))//2
    x = F.pad(x, (pad_len, pad_len), "constant", 0)
    
    if condition is not None:
      condition = torch.cat([torch.zeros(condition.shape[0], 1, device=condition.device), condition], dim=-1)
    else:
      condition = torch.cat([torch.ones(x.shape[0], 1, device=x.device), torch.zeros(x.shape[0], self.condition_dim, device=x.device)], dim=-1)
    condition_embed = self.act(self.condition_layer(condition))
      
    embed = torch.cat([embed, condition_embed], dim=-1)
     
    # Encoding path - store all intermediate outputs for skip connections
    encoding_outputs = []
    h = x
    
    for i in range(self.num_encoding_layers):
      h = self.encoding_convs[i](h)
      h += self.encoding_denses[i](embed)
      h = self.encoding_gnorms[i](h)
      h = self.act(h)
      encoding_outputs.append(h)

    # Decoding path
    # First decoding layer (no skip connection)
    h = self.decoding_tconvs[0](h)
    h += self.decoding_denses[0](embed)
    h = self.decoding_gnorms[0](h)
    h = self.act(h)
    
    # Middle decoding layers (with skip connections)
    for i in range(1, len(self.decoding_tconvs) - 1):
      # Concatenate with corresponding encoding output (reverse order)
      skip_idx = self.num_encoding_layers - 1 - i
      h = self.decoding_tconvs[i](torch.cat([h, encoding_outputs[skip_idx]], dim=1))
      h += self.decoding_denses[i](embed)
      h = self.decoding_gnorms[i](h)
      h = self.act(h)
    
    # Final decoding layer
    h = self.decoding_tconvs[-1](torch.cat([h, encoding_outputs[0]], dim=1))
    
    # Interpolate the output to the original length
    h = F.interpolate(h, size=seq_len, mode='linear', align_corners=False)

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None]
    h = h.squeeze()
    return h