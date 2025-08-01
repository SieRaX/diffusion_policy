import numpy as np
import torch

def marginal_prob_std(t, sigma, device):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """
  t = t.to(device)
  # t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time t.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  t = t.to(device)
  # t = torch.tensor(t, device=device)
  return sigma**t
  
# sigma =  50.0#@param {'type':'number'}
# marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
# diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)