import torch

def loss_fn(model, x, condition, marginal_prob_std, eps=1e-3):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t).reshape(-1, *[1 for _ in range( x.ndim-1)])
  perturbed_x = x + z * std

  # int_time = (random_t*100).to(torch.int32)
  # score = model(perturbed_x, int_time, condition)
  score = model(perturbed_x, random_t, condition)
  loss = torch.mean(torch.sum((score * std + z)**2, dim=(1)))
  return loss

def loss_fn_fixed_time(model, x, condition, marginal_prob_std, time):
  # time = torch.max(time, torch.tensor(1e-3))
  time = torch.fill(torch.zeros(x.shape[0], device=x.device), time)
  z = torch.randn_like(x)
  std = marginal_prob_std(time).reshape(-1, *[1 for _ in range( x.ndim-1)])
  perturbed_x = x + z * std
  normalized_score = model(perturbed_x, time, condition)
  loss = torch.mean(torch.sum((normalized_score * std + z)**2, dim=(1)))
  return loss