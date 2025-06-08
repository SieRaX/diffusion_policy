import torch
import tqdm

class BaseSampler:
    def __init__(self):
        raise NotImplementedError
        
    def sample(self, batch_size, device, **kwargs):
        raise NotImplementedError
    
class EulerMaruyamaSampler(BaseSampler):
    def __init__(self, sample_dim, batch_size, num_steps, eps):
        self.num_steps = num_steps
        self.eps = eps
        self.sample_dim = sample_dim
        self.batch_size = batch_size
        
    def sample(self, score_model, condition, marginal_prob_std, diffusion_coeff, device, **kwargs):
        """Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
            score_model: A PyTorch model that represents the time-dependent score-based model.
            marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
            diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
            batch_size: The number of samplers to generate by calling this function once.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.
            device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
            eps: The smallest time step for numerical stability.
        
        Returns:
            Samples.    
        """
        t = torch.ones(self.batch_size, device=device)
        
        time_steps = torch.linspace(1., self.eps, self.num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        x = torch.randn(self.batch_size, self.sample_dim, device=device) * marginal_prob_std(t)[:, None]
        tqdm_bar = tqdm.tqdm(time_steps, leave=False, desc=f"{'Conditional' if condition is not None else 'Unconditional'} Sampling")
        with torch.no_grad():
            for time_step in tqdm_bar:      
                batch_time_step = torch.ones(self.batch_size, device=device) * time_step
                g = diffusion_coeff(batch_time_step)
                score_batch_time_step = batch_time_step
                # score_batch_time_step = (batch_time_step*100).to(dtype=torch.int32)
                mean_x = x + (g**2)[:, None] * score_model(x, score_batch_time_step, condition) * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)      
        # Do not include any noise in the last sampling step.
        return mean_x