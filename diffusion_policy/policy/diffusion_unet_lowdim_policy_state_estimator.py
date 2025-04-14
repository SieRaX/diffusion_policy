from typing import Dict
import torch
from torch import vmap
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from einops import rearrange, reduce
from diffusion_policy.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.diffusion.kl_divergence import normal_kl, extract

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim+obs_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # Temporal Assumption for current experiment
        assert self.obs_as_global_cond, "Only support global conditioning for now"
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_kl_grad(self, obs_dict: Dict[str, torch.Tensor], n_samples: int = 128) -> torch.Tensor:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        assert obs_dict['obs'].shape[0] == 1 # assert that there is only one observation as input. (For now)
        assert self.obs_as_global_cond # only support global conditioning (For Now)
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        
        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            global_cond = global_cond.repeat(n_samples, 1)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True
        
        cond_data = cond_data.repeat(n_samples, 1, 1)
        cond_mask = cond_mask.repeat(n_samples, 1, 1)
            
        model = self.model
        scheduler = self.noise_scheduler
        
        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        t_lst = [1, 2, 5, 10, 20, 50, 60, 90]
        
        kl_grad_lst = list()
        for t in t_lst:
            noise = torch.randn_like(nsample)
            timesteps = torch.full((B*n_samples, ), t, device=nsample.device)
            x_t = scheduler.add_noise(nsample, noise, timesteps)
            
            true_mean, true_variance, true_log_variance_clipped = self.q_posterior(nsample, x_t, timesteps)
            
            global_cond_input = global_cond.clone().requires_grad_(True)
            model_output = model(x_t, timesteps, local_cond=local_cond, global_cond=global_cond_input)
            pred_mean, pred_variance, pred_log_variance_clipped = self.p_mean_variance(
            x_t, t, model_output)
            
            kl = normal_kl(true_mean, true_log_variance_clipped,
                    pred_mean, pred_log_variance_clipped)
            kl_mean = kl.mean()
            kl_grad = grad(kl_mean, global_cond_input, create_graph=True)[0].detach()
            kl_grad_lst.append(kl_grad.norm(dim=1).mean())
        
        return kl_grad.mean()
        
        
        
    def predict_kl_divergence(self, obs_dict: Dict[str, torch.Tensor], n_samples: int = 128) -> torch.Tensor:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        assert self.obs_as_global_cond # only support global conditioning (For Now)
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da+Do)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            global_cond = global_cond.repeat_interleave(n_samples, dim=0)
        else:
            # condition through impainting
            shape = (B, T, Da+Do+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True
        
        cond_data = cond_data.repeat_interleave(n_samples, dim=0)
        cond_mask = cond_mask.repeat_interleave(n_samples, dim=0)
            
        model = self.model
        scheduler = self.noise_scheduler
        
        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # Sample a random timestep for each image
        kl_lst = list()
        num_anomaly_lst = list()
        # for t in scheduler.timesteps[:-1]:
        for t in [5]:
            # get x_t
            noise = torch.randn_like(nsample)
            timesteps = torch.full((B*n_samples, ), t, device=nsample.device)
            x_t = scheduler.add_noise(nsample, noise, timesteps)
            true_mean, true_variance, true_log_variance_clipped = self.q_posterior(nsample, x_t, timesteps)
            
            model_output = model(x_t, timesteps, local_cond=local_cond, global_cond=global_cond)
            pred_mean, pred_variance, pred_log_variance_clipped = self.p_mean_variance(
                x_t, t, model_output)
            
            kl = normal_kl(true_mean, true_log_variance_clipped,
                pred_mean, pred_log_variance_clipped)
            kl = kl.mean(dim=[1,2])
            valid_mask = kl <= 1000
            num_anomaly = len(valid_mask) - valid_mask.sum()
            if num_anomaly > 0:
                    assert False, "num_anomaly > 0"
                    
            kl_mean = kl.reshape(B, n_samples).mean(dim=1).unsqueeze(1)
            # kl_mean = kl[valid_mask].mean() if valid_mask.any() else torch.zeros_like(kl).mean()
            kl_lst.append(kl_mean)
        kl_lst = torch.cat(kl_lst, dim=-1).squeeze()
        return kl_lst, None
        
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nsample.shape[0],), device=trajectory.device
        ).long()
        x_t = scheduler.add_noise(nsample, noise, timesteps)
        
        true_mean, true_variance, true_log_variance_clipped = self.q_posterior(
            nsample, x_t, timesteps)
        
        # get p_mean_variance
        model_output = model(x_t, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        pred_mean, pred_variance, pred_log_variance_clipped = self.p_mean_variance(
            x_t, timesteps, model_output)
        
        # calculate normal kl
        kl = normal_kl(true_mean, true_log_variance_clipped,
            pred_mean, pred_log_variance_clipped)
        kl = kl.mean(dim=[1,2])
        # Create a mask for valid KL values (not larger than 1000)
        valid_mask = kl <= 1000
        num_anomaly = len(valid_mask) - valid_mask.sum()
        # Calculate mean only over valid values
        kl_mean = kl[valid_mask].mean() if valid_mask.any() else torch.zeros_like(kl).mean()
        # If you need to keep the original tensor structure, but with proper mean:
        kl = torch.where(kl > 1000, torch.zeros_like(kl), kl)
        return kl_mean, num_anomaly
        # TODO:
        # 1. get p_mean_variance
        # 2. calculate normal kl
    
    def kl_divergence_drop(self, obs_dict: Dict[str, torch.Tensor], n_samples: int = 128) -> torch.Tensor:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        assert self.obs_as_global_cond # only support global conditioning (For Now)
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da+Do)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            global_cond = global_cond.repeat_interleave(n_samples, dim=0)
        else:
            # condition through impainting
            shape = (B, T, Da+Do+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True
        
        cond_data = cond_data.repeat_interleave(n_samples, dim=0)
        cond_mask = cond_mask.repeat_interleave(n_samples, dim=0)
            
        model = self.model
        scheduler = self.noise_scheduler
        
        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        delta_o = 0.1
        kl_dim_lst = list()
        for i in range(global_cond.shape[1]):
            global_cond_perturbed = global_cond.clone()
            global_cond_perturbed[:,i] = global_cond[:,i] + delta_o

            kl_lst = list()
            # for t in scheduler.timesteps[:-1]:
            for t in [5]:
                # get x_t
                noise = torch.randn_like(nsample)
                timesteps = torch.full((B*n_samples, ), t, device=nsample.device)
                x_t = scheduler.add_noise(nsample, noise, timesteps)
                true_mean, true_variance, true_log_variance_clipped = self.q_posterior(nsample, x_t, timesteps)
                
                model_output = model(x_t, timesteps, local_cond=local_cond, global_cond=global_cond_perturbed)
                pred_mean, pred_variance, pred_log_variance_clipped = self.p_mean_variance(
                    x_t, t, model_output)
                
                kl = normal_kl(true_mean, true_log_variance_clipped,
                pred_mean, pred_log_variance_clipped)
                kl = kl.mean(dim=[1,2])
                valid_mask = kl <= 1000
                num_anomaly = len(valid_mask) - valid_mask.sum()
                if num_anomaly > 0:
                    assert False, "num_anomaly > 0"
                    
                kl_mean = kl.reshape(B, n_samples).mean(dim=1).unsqueeze(1)
                # kl_mean = kl[valid_mask].mean() if valid_mask.any() else torch.zeros_like(kl).mean()
                kl_lst.append(kl_mean)
            kl_dim_lst.append(torch.cat(kl_lst, dim=1).mean(dim=1))
        
            
            # # Sample a random timestep for each image
            # timesteps = torch.randint(
            #     0, self.noise_scheduler.config.num_train_timesteps, 
            #     (nsample.shape[0],), device=trajectory.device
            # ).long()
            # x_t = scheduler.add_noise(nsample, noise, timesteps)
            
            # true_mean, true_variance, true_log_variance_clipped = self.q_posterior(
            #     nsample, x_t, timesteps)
            
            # # get p_mean_variance
            # model_output = model(x_t, timesteps, 
            #     local_cond=local_cond, global_cond=global_cond_perturbed)
            # pred_mean, pred_variance, pred_log_variance_clipped = self.p_mean_variance(
            #     x_t, timesteps, model_output)
            
            # # calculate normal kl
            # kl = normal_kl(true_mean, true_log_variance_clipped,
            #     pred_mean, pred_log_variance_clipped)
            # kl = kl.mean(dim=[1,2])
            # # Create a mask for valid KL values (not larger than 1000)
            # valid_mask = kl <= 1000
            # # Calculate mean only over valid values
            # kl_mean = kl[valid_mask].mean() if valid_mask.any() else torch.zeros_like(kl).mean()
            # # If you need to keep the original tensor structure, but with proper mean:
            # kl = torch.where(kl > 1000, torch.zeros_like(kl), kl)
            # kl_dim_lst.append(kl_mean)
        kl_dim_lst = torch.stack(kl_dim_lst, dim=1)
        kl_dim_lst = kl_dim_lst.mean(dim=-1)
        
        original_kl_divergence, _ = self.predict_kl_divergence(obs_dict, n_samples=n_samples)
        
        difference = (kl_dim_lst - original_kl_divergence)/delta_o
        return difference
    
    def p_mean_variance(self, x_t, t: int, model_output):
        """
        Compute the mean and variance of the diffusion posterior p(x_{t-1} | x_t).
        """
        
        # set step values
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=x_t.device)
        
        diffusion_step_output = scheduler.step(model_output, t, x_t)
        
        return diffusion_step_output.mean, diffusion_step_output.variance, torch.log(diffusion_step_output.variance.clamp(min=1e-20))
        
        # B = x_t.shape[0]
        # model_output = model_output.unsqueeze(1)
        # x_t = x_t.unsqueeze(1)
        
        # # We use iterative code for now... it would be better to somehow find batchwise computation...
        # mean_batch = []
        # variance_batch = []
        
        # for i in range(B):
        #     diffusion_step_output = scheduler.step(model_output[i], int(t[i]), x_t[i])
            
        #     mean_batch.append(diffusion_step_output.mean.detach())
        #     variance_batch.append(diffusion_step_output.variance.detach().reshape(-1, 1, 1))
        
        # mean = torch.cat(mean_batch, dim=0)
        # variance = torch.cat(variance_batch, dim=0)  
        # # def batch_wise_step(model_output, x_t, t):
            
        # #     diffusion_step_output = scheduler.step(
        # #         model_output, t, x_t)
            
        # #     return torch.cat([diffusion_step_output.mean, diffusion_step_output.variance])
        
        # # model_output = model_output.unsqueeze(1)
        # # x_t = x_t.unsqueeze(1)
        # # diffusion_step_output = vmap(batch_wise_step)(model_output, x_t, t)

        # # mean = diffusion_step_output.mean
        # # variance = diffusion_step_output.variance

        # return mean, variance, torch.log(variance.clamp(min=1e-20))
    
    def likelihood_gradient(self, obs_dict: Dict[str, torch.Tensor], n_samples: int = 128) -> torch.Tensor:
        """
        Compute the gradient of the likelihood of the model.
        """
        
        pass
    
    def q_posterior(self, x_start, x_t, t):
        """
        Compute the posterior distribution q(x_{t-1} | x_t, x_0).
        """
        
        B, T, D = x_start.shape
        device = x_start.device
        dtype = x_start.dtype
        betas = self.noise_scheduler.betas.to(device, dtype)
        alphas = self.noise_scheduler.alphas.to(device, dtype)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.clone().detach().to(device, dtype)
        
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=x_start.device), alphas_cumprod[:-1]]) 
        posterior_mean_coef1 = (betas*torch.sqrt(alphas_cumprod_prev)) / (1-alphas_cumprod)
        posterior_mean_coef2 = (1-alphas_cumprod_prev) * torch.sqrt(alphas) / (1-alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        
        posterior_mean = (
            extract(posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        posterior_variance = extract(posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da+Do)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        nstate_pred = nsample[..., Da:Da+Do]
        state_pred = self.normalizer['obs'].unnormalize(nstate_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
            state = state_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
            state = state_pred[:,start:]
        result = {
            'action': action,
            'action_pred': action_pred,
            'state': state,
            'state_pred': state_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = torch.cat([action, obs], dim=-1)
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, trajectory], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
