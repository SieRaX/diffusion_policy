import random, functools
import numpy as np
import torch
import hydra
import wandb
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset

from conditional_gradient_estimator.model.score_model import ScoreNet
from conditional_gradient_estimator.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDatasetWrapper
from conditional_gradient_estimator.util.sde import marginal_prob_std, diffusion_coeff
from conditional_gradient_estimator.util.loss_fn import loss_fn, loss_fn_fixed_time
from conditional_gradient_estimator.sampler.sampler import BaseSampler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class ConditionalGradientEstimationWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = cfg.training.device
        
        # configure sde
        self.marginal_prob_std = functools.partial(marginal_prob_std, sigma=cfg.training.sigma, device=self.device)
        self.diffusion_coeff = functools.partial(diffusion_coeff, sigma=cfg.training.sigma, device=self.device)
        
        # configure model
        self.model: ScoreNet
        self.model = hydra.utils.instantiate(cfg.score_model, marginal_prob_std=self.marginal_prob_std)
        self.model.to(self.device)
        
        # configure train and val dataset
        dataset: RobomimicReplayLowdimDatasetWrapper
        root_dataset = hydra.utils.instantiate(cfg.task.dataset)
        dataset = hydra.utils.instantiate(cfg.wrapper_dataset, root_dataset=root_dataset)
        
        self.raw_data = {"data": list(), "condition": list()}
        for data in dataset:
            self.raw_data["data"].append(data["data"])
            self.raw_data["condition"].append(data["condition"])
            
        self.raw_data["data"] = torch.stack(self.raw_data["data"], dim=0)
        self.raw_data["condition"] = torch.stack(self.raw_data["condition"], dim=0)
            
        assert isinstance(dataset, RobomimicReplayLowdimDatasetWrapper)
        assert isinstance(root_dataset, RobomimicReplayLowdimDataset)
        self.train_dataloader = DataLoader(dataset, **cfg.dataloader)
        self.normalizer = dataset.get_normalizer()
        
        # configure optimizer and its scheduler
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.training.num_epochs*len(self.train_dataloader), eta_min=0.0)
        
        self.num_epochs = cfg.training.num_epochs
        self.sde_loss_weight = cfg.training.sde_loss_weight
        self.dsm_loss_weight = cfg.training.dsm_loss_weight
        
        self.sde_min_time = cfg.training.sde_min_time
        
        self.sampler : BaseSampler
        self.sampler = hydra.utils.instantiate(cfg.sampler)
        self.sample_every = cfg.training.sample_every
        self.max_train_steps = cfg.training.max_train_steps
        
        if cfg.training.debug:
            self.num_epochs = 2
            self.max_train_steps = 10
            self.sample_every = 1
            
        # self.cfg for saving checkpoints
        self.cfg = cfg
        
        """
        We need
        1. Model o
        2. optimizer o
        3. cosine scheduler o
        4. dataset o
        5. dataloader o
        6. epoch o
        7. ploting interval
        8. checkpoint interval
        9. save last checkpoint
        10. save last snapshot
        11. save best checkpoint
        12. save best snapshot
        13. save last checkpoint
        14. save last snapshot
        15. wandb support
        
        """

    def run(self):
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **self.cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        
        
        global_step = 0
        
        for epoch in range(self.num_epochs):
            
            step_log = dict()
            train_losses = list()
            
            with tqdm.tqdm(self.train_dataloader, desc=f"Training epoch {epoch}|{self.num_epochs}", leave=False) as tepoch:
                for batch_idx, data in enumerate(tepoch):
                    x = self.normalizer['data'].normalize(data['data']).to(self.device)
                    y = self.normalizer['condition'].normalize(data['condition']).to(self.device)
                    
                    ## get loss
                    conditional_loss = loss_fn(self.model, x, y, self.marginal_prob_std, eps=self.sde_min_time)
                    unconditional_loss = loss_fn(self.model, x, None, self.marginal_prob_std, eps=self.sde_min_time)
                    fixed_time_loss_unconditional = loss_fn_fixed_time(self.model, x, None, self.marginal_prob_std, torch.tensor(self.sde_min_time))
                    fixed_time_loss_conditional = loss_fn_fixed_time(self.model, x, y, self.marginal_prob_std, torch.tensor(self.sde_min_time))
                    
                    loss = \
                        self.sde_loss_weight * (conditional_loss + unconditional_loss) + \
                        self.dsm_loss_weight * (fixed_time_loss_conditional + fixed_time_loss_unconditional)
                    
                    ## backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    train_losses.append(loss.item())
                    
                    step_log = {
                        'train_loss': loss.item(),
                        'global_step': global_step,
                        'epoch': epoch,
                        'lr': self.scheduler.get_last_lr()[0]
                    }
                    
                    is_last_batch = (batch_idx == (len(self.train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=global_step)
                        global_step += 1
                        
                    if (self.max_train_steps is not None) \
                            and batch_idx >= (self.max_train_steps-1):
                            break
                    
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss
            
            ## evaluation ##
            # 1. sample unconditional data and compare with groudn truth
            # 2. sample conditional data and compare with ground truth
            
            if (epoch % self.sample_every) == 0:
                self.model.eval()
                with torch.no_grad():
                    # gt_data = data['data'].to(self.device)
                    # gt_data = self.train_dataloader.dataset[:].to(self.device)
                    
                    unconditional_samples = self.sampler.sample(
                        self.model,
                        None,
                        self.marginal_prob_std,
                        self.diffusion_coeff,
                        device=self.device,
                    )
                    unconditional_samples = self.normalizer['data'].unnormalize(unconditional_samples)
                    
                    idx = torch.randint(0, len(self.train_dataloader.dataset), (1,))
                    data = self.train_dataloader.dataset[idx]
                    condition = self.normalizer['condition'].normalize(data['condition']).to(self.device)
                    conditional_gt = data['data'].to(self.device)
                    sampling_condition = condition.repeat(self.sampler.batch_size, 1).to(self.device)
                    
                    conditional_samples = self.sampler.sample(
                        self.model,
                        sampling_condition,
                        self.marginal_prob_std,
                        self.diffusion_coeff,
                        device=self.device,
                    )
                    conditional_samples = self.normalizer['data'].unnormalize(conditional_samples)
                    
                    # gt_data_np = gt_data.cpu().numpy()
                    conditional_samples_np = conditional_samples.cpu().numpy()
                    unconditional_samples_np = unconditional_samples.cpu().numpy()
                    
                # Plot in wandb
                tqdm_bar = tqdm.tqdm(range(self.raw_data["data"].shape[1]-1), leave=False, desc=f"Plotting")
                max_digit = len(str(self.raw_data["data"].shape[1]-1))
                for i in tqdm_bar:
                    
                    gt_scatter_data = self.raw_data["data"][:, [i, i+1]]
                    conditional_scatter_data = conditional_samples_np[:, [i, i+1]]
                    unconditional_scatter_data = unconditional_samples_np[:, [i, i+1]]
                    conditional_gt_data = conditional_gt.cpu().numpy()[[i, i+1]]
                    
                    # Create scatter plot
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.grid(True)
                    ax.scatter(gt_scatter_data[:,0], gt_scatter_data[:,1], alpha=0.5, label='Ground Truth')
                    ax.scatter(conditional_scatter_data[:,0], conditional_scatter_data[:,1], alpha=0.5, label='Conditional')
                    ax.scatter(conditional_gt_data[0], conditional_gt_data[1], label='Condition', marker='*', color='red', s=100)
                    ax.set_xlabel(f'x_{i}')
                    ax.set_ylabel(f'x_{i+1}')
                    ax.legend()
                    
                    # Convert plot to PIL Image
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    conditional_scatter_img = Image.fromarray(data)
                    plt.close()
                    
                    step_log[f"conditional_scatter/{i}_{i+1}"] = wandb.Image(conditional_scatter_img)
                    
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.grid(True)
                    ax.scatter(gt_scatter_data[:,0], gt_scatter_data[:,1], alpha=0.5, label='Ground Truth')
                    ax.scatter(unconditional_scatter_data[:,0], unconditional_scatter_data[:,1], alpha=0.5, label='Unconditional')
                    ax.set_xlabel(f'x_{i}')
                    ax.set_ylabel(f'x_{i+1}')
                    ax.legend()
                    
                    # Convert plot to PIL Image
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    unconditional_scatter_img = Image.fromarray(data)
                    plt.close()
                    step_log[f"unconditional_scatter/{str(i).zfill(max_digit)}_{str(i+1).zfill(max_digit)}"] = wandb.Image(unconditional_scatter_img)
                
                self.model.train()
                # save checkpoint
                self.save_checkpoint()
            
            wandb_run.log(step_log, step=global_step)
            global_step += 1
            
        # save checkpoint
        self.save_checkpoint()
            