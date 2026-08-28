from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.moh_conditional_unet1d import MoHConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class MoHDiffusionUnetHybridImagePolicy(BaseImagePolicy):
    """
    Mixture-of-Horizons on the hybrid (vision + robot state) Diffusion Policy.

    Same MoH machinery as MoHDiffusionUnetLowdimPolicy (per-horizon truncated
    UNet forwards, gate on per-step UNet features, balance loss, DDIM-space
    consensus for dynamic inference). The robomimic ObservationEncoder is
    shared: it runs ONCE per predict/loss call and all horizon branches reuse
    its global conditioning, so the MoH overhead here is only the UNet passes.
    """
    def __init__(self,
            shape_meta: dict,
            noise_scheduler,
            horizon,
            horizons: List[int],
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # MoH training (paper Eq. 17)
            lambda_ind=1.0,
            lambda_bal=1.0e-3,
            use_gate_noise=True,
            # MoH dynamic inference (paper Alg. 1)
            min_replan_steps=4,
            min_active_horizons=2,
            scale_ratio=1.1,
            downsample_factor=4,
            # parameters passed to scheduler.step
            **kwargs):
        super().__init__()
        assert obs_as_global_cond, \
            "MoH baseline only supports obs_as_global_cond=True"
        horizons = sorted(int(h) for h in horizons)
        assert horizons[-1] == horizon
        assert all(h % downsample_factor == 0 for h in horizons), \
            f"all horizons must be multiples of {downsample_factor}, got {horizons}"
        assert len(set(horizons)) == len(horizons)

        # ---- obs encoder (identical to DiffusionUnetHybridImagePolicy) ----
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        config = get_robomimic_config(
            algo_name='bc_rnn', hdf5_type='image',
            task_name='square', dataset_type='ph')
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name, config=config,
            obs_key_shapes=obs_key_shapes, ac_dim=action_dim, device='cpu')
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features))
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc))

        obs_feature_dim = obs_encoder.output_shape()[0]
        model = MoHConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim, obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True, action_visible=False)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.horizons = horizons
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.oa_step_convention = True  # hybrid policy uses start = To-1
        self.lambda_ind = lambda_ind
        self.lambda_bal = lambda_bal
        self.use_gate_noise = use_gate_noise
        self.min_replan_steps = min_replan_steps
        self.min_active_horizons = min_active_horizons
        self.scale_ratio = scale_ratio
        self.kwargs = kwargs

        self.gate_head = nn.Linear(model.feature_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.gate_noise_head = nn.Linear(model.feature_dim, 1)
        nn.init.zeros_(self.gate_noise_head.weight)
        nn.init.zeros_(self.gate_noise_head.bias)

        step_idx = torch.arange(horizon).unsqueeze(1)
        horizon_t = torch.tensor(horizons).unsqueeze(0)
        self.register_buffer('valid_mask', step_idx < horizon_t, persistent=False)
        self.register_buffer('n_active', self.valid_mask.sum(dim=-1), persistent=False)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.last_loss_info = dict()

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # ========= MoH internals (same as MoHDiffusionUnetLowdimPolicy) ============
    def _forward_all_horizons(self, trajectory, timesteps, global_cond):
        preds, feats = [], []
        for h in self.horizons:
            pred, feat = self.model(
                trajectory[:, :h], timesteps,
                global_cond=global_cond, return_features=True)
            preds.append(pred)
            feats.append(feat)
        return preds, feats

    def _gate_weights(self, feats, training):
        B = feats[0].shape[0]
        T = self.horizon
        N = len(self.horizons)
        device = feats[0].device
        dtype = feats[0].dtype
        neg_inf = torch.finfo(dtype).min
        logits = torch.full((B, T, N), neg_inf, device=device, dtype=dtype)
        for i, h in enumerate(self.horizons):
            g = self.gate_head(feats[i]).squeeze(-1)
            if training and self.use_gate_noise:
                std = F.softplus(self.gate_noise_head(feats[i]).squeeze(-1))
                g = g + torch.randn_like(g) * std
            logits[:, :h, i] = g
        return torch.softmax(logits, dim=-1)

    def _fuse(self, preds, alpha):
        B = preds[0].shape[0]
        T = self.horizon
        Da = preds[0].shape[-1]
        stacked = torch.zeros(
            (B, T, len(self.horizons), Da),
            device=preds[0].device, dtype=preds[0].dtype)
        for i, h in enumerate(self.horizons):
            stacked[:, :h, i] = preds[i]
        return (alpha.unsqueeze(-1) * stacked).sum(dim=2)

    def _balance_loss(self, alpha):
        eps = 1e-8
        boundaries = [0] + self.horizons
        cv2_list = []
        for i in range(len(self.horizons)):
            lo, hi = boundaries[i], boundaries[i + 1]
            active = alpha[:, lo:hi, i:]
            if active.shape[-1] <= 1 or active.shape[1] == 0:
                continue
            avg_usage = active.mean(dim=(0, 1))
            cv2 = avg_usage.var(unbiased=False) / (avg_usage.mean() ** 2 + eps)
            cv2_list.append(cv2)
        if len(cv2_list) == 0:
            return alpha.new_zeros(())
        return torch.stack(cv2_list).mean()

    def _ddim_prev_eps_coeff(self, t):
        scheduler = self.noise_scheduler
        alphas_cumprod = scheduler.alphas_cumprod
        t = int(t)
        step_ratio = scheduler.config.num_train_timesteps // self.num_inference_steps
        prev_t = t - step_ratio
        a_t = alphas_cumprod[t]
        if prev_t >= 0:
            a_prev = alphas_cumprod[prev_t]
        else:
            a_prev = getattr(scheduler, 'final_alpha_cumprod', torch.ones_like(a_t))
        c2 = (1 - a_prev).sqrt() - (a_prev * (1 - a_t) / a_t).sqrt()
        return c2.abs().item()

    # ========= inference ============
    def conditional_sample(self, cond_shape, global_cond,
            generator=None, accumulate_disagreement=False, **kwargs):
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=cond_shape, dtype=self.dtype, device=self.device,
            generator=generator)
        scheduler.set_timesteps(self.num_inference_steps)

        disagreement = None
        if accumulate_disagreement:
            disagreement = torch.zeros(
                cond_shape[:2], device=self.device, dtype=self.dtype)

        for t in scheduler.timesteps:
            preds, feats = self._forward_all_horizons(trajectory, t, global_cond)
            alpha = self._gate_weights(feats, training=False)
            fused = self._fuse(preds, alpha)
            if accumulate_disagreement:
                c2 = self._ddim_prev_eps_coeff(t)
                for i, h in enumerate(self.horizons):
                    diff = (preds[i] - fused[:, :h]).abs().sum(dim=-1)
                    disagreement[:, :h] += c2 * alpha[:, :h, i] * diff
            trajectory = scheduler.step(
                fused, t, trajectory, generator=generator, **kwargs).prev_sample
        return trajectory, disagreement

    def _consensus_exec_steps(self, disagreement, start):
        B, T = disagreement.shape
        n = self.min_replan_steps
        m = self.min_active_horizons
        max_exec = T - start
        if max_exec <= n:
            return torch.full((B,), max_exec,
                dtype=torch.long, device=disagreement.device)
        d = disagreement[:, start:]
        thres = d[:, :n].mean(dim=1) * self.scale_ratio
        active = self.n_active[start + n: start + max_exec]
        ok = (d[:, n:] <= thres.unsqueeze(1)) & (active.unsqueeze(0) > m)
        return n + ok.long().cumprod(dim=1).sum(dim=1)

    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
            use_dynamic=False) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict  # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        nsample, disagreement = self.conditional_sample(
            (B, T, Da), global_cond,
            accumulate_disagreement=use_dynamic,
            **self.kwargs)

        action_pred = self.normalizer['action'].unnormalize(nsample)
        start = To - 1

        result = {'action_pred': action_pred}
        if use_dynamic:
            n_exec = self._consensus_exec_steps(disagreement, start)
            n_exec_common = int(n_exec.min().item())
            result['action'] = action_pred[:, start:start + n_exec_common]
            result['replan_steps'] = n_exec
            result['disagreement'] = disagreement
        else:
            end = start + self.n_action_steps
            result['action'] = action_pred[:, start:end]
        return result

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_extra_losses(self):
        # picked up by the hybrid workspace's step_log
        return {f'train_{k}': v for k, v in self.last_loss_info.items()}

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        trajectory = nactions
        assert trajectory.shape[1] == self.horizon

        # shared obs encoding, one forward for all horizon branches
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        preds, feats = self._forward_all_horizons(
            noisy_trajectory, timesteps, global_cond)

        loss_ind = trajectory.new_zeros(())
        for i, h in enumerate(self.horizons):
            loss_h = F.mse_loss(preds[i], target[:, :h], reduction='none')
            loss_ind = loss_ind + reduce(loss_h, 'b ... -> b (...)', 'mean').mean()

        alpha = self._gate_weights(feats, training=self.training)
        fused = self._fuse(preds, alpha)
        loss_mix = F.mse_loss(fused, target, reduction='none')
        loss_mix = reduce(loss_mix, 'b ... -> b (...)', 'mean').mean()

        loss_bal = self._balance_loss(alpha)

        loss = loss_mix + self.lambda_ind * loss_ind + self.lambda_bal * loss_bal
        self.last_loss_info = {
            'loss_mix': loss_mix.item(),
            'loss_ind': loss_ind.item(),
            'loss_bal': loss_bal.item(),
        }
        return loss
