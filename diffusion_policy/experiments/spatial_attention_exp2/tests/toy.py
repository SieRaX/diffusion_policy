"""Toy deterministic policies for estimator tests (no GPU / robosuite)."""
import torch


class _Identity:
    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class DummyVelPolicy:
    """Velocity depends only on the observation (global_cond): v = ones * mean(global_cond).
    So ||v(o') - v(o)||^2 = H * Da * (mean(gc') - mean(gc))^2 — analytically checkable."""
    def __init__(self, H=4, Da=3, To=2, Do=5):
        self.horizon = H
        self.action_dim = Da
        self.n_obs_steps = To
        self.time_scale = 1000.0
        self.oa_step_convention = True
        self.num_inference_steps = 16
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self._init_noise = None
        self.normalizer = {'action': _Identity(), 'obs': _Identity()}

    def eval(self):
        return self

    def fm_global_cond(self, obs_dict):
        o = obs_dict['obs']
        return o[:, :self.n_obs_steps].reshape(o.shape[0], -1)

    def model(self, x, t, local_cond=None, global_cond=None):
        B, H, Da = x.shape
        g = global_cond.mean(dim=1).view(B, 1, 1)
        return torch.ones(B, H, Da) * g


class DummyEndpointPolicy:
    """Copy of the prelim's DummyPerturbPolicy: norm = eps0 + mean(obs); raw = 2*norm + 1."""
    def __init__(self, H=4, D=3, To=2):
        self.action_dim = D
        self.horizon = H
        self.n_obs_steps = To
        self.oa_step_convention = True
        self.num_inference_steps = 99
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self._init_noise = None

    def eval(self):
        return self

    def predict_action(self, obs_dict):
        eps = self._init_noise
        g = torch.stack([v.float().mean() for v in obs_dict.values()]).mean()
        norm = eps + g
        raw = norm * 2.0 + 1.0
        start = self.n_obs_steps - 1
        return {'naction_pred': norm, 'action_pred': raw,
                'naction': norm[:, start:start + 1], 'action': raw[:, start:start + 1]}
