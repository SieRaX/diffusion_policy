"""
Per-env wrapper that records entropy statistics of sampled action chunks.

Positioned like AttentionRecordingWrapper in
`robomimic_lowdim_ADP_jumpying_disturbance_runner_by_avg_length.py`: inner to
VideoRecordingWrapper and MultiStepWrapper (wrapper stack becomes
`MultiStep(Video(EntropyChange(Robomimic)))`). The outer eval runner samples
N candidate chunks per env on the main process, computes per-env entropy on
GPU, and pushes `(diffs, avg_entropy, h_star)` into each worker via
`env.call_each('record_entropy', args_list=...)`; gymnasium's Wrapper
`__getattr__` traverses MultiStep -> Video to reach this wrapper. An internal
step counter (incremented on each inner env step) provides `chunk_start_t`
relative to the VideoRecordingWrapper frame index.
"""
from typing import Callable, Dict, List

import gymnasium as gym
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

import pytorch3d.transforms as pt

from diffusion_policy.gym_util.multistep_wrapper import (
    MultiStepWrapper_Gymnasium,
    aggregate,
    dict_take_last_n,
)


def rot6d_to_quat_canonical(d6: torch.Tensor) -> torch.Tensor:
    """
    Sign-canonicalized unit quaternion (w, x, y, z) from 6D rotation.

    Used for entropy computation as a numerically stable alternative to
    axis-angle: no singularity at theta=pi (no antipodal flip inflating
    variance), only failure surface is the F.normalize inside
    rotation_6d_to_matrix which already has eps=1e-12 protection.
    """
    R = pt.rotation_6d_to_matrix(d6)
    q = pt.matrix_to_quaternion(R)             # (..., 4) [w, x, y, z]
    sign = torch.where(q[..., :1] < 0,
                       torch.full_like(q[..., :1], -1.0),
                       torch.full_like(q[..., :1], 1.0))
    return q * sign


def _batched_continuous_entropy(samples: torch.Tensor) -> torch.Tensor:
    """E = 0.5 * log((2 * pi * e)^d * det(Sigma)). samples: (H, N, d) -> (H,)."""
    H, N, d = samples.shape
    mean = samples.mean(dim=1, keepdim=True)
    centered = samples - mean
    centered_T = centered.transpose(1, 2)
    cov = torch.bmm(centered_T, centered) / (N - 1)
    reg = torch.eye(d, device=samples.device, dtype=samples.dtype)
    cov = cov + reg.unsqueeze(0) * 1e-8
    eigvals = torch.linalg.eigvalsh(cov)
    log_det = torch.sum(torch.log(torch.clamp(eigvals, min=1e-8)), dim=-1)
    entropy = 0.5 * (d * np.log(2.0 * np.pi * np.e) + log_det)
    return entropy


def _batched_discrete_entropy(samples: torch.Tensor) -> torch.Tensor:
    """E = -sum p(a) log p(a). samples: (H, N) -> (H,).

    Implementation note on eps: in float32 the machine epsilon at 1.0 is
    ~1.19e-7, so an eps below that makes `1.0 - eps == 1.0` and the upper
    clamp becomes a no-op. When all N samples land on the same side of
    0.5 (common at high-h on settled gripper predictions), p_close=1.0
    survives the clamp, p_open=0, and `0 * log(0)` produces NaN that
    then poisons the entropy curve via cumsum. eps=1e-6 is safely
    representable in float32.
    """
    N = samples.shape[1]
    p_close = (samples > 0.5).float().sum(dim=1) / N
    eps = 1e-6
    p_close = torch.clamp(p_close, eps, 1.0 - eps)
    p_open = 1.0 - p_close
    entropy = -(p_close * torch.log(p_close) + p_open * torch.log(p_open))
    return entropy


def compute_average_action_entropy(
    action_chunks: torch.Tensor,
    rot6d_to_rotvec: Callable[[torch.Tensor], torch.Tensor],
) -> dict:
    """
    action_chunks: (N, H, 10) — 10D = [3 pos, 6 rot6d, 1 gripper].
    rot6d_to_rotvec: (M, 6) -> (M, K). K is the dim of the rotation
        representation used for entropy (3 for axis-angle, 4 for quaternion).
    """
    N, H, D = action_chunks.shape
    assert D == 10, f"Expected action dim 10, got {D}"

    # Diagnostic: if the policy is emitting NaN/Inf at high-h on OOD obs,
    # the entropy curve will go NaN through cumsum. Surfacing it here
    # disambiguates "rotation conversion produced NaN" from "input was
    # already bad".
    finite_mask = torch.isfinite(action_chunks)
    if not finite_mask.all():
        bad_per_h = (~finite_mask).any(dim=0).any(dim=-1)  # (H,)
        bad_h = torch.where(bad_per_h)[0].tolist()
        print(f"[entropy] non-finite raw action samples at h={bad_h}")
    # Also surface extreme-but-finite magnitudes that can overflow bmm
    # inside _batched_continuous_entropy (float32 max ~3.4e38; bmm of
    # centered@centered grows quadratically, so any sample >~1e19 risks
    # Inf cov and NaN eigvalsh).
    abs_max = action_chunks.abs().max().item()
    if abs_max > 1e6:
        print(f"[entropy] extreme magnitude in action samples: abs_max={abs_max:.3e}")

    chunks = action_chunks.permute(1, 0, 2)

    trans = chunks[:, :, 0:3]
    trans_ent = _batched_continuous_entropy(trans)

    rot6d = chunks[:, :, 3:9].reshape(H * N, 6)
    rot_vec = rot6d_to_rotvec(rot6d)
    K = rot_vec.shape[-1]
    rot_vec = rot_vec.reshape(H, N, K)
    rot_ent = _batched_continuous_entropy(rot_vec)

    grip = chunks[:, :, 9]
    grip_ent = _batched_discrete_entropy(grip)

    # Per-component NaN attribution
    bad_trans = ~torch.isfinite(trans_ent)
    bad_rot = ~torch.isfinite(rot_ent)
    bad_grip = ~torch.isfinite(grip_ent)
    if bad_trans.any() or bad_rot.any() or bad_grip.any():
        msg = []
        if bad_trans.any():
            msg.append(f"trans@h={torch.where(bad_trans)[0].tolist()}")
        if bad_rot.any():
            msg.append(f"rot@h={torch.where(bad_rot)[0].tolist()}")
        if bad_grip.any():
            msg.append(f"grip@h={torch.where(bad_grip)[0].tolist()}")
        print(f"[entropy] NaN/Inf entropy component: {' | '.join(msg)}")

    total_per_step = trans_ent + rot_ent + grip_ent
    cumsum = torch.cumsum(total_per_step, dim=0)
    h_values = torch.arange(
        1, H + 1, dtype=total_per_step.dtype, device=total_per_step.device
    )
    avg_entropy = cumsum / h_values

    if H > 1:
        diffs = avg_entropy[1:] - avg_entropy[:-1]
        h_star = int(torch.argmax(diffs).item()) + 1
        h_star = max(h_star, 1)
    else:
        h_star = 1
        diffs = torch.tensor([0.0], device=total_per_step.device)

    return {
        "entropy_per_timestep": total_per_step,
        "translation_entropy": trans_ent,
        "rotation_entropy": rot_ent,
        "gripper_entropy": grip_ent,
        "average_entropy": avg_entropy,
        "diffs": diffs,
        "optimal_chunk_size": h_star,
    }


class EntropyChangeRecordingWrapper(gym.Wrapper):
    """
    Inner wrapper of a (MultiStep, Video, Entropy, Robomimic) stack. Step
    count is tracked internally since the underlying raw env does not expose
    a cumulative reward list. Each call to `step()` increments the counter by
    one, matching the frame index used by VideoRecordingWrapper.
    """

    def __init__(self, env, max_timesteps=400):
        super().__init__(env)
        self.max_timesteps = max_timesteps
        self.diffs_history: List[np.ndarray] = []
        self.avg_entropy_history: List[np.ndarray] = []
        self.entropy_per_step_history: List[np.ndarray] = []
        self.h_star_history: List[int] = []
        self.chunk_start_t: List[int] = []
        self._step_count: int = 0
        # Per-env adaptive chunk length consumed by
        # MultiStepWrapper_EntropyAdaptive on the next outer step().
        # Set fresh each iteration via record_entropy.
        self.pending_h_star: int = None

    def reset(self, **kwargs):
        self.diffs_history = []
        self.avg_entropy_history = []
        self.entropy_per_step_history = []
        self.h_star_history = []
        self.chunk_start_t = []
        self._step_count = 0
        self.pending_h_star = None
        return super().reset(**kwargs)

    def step(self, action):
        result = super().step(action)
        self._step_count += 1
        return result

    def record_entropy(
        self,
        diffs: np.ndarray,
        avg_entropy: np.ndarray,
        h_star: int,
        entropy_per_step: np.ndarray = None,
    ) -> None:
        """Called from main process via env.call_each before env.step."""
        h_star = int(h_star)
        self.diffs_history.append(np.asarray(diffs))
        self.avg_entropy_history.append(np.asarray(avg_entropy))
        self.h_star_history.append(h_star)
        self.chunk_start_t.append(int(self._step_count))
        # Hand off to MultiStepWrapper_EntropyAdaptive: the next outer
        # step() will execute exactly this many actions, then replan.
        self.pending_h_star = h_star
        if entropy_per_step is not None:
            self.entropy_per_step_history.append(np.asarray(entropy_per_step))

    def get_entropy_history(self) -> Dict[str, list]:
        return {
            "diffs": [np.asarray(x) for x in self.diffs_history],
            "avg_entropy": [np.asarray(x) for x in self.avg_entropy_history],
            "entropy_per_step": [np.asarray(x) for x in self.entropy_per_step_history],
            "h_star": list(self.h_star_history),
            "chunk_start_t": list(self.chunk_start_t),
        }

    # ------------------------------------------------------------------
    #  Render: compose env frame + 3 graph panels (avg_entropy, diffs, h*)
    # ------------------------------------------------------------------
    def render(self, mode='rgb_array', **kwargs):
        if mode != 'rgb_array':
            return super().render(mode, **kwargs)

        # current position within the latest action chunk (1-indexed h)
        if len(self.chunk_start_t) > 0:
            h_current = self._step_count - self.chunk_start_t[-1] + 1
        else:
            h_current = None

        # ---- Panel 1: avg_entropy concatenated over time ----
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        if len(self.avg_entropy_history) > 0:
            # collect all finite values for y-range
            all_finite = []
            for avg in self.avg_entropy_history:
                all_finite.extend(avg[np.isfinite(avg)].tolist())
            y_min = float(min(all_finite)) if all_finite else 0.0
            y_max = float(max(all_finite)) if all_finite else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0

            n_chunks = len(self.avg_entropy_history)
            for ci in range(n_chunks):
                avg_i = self.avg_entropy_history[ci]
                t0 = self.chunk_start_t[ci]
                x_i = np.arange(t0, t0 + len(avg_i))
                is_latest = (ci == n_chunks - 1)
                finite_m = np.isfinite(avg_i)
                nan_m = np.isnan(avg_i)
                inf_m = np.isinf(avg_i)
                avg_safe = np.where(finite_m, avg_i, np.nan)

                if is_latest:
                    ax1.plot(x_i, avg_safe, 'b-', linewidth=1.2)
                    ax1.scatter(x_i[finite_m], avg_i[finite_m],
                                color='blue', s=15)
                else:
                    ax1.plot(x_i, avg_safe, color='gray', alpha=0.4,
                             linewidth=1, linestyle='--')
                # NaN -> 'x' at y_min so it's not cropped
                if np.any(nan_m):
                    ax1.scatter(x_i[nan_m],
                                np.full(nan_m.sum(), y_min),
                                color='magenta', s=60, marker='x', zorder=5)
                # Inf -> 'o'
                if np.any(inf_m):
                    ax1.scatter(x_i[inf_m],
                                np.full(inf_m.sum(), y_min),
                                color='orange', s=60, marker='o', zorder=5)

            # red star at current timestep
            latest_avg = self.avg_entropy_history[-1]
            h_idx = self._step_count - self.chunk_start_t[-1]
            if 0 <= h_idx < len(latest_avg) and np.isfinite(latest_avg[h_idx]):
                ax1.scatter([self._step_count], [latest_avg[h_idx]],
                            color='red', s=120, marker='*', zorder=10,
                            label=f't={self._step_count}')
            # h* vertical line at latest chunk
            if len(self.h_star_history) > 0:
                h_star = self.h_star_history[-1]
                h_star_x = self.chunk_start_t[-1] + h_star - 1
                ax1.axvline(x=h_star_x, color='green', linestyle='--',
                            linewidth=1.5, label=f'h*={h_star}')
            if ax1.get_legend_handles_labels()[0]:
                ax1.legend(loc='upper right', fontsize=7)
            ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        ax1.set_xlim(0, self.max_timesteps)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Avg Entropy')
        ax1.set_title('Average Entropy')
        ax1.grid(True)
        fig1.tight_layout()
        fig1.canvas.draw()
        graph1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
        graph1 = graph1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
        g_h, g1_w = graph1.shape[:2]
        plt.close(fig1)

        # ---- Panel 2: diffs concatenated over time ----
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        if len(self.diffs_history) > 0:
            # collect all finite values for y-range
            all_finite = []
            for d in self.diffs_history:
                all_finite.extend(d[np.isfinite(d)].tolist())
            y_abs = float(max(abs(v) for v in all_finite)) if all_finite else 1.0

            n_chunks = len(self.diffs_history)
            for ci in range(n_chunks):
                diffs_i = self.diffs_history[ci]
                t0 = self.chunk_start_t[ci]
                # diffs starts at h=2, so offset by +1
                x_i = np.arange(t0 + 1, t0 + 1 + len(diffs_i))
                is_latest = (ci == n_chunks - 1)
                finite_m = np.isfinite(diffs_i)
                nan_m = np.isnan(diffs_i)
                inf_m = np.isinf(diffs_i)
                diffs_safe = np.where(finite_m, diffs_i, np.nan)

                if is_latest:
                    ax2.plot(x_i, diffs_safe, 'r-', linewidth=1.2)
                    ax2.scatter(x_i[finite_m], diffs_i[finite_m],
                                color='blue', s=15)
                else:
                    ax2.plot(x_i, diffs_safe, color='gray', alpha=0.4,
                             linewidth=1, linestyle='--')
                # NaN -> 'x'
                if np.any(nan_m):
                    ax2.scatter(x_i[nan_m],
                                np.zeros(nan_m.sum()), color='magenta',
                                s=60, marker='x', zorder=5)
                # Inf -> 'o'
                if np.any(inf_m):
                    ax2.scatter(x_i[inf_m],
                                np.zeros(inf_m.sum()), color='orange',
                                s=60, marker='o', zorder=5)

            # red star at current timestep
            latest_diffs = self.diffs_history[-1]
            h_idx = self._step_count - self.chunk_start_t[-1] - 1
            if 0 <= h_idx < len(latest_diffs) and np.isfinite(latest_diffs[h_idx]):
                ax2.scatter([self._step_count], [latest_diffs[h_idx]],
                            color='red', s=120, marker='*', zorder=10,
                            label=f't={self._step_count}')
            # h* vertical line
            if len(self.h_star_history) > 0:
                h_star = self.h_star_history[-1]
                h_star_x = self.chunk_start_t[-1] + h_star
                ax2.axvline(x=h_star_x, color='green', linestyle='--',
                            linewidth=1.5, label=f'h*={h_star}')
            if ax2.get_legend_handles_labels()[0]:
                ax2.legend(loc='upper right', fontsize=7)
            ax2.set_ylim(-1.1 * y_abs, 1.1 * y_abs)
        ax2.set_xlim(0, self.max_timesteps)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Δ Avg Entropy')
        ax2.set_title('Diffs (ΔAvgEntropy)')
        ax2.grid(True)
        fig2.tight_layout()
        fig2.canvas.draw()
        graph2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        graph2 = graph2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        _, g2_w = graph2.shape[:2]
        plt.close(fig2)

        # ---- Panel 3: h* over time ----
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        if len(self.h_star_history) > 0:
            ax3.step(self.chunk_start_t, self.h_star_history,
                     'g-', where='post', linewidth=1)
            ax3.scatter(self.chunk_start_t, self.h_star_history,
                        color='green', s=30, zorder=5)
            # red star at current timestep
            ax3.scatter([self._step_count], [self.h_star_history[-1]],
                        color='red', s=120, marker='*', zorder=10,
                        label=f't={self._step_count}')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.set_ylim(0, max(self.h_star_history) + 2)
        ax3.set_xlim(0, self.max_timesteps)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('h*')
        ax3.set_title('Optimal Chunk Size (h*)')
        ax3.grid(True)
        fig3.tight_layout()
        fig3.canvas.draw()
        graph3 = np.frombuffer(fig3.canvas.tostring_rgb(), dtype=np.uint8)
        graph3 = graph3.reshape(fig3.canvas.get_width_height()[::-1] + (3,))
        _, g3_w = graph3.shape[:2]
        plt.close(fig3)

        # ---- Panel 4: per-step entropy (non-cumulated) over time ----
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        if len(self.entropy_per_step_history) > 0:
            all_finite = []
            for eps in self.entropy_per_step_history:
                all_finite.extend(eps[np.isfinite(eps)].tolist())
            y_min4 = float(min(all_finite)) if all_finite else 0.0
            y_max4 = float(max(all_finite)) if all_finite else 1.0
            y_range4 = y_max4 - y_min4 if y_max4 != y_min4 else 1.0

            n_ch = len(self.entropy_per_step_history)
            for ci in range(n_ch):
                eps_i = self.entropy_per_step_history[ci]
                t0 = self.chunk_start_t[ci]
                x_i = np.arange(t0, t0 + len(eps_i))
                is_latest = (ci == n_ch - 1)
                finite_m = np.isfinite(eps_i)
                eps_safe = np.where(finite_m, eps_i, np.nan)
                if is_latest:
                    ax4.plot(x_i, eps_safe, 'm-', linewidth=1.2)
                    ax4.scatter(x_i[finite_m], eps_i[finite_m],
                                color='purple', s=15)
                else:
                    ax4.plot(x_i, eps_safe, color='gray', alpha=0.4,
                             linewidth=1, linestyle='--')
            # red star at current timestep
            latest_eps = self.entropy_per_step_history[-1]
            h_idx = self._step_count - self.chunk_start_t[-1]
            if 0 <= h_idx < len(latest_eps) and np.isfinite(latest_eps[h_idx]):
                ax4.scatter([self._step_count], [latest_eps[h_idx]],
                            color='red', s=120, marker='*', zorder=10,
                            label=f't={self._step_count}')
            if ax4.get_legend_handles_labels()[0]:
                ax4.legend(loc='upper right', fontsize=7)
            ax4.set_ylim(y_min4 - 0.1 * y_range4, y_max4 + 0.1 * y_range4)
        ax4.set_xlim(0, self.max_timesteps)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Per-Step Entropy (raw)')
        ax4.grid(True)
        fig4.tight_layout()
        fig4.canvas.draw()
        graph4 = np.frombuffer(fig4.canvas.tostring_rgb(), dtype=np.uint8)
        graph4 = graph4.reshape(fig4.canvas.get_width_height()[::-1] + (3,))
        _, g4_w = graph4.shape[:2]
        plt.close(fig4)

        # ---- Compose: env frame | graph1 | graph2 | graph3 | graph4 ----
        frame = self.env.render(mode, **kwargs)
        frame_h, frame_w = frame.shape[:2]
        aspect = frame_w / frame_h
        new_h = g_h
        new_w = int(new_h * aspect)
        frame_resized = np.array(
            Image.fromarray(frame).resize((new_w, new_h)))

        total_w = new_w + g1_w + g2_w + g3_w + g4_w
        canvas = np.zeros(
            (g_h + int(g_h * 0.2), total_w, 3), dtype=np.uint8)
        canvas[:g_h, :new_w] = frame_resized
        canvas[:g_h, new_w:new_w + g1_w] = graph1
        canvas[:g_h, new_w + g1_w:new_w + g1_w + g2_w] = graph2
        canvas[:g_h, new_w + g1_w + g2_w:new_w + g1_w + g2_w + g3_w] = graph3
        canvas[:g_h, new_w + g1_w + g2_w + g3_w:] = graph4
        return canvas

    def get_attr(self, name):
        return getattr(self, name)

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None


class MultiStepWrapper_EntropyAdaptive(MultiStepWrapper_Gymnasium):
    """
    AAC-style adaptive chunking wrapper.

    The outer runner passes a full-horizon action tensor of shape
    (n_action_steps, action_dim). Each env's
    EntropyChangeRecordingWrapper has set `pending_h_star` (via
    `record_entropy`) to that env's h*. We iterate only the first h*
    actions, then return — letting the runner replan.

    Falls back to the parent's behavior (execute the entire passed
    action) when `pending_h_star` is unset, e.g. on the very first
    step before any record_entropy call has fired.
    """

    def step(self, action):
        # MultiStep -> Video -> EntropyChange
        ent_wrapper = self.env.env

        # If this env has already terminated in a previous outer step,
        # return cached state without applying any inner actions. Mirrors
        # AsyncVectorEnv's "frozen after done" semantics and prevents the
        # underlying sim from being stepped post-termination.
        if len(self.done) > 0 and self.done[-1]:
            ent_wrapper.pending_h_star = None
            observation = self._get_obs(self.n_obs_steps)
            reward = aggregate(self.reward, self.reward_agg_method)
            terminated = aggregate(self.terminated, 'max')
            truncated = aggregate(self.truncated, 'max')
            info = dict_take_last_n(self.info, self.n_obs_steps)
            return observation, reward, terminated, truncated, info

        h_star = getattr(ent_wrapper, 'pending_h_star', None)
        n_total = len(action)
        if h_star is None:
            n_to_execute = n_total
        else:
            n_to_execute = max(8, min(int(h_star), n_total))
            if n_to_execute < 16:
                print(
                    f"Warning: h_star={h_star} leads to very short chunk length "
                    f"{n_to_execute}. Consider setting a higher minimum chunk "
                    f"length or investigating the entropy curve."
                )

        self.env.action_seq = action
        for i, act in enumerate(action):
            if i >= n_to_execute:
                break
            if len(self.done) > 0 and self.done[-1]:
                break
            observation, reward, terminated, truncated, info = \
                gym.Wrapper.step(self, act)
            done = terminated or truncated

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                    and (len(self.reward) >= self.max_episode_steps):
                done = True
                terminated = True
            self.done.append(done)
            self.terminated.append(terminated)
            self.truncated.append(truncated)
            self._add_info(info)

        # Consume the h* so a missed record_entropy in the next outer
        # iteration falls back to full-horizon rather than reusing
        # stale h*.
        ent_wrapper.pending_h_star = None

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        terminated = aggregate(self.terminated, 'max')
        truncated = aggregate(self.truncated, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, terminated, truncated, info
