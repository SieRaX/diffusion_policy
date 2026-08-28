"""
Train a seq2seq spatial-attention estimator (SA paper Eq. 19) from an
SA-labeled lowdim dataset (low_dim_abs_with_attention.hdf5, produced by
gather_spatial_attention_CGE.py).

Modernized version of train_seq2seq_attention_transformer.py: horizon,
obs_keys and output dir are CLI options (the original hardcoded horizon=32 and
single-arm obs keys), visualization stripped. Saves
seq2seq_attention_estimator.pth + normalizer.pth + command.json, the exact
artifact layout the +SA evaluation (scripts/eval_sa_sweep.py) consumes.

Usage (transport, horizon 48):
  python scripts/train_sa_estimator.py \
      -p data/sa_artifacts/transport/with_attention_normalized.hdf5 \
      -o data/sa_artifacts/transport/attention_estimator_horizon_48 \
      --horizon 48 --dual_arm
"""
import sys, os, json, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.model.transformer import Seq2SeqTransformer

SINGLE_ARM_KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
DUAL_ARM_KEYS = SINGLE_ARM_KEYS + ['robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos']


@click.command()
@click.option('-p', '--dataset_path', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('--horizon', default=48)
@click.option('--dual_arm', is_flag=True, default=False)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_epoch', default=200)
@click.option('--batch_size', default=256)
@click.option('--lr', default=1e-3)
@click.option('--n_obs_steps', default=2)
def main(dataset_path, output_dir, horizon, dual_arm, device, n_epoch,
         batch_size, lr, n_obs_steps):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs_keys = DUAL_ARM_KEYS if dual_arm else SINGLE_ARM_KEYS

    dataset = RobomimicReplayLowdimDataset(
        dataset_path, horizon=horizon, pad_before=n_obs_steps - 1,
        pad_after=horizon - 1, obs_keys=obs_keys, abs_action=True,
        rotation_rep='rotation_6d', use_legacy_normalizer=False, seed=42,
        val_ratio=0.02, max_train_episodes=None,
        info_keys=['spatial_attention'])
    val_dataset = dataset.get_validation_dataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    normalizer = dataset.get_normalizer()

    device = torch.device(device)
    sample = next(iter(train_loader))
    obs_dim = sample['obs'].shape[-1] * n_obs_steps
    action_dim = sample['action'].shape[-1]
    seq_len = sample['action'].shape[1]
    assert seq_len == horizon
    print(f"obs_dim {obs_dim} action_dim {action_dim} seq_len {seq_len} "
          f"train {len(dataset)} val {len(val_dataset)}")

    model = Seq2SeqTransformer(
        obs_dim=obs_dim, action_dim=action_dim, seq_len=seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epoch * len(train_loader), eta_min=1e-6)

    def batch_io(batch):
        B = batch['obs'].shape[0]
        nobs = normalizer['obs'].normalize(batch['obs'])
        naction = normalizer['action'].normalize(batch['action'])
        natt = normalizer['spatial_attention'].normalize(
            batch['spatial_attention'].unsqueeze(-1))
        return (nobs[:, :n_obs_steps, :].reshape(B, -1).to(device),
                naction.to(device), natt.to(device))

    best_val, best_state = float('inf'), None
    for epoch in range(n_epoch):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            obs, action_seq, attention = batch_io(batch)
            loss = criterion(model(obs, action_seq), attention)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                obs, action_seq, attention = batch_io(batch)
                val_loss += criterion(model(obs, action_seq), attention).item()
                n += 1
        val_loss /= max(n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == n_epoch - 1:
            print(f"epoch {epoch:3d} train {loss.item():.5f} "
                  f"val {val_loss:.5f} best {best_val:.5f}")

    torch.save(best_state, out / 'seq2seq_attention_estimator.pth')
    torch.save(normalizer.state_dict(), out / 'normalizer.pth')
    json.dump({'command': ' '.join(sys.argv), 'best_val_loss': best_val,
               'obs_keys': obs_keys, 'horizon': horizon},
              open(out / 'command.json', 'w'), indent=2)
    print(f"saved to {out} (best val {best_val:.5f})")


if __name__ == '__main__':
    main()
