import os
import re
import numpy as np
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json

@click.command()
@click.option('-p', '--path', required=True)
@click.option('-a', '--attention_estimator_seed_dir', required=True)
@click.option('-s', '--sub_dir')
@click.option('-d', '--device', default='cuda:0')
@click.option('-ci', '--c_init', default=1.0, type=float)
@click.option('-di', '--d_init', default=1.0, type=float)
@click.option('-m', '--max_steps', type=int)
def main(path, attention_estimator_seed_dir, sub_dir="", device='cuda:0', c_init=1.0, d_init=1.0, max_steps=100):
    '''
    From given path e.g. "outputs/hammer_lowdim_human-v3_reproduction/train_by_seed"
    find all the best checkpoints and run eval.py for each checkpoint
    '''
    # Regex to extract score from filename like 'epoch=1000-test_mean_score=0.560.ckpt'
    score_pattern = re.compile(r"test_mean_score=([\d\.eE+-]+)\.ckpt")

    seed_number=0
    
    json_log = dict()
    mean_scores_list = list()
    for seed_number in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # Find all directories starting with f"seed_{seed_number}" and get the last one
        seed_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith(f"seed_{seed_number}")]
        if not seed_dirs:
            print(f"\033[91mNo directories found starting with seed_{seed_number}\033[0m")
            continue
        entry = sorted(seed_dirs)[-1]  # Get the last directory alphabetically
        
        full_path = os.path.join(path, entry)
        
        att_seed_dirs = [d for d in os.listdir(attention_estimator_seed_dir) if os.path.isdir(os.path.join(attention_estimator_seed_dir, d)) and d.startswith(f"seed_{seed_number}")]
        if not att_seed_dirs:
            print(f"\033[91mNo directories found starting with seed_{seed_number} in {attention_estimator_seed_dir}\033[0m")
            continue
        att_entry = sorted(att_seed_dirs)[-1]
        attention_estimator_dir = os.path.join(attention_estimator_seed_dir, att_entry, sub_dir, "seq2seq_attention_estimator.pth")
        normalizer_dir = os.path.join(attention_estimator_seed_dir, att_entry, sub_dir, "normalizer.pth")
        
        if os.path.isdir(full_path):
            checkpoints_dir = os.path.join(full_path, "checkpoints")
            if not os.path.isdir(checkpoints_dir):
                continue
            checkpoint_list = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt') and 'test_mean_score=' in f])
            
            
            max_score = None
            max_file = None
            for fname in checkpoint_list:
                match = score_pattern.search(fname)
                if match:
                    try:
                        score = float(match.group(1))
                        if max_score is None or score > max_score:
                            max_score = score
                            max_file = fname
                    except ValueError:
                        continue
            print(f"\033[92m{entry}: {max_file} (score={max_score})\033[0m")
            max_file = max_file.replace("=", "\=")
            
            eval_uniform = False
            
            if eval_uniform:
                eval_uniform_dir="eval_uniform_by_length"
                false_true="true"
            else:
                eval_uniform_dir="eval_ADP_by_length"
                false_true="false"

            output_dir = os.path.join(full_path, eval_uniform_dir, '16')
            command = f"python eval.py --checkpoint=\'{os.path.join(full_path, 'checkpoints', max_file)}\'  --output_dir={output_dir} --device=cuda:0 --n_action_steps=16"
            
            command = f"python eval_AHC_attention_estimator_by_length_cge_hydra_version.py \
                --config-name=eval_config.yaml\
                checkpoint=\'{os.path.join(full_path, 'checkpoints', max_file)}\'\
                attention_estimator_dir=\'{attention_estimator_dir}\'\
                normalizer_dir=\'{normalizer_dir}\'\
                output_dir=\'{output_dir}\'\
                device={device}\
                env_runner.n_action_steps=16\
                env_runner.max_steps={max_steps}\
                env_runner.max_attention=15.0\
                env_runner.min_n_action_steps=2 env_runner.attention_exponent=1.0 env_runner.n_test=100 env_runner.n_test_vis=100 env_runner.uniform_horizon=false \
                init_catt={c_init} init_dcatt={d_init}\
            "
                        
            os.system(command)

            # take the result of the eval and save it to a json file
            eval_log = json.load(open(os.path.join(output_dir, 'eval_log.json')))
            mean_score = eval_log['test/mean_score']
            json_log[output_dir] = mean_score
            mean_scores_list.append(mean_score)

    json_log['mean_scores_mean'] = np.mean(mean_scores_list)
    json_log['mean_scores_std'] = np.std(mean_scores_list)
    json_log['mean_scores_min'] = np.min(mean_scores_list)
    json_log['mean_scores_max'] = np.max(mean_scores_list)
    json_log['command'] = ' '.join(sys.argv)
    json.dump(json_log, open(os.path.join(path, 'ADP_total_eval_statistics.json'), 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()