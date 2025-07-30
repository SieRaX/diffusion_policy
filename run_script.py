import os
import re

# Path to the target directory
path = "outputs/hammer_lowdim_human-v3_reproduction/train_by_seed"

# Regex to extract score from filename like 'epoch=1000-test_mean_score=0.560.ckpt'
score_pattern = re.compile(r"test_mean_score=([\d\.eE+-]+)\.ckpt")

for entry in os.listdir(path):
    full_path = os.path.join(path, entry)
    if os.path.isdir(full_path):
        checkpoints_dir = os.path.join(full_path, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            continue
        checkpoint_list = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt') and 'test_mean_score=' in f]
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
        command = f"python eval_AHC_attention_estimator_by_length_cge_hydra_version.py --config-name=eval_config_d4rl.yaml checkpoint=\'{os.path.join(full_path, 'checkpoints', max_file)}\' normalizer_dir=outputs/cge/hammer_lowdim_human-v3/2025.07.09_23.26.58_train_conditional_gradient_lowdim_hammer_lowdim/normalizer.pth attention_estimator_dir=outputs/cge/hammer_lowdim_human-v3/2025.07.09_23.26.58_train_conditional_gradient_lowdim_hammer_lowdim/seq2seq_attention_estimator.pth output_dir={os.path.join(full_path, eval_uniform_dir, '16')} device=cuda:0 env_runner.n_action_steps=16 ~env_runner.max_steps env_runner.max_attention=2.5 env_runner.min_n_action_steps=4 env_runner.attention_exponent=1.0 env_runner.n_test=100 env_runner.n_test_vis=100 env_runner.uniform_horizon={false_true}"
        
        os.system(command)
