#!/usr/bin/env bash
for seed in {100000..100199}
do
    echo -e "\033[32m[Evaluating seed: ${seed}]\033[0m"
    python eval_AHC_attention_estimator_by_seed.py --checkpoint outputs/can_ph_lowdim_reproduction/2025.04.24_18.13.14_train_diffusion_unet_lowdim_can_lowdim_cnn_32/checkpoints/epoch=0200-test_mean_score=1.000.ckpt --output_dir outputs/can_ph_lowdim_reproduction/2025.04.24_18.13.14_train_diffusion_unet_lowdim_can_lowdim_cnn_32/eval_AHC_disturbance_by_seed/eval_AHC_seed_${seed}_jumping_disturbance_num_distrub_1_pos_0.03_gripper_dir_near_gripper_0.025 --seed ${seed} --n_test_vis 5 --device cuda:1 --env_runner diffusion_policy.env_runner.robomimic_lowdim_AHC_jumping_disturbance_attention_estimator_runner_by_seed.RobomimicLowdimAHCRunner --max_steps 150
done

# --checkpoint data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/checkpoints/epoch=0200-test_mean_score=1.000.ckpt --output_dir data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/eval_AHC_seed_100000_jumping_disturbance_num_distrub_1_pos_0.03_gripper_dir_near_gripper_0.025 --seed 100000 --n_test_vis 5 --device cuda:0 --env_runner diffusion_policy.env_runner.robomimic_lowdim_AHC_jumping_disturbance_attention_estimator_runner_by_seed.RobomimicLowdimAHCRunner