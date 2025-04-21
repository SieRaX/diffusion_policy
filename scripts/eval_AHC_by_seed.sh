for seed in {100025..100049}
do
    echo "Evaluating seed: ${seed}"
    python eval_AHC_by_seed.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/AHC_total_evaluation_by_seed_jumping_disturbance/eval_AHC_seed_${seed}_jumping_disturbance_num_distrub_1_pos_0.05_gripper_dir_near_gripper_0.025 --seed ${seed} --n_test_vis 5 --device cuda:1 --env_runner diffusion_policy.env_runner.robomimic_lowdim_AHC_jumping_disturbance_runner_by_seed.RobomimicLowdimAHCRunner
done

# for seed in 100000 100001 100002 100008 100009 100010 100011
# do
#     echo "Evaluating seed: ${seed}"
#     python eval_AHC_by_seed.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/AHC_total_evaluation_by_seed_20250419/eval_AHC_seed_${seed}_distrubance_prob_1.0_vel_0.0017_gripper_dir_near_gripper_0.05_50episodes --seed ${seed} --n_test_vis 5 --device cuda:1
# done