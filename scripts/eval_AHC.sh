# Eval AHC in lift
# python eval_AHC.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/debug --c_att 0.001   --n_test_vis 5 --device cuda:0

# Eval AHC in square
# python eval_AHC.py --checkpoint outputs/square_lowdim_ph_reproduction/2025.04.12_12.23.07_train_diffusion_unet_lowdim_square_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/square_lowdim_ph_reproduction/2025.04.12_12.23.07_train_diffusion_unet_lowdim_square_lowdim_16_state_estimator/eval_disturbance_horizon_AHC_distrubance_prob_1.0_vel_0.0015_gripper_dir_near_gripper_0.05_50episodes --c_att 0.03   --n_test_vis 50 --device cuda:0

# for c_att in 0.001 0.003 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045
for c_att in 0.035 0.04 0.045
do
    echo "Evaluating c_att: ${c_att}"
    python eval_AHC.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/AHC_total_evaluation/eval_AHC_c_att_${c_att}_distrubance_prob_1.0_vel_0.0017_gripper_dir_near_gripper_0.05_50episodes --c_att ${c_att}   --n_test_vis 50 --device cuda:0
done


python eval_AHC_by_seed.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/AHC_total_evaluation_by_seed/eval_AHC_seed_100000_distrubance_prob_1.0_vel_0.0017_gripper_dir_near_gripper_0.05_50episodes --seed 100000 --n_test_vis 5 --device cuda:0