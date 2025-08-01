#!/usr/bin/env bash
# python eval_likelihood.py --checkpoint data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/checkpoints/epoch=0650-test_mean_score=0.894.ckpt --output_dir data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/total_likelihood_diff_0.01_mean   --n_test_vis 50 --device cuda:0

# # Eval likelihood in Lift task
# python eval_likelihood.py --checkpoint outputs/square_lowdim_ph_reproduction/2025.04.12_12.23.07_train_diffusion_unet_lowdim_square_lowdim_16_state_estimator/checkpoints/epoch=0650-test_mean_score=1.000.ckpt --output_dir outputs/square_lowdim_ph_reproduction/2025.04.12_12.23.07_train_diffusion_unet_lowdim_square_lowdim_16_state_estimator/debug --n_action_steps 14 --n_test_vis 50 --device cuda:1

# for horizon in {14..1}
# do
#     echo -e "\033[32mEvaluating horizon: ${horizon}\033[0m"
#     python eval_likelihood.py --checkpoint data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/checkpoints/epoch=0200-test_mean_score=1.000.ckpt --output_dir data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/uniform_horizon_jumpying_disturbance_2/eval_jumping_disturbance_horizon_${horizon}_num_distrub_1_pos_0.03_gripper_dir_near_gripper_0.025_50episodes  --n_action_steps ${horizon} --n_test_vis 100 --device cuda:0 --env_runner diffusion_policy.env_runner.robomimic_lowdim_likelihood_jummping_disturbance_runner.RobomimicLowdimLikelihoodDisturbanceRunner
# done
# python eval_likelihood.py --checkpoint data/outputs/square_reproduction/2025.02.28/07.43.04_train_diffusion_unet_lowdim_square_lowdim/checkpoints/epoch=0150-test_mean_score=0.960.ckpt --output_dir data/outputs/square_reproduction/2025.02.28/07.43.04_train_diffusion_unet_lowdim_square_lowdim/ddeubg --n_action_steps 8 --n_test_vis 50 --device cuda:0 --env_runner diffusion_policy.env_runner.robomimic_lowdim_likelihood_runner.RobomimicLowdimLikelihoodRunner


# # Eval likelihood in Lift task
# python eval_likelihood.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/eval_disturbance_measure_pick_position_and_time_prob_1.0_vel_0.0017 --n_action_steps 14 --n_test_vis 20 --device cuda:0

# # Eval likelihood in Square task
# python eval_likelihood.py --checkpoint outputs/square_lowdim_ph_reproduction/2025.04.12_12.23.07_train_diffusion_unet_lowdim_square_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/square_lowdim_ph_reproduction/2025.04.12_12.23.07_train_diffusion_unet_lowdim_square_lowdim_16_state_estimator/eval_disturbance_horizon_14_distrubance_prob_1.0_vel_0.0007_gripper_dir_near_gripper_0.10_minus_gripper_direction_50episodes --max_steps 300 --n_action_steps 14 --n_test_vis 10 --device cuda:0

# --checkpoint data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/checkpoints/epoch=0650-test_mean_score=0.894.ckpt --output_dir data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/debug  --n_test_vis 50 --device cuda:0


# # Loop through different action steps
# for n_steps in {6..14}
# do
#     echo "Evaluating action steps: ${n_steps}"
#     python eval_likelihood.py \
#         --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt \
#         --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/uniform_horizon_jumping_disturbance/eval_jumping_disturbance_horizon_${n_steps}_num_distrub_1_pos_0.05_gripper_dir_near_gripper_0.025_200episodes \
#         --n_action_steps ${n_steps} \
#         --n_test_vis 50 \
#         --device cuda:1 \
#         --env_runner diffusion_policy.env_runner.robomimic_lowdim_likelihood_jummping_disturbance_runner.RobomimicLowdimLikelihoodDisturbanceRunner
# done

# eval_likelihood.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/eval_disturbance_horizon_3_disturbance_prob_1.0_vel_0.0015_gripper_dir_near_gripper_0.05_100episode --n_action_steps 3 --n_test_vis 100 --device cuda:0 --env_runner diffusion_policy.env_runner.2025_04_20_robomimic_lowdim_likelihood_disturbance_runner_better_performance_on_horizon3.RobomimicLowdimLikelihoodDisturbanceRunner

# eval_likelihood.py --checkpoint outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/checkpoints/epoch=0400-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_ph_reproduction/2025.04.12_11.32.56_train_diffusion_unet_lowdim_lift_lowdim_16_state_estimator/eval_jumping_disturbance_horizon_14_num_distrub_1_pos_0.03_gripper_dir_near_gripper_0.025 --n_action_steps 14 --n_test_vis 10 --device cuda:0 --env_runner diffusion_policy.env_runner.robomimic_lowdim_likelihood_jummping_disturbance_runner.RobomimicLowdimLikelihoodDisturbanceRunner

# horizon=3
# python eval_likelihood.py --checkpoint outputs/square_lowdim_reproduction/08.45.33_train_diffusion_unet_lowdim_square_lowdim_cnn_16/checkpoints/epoch=0550-test_mean_score=0.960.ckpt --output_dir outputs/square_lowdim_reproduction/08.45.33_train_diffusion_unet_lowdim_square_lowdim_cnn_16/eval_jumping_disturbance_horizon_${horizon}_num_distrub_1_pos_0.04_gripper_dir_near_gripper_0.1_50episodes  --n_action_steps ${horizon} --n_test_vis 20 --device cuda:0 --env_runner diffusion_policy.env_runner.robomimic_lowdim_likelihood_jummping_disturbance_runner.RobomimicLowdimLikelihoodDisturbanceRunner --max_steps 300

n_test=200
n_test_vis=0
cuda_device=0
pos=0.03
gripper_distance=0.025
max_steps=200
for horizon in 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 126
# for horizon in 126 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8
do
    echo -e "\033[32mEvaluate in horizon ${horizon}\033[0m"
    python eval_likelihood.py --checkpoint outputs/lift_lowdim_mh_reproduction/2025.04.29_05.39.45_train_diffusion_unet_lowdim_lift_lowdim_cnn_128/checkpoints/epoch=0100-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_mh_reproduction/2025.04.29_05.39.45_train_diffusion_unet_lowdim_lift_lowdim_cnn_128/uniform/eval_jumping_disturbance_num_distrub_1_pos_${pos}_gripper_dir_near_gripper_${gripper_distance}_${n_test}episodes_max_steps_${max_steps}_horizon_${horizon}  --n_action_steps ${horizon} --n_test ${n_test} --n_test_vis ${n_test_vis} --device cuda:${cuda_device} --env_runner diffusion_policy.env_runner.robomimic_lowdim_likelihood_jummping_disturbance_runner.RobomimicLowdimLikelihoodDisturbanceRunner --max_steps ${max_steps}
done

# horizon=8
# n_test_vis=20
# cuda_device=0
# pos=0.03
# gripper_distance=0.04
# python eval_likelihood.py --checkpoint outputs/can_ph_lowdim_reproduction/2025.04.24_18.13.14_train_diffusion_unet_lowdim_can_lowdim_cnn_32/checkpoints/epoch=0200-test_mean_score=1.000.ckpt --output_dir outputs/can_ph_lowdim_reproduction/2025.04.24_18.13.14_train_diffusion_unet_lowdim_can_lowdim_cnn_32/eval_jumping_disturbance_num_distrub_1_pos_${pos}_gripper_dir_near_gripper_${gripper_distance}_${n_test_vis}episodes_horizon_${horizon}  --n_action_steps ${horizon} --n_test_vis ${n_test_vis} --device cuda:${cuda_device} --env_runner diffusion_policy.env_runner.robomimic_lowdim_likelihood_jummping_disturbance_runner.RobomimicLowdimLikelihoodDisturbanceRunner --max_steps 150 