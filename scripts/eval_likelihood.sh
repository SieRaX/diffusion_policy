python eval_likelihood.py --checkpoint data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/checkpoints/epoch=0650-test_mean_score=0.894.ckpt --output_dir data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/total_likelihood_diff_0.01_mean   --n_test_vis 50 --device cuda:0


# # Eval likelihood in Lift task
# python eval_likelihood.py --checkpoint data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/checkpoints/epoch=0250-test_mean_score=1.000.ckpt --output_dir data/outputs/lift_lowdim_ph_reproduction/horizon_16/2025.03.11/10.57.22_train_diffusion_unet_lowdim_lift_lowdim_transformer_128/total_likelihood_diff_0.1_mean   --n_test_vis 50 --device cuda:0

# Eval likelihood in Square task
# python eval_likelihood.py --checkpoint data/outputs/square_reproduction/2025.02.28/07.43.04_train_diffusion_unet_lowdim_square_lowdim/checkpoints/epoch=0150-test_mean_score=0.960.ckpt --output_dir data/outputs/square_reproduction/2025.02.28/07.43.04_train_diffusion_unet_lowdim_square_lowdim/total_likelihood_diff_0.1_mean   --n_test_vis 50 --device cuda:0

# --checkpoint data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/checkpoints/epoch=0650-test_mean_score=0.894.ckpt --output_dir data/outputs/pusht_lowdim_mh_reproduction/2025.04.08_04.43.29_train_diffusion_unet_lowdim_pusht_lowdim_16_obs_as_global_cond/debug  --n_test_vis 50 --device cuda:0
