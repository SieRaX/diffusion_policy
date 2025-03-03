# for n_action_step in 8 16 32 62; do

#     rm -r data/outputs/2025.02.27/change_by_horizon_length_pusht_image/04.20.39_train_diffusion_unet_image_pusht_image_Th_64/eval/n_action_step_${n_action_step};
    
# done

# rm -r data/outputs/square_hybrid_reproduction/2025.03.01/07.46.12_train_diffusion_unet_hybrid_square_image/eval;
for n_action_step in 1 2; do
	echo "Evaluating n_action_step: ${n_action_step}";
	python eval.py --checkpoint data/outputs/square_hybrid_reproduction/2025.03.02/12.38.32_train_diffusion_unet_hybrid_square_image_64/checkpoints/epoch=0350-test_mean_score=0.780.ckpt --output_dir data/outputs/square_hybrid_reproduction/2025.03.02/12.38.32_train_diffusion_unet_hybrid_square_image_64/eval/n_action_step_${n_action_step} --n_action_steps ${n_action_step}
done