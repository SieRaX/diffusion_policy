# for n_action_step in 8 16 32 62; do

#     rm -r data/outputs/2025.02.27/change_by_horizon_length_pusht_image/04.20.39_train_diffusion_unet_image_pusht_image_Th_64/eval/n_action_step_${n_action_step};
    
# done

# rm -r data/outputs/pusht_hybrid_reproduction/2025.02.28/08.37.58_train_diffusion_unet_hybrid_pusht_image/debug/;
for n_action_step in 4; do
	echo "Evaluating n_action_step: ${n_action_step}";
	python eval.py --checkpoint data/outputs/square_hybrid_mh_reproduction/09.02.41_train_diffusion_transformer_hybrid_square_image_cnn_128/checkpoints/epoch=0550-test_mean_score=0.740.ckpt --output_dir data/outputs/square_hybrid_mh_reproduction/09.02.41_train_diffusion_transformer_hybrid_square_image_cnn_128/debug/n_action_step_${n_action_step} --n_action_steps ${n_action_step} \
	--n_test_vis 50 --device cuda:1
done