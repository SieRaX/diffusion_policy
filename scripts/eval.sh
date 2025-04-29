# for n_action_step in 8 16 32 62; do

#     rm -r data/outputs/2025.02.27/change_by_horizon_length_pusht_image/04.20.39_train_diffusion_unet_image_pusht_image_Th_64/eval/n_action_step_${n_action_step};
    
# done

# rm -r data/outputs/pusht_hybrid_reproduction/2025.02.28/08.37.58_train_diffusion_unet_hybrid_pusht_image/debug/;
max_steps=150
for n_action_step in 8; do
	echo -e "\033[32mEvaluating n_action_step: ${n_action_step}\033[0m";
	python eval.py --checkpoint outputs/lift_lowdim_mh_reproduction/2025.04.29_05.39.45_train_diffusion_unet_lowdim_lift_lowdim_cnn_128/checkpoints/epoch=0100-test_mean_score=1.000.ckpt --output_dir outputs/lift_lowdim_mh_reproduction/2025.04.29_05.39.45_train_diffusion_unet_lowdim_lift_lowdim_cnn_128/normal_eval_max_steps_${max_steps}/ --n_action_steps ${n_action_step} \
	--n_test_vis 50 --device cuda:1 \
	--max_steps ${max_steps}
done