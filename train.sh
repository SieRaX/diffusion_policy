# for horizon in 64; do
#     python train.py --config-name=train_diffusion_unet_hybrid_workspace \
#         task=square_image_abs \
#         training.seed=42 \
#         horizon=$horizon \
#         training.device=cuda:1 \
#         logging.group=Change_Horizon_Square_Image \
#         logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_Th_${horizon}' \
#         hydra.run.dir='data/outputs/${now:%Y.%m.%d}/change_by_horizon_length_square_image/${now:%H.%M.%S}_${name}_${task_name}_Th_${horizon}'
# done


python train.py --config-dir=. --config-name=image_square_diffusion_policy_cnn.yaml training.seed=42 hydra.run.dir='data/outputs/square_hybrid_reproduction/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'_lowunetparam \
logging.group=Square_Hybrid_Reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_lowunetparam' \
checkpoint.topk.k=2 \
training.device=cuda:1 \
horizon=64 \
policy.horizon=64 \
task.dataset.horizon=64 \
task.dataset.pad_after=63 \
policy.down_dims=[1024,2048,4096]



# policy.num_inference_steps=1000 \
# policy.noise_scheduler.num_train_timesteps=1000
# Square_Hybrid_Reproduction