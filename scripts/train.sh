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

# python train.py --config-dir=. --config-name=low_dim_square_mh_diffusion_policy_cnn.yaml training.seed=42 hydra.run.dir='data/outputs/square_lowdim_mh_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_128' \
# logging.group=Square_lowdim_Reproduction \
# logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_mh_cnn_128' \
# checkpoint.topk.k=5 \
# training.device=cuda:0 \
# horizon=128 \
# policy.horizon=128 \
# task.dataset.horizon=128 \
# task.dataset.pad_after=127 ;

# python train.py --config-dir=. --config-name=low_dim_square_mh_diffusion_policy_cnn.yaml training.seed=42 hydra.run.dir='data/outputs/square_lowdim_mh_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
# logging.group=Square_lowdim_Reproduction \
# logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_mh_cnn_16' \
# checkpoint.topk.k=5 \
# training.device=cuda:0 ;

# python train.py --config-dir=. --config-name=image_square_mh_diffusion_policy_transformer.yaml training.seed=42 hydra.run.dir='data/outputs/square_hybrid_mh_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
# logging.group=Square_Hybrid_Reproduction \
# logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_mh_transformer_16' \
# checkpoint.topk.k=5 ;

# python train.py --config-dir=. --config-name=image_square_mh_diffusion_policy_transformer.yaml training.seed=42 hydra.run.dir='data/outputs/square_hybrid_mh_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_128' \
# logging.group=Square_Hybrid_Reproduction \
# logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_mh_transformer_128' \
# checkpoint.topk.k=5 \
# training.device=cuda:0 \
# horizon=128 \
# policy.horizon=128 \
# task.dataset.horizon=128 \
# task.dataset.pad_after=127 ;
# # policy.num_inference_steps=1000 \
# # policy.noise_scheduler.num_train_timesteps=1000
# # Square_Hybrid_Reproduction

# python train.py --config-dir=. --config-name=low_dim_square_diffusion_policy_cnn.yaml \
# training.seed=42 \
# hydra.run.dir='outputs/square_lowdim_reproduction/${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}_cnn_32' \
# logging.group=Square_lowdim_Reproduction \
# logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_32' \
# checkpoint.topk.k=5 \
# training.device=cuda:0 \
# horizon=32 \
# policy.horizon=32 \
# task.dataset.horizon=32 \
# task.dataset.pad_after=31 ;

python train.py --config-dir=. --config-name=low_dim_can_ph_diffusion_policy_cnn.yaml \
training.seed=42 \
hydra.run.dir='outputs/can_ph_lowdim_reproduction/${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}_cnn_32' \
logging.group=can_ph_lowdim_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_32' \
checkpoint.topk.k=5 \
training.device=cuda:0 \
horizon=32 \
policy.horizon=32 \
task.dataset.horizon=32 \
task.dataset.pad_after=31 ;

python train.py --config-dir=. --config-name=low_dim_can_mh_diffusion_policy_cnn.yaml \
training.seed=42 \
hydra.run.dir='outputs/can_lowdim_mh_reproduction/${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}_cnn_128' \
logging.group=can_lowdim_mh_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_128' \
checkpoint.topk.k=5 \
training.device=cuda:0 \
horizon=128 \
policy.horizon=128 \
task.dataset.horizon=128 \
task.dataset.pad_after=127 ;

python train.py --config-dir=. --config-name=image_tool_hang_diffusion_policy_cnn.yaml \
training.seed=42 \
hydra.run.dir='outputs/toolhang_ph_image_reproduction/${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
logging.group=toolhang_ph_image_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_16' \
checkpoint.topk.k=5 \
training.device=cuda:1;
