python train.py --config-dir=. --config-name=low_dim_pusht_diffusion_policy_cnn.yaml training.seed=42 hydra.run.dir='data/outputs/pusht_lowdim_reproduction/debug/${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
logging.group=Debug \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}' \
checkpoint.topk.k=5 ;