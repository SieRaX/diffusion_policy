python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml training.seed=42 hydra.run.dir='data/outputs/pusht_lowdim_mh_reproduction/${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}_16_obs_as_global_cond' \
logging.group=pusht_lowdim_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_16_obs_as_global_cond' \
training.num_epochs=1000 \
training.device=cuda:0