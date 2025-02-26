for horizon in 32 64 128; do
    python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
        training.seed=42 \
        horizon=$horizon \
        logging.group=Change_Horizon \
        logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_Th_${horizon}' \
        hydra.run.dir='data/outputs/${now:%Y.%m.%d}/change_by_horizon_length/${now:%H.%M.%S}_${name}_${task_name}_Th_${horizon}'
done
