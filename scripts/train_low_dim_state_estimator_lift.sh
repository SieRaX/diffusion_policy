python train.py --config-name=train_diffusion_unet_lowdim_state_estimator_workspace.yaml \
task=lift_lowdim_abs \
policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_state_estimator.DiffusionUnetLowdimPolicy \
training.seed=42 \
hydra.run.dir='outputs/lift_lowdim_ph_reproduction/${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}_16_state_estimator' \
logging.group=Lift_Lowdim_Reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_16_state_estimator' \
training.num_epochs=1000 \
training.device=cuda:0

# --config-name=train_diffusion_unet_lowdim_state_estimator_workspace.yaml task.name=lift_lowdim_abs policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_state_estimator.DiffusionUnetLowdimPolicy training.seed=42 hydra.run.dir='outputs/lift_lowdim_ph_reproduction/${now:%Y.%m.%d}_${now:%H.%M.%S}_${name}_${task_name}_16_state_estimator' logging.group=Lift_Lowdim_Reproduction logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_16_state_estimator' training.num_epochs=1000 training.device=cuda:0 training.debug=true