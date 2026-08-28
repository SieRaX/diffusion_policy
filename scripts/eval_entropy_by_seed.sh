CKPT_DIRS=(
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed0_2026.02.26-12.41.40_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=0800-test_mean_score=0.940.ckpt
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed0_2026.02.26-12.41.40_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=1250-test_mean_score=0.960.ckpt
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed0_2026.02.26-12.41.40_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=1300-test_mean_score=0.960.ckpt
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed1_2026.02.27-10.31.13_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=0650-test_mean_score=0.940.ckpt
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed1_2026.02.27-10.31.13_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=0750-test_mean_score=0.940.ckpt
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed1_2026.02.27-10.31.13_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=0850-test_mean_score=0.920.ckpt
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed1_2026.02.27-10.31.13_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=1150-test_mean_score=0.940.ckpt
    outputs_HDD4/square_lowdim_ph_reproduction/train_by_seed_ddpm/seed1_2026.02.27-10.31.13_train_diffusion_unet_lowdim_square_lowdim_cnn_48/checkpoints/epoch=1800-test_mean_score=0.940.ckpt
)

for CKPT_DIR in "${CKPT_DIRS[@]}"; do
    OUTPUT_DIR=${CKPT_DIR%.ckpt}/eval_entropy_by_seed

    python eval_entropy_by_seed.py \
        -c "$CKPT_DIR" \
        -o "$OUTPUT_DIR" \
        -d cuda:0 \
        -v 100 \
        -n 64 \
        -t 250
done
