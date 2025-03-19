export WANDB_DISABLED=true

SEED=41
ROOT_DIR="~/maplecg_nfs_public/watermark_tofu/main_results/seed_${SEED}/watermarked_results_remove-5pct_semdup-5pct"

python scr/train.py \
    --seed $SEED \
    --dataset_name "tofu" \
    --data_config_path "config/tofu/data_forget05_semdup05.yaml" \
    --train_config_path "config/tofu/finetune_lora.yaml" \
    --output_dir "${ROOT_DIR}"
