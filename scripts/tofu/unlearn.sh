export WANDB_DISABLED=true

for SEED in 41 42 43
do
    for METHOD in "grad_ascent" "grad_diff" "dpo" "scrub" "finetune"
    do
    ROOT_DIR="~/maplecg_nfs_public/watermark_tofu/main_results/seed_${SEED}/results_remove-1pct_semdup-1pct"

    python scr/unlearn.py \
        --seed $SEED \
        --dataset_name "tofu" \
        --data_config_path "config/tofu/data_unwatermark_forget01_semdup01.yaml" \
        --orig_model_path "${ROOT_DIR}/original.pt" \
        --unlearn_config_path "config/tofu/unlearn_lora.yaml" \
        --unlearn_method ${METHOD} \
        --output_dir ${ROOT_DIR}
    done
done