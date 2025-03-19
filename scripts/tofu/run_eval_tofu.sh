SEED=43

ROOT_DIR="~/maplecg_nfs_public/watermark_tofu/main_results/seed_${SEED}/watermarked_results_remove-10pct_dup-10pct"

for METHOD in "retraining" "dpo" "finetune" "grad_ascent" "grad_diff" "KL" "original" "scrub" "tv"
do
python scr/evaluate_methods/eval_everything_tofu.py \
    --seed $SEED \
    --model_path "${ROOT_DIR}/${METHOD}.pt" \
    --data_config_path "config/tofu/data_forget10_dup10.yaml" \
    --model_config_path "config/tofu/finetune_lora.yaml" \
    --output_dir "${ROOT_DIR}/eval/${METHOD}" \
    --retrain_output_dir "${ROOT_DIR}/eval/retraining" 
done
