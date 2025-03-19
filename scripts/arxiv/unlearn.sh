export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=2
SEED=42
METHOD="scrub"

python scr/unlearn.py \
    --seed $SEED \
    --dataset_name "arxiv" \
    --data_config_path "config/arxiv/data_forget01_dup01.yaml" \
    --orig_model_path "/home/xinyang/maplecg_nfs_public/watermark_arxiv/main_results/seed_${SEED}/watermarked_results_remove-1class/original.pt" \
    --unlearn_config_path "config/arxiv/unlearn_lora.yaml" \
    --unlearn_method $METHOD \
    --output_dir "main_results/watermark_arxiv/0.01_split/seed_${SEED}"

