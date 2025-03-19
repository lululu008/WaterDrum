export CUDA_VISIBLE_DEVICES=2

SEED=42

python scr/train.py \
    --seed $SEED \
    --dataset_name "arxiv" \
    --data_config_path "config/arxiv/data_forget01.yaml" \
    --train_config_path "config/arxiv/finetune_lora.yaml" \
    --output_dir "main_results/watermark_arxiv/0.01_split/seed_${SEED}"