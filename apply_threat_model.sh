# Generate and verify for data owner with interclass data, and for extra interclass data not in training set
for seed in 41 42 43; do
    python generate_and_verify_watermark.py \
    --dataset="Glow-AI/WaterDrum-Ax" \
    --dataset_column="threat_model" \
    --dataset_split="watermarked_intraclass" \
    --idstart=0 \
    --idend=19 \
    --postfix="_intra" \
    --directory_path=watermark_arxiv/main_results/seed_$seed/watermarked_results_duplicate_interclass \
    --model_filter="model" \
    --verification \

    wait

    python verify_vllm.py \
    --dataset="Glow-AI/WaterDrum-Ax" \
    --dataset_column="threat_model" \
    --dataset_split="unwatermarked_intraclass" \
    --idstart=0 \
    --idend=19 \
    --postfix="_intra_extra" \
    --directory_path=watermark_arxiv/main_results/seed_$seed/watermarked_results_duplicate_interclass \
    --model_filter="model" \
    --verification \

    wait
done

python apply_threat_model.py