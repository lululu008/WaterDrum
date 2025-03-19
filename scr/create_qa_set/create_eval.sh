export CUDA_VISIBLE_DEVICES=5
export OPENAI_API_KEY=""

for SEED in 42
do
python eval/create_eval.py \
    --seed $SEED \
    --dataset "arxiv" \
    --pretrained_model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "eval/" \
    --watermarked_dir "/../../maplecg_nfs_public/watermark_arxiv/arxiv_added.pkl" 
done