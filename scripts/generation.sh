export CUDA_VISIBLE_DEVICES=1

python model_api.py \
    --model_dir "main_results/seed_42/watermarked_results_remove-5pct/ga.pt" \
    --tokenizer_dir "meta-llama/Llama-2-7b-chat-hf"
