export CUDA_VISIBLE_DEVICES=1


for SEED in 42
do
for UNLEARN_METHOD in retraining 
# for UNLEARN_METHOD in retraining original finetune ga gdiff KL dpo tv
do
python scr/evaluate_methods/eval_everything_arxiv.py \
    --seed $SEED \
    --model_path "/home/xinyang/watermark_metric/watermark_metric/main_results/watermark_arxiv/0.1_split/seed_${SEED}/model.pt" \
    --data_config_path "config/arxiv/data_forget01_dup01.yaml" \
    --model_config_path "config/arxiv/unlearn_lora.yaml" \
    --unlearn_method $UNLEARN_METHOD \
    --run_mia \
    --run_rouge \
    --run_knowmem
done
done