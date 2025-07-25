#! /bin/bash

### Arxiv dataset
mkdir -p watermark_arxiv
# Watermark the arxiv dataset
for i in {0..9}; do
    python watermark_dataset.py --id=$i --dataset=Glow-AI/WaterDrum-Ax --dataset_subset=unwatermarked_forget_01 --dataset_split=full --start_idx=$((i*400)) --batch=400 --dataset_column=text
done
python -c 'import pandas as pd; pd.concat([pd.read_pickle(f"watermarked/Glow-AI/WaterDrum-Ax/unwatermarked_forget_01/full_{i}_2_1_meta-llama/Meta-Llama-3.1-8B-Instruct_fourier_{i*400}_400_group/output.pkl") for i in range(10)]).to_pickle("watermark_arxiv/watermarked_arxiv.pkl")'

# Watermark the paraphrased arxiv dataset
python watermark_dataset.py --id=0 --dataset=Glow-AI/WaterDrum-Ax --dataset_subset=unwatermarked_forget_01 --dataset_split=semantic_duplicate --dataset_column=text
cp watermarked/watermark_arxiv/paraphrased_arxiv_full/train_0_2_1_meta-llama/Meta-Llama-3.1-8B-Instruct_fourier_7600_400_group/output.pkl watermark_arxiv/watermarked_para.pkl

# Watermark the intra class arxiv dataset (Maths)
python watermark_dataset.py --id=0 --dataset=Glow-AI/WaterDrum-Ax --dataset_subset=unwatermarked_forget_01 --dataset_split=exact_duplicate --dataset_column=text
cp watermarked/watermark_arxiv/mathPR/train_0_2_1_meta-llama/Meta-Llama-3.1-8B-Instruct_fourier_0_400_group/output.pkl watermark_arxiv/watermarked_intra.pkl

### TOFU dataset
mkdir -p watermark_tofu
# Watermark the TOFU dataset
python watermark_dataset.py --id=0 --dataset=Glow-AI/WaterDrum-TOFU --dataset_subset=unwatermarked_forget_01 --dataset_split=full --dataset_column=answer --question_column=question
python watermark_dataset.py --id=1 --start_idx=3600 --batch=400 --dataset=Glow-AI/WaterDrum-TOFU --dataset_subset=unwatermarked_forget_01 --dataset_split=full --dataset_column=answer --question_column=question
python -c 'import pandas as pd; [pd.concat([pd.read_pickle("watermarked/Glow-AI/WaterDrum-TOFU/unwatermarked_forget_01/full_0_2_1_meta-llama/Meta-Llama-3.1-8B-Instruct_fourier_0_None_group/output.pkl").iloc[:-(4000//100)*i], pd.read_pickle("watermarked/Glow-AI/WaterDrum-TOFU/unwatermarked_forget_01/full_1_2_1_meta-llama/Meta-Llama-3.1-8B-Instruct_fourier_3600_400_group/output.pkl").iloc[-(4000//100)*i:]]).to_pickle(f"watermark_tofu/watermarked_tofu_{i}pct.pkl") for i in [1,5,10]]'