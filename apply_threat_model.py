from sentence_transformers import SentenceTransformer
from datasets import load_from_disk, load_dataset
import argparse
from glob import glob
import torch
from transformers import AutoTokenizer
import os
import numpy as np
from matplotlib import pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=41)
argparser.add_argument("--parent_dir", type=str, default="watermark_arxiv")
argparser.add_argument("--subdir", type=str, default="watermarked_results_duplicate_interclass")
argparser.add_argument("--dataset", type=str, default="Glow-AI/WaterDrum-Ax")
argparser.add_argument("--dataset_subset", type=str, default="forget_01")
argparser.add_argument("--forget_size", type=int, default=400)
argparser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")

args = argparser.parse_args()

# Load the dataset
generated_texts = []
generated_qs = []
for seed in (41,42,43):
    path = f"{args.parent_dir}/main_results/seed_{seed}/{args.subdir}/watermark_verify/"

    # load intra-class generations (used in training)
    # To check how much threat model affect other data owner with intra-class data
    generated_path = glob(f"{path}/model_verify_*intra")
    assert len(generated_path) == 1, f"{len(generated_path)} files found in {path}, expecting only 1"
    generated_dataset = load_from_disk(generated_path[0])
    assert len(generated_dataset) == args.forget_size, f"Expecting {args.forget_size} samples, got {len(generated_dataset)}"
    generated_texts.append(generated_dataset["generation"])

    # load extra intra-class generations (not used in training)
    # To check how much watermark leaks through in extra intra-class data
    generated_path = glob(f"{path}/model_verify_*intra_extra")
    assert len(generated_path) == 1, f"{len(generated_path)} files found in {path}, expecting only 1"
    generated_dataset = load_from_disk(generated_path[0])
    assert len(generated_dataset) == args.forget_size, f"Expecting {args.forget_size} samples, got {len(generated_dataset)}"
    generated_texts.append(generated_dataset["generation"])
    generated_qs.append(generated_dataset["q"])

generated_qs = np.array(generated_qs)
generated_qs /= generated_qs.mean(axis=(1,2), keepdims=True)

# Load and prepare training dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
forget_dataset = load_dataset(args.dataset, args.dataset_subset, split="forget")["text"]
forget_dataset = [tokenizer.decode(tokenizer.encode(i, add_special_tokens=False)[50:]) for i in forget_dataset]

# Calculate the embeddings
sts_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=f"cuda:0")

all_text = forget_dataset + [k for i in generated_texts for j in i for k in j]
wat_encoding = sts_model.encode(all_text, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)

forget_encoding = wat_encoding[:len(forget_dataset)]
generated_encoding = wat_encoding[len(forget_dataset):]

# Calculate the cosine similarity
cos_sim = torch.einsum('ik,jk->ij', forget_encoding, generated_encoding).half().cpu().numpy()
cos_sim = cos_sim.max(axis=0).reshape(3, 2, -1, 10)  # 3 seeds, 2 subsets, 400 queries, 10 samples per query

# Sweep STS threshold
sweep_num = 100
cos_sim = np.tile(cos_sim[...,None],sweep_num)
generated_qs = np.tile(generated_qs[...,-1,None],sweep_num)
thresholds = np.linspace(np.min(cos_sim), np.max(cos_sim), sweep_num)
filtered_by_threshold = cos_sim > thresholds

generated_qs[filtered_by_threshold[:,1]] = 0    # For intra-class extra, set q to 0 if not passed through threshold

fig = plt.figure(figsize=(3, 1.8))

false_intercept_rate = filtered_by_threshold[:,0].mean(axis=(1,2))    # For intra-class, calculate false intercept rate
forget_watermark_strength = generated_qs.mean(axis=(1,2))

plt.plot(false_intercept_rate.mean(axis=0), forget_watermark_strength.mean(axis=0))
plt.fill_between(false_intercept_rate.mean(axis=0), forget_watermark_strength.max(axis=0), forget_watermark_strength.min(axis=0), alpha=0.4)
plt.xlabel("Falsely intercepted outputs")
plt.ylabel("Forget watermark strength")

plt.xticks(np.linspace(0,1,6), [f"{i*100:.0f}%" for i in np.linspace(0,1,6)], rotation=45)
plt.yticks(np.linspace(0,1,6))
plt.title("Resilience of WaterDrum")
plt.savefig(f"plots/threat_model.pdf", bbox_inches="tight")