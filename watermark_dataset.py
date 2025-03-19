import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Waterfall.Watermark.WatermarkingFnFourier import WatermarkingFnFourier
from Waterfall.Watermark.WatermarkingFnSquare import WatermarkingFnSquare
from Waterfall.Watermark.WatermarkerBase import Watermarker
from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import traceback
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt = (
    "Paraphrase the user provided text while preserving semantic similarity. "
    "Do not include any other sentences in the response, such as explanations of the paraphrasing. "
    "Retain all factual information from the provided text, and do not add any other information not present in the original text. "
    "Keep the paraphrased text similar in length to the provided text. Do not summarize."
)
pre_paraphrased = "Here is a paraphrased version of the text while preserving the semantic similarity:\n\n"

qa_prompt = (
    "Paraphrase the answer for the following question and answer pair, while preserving semantic similarity. "
    "Do not include any other sentences in the response, such as explanations of the paraphrasing. "
    "Retain all factual information from the provided answer, and do not add any other information not present in the original answer. "
    "Keep the paraphrased answer similar in length to the provided answer. Do not summarize."
)
qa_pre_paraphrased = "Here is the question with the paraphrased version of the answer while preserving the semantic similarity:\n\n"

def watermark(data, dataset_column, seed=42, question_column = None):
    T_o = data[dataset_column]
    # Generate watermarked text
    if question_column is not None:
        paraphrasing_prompt = tokenizer.apply_chat_template(
            [
                {"role":"system", "content":qa_prompt},
                {"role":"user", "content":f"Question:\n{data[question_column]}\n\nAnswer:\n"+T_o.replace("\n", " ")},
            ], tokenize=False, add_generation_prompt = True) + f"{pre_paraphrased}Question:\n{data[question_column]}\n\nParaphrased answer:\n"
    else:
        paraphrasing_prompt = tokenizer.apply_chat_template(
            [
                {"role":"system", "content":prompt},
                {"role":"user", "content":T_o.replace("\n", " ")},
            ], tokenize=False, add_generation_prompt = True) + f"{pre_paraphrased}\n\n"
    torch.manual_seed(seed)
    try:
        watermarked = watermarker.generate(
            paraphrasing_prompt, 
            return_scores=True,
            max_new_tokens=min(1000, len(paraphrasing_prompt)),
            do_sample=False, temperature=None, top_p=None,
            num_beams=8, num_beam_groups=4, num_return_sequences=8, diversity_penalty = 0.5
            )
    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        try:
            print("Retrying with lower beam settings of 4 beams and 2 beam groups")
            watermarked = watermarker.generate(
                paraphrasing_prompt, 
                return_scores=True,
                max_new_tokens=min(1000, len(paraphrasing_prompt)),
                do_sample=False, temperature=None, top_p=None,
                num_beams=4, num_beam_groups=2, num_return_sequences=4, diversity_penalty = 0.5
                )
        except:
            print(e)
            torch.cuda.empty_cache()
            try:
                print("Retrying with even lower beam settings of 2 beams and 1 beam group")
                watermarked = watermarker.generate(
                    paraphrasing_prompt, 
                    return_scores=True,
                    max_new_tokens=min(1000, len(paraphrasing_prompt)),
                    do_sample=False, temperature=None, top_p=None,
                    num_beams=2, num_beam_groups=1, num_return_sequences=2, diversity_penalty = 0.5
                    )
            except:
                torch.cuda.empty_cache()
                print("Failed to generate watermarked text")
                return [T_o], watermarker.verify(T_o)[:,watermarker.k_p-1], torch.ones(1), torch.zeros(1), 0
    sts_scores = sts_model.encode([T_o, *watermarked["text"]], convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(sts_scores[0], sts_scores[1:], dim=1).cpu()
    selection_score = cos_sim + torch.from_numpy(watermarked["q_score"])
    selection = torch.argmax(selection_score)
    return watermarked["text"], watermarked["q_score"], cos_sim, selection_score.float(), selection.item(), watermarked["text"][selection.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    parser.add_argument('--id',default=0,type=int,
            help='id: unique ID')
    parser.add_argument('--kappa',default=2,type=float,
            help='kappa: watermarking strength')
    parser.add_argument('--k_p', default=1, type=int,
            help="k_p: Perturbation key")
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str,
            help="model")
    parser.add_argument('--watermark_fn', default='fourier', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='locuslab/TOFU', type=str)
    parser.add_argument('--dataset_split', default=None, type=str)
    parser.add_argument('--dataset_key', default="train", type=str)
    parser.add_argument('--dataset_column', default="answer", type=str)
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--batch', default=None, type=int)
    parser.add_argument('--question_column', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--outdir', default="watermarked", type=str)

    args = parser.parse_args()

    print(args)

    id = args.id
    kappa = args.kappa
    k_p = args.k_p
    model_name_or_path = args.model
    dataset_dir = args.dataset
    start_idx = args.start_idx

    # Initialize tokenizer and model
    print("Initializing tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    id = args.id
    print(f"Starting from index {start_idx}")
    print("Loading dataset")
    if "." in dataset_dir:
        if dataset_dir.endswith(".pkl"):
            dataset = pd.read_pickle(dataset_dir)
        dataset = dataset.iloc[start_idx:]
        if args.batch is not None:
            dataset = dataset.iloc[:args.batch]
    else:
        try:
            dataset = load_dataset(dataset_dir, args.dataset_split)
            dataset = dataset[args.dataset_key]
        except:
            dataset = load_from_disk(dataset_dir)
        dataset = dataset.select(range(start_idx, len(dataset)))
        if args.batch is not None:
            dataset = dataset.select(range(min(len(dataset), args.batch)))
        dataset = dataset.to_pandas()
    print(f"Dataset loaded with {len(dataset)} samples")

    print("Initializing watermarker")
    if args.watermark_fn == 'fourier':
        watermarkingFnClass = WatermarkingFnFourier
    elif args.watermark_fn == 'square':
        watermarkingFnClass = WatermarkingFnSquare
    else:
        raise ValueError("Invalid watermarking function")

    outdir = f"{args.dataset.rsplit('.',1)[0]}/{args.dataset_split + '/' if args.dataset_split is not None else ''}{args.dataset_key}_{id}_{kappa}_{k_p}_{args.model}_{args.watermark_fn}_{start_idx}_{args.batch}_group"
    outdir = os.path.join(args.outdir, outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Watermarking dataset")

    if args.debug:
        model = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16,
            device_map = "auto",
            )
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        sts_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=args.device)
    watermarker = Watermarker(model, tokenizer, id, kappa, k_p, watermarkingFnClass=watermarkingFnClass)

    watermarked_dataset = dataset.iloc[:2].progress_apply(
        partial(watermark, dataset_column=args.dataset_column, question_column=args.question_column),
        axis=1
        )

    watermarked_dataset = pd.DataFrame(list(watermarked_dataset), columns=["watermarked_texts", "q_score", "cos_sim", "selection_score", "selection", "watermarked"])
    watermarked_dataset.to_pickle(outdir+f"/output{'_'+args.question_column if args.question_column is not None else ''}.pkl")