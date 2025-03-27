import torch
if torch.cuda.device_count() != 0 and __name__ == "__main__":
    _ = torch.empty(0).cuda()
    print("initialized cuda")
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset, Sequence, Value
from Waterfall.Watermark.WatermarkerBase import Watermarker
from Waterfall.Watermark.WatermarkingFnFourier import WatermarkingFnFourier
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os
import shutil
from safetensors.torch import save_file
import gc
from multiprocessing import Process
from copy import deepcopy
import glob
from tqdm import tqdm
import traceback
import pandas as pd
import time
from typing import Tuple, List
import gc
import time
import numpy as np
from itertools import repeat
from copy import deepcopy
from bdb import BdbQuit
from threading import Thread
from itertools import count

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Wrapper for the LLM class to allow for lazy initialization
class LazyLLM(LLM):
    def __init__(self, lazy=True, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.initialized = False
        self.thread = None
        self.thread = Thread(target=super().__init__, args=args, kwargs=kwargs)
        self.thread.start()
        if not lazy:
            self.finish_init()
    def finish_init(self):
        if not self.initialized:
            self.thread.join()
            self.initialized = True

# Add the verification results to the dataset and save to disk
def add_columns_and_save(outfile, verification_output, dataset):
    start_time = time.time()
    tqdm.write("adding columns")
    def create_nested_feature(nesting_level, dtype):
        if nesting_level == 0:
            return Value(dtype)
        else:
            return Sequence(create_nested_feature(nesting_level - 1, dtype))
    def nested_lists_of_np_array_to_list(data):
        if isinstance(data, list):
            return [nested_lists_of_np_array_to_list(i) for i in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    for k, v in verification_output.items():
        tmp = v
        nesting_level = 0
        while True:
            if isinstance(tmp, list):
                tmp = tmp[0]
                nesting_level += 1
            elif isinstance(tmp, np.ndarray):
                nesting_level += len(tmp.shape) - 1
                break
            else:
                break
        v = nested_lists_of_np_array_to_list(v)
        feature_type = create_nested_feature(nesting_level, str(tmp.dtype))
        dataset = dataset.add_column(k, v, feature=feature_type)
    tqdm.write(f"adding columns took {time.time() - start_time} seconds")
    gc.collect()
    if args.debug:
        breakpoint()

    # Save the results
    start_time = time.time()
    tqdm.write(f"Saving to {outfile}")
    dataset.save_to_disk(outfile)
    tqdm.write(f"Saving took {time.time() - start_time} seconds")

    start_time = time.time()
    tqdm.write("saving q")
    lens = verification_output["len"]
    lens_sum = lens.sum(axis=1)
    lens_sum[lens_sum == 0] = 1
    q = verification_output["q"]
    q_scores = (lens[:,None,:] @ q).squeeze() / lens_sum[:,None]
    np.save(f"{outfile.split('_verify_')[0]}_q_all.npy", q_scores)
    tqdm.write(f"saving q took {time.time() - start_time} seconds")

# Verify the generation to get the wateramrk strengths, and save the results
def verify(dataset: Dataset, outfile, ids, k_p):
    if os.path.exists(outfile):
        process = Process()
        process.start()
        return process
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    tqdm.write(f"Verifying {outfile}")

    res = {
        "q": [], 
        "ranking": [], 
        "indices": [], 
        }
    texts = [j for i in dataset["generation"] for j in i]
    start_time = time.time()
    try:
        id_res = watermarker.verify(texts, ids, k_p, return_ranking=True, return_unshuffled_indices=True, use_tqdm=True, batch_size=2**8)
    except Exception as e:
        if args.debug:
            breakpoint()
        else:
            raise e

    tqdm.write(f"Verification took {time.time() - start_time} seconds")
    gc.collect()
    start_time = time.time()

    q_score = id_res["q_score"].reshape(len(dataset), -1, *id_res["q_score"].shape[1:])   # [sample x generations x id x k_p]
    q_score = q_score[...,0]
    ranking = id_res["ranking"].reshape(len(dataset), -1, *id_res["ranking"].shape[1:])   # [sample x generations x id x k_p]
    ranking = ranking[...,0]

    res["indices"] = [id_res["unshuffled_indices"][i * q_score.shape[1]:(i + 1) * q_score.shape[1]] for i in range(len(q_score))]

    res["q"] = q_score
    res["ranking"] = ranking
    res["len"] = np.array([[j.shape[1] for j in i] for i in res["indices"]], dtype = int)
    res["len"] = res["len"].astype(np.min_scalar_type(res["len"].max()))

    tqdm.write(f"Processing took {time.time() - start_time} seconds")

    gc.collect()
    if args.debug:
        add_columns_and_save(outfile, res, dataset)
        process = Process()
    else:
        process = Process(target=add_columns_and_save, args = (outfile, res, dataset), daemon=False)
    process.start()
    return process

# Prepare the adapter for the fine-tuned model
# vllm expectes safetensors
def prepare_adapter(ft_model_path, tokenizer):
    newpath = ft_model_path.split(".",1)[0]
    tqdm.write(f"Generating for {newpath}")
    if ft_model_path != "":
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        try:
            tensors = torch.load(ft_model_path, weights_only=True, map_location=torch.device('cpu'))
        except Exception as e:
            shutil.rmtree(newpath)
            raise e
        tensors = {k.rsplit("default.",1)[0] + "weight": v for k, v in tensors.items()}
        save_file(tensors, newpath + "/adapter_model.safetensors")
        if "phi" in ft_model_path:
            shutil.copyfile("adapter_config_phi.json", newpath + "/adapter_config.json")
        else:
            shutil.copyfile("adapter_config.json", newpath + "/adapter_config.json")
        tokenizer.save_pretrained(newpath)
    return newpath

# Delete the folder created for the adapter
def cleanup_adapter(newpath):
    if os.path.exists(newpath):
        shutil.rmtree(newpath)

# Lora adapter unique id
lora_adapter_unique_id = count(1)

def generate(
        ft_model_path, base_model: LazyLLM, 
        args, 
        outfile = None,
        ):

    samples = args.samples
    max_tokens = args.max_tokens
    logprobs = args.logprobs
    seed = args.seed

    if os.path.exists(outfile):
        dataset = load_from_disk(outfile)
        if len(dataset) > 0:
            if args.start is not None:
                dataset = dataset.select(range(args.start, len(dataset)))
            if args.end is not None:
                dataset = dataset.select(range(args.end))
            return dataset

    if torch.cuda.device_count() == 0:
        return None

    tqdm.write(f"Generating {outfile}")

    if args.debug:
        breakpoint()
        return None

    start_time = time.time()
    newpath = prepare_adapter(ft_model_path, tokenizer)

    dataset = load_dataset_(tokenizer, args)
    try:
        kwargs = {
            "sampling_params": SamplingParams(
                n = samples, 
                max_tokens = max_tokens, 
                seed=seed,
                logprobs=logprobs,
                ), 
            "lora_request": LoRARequest(newpath, lora_adapter_unique_id.__next__(), newpath) if newpath != "" else None
            }
        base_model.finish_init()
        outputs = base_model.generate(
            dataset["prompt"], 
            **kwargs
            )
    except Exception as e:
        cleanup_adapter(newpath)
        if isinstance(e, (KeyboardInterrupt, BdbQuit)):
            raise e
        traceback.print_exc()
        tqdm.write(f"Failed to generate for {ft_model_path}")
        raise e
    tqdm.write(f"Generated {outfile}, took {time.time() - start_time} seconds")
    start_time = time.time()
    tqdm.write(f"Adding columns to {outfile}")

    log_probs = [[j.cumulative_logprob for j in i.outputs] for i in outputs]
    outputs = [[j.text for j in i.outputs] for i in outputs]

    dataset = dataset.add_column("generation", outputs)
    dataset = dataset.add_column("log_probs", log_probs, feature=Sequence(Value("float32")))
    tqdm.write(f"Added columns to {outfile}, took {time.time() - start_time} seconds")
    if outfile is not None:
        start_time = time.time()
        tqdm.write(f"Saving to {outfile}")
        dataset.save_to_disk(outfile)
        tqdm.write(f"Saved to {outfile}, took {time.time() - start_time} seconds")
    cleanup_adapter(newpath)
    torch.cuda.empty_cache()

    return dataset

# Load the dataset and format the prompts
def load_dataset_(tokenizer, args):
    dataset = args.dataset
    dataset_column = args.dataset_column
    start = args.start
    end = args.end
    dataset = load_dataset(dataset, dataset_column, split=args.dataset_split)
    if args.dataset == "Glow-AI/WaterDrum-Ax":
        dataset = dataset.map(
            lambda x: {"prompt": tokenizer.decode(tokenizer.encode(x["text"], add_special_tokens=False)[:args.completion_prompt_length])},
        )
    elif args.dataset == "Glow-AI/WaterDrum-TOFU":
        dataset = dataset.map(
            lambda x: {"prompt": tokenizer.apply_chat_template([{"role": "user", "content": x["question"]}], tokenize=False)},
        )
    else:
        raise NotImplementedError
    if start is not None:
        dataset = dataset.select(range(start, len(dataset)))
    if end is not None:
        dataset = dataset.select(range(end))
    return dataset

# Generate and verify for a single model
# Verification is ran as a separate process to allow generate to fully utilize the GPU
def run(base_model, ft_model_path, outdir_subdir, watermark_identifier, ids, args) -> Process:
    outdir = ft_model_path.rsplit('/',1)[0] + outdir_subdir + '/' + ft_model_path.rsplit('/',1)[1].rsplit('.',1)[0]
    outfile = f"{outdir}_gen_{watermark_identifier}" + args.postfix
    dataset_ = generate(ft_model_path, base_model, args, outfile=outfile)
    if dataset_ is not None and args.verification:
        outfile_verify = f"{outdir}_verify_{watermark_identifier}" + args.postfix
        process = verify(dataset_, outfile_verify, ids, args.k_p)
    else:
        process = Process()
        process.start()
    gc.collect()
    return process

# Iterate through directory_path folder and run the watermarking and verification
def main(args):
    global lora_adapter_unique_id
    try:
        if "tofu" in args.directory_path.lower():
            assert "Glow-AI/WaterDrum-TOFU" in args.dataset, f"Generating for {args.dataset} but directory path is {args.directory_path}"
        if "arxiv" in args.directory_path.lower():
            assert "Glow-AI/WaterDrum-Ax" in args.dataset, f"Generating for {args.dataset} but directory path is {args.directory_path}"
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, BdbQuit)):
            raise e
        traceback.print_exc()
        if args.debug:
            breakpoint()
        raise e

    if not os.access(args.directory_path, os.W_OK):
        raise Exception(f"Cannot write to {args.directory_path}")

    watermark_identifier = f"{'_'.join([str(i) for i in ids])}_{args.k_p}"
    watermark_identifier = watermark_identifier + "_" + str(args.seed)

    to_break = False
    files = glob.glob(os.path.join(args.directory_path, '*.pt'))
    if args.model_filter is not None:
        files = [i for i in files if args.model_filter in i]
    if len(files) == 0:
        return

    pbar = tqdm(total=len(files), desc=args.directory_path.split("seed_")[-1])

    for file_name in sorted(files):
        if (os.path.isdir(file_name.rsplit(".",1)[0]) 
            or "tv_ft" in file_name 
            or "test" in file_name
            or "archive" in file_name
            or "old" in file_name
            ):
            tqdm.write(f"Skipping {file_name}")
            pbar.update(1)
            continue

        for process in processes:
            process.join(timeout=1)
            if not process.is_alive():
                processes.remove(process)
                pbar.update(1)

        # limit maximum number of processs
        while len(processes) >= 4:
            time.sleep(1)
            for process in processes:
                process.join(timeout=0)
                if not process.is_alive():
                    processes.remove(process)
                    pbar.update(1)

        try:
            process = run(base_model, file_name, outdir_subdir, watermark_identifier, ids, args)
            processes.append(process)
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, BdbQuit)):
                raise e
            traceback.print_exc()
            tqdm.write(f"Failed to run {file_name}")
            gc.collect()
            if to_break:
                breakpoint()
            pbar.update(1)

if __name__ == "__main__":
    # Define the watermarking parameters
    parser = argparse.ArgumentParser(description='Verify a fine-tuned model')
    parser.add_argument('--idstart',default=0,type=int,
            help='for verification, start from this ID')
    parser.add_argument('--idend',default=1,type=int,
            help='for verification, end at this ID')
    parser.add_argument('--k_p',default=1,type=int,
            help='k_p: Perturbation key')
    parser.add_argument('--wt_model', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str)
    parser.add_argument('--base_model', default=None, type=str)
    parser.add_argument('--dataset', default="Glow-AI/WaterDrum-Ax", type=str, help="Dataset to use")
    parser.add_argument('--dataset_column', default='forget_01', type=str, help="Column to use for the dataset")
    parser.add_argument('--dataset_split', default='full', type=str, help="Split to use for the dataset")
    parser.add_argument('--seed', default=0, type=int, help="Seed for generation")
    parser.add_argument('--samples', default=10, type=int, help="Number of generatios per prompt")
    parser.add_argument('--start', default=None, type=int, help="Start index for the dataset")
    parser.add_argument('--end', default=None, type=int, help="End index for the dataset")
    parser.add_argument("--logprobs", type=int, default=1, help="Number of logprobs to save")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--directory_path", type=str, default=None, help="Path to the directory containing the fine-tuned models")
    parser.add_argument("--verification", action='store_true')
    parser.add_argument("--postfix", type=str, default="", help="Postfix for the output file")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--one_per_dir", action='store_true')
    parser.add_argument("--filter", type=str, default=None, help="Filter for directory path")
    parser.add_argument("--model_filter", type=str, default=None, help="Filter for model path")
    parser.add_argument("--completion_prompt_length", type=int, default=50, help="Prompt length for completion")

    args = parser.parse_args()

    if args.base_model is None and args.directory_path is not None:
        if "arxiv" in args.directory_path:
            args.base_model="meta-llama/Llama-2-7b-hf"
        elif "tofu" in args.directory_path:
            args.base_model="meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    wt_tokenizer = AutoTokenizer.from_pretrained(args.wt_model)
    if torch.cuda.device_count() == 0:
        base_model = None
    else:
        base_model = LazyLLM(
            (args.directory_path=="tofu" or args.directory_path=="arxiv"), 
            args.base_model, 
            tokenizer=args.base_model, dtype="bfloat16", 
            enable_lora=True, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9, 
            max_model_len=args.completion_prompt_length + args.max_tokens)


    # Define the watermarking function
    watermarker = Watermarker(None, wt_tokenizer, 0, 0, args.k_p, watermarkingFnClass=WatermarkingFnFourier)

    processes: List[Process] = []

    if torch.cuda.device_count() == 0:
        args.verification = True
    all_args = [args]

    # If directory_path is not set to a specific subfolder, iterate through all subfolders
    to_iterate = args.directory_path == "arxiv" or args.directory_path == "tofu"
    if to_iterate:
        parent_dir = f"watermark_{args.directory_path}/main_results"
        all_args = []
        folders = sorted(
            [i for i in os.listdir(parent_dir) if "seed_" in i and "archive" not in i],
            )
        start = repeat(args.start)
        end = repeat(args.end)
        for folder in folders:
            folder = os.path.join(parent_dir, folder)
            subfolders = sorted([i for i in os.listdir(folder)], reverse=True)
            for subfolder in subfolders:
                if "watermarked" not in subfolder and torch.cuda.device_count() == 0:
                    continue
                if ("phi" in subfolder) != ("phi" in args.base_model):
                    continue
                args_ = deepcopy(args)
                args_.directory_path = os.path.join(folder, subfolder)
                if "archive" in args_.directory_path or "old" in args_.directory_path or "tmp" in args_.directory_path or "copy" in args_.directory_path:
                    continue
                if args.directory_path == "arxiv":
                    args_.dataset = "Glow-AI/WaterDrum-Ax"
                elif args.directory_path == "tofu":
                    args_.dataset = "Glow-AI/WaterDrum-TOFU"
                else:
                    raise NotImplementedError
                if subfolder.startswith("watermarked"):
                    args_.dataset_column = "forget_01"
                else:
                    args_.dataset_column = "unwatermarked_forget_01"
                all_args.append(args_)

        if args.filter is not None:
            all_args = [i for i in all_args if args.filter in i.directory_path]

    outdir_subdir = "/watermark_verify"

    ids = list(range(args.idstart, args.idend+1))

    pbar = tqdm(total=len(all_args), leave=False)

    try:
        for args_ in all_args:
            pbar.set_description(f"Processing {args_.directory_path}")
            try:
                main(args_)
            except Exception as e:
                if isinstance(e, (KeyboardInterrupt, BdbQuit)):
                    raise e
                traceback.print_exc()
                tqdm.write(f"Failed to run {args_.directory_path}")
            pbar.update(1)
    except Exception as e:
        if base_model is not None:
            base_model.finish_init()
        raise e
    if base_model is not None:
        base_model.finish_init()