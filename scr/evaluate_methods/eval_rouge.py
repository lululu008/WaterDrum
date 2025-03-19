import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import random
import datasets

def load_gen(data_path):
    gen_result = datasets.load_from_disk(data_path)
    full_text = gen_result['watermarked']
    input_strings = gen_result['prompt']
    output_strings = [sublist[0] for sublist in gen_result['generation']]
    if full_text.startswith(input_strings):
        groundtruth = full_text[len(input_strings):]
    else:
        # If `input_strings` is not a prefix, handle this accordingly
        print("The input_strings is not a prefix of full_text.")
        groundtruth = full_text
    return input_strings, output_strings, groundtruth

def run_generation(input_ids, model, tokenizer, max_new_tokens):
    model.eval()
    input = input_ids[:, :100]
    input_strings = tokenizer.batch_decode(input, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    gt = input_ids[:, 100:200]
    gt = tokenizer.batch_decode(gt, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    
    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    # now generate
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return input_strings, strs, gt

def eval_gen(
    model, tokenizer,
    data, 
    max_new_tokens: int = 100,
    max_samples : int = 512
):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []
    num_samples = 0
    
    for sample in tqdm(data):
        input_ids, attention_mask, idx = sample

        all_indices.extend(idx.cpu().numpy().tolist())
        with torch.no_grad():
            input_string, gen_output, gt = run_generation(input_ids, model, tokenizer=tokenizer, max_new_tokens=max_new_tokens) 
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
        num_samples += len(input_ids)
        
    for gen, gt, idx in zip(gen_outputs, ground_truths, all_indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

def eval_load(data_path):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []
    num_samples = 0
    
    for sample in tqdm(data):
        input_ids, attention_mask, idx = sample

        all_indices.extend(idx.cpu().numpy().tolist())
        with torch.no_grad():
            input_string, gen_output, gt = load_gen(data_path)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
        num_samples += len(input_ids)
        
    for gen, gt, idx in zip(gen_outputs, ground_truths, all_indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}


def aggregate_results(eval_result_dict):
    eval_task_dict = {
        'eval_rouge.json': 'Retain',
        'eval_rouge_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[metric + ' ' + eval_task_dict[eval_task]] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'ROUGE {eval_task_dict[k]}'] = avg_rouge
    return output_result