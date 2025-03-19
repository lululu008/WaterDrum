import os

import torch
import zlib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve


def compute_ppl(text, model, tokenizer, device='cuda'):
    '''
    Compute perplexity of a given text 
    '''
    input_ids = text[0].unsqueeze(0).to(device)
    labels = text[1].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]

    log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        log_prob = log_probabilities[0, i, token_id].item()
        all_prob.append(log_prob)

    ppl = torch.exp(loss).item()

    return ppl, all_prob, loss.item()


def inference(text , model, tokenizer) -> dict:
    model.eval()
    pred = {}

    _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    decoded_text = tokenizer.decode(text[0], skip_special_tokens=True)
    zlib_entropy = len(zlib.compress(bytes(decoded_text, 'utf-8')))

    pred['PPL'] = float(p1_likelihood)
    pred['PPL/zlib'] = float(p1_likelihood / zlib_entropy)
    print('PPL', pred['PPL'])
    print('PPL/zlib', pred['PPL/zlib'])

    # min-k prob
    for ratio in [0.4]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob, axis=-1)[:k_length]
        pred[f'Min-{int(ratio*100)}%'] = float(-np.mean(topk_prob).item())

    return pred


def eval_data(data, model, tokenizer):
    out = {}
    for qa in tqdm(data):
        result = {'text': qa} | inference(qa, model, tokenizer)
        if not out:  # Initialize output dictionary based on the first result
            out = {key: [] for key in result.keys()}
        for key, value in result.items():
            out[key].append(value)
    return out


def sweep(ppl, y):
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, get_auc(fpr, tpr), acc


def eval_mia(
    forget_data,
    retain_data,
    holdout_data,
    model, 
    tokenizer,
    cache_dir: str = None,
):
    log = {}
    print('Evaluating on the forget set...')
    log['forget'] = eval_data(forget_data, model, tokenizer)
    print('Evaluating on the retain set...')
    log['retain'] = eval_data(retain_data, model, tokenizer)
    print('Evaluating on the holdout set...')
    log['holdout'] = eval_data(holdout_data, model, tokenizer)

    auc = {}
    ppl_types = list(log['forget'].keys())
    ppl_types.remove('text')
    for split0 in ['forget', 'retain', 'holdout']:
        for split1 in ['forget', 'retain', 'holdout']:
            log0, log1 = log[split0], log[split1]
            for ppl_type in ppl_types:
                ppl_nonmember = log0[ppl_type]
                ppl_member = log1[ppl_type]
                ppl = np.array(ppl_nonmember + ppl_member)
                y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                _, _, auc_score, _ = sweep(ppl, y)
                auc[f'{split0}_{split1}_{ppl_type}'] = auc_score

    return auc, log