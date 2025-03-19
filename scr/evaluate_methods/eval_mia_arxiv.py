from argparse import ArgumentParser, Namespace
import sys
import os
import json
# Add the submodule path to the system path
sys.path.append(os.path.join(os.getcwd(), 'tofu'))

from typing import List, Dict
import torch
from tqdm import tqdm
import zlib
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model

from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve
import datasets
from scr.train import *


def compute_ppl(text, model, tokenizer, device='cuda'):
    model.eval()
    input_ids = text[0].to(device)
    labels = text[1].to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    ppl = torch.exp(loss).item()
    return ppl, all_prob, loss.item()


def inference(text, model, tokenizer) -> Dict:
    pred = {}

    _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    # _, _, p_lower_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    # decoded_text = tokenizer.decode(text[0][0], skip_special_tokens=True)
    # zlib_entropy = len(zlib.compress(bytes(decoded_text, 'utf-8')))

    pred["PPL"] = float(p1_likelihood)
    # pred["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
    # pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)

    # print("PPL", pred["PPL"])
    # print("PPL/lower", pred["PPL/lower"])
    # print("PPL/zlib", pred["PPL/zlib"])

    # min-k prob
    # for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    for ratio in [0.4]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())

    return pred


# def eval_data(data, model, tokenizer):
#     out = []
#     for qa in tqdm(data):
#         out.append({'text': qa} | inference(qa, model, tokenizer))
#         import pdb; pdb.set_trace()
#     return out
def eval_data(data, model, tokenizer, data_collator=None):
    out = {}
    for sample in tqdm(data):
        if data_collator is not None:
            qa = data_collator([sample])
        else:
            qa = sample
        result = {'text': qa} | inference(qa, model, tokenizer)
        if not out:  # Initialize output dictionary based on the first result
            out = {key: [] for key in result.keys()}
        for key, value in result.items():
            out[key].append(value)
    return out


def sweep(ppl, y):
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    
    # roc_auc = get_auc(fpr, tpr)
    
    # import matplotlib.pyplot as plt

    # # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Min-40%_AUC_pretrain.png', format='png', bbox_inches='tight')

    return fpr, tpr, get_auc(fpr, tpr), acc


def eval_mia(
    forget_data,
    retain_data,
    holdout_data,
    model, tokenizer,
    data_collator=None
):
    log = {}
    print("Evaluating on the forget set...")
    log['forget'] = eval_data(forget_data, model, tokenizer, data_collator)
    print("Evaluating on the retain set...")
    log['retain'] = eval_data(retain_data, model, tokenizer, data_collator)
    print("Evaluating on the holdout set...")
    log['holdout'] = eval_data(holdout_data, model, tokenizer, data_collator)

    auc = {}
    ppl_types = list(log['forget'].keys())
    ppl_types.remove('text')
    
        
    # ppl_nonmember = log['holdout']['PPL']
    # ppl_member = log['forget']['PPL']

    # min_size = min(len(ppl_nonmember), len(ppl_member))
    # # Shuffle and select subsets of equal size
    # if len(ppl_nonmember) > min_size:
    #     ppl_nonmember = np.random.choice(ppl_nonmember, min_size, replace=False).tolist()
    # else:
    #     ppl_member = np.random.choice(ppl_member, min_size, replace=False).tolist()
                    
    # import matplotlib.pyplot as plt
    # # Creating the plot
    # plt.figure(figsize=(10, 6))
    # plt.hist(ppl_nonmember, bins=30, alpha=0.5, label='Non-member')
    # plt.hist(ppl_member, bins=30, alpha=0.5, label='Member')

    # # Adding titles and labels
    # plt.title('Loss Distribution')
    # plt.xlabel('Loss Value')
    # plt.ylabel('Frequency')
    # plt.legend(loc='upper right')

    # # Save plot to a file
    # plt.savefig('PPL_distribution_pretrain.png', format='png', bbox_inches='tight')
    
    for split0 in ['forget']:
        for split1 in ['holdout']:
            log0, log1 = log[split0], log[split1]
            for ppl_type in ppl_types:
                ppl_nonmember = log0[ppl_type]
                ppl_member = log1[ppl_type]
                
                min_size = min(len(ppl_nonmember), len(ppl_member))
                # Shuffle and select subsets of equal size
                if len(ppl_nonmember) > min_size:
                    ppl_nonmember = np.random.choice(ppl_nonmember, min_size, replace=False).tolist()
                else:
                    ppl_member = np.random.choice(ppl_member, min_size, replace=False).tolist()
                
                ppl = np.array(ppl_nonmember + ppl_member)
                y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                
                # combined = np.column_stack((ppl, y))
                # np.random.shuffle(combined)
                # shuffled_losses = combined[:, 0].reshape(-1, 1)  # Reshape to 2D array
                # shuffled_members = combined[:, 1]  # This is already a 1D array
                
                # from sklearn import linear_model, model_selection
                # scoring = {
                #     'accuracy': 'accuracy',
                #     'roc_auc': 'roc_auc'
                # }
                # attack_model = linear_model.LogisticRegression()
                # cv = model_selection.StratifiedShuffleSplit(n_splits=5, random_state=42)
                # results =  model_selection.cross_validate(
                #         attack_model, shuffled_losses, shuffled_members, cv=cv, scoring=scoring, return_train_score=True
                #     )
                
                # result_dict = {metric: results['test_' + metric].tolist() for metric in scoring}
                # result_dict['train_accuracy'] = results['train_accuracy'].tolist() # Optionally save train scores if needed
                # result_dict['train_roc_auc'] = results['train_roc_auc'].tolist()  # Optionally save train scores if needed
                # auc_score = results['test_' + 'roc_auc']
                
                _, _, auc_score, _ = sweep(ppl, y)
                auc[f"{split0}_{split1}_{ppl_type}"] = np.mean(auc_score)
    return auc, log

if __name__ == "__main__":
    UNLEARN_METHODS = [
        "original", 
        "retraining", 
        "finetune", 
        "ga", 
        "gdiff",
        "KL",
        "dpo",
        "scr_newton", 
        "scrub",
    ]
    parser = ArgumentParser() 
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--unlearned_model_path", type=str, default="main_results/model.pt",
                        help="Path of the unlearned model")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Name or path to the model locally or on HuggingFace Hub")
    parser.add_argument("--unlearn_method", type=str, default="retraining",
                        choices=UNLEARN_METHODS, 
                        help="Unlearning method")            
    parser.add_argument("--remove_pct", type=str, default="1",
                        help="Removal percentage")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    privleak_auc_key = 'forget_holdout_Min-40%'
    
    out_dir = args.unlearned_model_path.rsplit('/', 1)[0] + '/eval'
    mia_path = f"{out_dir}/mia_{args.unlearn_method}.json"
    retrain_path = f"{out_dir}/mia_retraining.json"

    with open(mia_path, "r") as f:
        auc = json.load(f)
    with open(retrain_path, "r") as f:
        AUC_RETRAIN = json.load(f)
        
    privleak = (auc[privleak_auc_key] - AUC_RETRAIN[privleak_auc_key]) / AUC_RETRAIN[privleak_auc_key] * 100
    privleak = {'privleak': privleak}
    
    with open(f"{out_dir}/privleak_{args.unlearn_method}.json", "a") as f:
        json.dump(privleak, f)
    
    # with open(f"config/train.json", "r") as f:
    #     config = json.load(f)
    #     print("train_config:", config)
    #     cfg = Namespace(**config)    # so that can access attributes through . operation
        
    # forget_split = "forget" + args.remove_pct.zfill(2)
    # retain_split = "retain" + str(100 - int(args.remove_pct)).zfill(2)

    # # Load pretrained model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, 
    #                                             torch_dtype=torch.bfloat16, 
    #                                             device_map='auto',
    #                                             )
    # # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    # model.generation_config.do_sample = True 
    # model.gradient_checkpointing_enable()

    # # Load peft configuration and peft model
    # config = LoraConfig(
    #     r=cfg.lora_r,
    #     lora_alpha=cfg.lora_alpha, 
    #     target_modules=find_all_linear_names(model), 
    #     lora_dropout=cfg.lora_dropout,
    #     bias=cfg.lora_bias, 
    #     task_type=cfg.lora_task_type
    # )
    # model = get_peft_model(model, config)
    # model.enable_input_require_grads()

    # model.load_state_dict(torch.load(args.unlearned_model_path), strict=False)

    # if "llama" in args.pretrained_model_name_or_path.lower():
    #     model_family = "llama2-7b"
    # else:
    #     raise NotImplementedError
    # forget_data = datasets.load_dataset("locuslab/TOFU", forget_split)["train"]
    # forget_data = WatermarkTextDatasetQA(data=forget_data, tokenizer=tokenizer, model_family=model_family,
    #                             question_key='question', answer_key='answer')
    # retain_data = datasets.load_dataset("locuslab/TOFU", retain_split)["train"]
    # retain_data = WatermarkTextDatasetQA(data=retain_data, tokenizer=tokenizer, model_family=model_family,
    #                             question_key='question', answer_key='answer')
    # holdout_data = datasets.load_dataset("locuslab/TOFU", 'real_authors')["train"]
    # holdout_data = WatermarkTextDatasetQA(data=holdout_data, tokenizer=tokenizer, model_family=model_family,
    #                             question_key='question', answer_key='answer')

    # auc, log = eval_mia(forget_data, retain_data, holdout_data, model, tokenizer)
    # print("AUC: ", auc)
    
    # out_dir = args.unlearned_model_path.rsplit('/', 1)[0] + '/eval'
    # os.makedirs(out_dir, exist_ok=True)

    # # Save the results
    # with open(f"{out_dir}/mia_{args.unlearn_method}.json", "w") as f:
    #     json.dump(auc, f)
    