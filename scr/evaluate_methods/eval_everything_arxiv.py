import os
import csv
import json
import yaml
from argparse import ArgumentParser, Namespace
import torch
import utils
import pandas as pd
import datasets

from scr.train import load_model_and_tokenizer
from scr.base_data import load_arxiv_train_dataset
from scr.evaluate_methods import eval_mia_arxiv, eval_rouge, eval_muse
from scr.wtm_arxiv_data_module import UnwatermarkedTextDataset, WatermarkTextDataset
from scr.wtm_arxiv_data_module import custom_data_collator as arxiv_data_collator, custom_data_collator_with_indices


def load_trainable_model(model, path):
    checkpoint = torch.load(path)
    trainable_layers = list(checkpoint.keys())
    for name in trainable_layers:
        if checkpoint[name].shape[0] == 0:
            checkpoint.pop(name) 
            print(f'Discard {name} because of 0 parameters')
    trainable_layers = set(checkpoint.keys())
    all_layers = set(model.state_dict().keys())
    num_match_layers = len(trainable_layers.intersection(all_layers))
    print('Load trainable parameters for {}/{} layers'.format(num_match_layers, len(all_layers)))
    model.load_state_dict(checkpoint, strict=False)
    return

if __name__ == "__main__":
    UNLEARN_METHODS = [
        "pretrain",
        "original", 
        "retraining", 
        "finetune", 
        "ga", 
        "gdiff",
        "KL",
        "dpo",
        "tv",
        "scr_newton", 
        "scrub",
    ]
    parser = ArgumentParser() 
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument('--data_config_path', type=str,
                        help='Path to dataset and split config')
    parser.add_argument('--model_config_path', type=str,
                        help='Path to model config')
    parser.add_argument("--model_path", type=str, default="main_results/model.pt",
                        help="Path of the unlearned model")
    parser.add_argument("--unlearn_method", type=str, default="retraining",
                        choices=UNLEARN_METHODS, 
                        help="Unlearning method")            
    parser.add_argument("--run_mia", action="store_true",
                        help="Whether to run mia evaluation")
    parser.add_argument("--run_rouge", action="store_true",
                        help="Whether to run rouge evaluation")
    parser.add_argument("--run_knowmem", action="store_true",
                        help="Whether to run knowmem evaluation")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)
    
    # load data config
    with open(args.data_config_path, 'r') as f:
        data_config = Namespace(**yaml.safe_load(f))
    print('data_config:', vars(data_config))

    # load model config
    with open(args.model_config_path, 'r') as f:
        config = Namespace(**yaml.safe_load(f))
    print('model_config:', vars(config))
    
    forget_ratio = data_config.forget_ratio
    forget_pct = forget_ratio * 100
    forget_calibration_ratio = data_config.forget_calibration_ratio
    forget_calibration_pct = forget_calibration_ratio * 100
    if forget_calibration_ratio != forget_ratio:
        calibration = True
    else:
        calibration = False
    
    # create and load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    utils.load_trainable_model(model, args.model_path)
    print(f'Loaded model from {args.model_path}')
    
    # load dataset used by the correpsonding unlearning algo
    train_data, forget_data, retain_data = load_arxiv_train_dataset(**vars(data_config))
    data_collator = arxiv_data_collator
    if data_config.is_wtm:
        get_encoded_dataset = lambda data: WatermarkTextDataset(
            data,
            tokenizer,
            max_length=config.max_seq_length,
        )
    else:
        get_encoded_dataset = lambda data: UnwatermarkedTextDataset(
            data,
            tokenizer,
            max_length=config.max_seq_length,
        )
    train_data = get_encoded_dataset(train_data)
    forget_data = get_encoded_dataset(forget_data)
    retain_data = get_encoded_dataset(retain_data)
    print('train_data_len:', len(train_data))
    print('forget_data_len:', len(forget_data))
    print('retain_data_len:', len(retain_data))
    
    df = pd.read_pickle(data_config.eval_dir)
    eval_text = df["Summary"].to_list()
    holdout_data = datasets.Dataset.from_dict({'text': eval_text})
    if data_config.is_wtm:
        holdout_data = WatermarkTextDataset(data=holdout_data, tokenizer=tokenizer, 
                                        max_length=config.max_seq_length)
    else:
        holdout_data = UnwatermarkedTextDataset(data=holdout_data, tokenizer=tokenizer, 
                                        max_length=config.max_seq_length)

    out_dir = args.model_path.rsplit('/', 1)[0] + '/eval'
    os.makedirs(out_dir, exist_ok=True)
    
    
    
    # Run Membership Inference Attack
    if args.run_mia:
        mia_auc, mia_log = eval_mia_arxiv.eval_mia(forget_data, retain_data, holdout_data, model, tokenizer, arxiv_data_collator)
        print("MIA AUC: ", mia_auc)
        if calibration:
            with open(f"{out_dir}/mia_{args.unlearn_method}_{forget_calibration_pct}pct.json", "w") as f:
                json.dump(mia_auc, f)
        else:
            with open(f"{out_dir}/mia_{args.unlearn_method}.json", "w") as f:
                json.dump(mia_auc, f)
    
    
    
    # # Run ROUGE evaluation
    if args.run_rouge:
        save_folder = f"{out_dir}/{args.unlearn_method}"
        os.makedirs(save_folder, exist_ok=True)
        aggregated_eval_rouge = {}
        eval_tasks = ["eval_rouge", "eval_rouge_forget"]
        if hasattr(data_config, 'gen_path'):
            rouge = eval_rouge.eval_load(data_config.gen_path, train_data)
            if calibration:
                forget_len = int(0.05*len(rouge))
            else:
                forget_len = int(forget_ratio*len(rouge))

            forget_rouge = rouge[-forget_len:]
            retain_rouge = rouge[:-forget_len]
            
            datas = [retain_rouge, forget_rouge]
            for i, (data, eval_task) in enumerate(zip(datas, eval_tasks)):
                rouge = {idx: score for idx, score in enumerate(data)}
                result = {'rougeL_recall': rouge}
                if calibration:
                    with open(f'{save_folder}/{forget_calibration_pct}pct_{eval_task}.json', "w") as f:
                        json.dump(result, f, indent=4)
                else:
                    with open(f'{save_folder}/{eval_task}.json', "w") as f:
                        json.dump(result, f, indent=4)
                aggregated_eval_rouge[f'{eval_task}.json'] = result
        else:
            datas = [retain_data, forget_data]
            for i, (data, eval_task) in enumerate(zip(datas, eval_tasks)):
                print(f'Working on eval task {eval_task}')
            
                dataloader = torch.utils.data.DataLoader(
                    data, batch_size=config.eval_batch_size, collate_fn=custom_data_collator_with_indices
                )
                rouge = eval_rouge.eval_gen(model, tokenizer, dataloader)
                
                if calibration:
                    with open(f'{save_folder}/{forget_calibration_pct}pct_{eval_task}.json', "w") as f:
                        json.dump(rouge, f, indent=4)
                else:
                    with open(f'{save_folder}/{eval_task}.json', "w") as f:
                        json.dump(rouge, f, indent=4)
                aggregated_eval_rouge[f'{eval_task}.json'] = rouge
        
        if calibration:
            aggregated_eval_log_filename = f"{out_dir}/{args.unlearn_method}/{forget_calibration_pct}pct_eval_rouge_aggregated.json"
            # aggregated_retrain_log_filename = f"{out_dir}/retraining/{forget_pct}pct_eval_rouge_aggregated.json"
        else:
            aggregated_eval_log_filename = f"{out_dir}/{args.unlearn_method}/eval_rouge_aggregated.json"
            # aggregated_retrain_log_filename = f"{out_dir}/retraining/eval_rouge_aggregated.json"
        with open(aggregated_eval_log_filename, "w") as f:
            json.dump(aggregated_eval_rouge, f, indent=4)
            
        # retain_result = json.load(open(aggregated_retrain_log_filename))
        ckpt_result = json.load(open(aggregated_eval_log_filename))

        aggregated_results = eval_rouge.aggregate_results(ckpt_result)

        if calibration:
            save_file = f"{out_dir}/rouge_{args.unlearn_method}_{forget_calibration_pct}pct.csv"
        else:
            save_file = f"{out_dir}/rouge_{args.unlearn_method}.csv"
        # dump the model utility to a temp.csv
        print('saveing file to ', save_file)
        with open(save_file, 'w') as f:  # You will need 'wb' mode in Python 2.x
            w = csv.DictWriter(f, aggregated_results.keys())
            w.writeheader()
            w.writerow(aggregated_results)




    # Run Knowmem Evaluation
    
    if args.run_knowmem:
        assert data_config.qa_file is not None
        knwomem_output_path = f"{out_dir}/knowmem/{forget_pct}/{args.unlearn_method}"
        qa_data = pd.read_csv(data_config.qa_file)

        # Split dataset into forget and retain_qa
        if calibration:
            split_index = int(len(qa_data) * 0.95)
        else:
            split_index = int(len(qa_data) * (1 - forget_ratio))
        retain_qa = qa_data.iloc[:split_index]
        forget_qa = qa_data.iloc[split_index:]
        
        # Extract questions and answers
        retain_questions = retain_qa.iloc[:, 0].tolist()
        retain_answers = retain_qa.iloc[:, 1].tolist()
        forget_questions = forget_qa.iloc[:, 0].tolist()
        forget_answers = forget_qa.iloc[:, 1].tolist()
        
        knowmem_results = {}
        for split, questions, answers, eval_task in zip(
            ["retain", "forget"],
            [retain_questions, forget_questions],
            [retain_answers, forget_answers],
            ["eval_knowmem_retain", "eval_knowmem_forget"]
        ):
            print(f"Running KnowMem evaluation for {split} set")

            # Perform evaluation using KnowMem
            knowmem_score, knowmem_log = eval_muse.eval(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                answers=answers,
                max_new_tokens=32, # Modify: max number of tokens
                batch_size=config.eval_batch_size
            )
            
            # Save evaluation logs
            save_path = os.path.join(knwomem_output_path, f"{eval_task}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(knowmem_log, f, indent=4)

            knowmem_results[eval_task] = knowmem_score

        # Save aggregated KnowMem results
        aggregated_results = {
            "KnowMem Forget": knowmem_results["eval_knowmem_forget"],
            "KnowMem Retain": knowmem_results["eval_knowmem_retain"],
        }
        knwomem_output_file_aggregated = f"{knwomem_output_path}/aggregated.json"
        with open(knwomem_output_file_aggregated, "w") as f:
            json.dump(aggregated_results, f, indent=4)
    