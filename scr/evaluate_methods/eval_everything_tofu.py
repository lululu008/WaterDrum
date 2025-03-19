import os
import csv
import json
import yaml
from argparse import ArgumentParser, Namespace

import utils
from thirdparty.tofu.data_module import TextDatasetQA
from scr.base_data import load_tofu_train_dataset
from scr.train import load_model_and_tokenizer
from scr.evaluate_methods import eval_mia_tofu, eval_tofu
from scr.wtm_tofu_data_module import WatermarkTextDatasetQA, get_eval_dataloader

question_key = 'question'
answer_key = 'answer'
wtm_answer_key = 'answer_split'
base_answer_key = 'paraphrased_answer'
perturbed_answer_key = 'perturbed_answer'
holdout_split = 'real_authors'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--model_path', type=str, default='main_results/model.pt',
                        help='Path of the model')
    parser.add_argument('--data_config_path', type=str,
                        help='Path to dataset and split config')
    parser.add_argument('--model_config_path', type=str,
                        help='Path to model config')
    parser.add_argument('--output_dir', type=str, default='results/eval/',
                        help='Directory to save results and models')
    parser.add_argument('--retrain_output_dir', type=str, default=None,
                        help='Directory of retraining results')
    args = parser.parse_args()
    utils.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.retrain_output_dir is None:
        args.retrain_output_dir = args.output_dir
        print('retrain_output_dir is set to output_dir')

    # load data config
    with open(args.data_config_path, 'r') as f:
        data_config = Namespace(**yaml.safe_load(f))
    print('data_config:', vars(data_config))

    # load model config
    with open(args.model_config_path, 'r') as f:
        config = Namespace(**yaml.safe_load(f))
    print('model_config:', vars(config))

    # create and load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    utils.load_trainable_model(model, args.model_path)
    print(f'Loaded model from {args.model_path}')

    # load train data and splits
    train_data, forget_data, retain_data = load_tofu_train_dataset(**vars(data_config))
    get_encoded_dataset = lambda data: WatermarkTextDatasetQA(
        data,
        tokenizer,
        model_family=config.model_family,
        max_length=config.max_seq_length,
        question_key=question_key,
        answer_key=wtm_answer_key if data_config.is_wtm else answer_key
    )
    train_data = get_encoded_dataset(train_data)
    forget_data = get_encoded_dataset(forget_data)
    retain_data = get_encoded_dataset(retain_data)
    print('train_data_len:', len(train_data))
    print('forget_data_len:', len(forget_data))
    print('retain_data_len:', len(retain_data))

    # load holdout data
    holdout_data = TextDatasetQA(data_path=data_config.hf_dataset_name,
                                 split=holdout_split,
                                 tokenizer=tokenizer,
                                 model_family=config.model_family,
                                 max_length=config.generation_max_length,
                                 question_key=question_key,
                                 answer_key=answer_key)
    print('holdout_data_len:', len(holdout_data))

    # Run Membership Inference Attack
    print('=' * 10, 'Running MIA', '=' * 10)
    mia_auc, mia_log = eval_mia_tofu.eval_mia(forget_data, 
                                         retain_data, 
                                         holdout_data, 
                                         model, 
                                         tokenizer=tokenizer)
    print('MIA AUC: ', mia_auc)
    path = os.path.join(args.output_dir, 'mia.json')
    with open(path, 'w') as f:
        json.dump(mia_auc, f, indent=4)

    # # Run TOFU evaluation
    eval_split = 'forget{:02d}_perturbed'.format(int(data_config.forget_ratio * 100))     # forget_ratio = 0.05 -> eval_split = forget05_perturbed
    split_list = ['retain_perturbed', 'real_authors_perturbed', 'world_facts_perturbed', eval_split]    # original and paraphrased splits are inclusive
    question_keys = ['question', 'question', 'question', 'question']
    answer_keys = ['answer', 'answer', 'answer', 'answer']
    base_answer_keys = ['paraphrased_answer', 'answer', 'answer', 'paraphrased_answer']
    perturbed_answer_keys = ['perturbed_answer', 'perturbed_answer', 'perturbed_answer', 'perturbed_answer']
    eval_tasks = ['eval_log', 'eval_real_author_wo_options', 'eval_real_world_wo_options', 'eval_log_forget']

    aggregated_eval_logs = {}
    for split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key in \
        zip(split_list, question_keys, answer_keys, eval_tasks, base_answer_keys, perturbed_answer_keys):
        print(f'Working on eval task {eval_task} with split {split}')

        eval_loaders = get_eval_dataloader(data_config.hf_dataset_name,
                                           split,
                                           tokenizer,
                                           config.model_family,
                                           question_key,
                                           answer_key,
                                           base_answer_key,
                                           perturbed_answer_key,
                                           eval_batch_size=config.eval_batch_size,
                                           max_length=config.max_seq_length)
        eval_loader, base_eval_loader, perturb_loader = eval_loaders

        normalize_gt = False
        if 'eval_log' not in eval_task:
            normalize_gt = True
        eval_logs = eval_tofu.eval(model,
                                   tokenizer,
                                   config.model_family,
                                   eval_loader,
                                   base_eval_loader,
                                   perturb_loader,
                                   normalize_gt=normalize_gt,
                                   config=config)
        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

        path = os.path.join(args.output_dir, f'tofu_{eval_task}.json')
        with open(path, 'w') as f:
            json.dump(eval_logs, f, indent=4)

    path = os.path.join(args.output_dir, 'tofu_eval_log_aggregated.json')
    with open(path, 'w') as f:
        json.dump(aggregated_eval_logs, f, indent=4)

    retain_result = json.load(open(os.path.join(args.retrain_output_dir, 'tofu_eval_log_aggregated.json')))
    ckpt_result = json.load(open(os.path.join(args.output_dir, 'tofu_eval_log_aggregated.json')))

    model_utility = eval_tofu.get_model_utility(ckpt_result)
    forget_quality = eval_tofu.get_forget_quality(ckpt_result, retain_result)
    model_utility['Forget Quality'] = forget_quality['Forget Quality']

    # dump the model utility to a temp.csv
    path = os.path.join(args.output_dir, 'tofu.csv')
    with open(path, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, model_utility.keys())
        w.writeheader()
        w.writerow(model_utility)
