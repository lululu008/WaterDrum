from argparse import ArgumentParser, Namespace
import os
import yaml

import utils
from thirdparty.tofu.dataloader import CustomTrainer, CustomTrainerForgetting
from thirdparty.tofu.data_module import TextDatasetQA, TextForgetDatasetQA
from thirdparty.tofu.data_module import custom_data_collator as tofu_data_collator
from thirdparty.tofu.dataloader import custom_data_collator_forget as tofu_data_collator_forget
from scr.base_data import load_tofu_train_dataset, load_arxiv_train_dataset
from scr.wtm_tofu_data_module import WatermarkTextDatasetQA, WatermarkTextForgetDatasetQA
from scr.wtm_arxiv_data_module import WatermarkTextDataset, UnwatermarkedTextDataset, WatermarkTextForgetDataset, UnwatermarkedTextForgetDataset
from scr.wtm_arxiv_data_module import custom_data_collator as arxiv_data_collator
from scr.wtm_arxiv_data_module import custom_data_collator_forget as arxiv_data_collator_forget
from scr.train import load_model_and_tokenizer, load_training_arguments
from scr.train import load_model_and_tokenizer
from scr.unlearning import task_vector, scrub


UNLEARN_METHODS = [
    'original', 'retraining', 'finetune',
    'grad_ascent', 'grad_diff', 'KL',
    'dpo', 'tv', 'scrub',
]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--dataset_name', type=str, choices=['tofu', 'arxiv'],
                        help='Dataset name')
    parser.add_argument('--data_config_path', type=str,
                        help='Path to dataset and split config')
    parser.add_argument('--unlearn_config_path', type=str,
                        help='Path to unlearning config')
    parser.add_argument('--orig_model_path', type=str,
                        help='Path to model that contains trainable parameters (peft parameters)')
    parser.add_argument('--unlearn_method', type=str, choices=UNLEARN_METHODS,
                        help='Unlearning method')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save results and models')
    args = parser.parse_args()
    utils.set_seed(args.seed)
    print(f'Unlearning method: {args.unlearn_method}')

    # load data config
    with open(args.data_config_path, 'r') as f:
        data_config = Namespace(**yaml.safe_load(f))
    print('data_config:', vars(data_config))

    # load unlearning config
    with open(args.unlearn_config_path, 'r') as f:
        config = yaml.safe_load(f)
        if args.unlearn_method in config:   # update method-specific hyperparameters
            config.update(config[args.unlearn_method])
        config = Namespace(**config)

    print('unlearn_config:', vars(config))

    # create and load finetuned model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    if args.unlearn_method != 'retraining':
        print(f'Loading trainable parameters from {args.orig_model_path}')
        utils.load_trainable_model(model, args.orig_model_path)

    # load dataset and splits
    if args.dataset_name == 'tofu':
        train_data, forget_data, retain_data = load_tofu_train_dataset(**vars(data_config))
        data_collator = tofu_data_collator
    elif args.dataset_name == 'arxiv':
        train_data, forget_data, retain_data = load_arxiv_train_dataset(**vars(data_config))
        data_collator = arxiv_data_collator
    else:
        raise NotImplementedError
    print('num_train_rows:', len(train_data))
    print('num_forget_rows:', len(forget_data))
    print('num_retain_rows:', len(retain_data))

    # create necessary datasets for the unlearner
    if args.unlearn_method in ('retraining', 'finetune', 'scr_newton', 'tv'):
        train_data = forget_data if args.unlearn_method == 'tv' else retain_data
        if args.dataset_name == 'tofu':
            data_collator = tofu_data_collator
            if data_config.is_wtm:
                train_data = WatermarkTextDatasetQA(train_data,
                                                    tokenizer,
                                                    model_family=config.model_family,
                                                    max_length=config.max_seq_length,
                                                    question_key='question',
                                                    answer_key='answer_split')
            else:
                train_data = TextDatasetQA(train_data,
                                           tokenizer,
                                           model_family=config.model_family,
                                           max_length=config.max_seq_length,
                                           question_key='question',
                                           answer_key='answer')
        elif args.dataset_name == 'arxiv':
            data_collator = arxiv_data_collator
            if data_config.is_wtm:
                train_data = WatermarkTextDataset(train_data,
                                                  tokenizer,
                                                  max_length=config.max_seq_length)
            else:
                train_data = UnwatermarkedTextDataset(train_data,
                                                  tokenizer,
                                                  max_length=config.max_seq_length)
    else:
        oracle_model = None
        forget_loss = 'idk' if args.unlearn_method == 'dpo' else args.unlearn_method
        if args.unlearn_method in ('KL', 'scrub'):
            oracle_model, _ = load_model_and_tokenizer(config)
            oracle_model.load_state_dict(model.state_dict())    # assign copy of model's parameters to oracle_model
        if args.dataset_name == 'tofu':
            data_collator = tofu_data_collator_forget
            if data_config.is_wtm:
                train_data = WatermarkTextForgetDatasetQA(
                    forget_data,
                    retain_data,
                    tokenizer,
                    model_family=config.model_family,
                    max_length=config.max_seq_length,
                    loss_type=forget_loss,
                )
            else:
                train_data = TextForgetDatasetQA(
                    forget_data,
                    retain_data,
                    tokenizer,
                    model_family=config.model_family,
                    max_length=config.max_seq_length,
                    loss_type=forget_loss,
                )
        elif args.dataset_name == 'arxiv':
            data_collator = arxiv_data_collator_forget
            if data_config.is_wtm:
                train_data = WatermarkTextForgetDataset(
                    forget_data,
                    retain_data,
                    tokenizer,
                    max_length=config.max_seq_length,
                    loss_type=forget_loss,
                )
            else:
                train_data = UnwatermarkedTextForgetDataset(
                    forget_data,
                    retain_data,
                    tokenizer,
                    max_length=config.max_seq_length,
                    loss_type=forget_loss,
                )

    # start unlearning
    training_args = load_training_arguments(args, config, num_train_samples=len(train_data))
    model.config.use_cache = False      # disable KV cache
    if args.unlearn_method in ('retraining', 'finetune', 'tv'):
        trainer = CustomTrainer(
            model=model,
            train_dataset=train_data,
            args=training_args,
            data_collator=data_collator,
        )
        trainer.train()

        if args.unlearn_method == 'tv':
            pt_model, _ = load_model_and_tokenizer(config)
            utils.load_trainable_model(pt_model, args.orig_model_path)  # load the original model to unlearn
            model = task_vector.unlearn(pt_model, forget_model=model, alpha=5.0)

    elif args.unlearn_method == 'scr_newton':
        trainer_init_kwargs = Namespace(
            model=model,
            train_dataset=train_data,
            args=training_args,
            data_collator=data_collator,
        )
        scr_newton.unlearn(model, train_data, scr_newton_cfg, trainer_init_kwargs)

    else:
        trainer = CustomTrainerForgetting(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            compute_metrics=None,   # the callback for computing metrics, None in this case since you're doing it in your callback
            args=training_args,
            data_collator=data_collator,
            oracle_model=oracle_model,
            forget_loss=forget_loss,
            eval_cfg=None,      # turn off evaluate during unlearning
        )
        if args.unlearn_method == 'scrub':
            scrub.unlearn(trainer, config)
        elif args.unlearn_method == "scr_newton":
            scr_newton.unlearn(trainer, config)
        else:
            trainer.train()

    # save model
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f'{args.unlearn_method}.pt')
    utils.save_trainable_model(model, path)
    print(f'Saved unlearned model to {path}')
