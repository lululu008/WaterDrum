import os
import yaml
from argparse import ArgumentParser, Namespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

import utils
from thirdparty.tofu.dataloader import CustomTrainer
from thirdparty.tofu.data_module import TextDatasetQA
from thirdparty.tofu.data_module import custom_data_collator as tofu_data_collator
from scr.base_data import load_tofu_train_dataset, load_arxiv_train_dataset
from scr.wtm_tofu_data_module import WatermarkTextDatasetQA
from scr.wtm_arxiv_data_module import WatermarkTextDataset, UnwatermarkedTextDataset
from scr.wtm_arxiv_data_module import custom_data_collator as arxiv_data_collator


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:      # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_model_and_tokenizer(config):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load model
    model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name_or_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map='auto',
                                                 #  attn_implementation='flash_attention_2',
                                                 )
    # tofu authors' suggestion: hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    model.gradient_checkpointing_enable()

    # load peft configuration and peft model
    if 'lora' in config:
        lora_config = LoraConfig(
            target_modules=find_all_linear_names(model),
            **config.lora,
        )
        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    return model, tokenizer


def load_training_arguments(args, config, num_train_samples):
    num_devices = 1
    max_steps = int(config.num_epochs * num_train_samples)
    max_steps = max_steps // (config.train_batch_size * config.gradient_accumulation_steps * num_devices)

    training_args = TrainingArguments(
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=max(1, max_steps//2),
        max_steps=max_steps,
        learning_rate=config.learning_rate,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, max_steps//20),
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        optim='paged_adamw_32bit',
        save_steps=max_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        evaluation_strategy='no',
        weight_decay=config.weight_decay,
        seed=args.seed,
    )

    return training_args


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--dataset_name', type=str, choices=['tofu', 'arxiv'],
                        help='Dataset name')
    parser.add_argument('--data_config_path', type=str,
                        help='Path to dataset and split config')
    parser.add_argument('--train_config_path', type=str,
                        help='Path to training config')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save results and models')
    args = parser.parse_args()
    utils.set_seed(args.seed)

    # load data config
    with open(args.data_config_path, 'r') as f:
        data_config = Namespace(**yaml.safe_load(f))
    print('data_config:', vars(data_config))

    # load training config
    with open(args.train_config_path, 'r') as f:
        config = Namespace(**yaml.safe_load(f))
    print('train_config:', vars(config))

    # load tokenizer and model
    model, tokenizer = load_model_and_tokenizer(config)

    # load dataset
    if args.dataset_name == 'tofu':
        train_data, _, _ = load_tofu_train_dataset(**vars(data_config))
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
        data_collator = tofu_data_collator
    elif args.dataset_name == 'arxiv':
        train_data, _, _ = load_arxiv_train_dataset(**vars(data_config))
        if data_config.is_wtm:
            train_data = WatermarkTextDataset(train_data,
                                              tokenizer,
                                              max_length=config.max_seq_length)
        else:
            train_data = UnwatermarkedTextDataset(train_data, 
                                                  tokenizer,
                                                  max_length=config.max_seq_length)
        data_collator = arxiv_data_collator
    else:
        raise NotImplementedError
    print('num_train_rows:', len(train_data))

    # create trainer
    training_args = load_training_arguments(args, config, num_train_samples=len(train_data))
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=training_args,
        data_collator=data_collator,
    )
    model.config.use_cache = False  # disable KV cache during finetuning

    # start training
    trainer.train()

    # save model (trainable parameters only)
    path = os.path.join(args.output_dir, 'original.pt')
    utils.save_trainable_model(model, path)
    print(f'Saved finetuned model (trainable parameters only) to {path}')
