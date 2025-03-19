import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
import json


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Chatbot with LLaMA-2 model')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the fine-tuned model directory')
    parser.add_argument('--tokenizer_dir', type=str, required=True, help='Path to the tokenizer directory')
    return parser.parse_args()

def main(model_dir, tokenizer_dir):

    with open(f"config/train.json", "r") as f:
        config = json.load(f)
        print("train_config:", config)
        cfg = argparse.Namespace(**config)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)

    model.generation_config.do_sample = True 
    model.gradient_checkpointing_enable()
    
    # Load peft configuration and peft model
    config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias, 
        task_type=cfg.lora_task_type
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()


    print(f"Loading trainable parameters from {args.model_dir}")
    model.load_state_dict(torch.load(args.model_dir), strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token

    def chat(input_text):
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda") 
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    print("Chatbot is ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat(user_input)
        
        print("====new response====")
        print(f"User: {user_input}")
        print(f"Model: {response}")
        print("====end response====")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.model_dir, args.tokenizer_dir)
