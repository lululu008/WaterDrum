from pathlib import Path
import torch
from typing import *
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

def get_trainable_parameters(model):
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param
    return state_dict 

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

def load_model(
    model_dir: str,
    device: str = 'cuda'
) -> AutoModelForCausalLM:
    
    if Path(model_dir).suffix == '.pt':
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            torch_dtype=torch.bfloat16
        )
        
        peft_config = LoraConfig(
                            r=8,
                            lora_alpha=32, 
                            target_modules=find_all_linear_names(model), 
                            lora_dropout=0.05,
                            bias="none",
                            task_type=TaskType.CAUSAL_LM,
                            )
        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load(model_dir), strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()

    model.to(device)
    return model


def compare(model1, model2) -> bool:
    """Compares two models.

    Args:
        model1 (_type_): _description_
        model2 (_type_): _description_

    Returns:
        bool: _description_
    """
    dict1, dict2 = model1.state_dict(), model2.state_dict()
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True


def unlearn(
    model,
    forget_model,
    # model_dir: str,
    # some_pt_model_dir=None,
    # some_ft_model_dir=None,
    # some_pt_model_dir: str | None = None,
    # some_ft_model_dir: str | None = None,
    alpha: float = 1.0
):
    # if some_pt_model_dir is None or some_ft_model_dir is None:
    #     raise ValueError("Task vector (ilharco2023) requires some pretrained & finetuned models!")

    task_vector = TaskVector(
        pretrained_state_dict=model.state_dict(),
        finetuned_state_dict=forget_model.state_dict(),
        # pretrained_state_dict=load_model(some_pt_model_dir).state_dict(),
        # finetuned_state_dict=load_model(some_ft_model_dir).state_dict()
    )

    if not task_vector.is_nonzero():
        raise ValueError("Zero task vector encountered!")

    neg_task_vector = -task_vector

    # model = load_model(model_dir)
    neg_task_vector.apply_to(pretrained_model=model, scaling_coef=alpha, in_place=True)
    # del model
    # new_model = load_model(model_dir)
    # model.load_state_dict(new_state_dict, strict=False)

    return model


class TaskVector():
    def __init__(self,
                 pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None,
                 pretrained_state_dict=None, finetuned_state_dict=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                (pretrained_checkpoint is not None and finetuned_checkpoint is not None)
                or
                (pretrained_state_dict is not None and finetuned_state_dict is not None)
            )
            with torch.no_grad():
                if pretrained_state_dict is None:
                    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                if finetuned_state_dict is None:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def is_nonzero(self):
        return any([(self.vector[key] != 0).any() for key in self.vector])

    def apply_to(self, pretrained_model, scaling_coef=1.0, in_place=False):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        if in_place:
            pretrained_model.load_state_dict(new_state_dict, strict=False)
        return new_state_dict
