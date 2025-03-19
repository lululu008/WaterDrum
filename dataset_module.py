import datasets
import torch
import numpy as np

from torch.utils.data import Dataset
from tofu.data_module import TextDatasetQA, custom_data_collator_with_indices
from tofu.utils import get_model_identifiers_from_yaml, add_dataset_index
from tofu.data_module import convert_raw_data_to_model_format

def get_eval_dataloader(cfg, eval_task, tokenizer, folder, split, model_family, question_key, answer_key, base_answer_key, perturbed_answer_key):
    eval_dataset = TextDatasetQA(folder, tokenizer=tokenizer, model_family=model_family, max_length=cfg.generation_max_length, 
                                split=split, question_key=question_key, answer_key=answer_key)
    base_eval_dataset = TextDatasetQA(folder, tokenizer=tokenizer, model_family=model_family, max_length=cfg.generation_max_length, 
                                split=split, question_key=question_key, answer_key=base_answer_key)
    perturb_eval_dataset = TextDatasetQA(folder, tokenizer=tokenizer, model_family=model_family, max_length=cfg.generation_max_length, 
                                split=split, question_key=question_key, answer_key=perturbed_answer_key)

    batch_size = cfg.eval_batch_size
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=custom_data_collator_with_indices
    )
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_eval_dataset, batch_size=batch_size//4, collate_fn=custom_data_collator_with_indices
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_eval_dataset, batch_size=batch_size//4, collate_fn=custom_data_collator_with_indices
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

class WatermarkTextDatasetQA(TextDatasetQA):
    def __init__(self, data_path=None, data=None, tokenizer=None, model_family=None, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        if data is not None:
            self.data = data
        else:
            self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key
        
        self.data = self._convert_index(self.data)
    
    def _convert_numpy_to_python(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_numpy_to_python(i) for i in obj)
        return obj

    def _convert_index(self, data):
        # Convert numpy integers to Python integers in the data
        return [self._convert_numpy_to_python(entry) for entry in data]



class WatermarkTextForgetDatasetQA(Dataset):
    def __init__(self, data_path=None, data=None, tokenizer=None, model_family=None, 
    max_length=512, split = "forget10", loss_type="idk", duplicate=None):
        super(WatermarkTextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if data is not None:
            forget_pct = int(split.replace("forget", ""))
            retain_pct = 100 - forget_pct
            forget_num_rows = int(len(data) * forget_pct / 100)
            retain_num_rows = int(len(data) * retain_pct / 100)
            self.forget_data = data.select(range(len(data) - forget_num_rows, len(data)))
            self.retain_data = data.select(range(0, retain_num_rows))
            if duplicate is not None:
                self.retain_data = datasets.concatenate_datasets([self.retain_data, duplicate])
            self.answer_key = 'answer_split'
        else: # load unwatermarked dataset
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
            if duplicate is not None:
                self.retain_data = datasets.concatenate_datasets([self.retain_data, duplicate])
            self.answer_key = 'answer'
        
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "tofu/data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"
            
    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx][self.answer_key]

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets