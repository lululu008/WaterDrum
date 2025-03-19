import datasets
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from thirdparty.tofu.data_module import TextDatasetQA, convert_raw_data_to_model_format, custom_data_collator_with_indices
from thirdparty.tofu.utils import get_model_identifiers_from_yaml


def get_eval_dataloader(
    data_path: str,
    split: str,
    tokenizer = None,
    model_family: str = None,
    question_key: str = None,
    answer_key: str = None,
    base_answer_key: str = None,
    perturbed_answer_key: str = None,
    eval_batch_size: int = 64,
    max_length: int = 128,
):
    init_eval = lambda q_key, a_key: TextDatasetQA(
        data_path=data_path,
        split=split,
        tokenizer=tokenizer,
        model_family=model_family,
        max_length=max_length,
        question_key=q_key,
        answer_key=a_key,
    )

    eval_data = init_eval(question_key, answer_key)
    base_eval_data = init_eval(question_key, base_answer_key)
    perturb_eval_data = init_eval(question_key, perturbed_answer_key)

    eval_loader = DataLoader(eval_data,
                             batch_size=eval_batch_size,
                             collate_fn=custom_data_collator_with_indices)
    base_eval_loader = DataLoader(base_eval_data,
                                  batch_size=eval_batch_size//4,
                                  collate_fn=custom_data_collator_with_indices)
    perturb_loader = DataLoader(perturb_eval_data,
                                batch_size=eval_batch_size//4,
                                collate_fn=custom_data_collator_with_indices)

    return eval_loader, base_eval_loader, perturb_loader


class WatermarkTextDatasetQA(TextDatasetQA):

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
    def __init__(
        self,
        forget_data=None,
        retain_data=None,
        tokenizer=None,
        model_family=None,
        max_length=512,
        loss_type='idk',
    ):
        super(WatermarkTextForgetDatasetQA, self).__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.tokenizer = tokenizer
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.max_length = max_length
        self.answer_key = 'answer_split'
        self.loss_type = loss_type

        if self.loss_type == 'idk':
            self.split1, self.split2 = 'idk', 'retain'
            self.idontknowfile = 'thirdparty/tofu/config/idontknow.jsonl'
            self.idk = open(self.idontknowfile, 'r').readlines()
        else:
            self.split1, self.split2 = 'forget', 'retain'

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == 'retain' else self.forget_data
            idx = idx if data_type != 'retain' else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx][self.answer_key]

            if data_type == 'idk':
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class WatermarkTextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path=None, data=None, tokenizer=None, model_family=None, max_length=512, split = 'forget10', ):
        super(WatermarkTextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if data is not None:
            forget_pct = int(split.replace('forget', ''))
            retain_pct = 100 - forget_pct
            forget_num_rows = int(len(data) * forget_pct / 100)
            retain_num_rows = int(len(data) * retain_pct / 100)
            self.forget_data = data.select(range(len(data) - forget_num_rows, len(data)))
            self.retain_data = data.select(range(0, retain_num_rows))
            self.answer_key = 'answer_split'
        else:
            self.forget_data = datasets.load_dataset(data_path, split)['train']
            retain_split = 'retain' + str(100 - int(split.replace('forget', ''))).zfill(2)
            self.retain_data = datasets.load_dataset(data_path, retain_split)['train']
            self.answer_key = 'answer'
        self.idontknowfile = 'tofu/data/idontknow.jsonl'
        self.idk = open(self.idontknowfile, 'r').readlines()
        self.model_configs = get_model_identifiers_from_yaml(model_family)


    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ['idk', 'forget', 'retain']:
            data = self.forget_data if data_type != 'retain' else self.retain_data
            idx = idx if data_type != 'retain' else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)

            question = data[idx]['question']

            if data_type != 'idk':
                answer = data[idx][self.answer_key]
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets
