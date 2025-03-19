import torch
import numpy as np
from torch.utils.data import Dataset

def custom_data_collator(samples):
    input_ids = torch.stack([s[0] for s in samples])
    attention_mask = torch.stack([s[1] for s in samples])
    labels = input_ids.clone()
    return (input_ids, labels, attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    attention_mask = [s[1] for s in samples]
    indices = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(attention_mask), torch.stack(indices)

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]

    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = torch.stack([s[0] for s in data])
        attention_mask = torch.stack([s[1] for s in data])
        labels = input_ids.clone()
        rets.append((input_ids, labels, attention_mask))
        
    return rets


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)

    return dataset


def convert_raw_data_to_model_format(tokenizer, max_length, text):
    encoded = tokenizer(
        text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    return torch.tensor(pad_input_ids),torch.tensor(pad_attention_mask)


class WatermarkTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        super(WatermarkTextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

        self.data = add_dataset_index(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        indices = self.data[idx]['index']

        pad_input_ids, pad_attention_mask = convert_raw_data_to_model_format(self.tokenizer, self.max_length, text)

        return pad_input_ids, pad_attention_mask, torch.tensor(indices)


class WatermarkTextForgetDataset(Dataset):
    def __init__(self, forget_data, retain_data, tokenizer, max_length=512, loss_type: str = "idk"):
        super(WatermarkTextForgetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.forget_data = forget_data
        self.retain_data = retain_data
        
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
            text = data[idx]['text']
            # indices = data[idx]['index']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, text)
            rets.append(converted_data)
        return rets 

    
class UnwatermarkedTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        super(UnwatermarkedTextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data 
        if 'index' not in self.data.column_names:
            self.data = self.data.add_column('index', list(range(len(self.data))))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        idx = int(idx)
        text = self.data[idx]['text']
        indices = self.data[idx]['index']
        pad_input_ids, pad_attention_mask = convert_raw_data_to_model_format(self.tokenizer, self.max_length, text)
        return pad_input_ids, pad_attention_mask, torch.tensor(indices)
    
class UnwatermarkedTextForgetDataset(Dataset):
    def __init__(self, forget_data, retain_data, tokenizer, max_length=512, loss_type="idk"):
        super(UnwatermarkedTextForgetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.loss_type = loss_type
    def __len__(self):
        return len(self.forget_data)
    def __getitem__(self, idx):
        rets = []
        for data_type in ["forget", "retain"]:
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            text = data[idx]['text']
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, text)
            rets.append(converted_data)
        return rets