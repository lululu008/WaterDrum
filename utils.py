import random
import os
import gc

import numpy as np
import torch


def set_seed(seed):
    '''
    Set random seed (for reproducbility)
    References: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    '''

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')


def save_trainable_model(model, path):
    res = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            res[name] = param.detach()
    torch.save(res, path)
    return


def load_trainable_model(model, path):
    checkpoint = torch.load(path)
    trainable_layers = set(checkpoint.keys())
    all_layers = set(model.state_dict().keys())
    num_match_layers = len(trainable_layers.intersection(all_layers))
    print('Load trainable parameters for {}/{} layers'.format(num_match_layers, len(all_layers)))
    model.load_state_dict(checkpoint, strict=False)
    return model


def count_trainable_parameters(model):
    count = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            count += param.numel()
    print('Number of trainable parameters: {:,}'.format(count))
    return


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()
