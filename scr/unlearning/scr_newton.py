from functools import partial

import torch
import numpy as np
from torch.utils.data import Subset

from tofu.dataloader import CustomTrainer
from scr.unlearning import sto_cubic_func, sto_cubic_func_hf

def unlearn(model,
            retain_set,
            config,
            trainer_init_kwargs,
            ):
    """
    Lessons:
    1. Approximation gets better when sample size increases. 
    2. On the contrary, stochastic step benefits the optimization by adding noise to the descent direction, degrading the mutual information between the remaining and the forgotten data.
    """
    trainer_init_kwargs.model = model
    trainer = CustomTrainer(**vars(trainer_init_kwargs))
    scr_step_func = partial(sto_cubic_func_hf.hf_stochastic_cubic_step, 
                            trainer=trainer)

    learning_rate = config.learning_rate
    # lr_scheduler = lambda step, lr: lr * 0.5 if (step % 5 == 0) else lr
    lr_scheduler = lambda step, lr: lr
    for step in range(config.num_outer_steps):
        print(f"Stochastic Step {step + 1}, lr = {learning_rate}")
        num_train_samples = len(retain_set)
        train_ids = list(range(num_train_samples))
        grad_ids = np.random.choice(train_ids, 
                                    config.grad_sample_size, 
                                    replace=False)
        hess_ids = np.random.choice(train_ids,
                                    config.hess_sample_size,
                                    replace=False)
        grad_batch = sample(retain_set, grad_ids) 
        hess_batch = sample(retain_set, hess_ids)

        trainer.train_dataset = grad_batch
        trainer._train_batch_size = config.grad_sample_size    # so that dataloader contains 1 batch
        grad_batchloader = trainer.get_train_dataloader()
        trainer.train_dataset = hess_batch
        trainer._train_batch_size = config.hess_sample_size
        hess_batchloader = trainer.get_train_dataloader()

        scr_step_func(model, 
                      grad_batchloader=grad_batchloader,
                      hess_batchloader=hess_batchloader,
                      M=config.M,
                      num_steps=config.num_inner_steps,
                      learning_rate=learning_rate,
                      device="cuda")
        learning_rate = lr_scheduler(step + 1, learning_rate) 

def sample(dataset, sample_ids):
    if isinstance(dataset, Subset):
        mapped_sample_ids = [dataset.indices[id] for id in sample_ids]
        res = Subset(dataset.dataset, mapped_sample_ids)
    else:
        res = Subset(dataset, sample_ids)
    return res