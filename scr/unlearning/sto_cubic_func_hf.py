from functools import partial
import torch
import gc

from scr.unlearning import cubic_func, sto_cubic_func


def hf_compute_loss(model, 
                    dataloader, 
                    trainer, 
                    device,
                    *tuple_params,
                    ):
    names = list(name for name, param in model.named_parameters() if param.requires_grad)
    loss = 0.0
    n_sample = 0
    for batch in dataloader:
        inputs = trainer._prepare_inputs(batch)
        input_ids, labels, attention_mask = inputs[0], inputs[1], inputs[2]
        kwargs={'labels': labels, 'attention_mask': attention_mask}
        outputs = torch.func.functional_call(model, {n: p for n, p in zip(names, tuple_params)}, input_ids, kwargs)
        if isinstance(outputs, dict):
            loss += outputs["loss"]
        else:
            loss += outputs[0]
        n_sample += input_ids.shape[0]
    loss /= n_sample
    return loss
    
def hf_gradient(model, dataloader, trainer, device=None):
    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)
    loss = hf_compute_loss(model, 
                           dataloader, 
                           trainer, 
                           device,
                           *tuple_params)
    grads = torch.autograd.grad(loss, tuple_params)
    grads = tuple(g.detach() for g in grads)
    del loss
    return grads

def hf_hvp_func(model, dataloader, trainer, device=None):
    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)
    compute_loss_fn = partial(hf_compute_loss, 
                              model, 
                              dataloader, 
                              trainer,
                              device)
    res = partial(torch.autograd.functional.hvp, compute_loss_fn, tuple_params)
    return res

def hf_stochastic_cubic_step(model,
                             grad_batchloader,
                             hess_batchloader,
                             M: float = 1,
                             num_steps: int = 5,
                             learning_rate: float = 0.001,
                             trainer=None,
                             device=None,
                             ):

    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)

    grad = hf_gradient(model, 
                       grad_batchloader, 
                       trainer,
                       device=device)
    grad = tuple(x.detach().cpu() for x in grad)
    clear_cache()
    compute_hvp = hf_hvp_func(model, 
                              hess_batchloader,
                              trainer,
                              device=device)

    dw = sto_cubic_func.gd_cubic_subsolver(tuple_params,
                                           compute_hvp,
                                           grad,
                                           M=M,
                                           num_steps=num_steps,
                                           learning_rate=learning_rate,
                                           device=device)
    tuple_param_update = cubic_func.decompose_param_vector(dw, tuple_params)
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            p.data += tuple_param_update[i]
            i += 1

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()