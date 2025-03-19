from functools import partial

import torch

from helper import cubic_func, sto_cubic_func, utils


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
        args = inputs.pop("input_ids")
        kwargs = inputs
        outputs = torch.func.functional_call(model, {n: p for n, p in zip(names, tuple_params)}, args, kwargs)
        if isinstance(outputs, dict):
            loss += outputs["loss"]
        else:
            loss += outputs[0]
        n_sample += args.shape[0]
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
    utils.clear_cache()
    compute_hvp = hf_hvp_func(model, 
                              hess_batchloader,
                              trainer,
                              device=device)

    dw = sto_cubic_func.gd_cubic_subsolver(model,
                                           tuple_params,
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

# def gd_cubic_subsolver(args, tuple_params, tuple_hvp_fn, tuple_g, eps: float = 0.01):
#     # Take Cauchy step if far away from the solution
#     hvp_g = hvp(tuple_hvp_fn, tuple_g)
#     hvp_g = cubic_func.compose_param_vector(hvp_g, tuple_params).cpu().detach().numpy()
#     g = cubic_func.compose_param_vector(tuple_g, tuple_params).cpu().detach().numpy()
#     norm_g = numpy.linalg.norm(g)
#     Rc = (g.T @ hvp_g) / (args.M * norm_g ** 2) 
#     Rc = -Rc + numpy.sqrt(Rc ** 2 + (2 * norm_g / args.M))
#     # print('cauchy_radius:', Rc)
#     s = -Rc * g / norm_g
#     if norm_g < (args.L ** 2 / args.M):
#         # GD-based cubic step with perturbed gradient
#         c = 0.01
#         sigma = c * numpy.sqrt(eps * args.M) / args.L
#         perturb = numpy.random.randn(*g.shape)
#         perturb = perturb / numpy.linalg.norm(perturb)  # see (Muller, 1959) and (Marsaglia, 1972)
#         g_perturb = g + sigma * perturb
#         eta = 1 / (20 * args.L)
#         for _ in range(5):
#             tuple_s = cubic_func.decompose_param_vector(s, tuple_params)
#             hvp_s = hvp(tuple_hvp_fn, tuple_s)
#             norm_s = numpy.linalg.norm(s)
#             hvp_s = cubic_func.compose_param_vector(hvp_s, tuple_params).cpu().detach().numpy()
#             s = s - eta * (g_perturb + hvp_s + args.M/2 * norm_s * s)
#             # print('norm_s:', norm_s)
#     return s
    