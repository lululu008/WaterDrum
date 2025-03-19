import torch
import numpy as np
from functools import partial
import gc

from scr.unlearning import cubic_func


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()
    
def convert_torch_to_numpy(tensor):
    try:
        res = tensor.numpy()
    except TypeError as error:
        # avoid typecast error from torch.bfloat16 to numpy
        print("Encountered TypeError:", str(error))
        print("Trying tensor.float().numpy()")
        res = tensor.float().numpy()
        print("Successfully converted")
    return res

def stochastic_cubic_step(model,
                          loss_fn,
                          grad_batchloader,
                          hess_batchloader,
                          M: float = 1,
                          num_steps: int = 5,
                          learning_rate: float = 0.001,
                          device=None,
                          ):

    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)
    
    grad = cubic_func.gradient(model,
                               loss_fn,
                               grad_batchloader,
                               device=device)
    grad = tuple(x.detach().cpu() for x in grad)
    clear_cache()
    compute_hvp = hvp_func(model,
                           loss_fn,
                           hess_batchloader,
                           device=device)                               
        
    dw = gd_cubic_subsolver(tuple_params, 
                            compute_hvp, 
                            grad,
                            M=M,
                            num_steps=num_steps,
                            learning_rate=learning_rate,
                            device=device)
    tuple_param_update = cubic_func.decompose_param_vector(dw, tuple_params)
    i = 0
    for param in model.parameters():
        if param.requires_grad:
            param.data += tuple_param_update[i]
            i += 1

def gd_cubic_subsolver(tuple_params,
                       tuple_hvp_fn: callable,
                       tuple_grad, 
                       M: float,
                       num_steps: int,
                       learning_rate: float,
                       device=None,
                       ):
    """
    Solve cubic subproblem using gradient descent 
    Algorithm 2 in Carmon and Duchi (2016) (https://arxiv.org/pdf/1612.00547)
    """

    grad = cubic_func.compose_param_vector(tuple_grad, tuple_params)
    grad = convert_torch_to_numpy(grad)
    grad_norm = np.linalg.norm(grad)
    print("grad_norm:", grad_norm)

    print("Setting x0 = cauchy point") 
    B_grad = hvp(tuple_hvp_fn, 
                 tuple_grad, 
                 device=device)
    clear_cache()
    B_grad = cubic_func.compose_param_vector(B_grad, tuple_params)
    B_grad = B_grad.detach().cpu()
    B_grad = convert_torch_to_numpy(B_grad)

    left_term = (grad.T @ B_grad) / (M * grad_norm**2) 
    right_term = np.sqrt(left_term**2 + 2*grad_norm/M) 
    Rc = -left_term + right_term
    s = -Rc * grad/grad_norm
    del left_term, right_term, Rc
    clear_cache()
    # Rc = -Rc + np.sqrt(Rc ** 2 + (2 * norm_g / args.M))
    # print('cauchy_radius:', Rc)

    print("Doing gradient descent...")
    sigma = 0.001
    perturb = np.random.randn(*grad.shape)
    perturb = perturb / np.linalg.norm(perturb)  # see (Muller, 1959) and (Marsaglia, 1972)
    grad = grad + sigma * perturb
    del perturb
    clear_cache()
    
    # s = np.zero(*grad.shape)
    for step in range(num_steps):
        print(f"GD step {step + 1}")
        tuple_s = cubic_func.decompose_param_vector(s, tuple_params)
        B_s = hvp(tuple_hvp_fn, 
                  tuple_s,
                  device=device)
        B_s = cubic_func.compose_param_vector(B_s, tuple_params)
        B_s = B_s.detach().cpu()
        B_s = convert_torch_to_numpy(B_s)
        s_norm = np.linalg.norm(s)
        s = s - learning_rate * (grad + B_s + M/2*s_norm*s)

        tmp = grad + B_s + M/2*s_norm*s
        print("descent_grad_norm:", np.linalg.norm(tmp))
        del B_s
        clear_cache()
    
    # import pdb; pdb.set_trace()
    # exit()
    return s
    
def hvp(compute_hvp: callable, tuple_v, device=None):
    v = tuple(x.to(device) for x in tuple_v)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        res = compute_hvp(v)[1]
    return res

def hvp_func(model, loss_fn, dataloader, device=None):
    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)
    compute_loss_fn = partial(cubic_func.compute_loss, 
                              model, 
                              loss_fn, 
                              dataloader,
                              device)
    res = partial(torch.autograd.functional.hvp,
                  compute_loss_fn, 
                  tuple_params)
    return res

