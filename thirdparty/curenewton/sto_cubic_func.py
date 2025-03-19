import torch
import numpy as np
from functools import partial

from helper import cubic_func, utils


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
    utils.clear_cache()
    compute_hvp = hvp_func(model,
                           loss_fn,
                           hess_batchloader,
                           device=device)                               
        
    dw = gd_cubic_subsolver(model,
                            tuple_params, 
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

def gd_cubic_subsolver(model,
                       tuple_params,
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
    grad = utils.convert_torch_to_numpy(grad)
    grad_norm = np.linalg.norm(grad)
    print("grad_norm:", grad_norm)

    print("Setting x0 = cauchy point") 
    try:
        B_grad = hvp(tuple_hvp_fn, 
                     tuple_grad, 
                     device=device)
    except RuntimeError as error:
        if "out of memory" in str(error):
            print("Bad batch. Please try another batch.")
            for param in model.parameters():
                if param.grad is not None:
                    del param.grad
            utils.clear_cache()
        raise error
    
    utils.clear_cache()
    B_grad = cubic_func.compose_param_vector(B_grad, tuple_params)
    B_grad = B_grad.detach().cpu()
    B_grad = utils.convert_torch_to_numpy(B_grad)

    left_term = (grad.T @ B_grad) / (M * grad_norm**2) 
    right_term = np.sqrt(left_term**2 + 2*grad_norm/M) 
    Rc = -left_term + right_term
    s = -Rc * grad/grad_norm
    del left_term, right_term, Rc, B_grad
    utils.clear_cache()

    print("Doing gradient descent...")
    sigma = 0.001
    perturb = np.random.randn(*grad.shape)
    perturb = perturb / np.linalg.norm(perturb)  # see (Muller, 1959) and (Marsaglia, 1972)
    grad = grad + sigma * perturb
    del perturb
    utils.clear_cache()
    
    # s = np.zero(*grad.shape)
    for step in range(num_steps):
        print(f"GD step {step + 1}")
        tuple_s = cubic_func.decompose_param_vector(s, tuple_params)
        B_s = hvp(tuple_hvp_fn, 
                  tuple_s,
                  device=device)
        B_s = cubic_func.compose_param_vector(B_s, tuple_params)
        B_s = B_s.detach().cpu()
        B_s = utils.convert_torch_to_numpy(B_s)
        s_norm = np.linalg.norm(s)
        s = s - learning_rate * (grad + B_s + M/2*s_norm*s)

        tmp = grad + B_s + M/2*s_norm*s
        print("descent_grad_norm:", np.linalg.norm(tmp))
        del B_s
        utils.clear_cache()
    
    return s



            
            
        
    
    # norm_g = numpy.linalg.norm(g)
    # Rc = (g.T @ hvp_g) / (args.M * norm_g ** 2) 
    # Rc = -Rc + numpy.sqrt(Rc ** 2 + (2 * norm_g / args.M))
    # # print('cauchy_radius:', Rc)
    # s = -Rc * g / norm_g
    # if norm_g < (args.L ** 2 / args.M):
    #     # GD-based cubic step with perturbed gradient
    #     c = 0.01
    #     sigma = c * numpy.sqrt(eps * args.M) / args.L
    #     perturb = numpy.random.randn(*g.shape)
    #     perturb = perturb / numpy.linalg.norm(perturb)  # see (Muller, 1959) and (Marsaglia, 1972)
    #     g_perturb = g + sigma * perturb
    #     eta = 1 / (20 * args.L)
    #     for _ in range(5):
    #         tuple_s = cubic_func.decompose_param_vector(s, tuple_params)
    #         hvp_s = hvp(tuple_hvp_fn, tuple_s)
    #         norm_s = numpy.linalg.norm(s)
    #         hvp_s = cubic_func.compose_param_vector(hvp_s, tuple_params).cpu().detach().numpy()
    #         s = s - eta * (g_perturb + hvp_s + args.M/2 * norm_s * s)
    #         # print('norm_s:', norm_s)
    # return s
    
def hvp(compute_hvp: callable, tuple_v, device=None):
    v = tuple(x.to(device) for x in tuple_v)
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


# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# args, model, loss_fn, batch_gradient)
# tuple_hvp_fn = hvp_func(args, model, loss_fn, batch_hessian)

    # args, model, loss_fn, batch_gradient, batch_hessian):
    # batch_gradient = torch.utils.data.TensorDataset(*batch_gradient)
    # batch_gradient = torch.utils.data.DataLoader(batch_gradient, batch_size=len(batch_gradient), shuffle=False)
    # batch_hessian = torch.utils.data.TensorDataset(*batch_hessian)
    # batch_hessian = torch.utils.data.DataLoader(batch_hessian, batch_size=len(batch_hessian), shuffle=False)