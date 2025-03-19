import os
from functools import partial

import torch
import numpy as np
import scipy


def full_cubic_step(model, 
                    loss_fn, 
                    dataloader,
                    M: float = 1,   # Hessian Lipschitz constant
                    save_hess: bool = False,
                    temp_dir: str = None,
                    device=None,
                    ):
    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)
    grad = gradient(model, 
                    loss_fn, 
                    dataloader,
                    device=device)
    grad = -compose_param_vector(grad, tuple_params)
    grad = grad.detach().cpu().numpy()
    hess = hessian(model, 
                   loss_fn, 
                   dataloader,
                   device=device)
    hess = compose_param_matrix(hess, tuple_params)
    hess = hess.detach().cpu().numpy()

    if save_hess:
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, "hess.npy")
        with open(path, "wb") as f:
            np.save(f, hess)

    dw, alpha = solve_cubic_dual(hess, grad, M)
    print('alpha: {}'.format(alpha))
    tuple_param_update = decompose_param_vector(dw, tuple_params)
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            p.data += tuple_param_update[i]
            i += 1
    return alpha

def solve_cubic_dual(hess, 
                     grad, 
                     M: float, 
                     tol: float = 1e-4,
                     ):
    """
    Solve cubic dual problem by Trust-Region method
    Constraint: r <= lambda / M
    """
    def _compute_s(hess, grad, alpha, err_const):
        while True:
            try:
                # Cholesky decomposition for a positive definite matrix
                L = np.linalg.cholesky(hess + (alpha + err_const) * np.eye(*hess.shape))
                break
            except np.linalg.LinAlgError:
                err_const *= 2
        s = scipy.linalg.cho_solve((L, True), grad)    # find s = (H~)^(-1)g via L(L^T)s = g
        return L, s, err_const

    lambda_n = scipy.linalg.eigvalsh(hess)[0]
    alpha = max(-lambda_n, 0)
    err_const = (1 + alpha) * np.sqrt(np.finfo(float).eps)
    r = 2.0 * alpha / M    # lambda := Mr/2
    L, s, err_const = _compute_s(hess, grad, alpha, err_const)
    norm_s = np.linalg.norm(s)
    if norm_s > r:   # boundary solution
        max_iter = 200
        it = 0
        # Find root of 1/norm2(s) = 1/r (secular eqn for norm(s) = r)
        while not (abs(norm_s - r) <= tol) and it < max_iter:
            w = scipy.linalg.solve_triangular(L, s, lower=True)     # Lw = s
            norm_w = np.linalg.norm(w)
            phi = 1.0 / norm_s - 1.0 * M / (2 * alpha)
            phi_prime = (norm_w ** 2) / (norm_s ** 3) + 1.0 * M / (2 * alpha ** 2)
            alpha = alpha - phi / phi_prime
            L, s, _ = _compute_s(hess, grad, alpha, err_const=err_const)
            r = 2.0 * alpha / M
            norm_s = np.linalg.norm(s)
            it += 1
        if it == max_iter:
            print('[DEBUG] alpha not converged with err = %.3f' % (abs(np.linalg.norm(s) - r)))
    else:   # interior solution
        print('[DEBUG] to find interior solution, norm_s ({}) vs. r ({})'.format(np.linalg.norm(s), r))
        if alpha == 0 or (abs(norm_s - r) <= tol):
            return s, alpha
        else:
            eigvals, Q = scipy.linalg.eigh(hess + alpha * np.eye(*hess.shape))
            inv_Lambda = np.linalg.pinv(np.diag(eigvals))
            s = -(Q @ inv_Lambda @ Q.T).dot(grad)
            u_n = Q[:,0]
            alpha = max(np.roots([u_n.dot(u_n), 2 * u_n.dot(s), s.dot(s) - r**2]))
            s = s + alpha * u_n
    return s, alpha

def compute_loss(model, 
                 loss_fn, 
                 dataloader, 
                 device, 
                 *tuple_params,
                 ):
    names = list(name for name, param in model.named_parameters() if param.requires_grad)
    loss = 0.0
    n_sample = 0
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.backends.cudnn.flags(enabled=False):
            preds = torch.func.functional_call(model, {n: p for n, p in zip(names, tuple_params)}, x)
        batch_size = x.shape[0]
        loss += batch_size * loss_fn(preds, y) 
        n_sample += batch_size
    loss /= n_sample
    return loss

def gradient(model, loss_fn, dataloader, device=None):
    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)
    loss = compute_loss(model, 
                        loss_fn, 
                        dataloader, 
                        device,
                        *tuple_params)
    grad = torch.autograd.grad(loss, tuple_params)
    return grad

def hessian(model, loss_fn, dataloader, device=None):
    tuple_params = tuple(param for param in model.parameters() if param.requires_grad)
    compute_loss_fn = partial(compute_loss, 
                              model, 
                              loss_fn, 
                              dataloader,
                              device)
    hess = torch.autograd.functional.hessian(compute_loss_fn, tuple_params)
    return hess

def compose_param_vector(tuple_v, tuple_params):
    layer_params = []
    for each in tuple_v:
        layer_params.append(each.view(-1))
    return torch.concat(layer_params)

def decompose_param_vector(v, tuple_params):
    from_id = 0
    layer_params = []
    for p in tuple_params:
        to_id = from_id + np.prod(p.data.shape)
        if not torch.is_tensor(v):
            layer_w = torch.from_numpy(v[from_id:to_id])
        else:
            layer_w = v[from_id:to_id]
        layer_params.append(layer_w.reshape(p.data.shape).to(p.data.device))
        from_id = to_id 
    return tuple(layer_params)

def compose_param_matrix(tuple_A, tuple_params):
    n_layer = len(tuple_params)
    layer_shape = [layer_params.shape for layer_params in tuple_params]
    layer_params = [[] for _ in range(n_layer)]
    for i in range(n_layer):
        for j in range(n_layer):
            flat_param_shape_i = np.prod(layer_shape[i])
            flat_param_shape_j = np.prod(layer_shape[j])
            layer_params[i].append(tuple_A[i][j].reshape((flat_param_shape_i, flat_param_shape_j)))
        layer_params[i] = torch.hstack(layer_params[i])
    return torch.vstack(layer_params)
