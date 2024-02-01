import os
from math import log, pi
from typing import Callable


import torch

if int(torch.__version__.split('.')[0]) >= 2: from torch.func import vmap, hessian, jacrev
else: from functorch import vmap, hessian, jacrev

from matplotlib import pyplot as plt


from regressor import Regressor


def gradient_norm(model: Regressor) -> torch.Tensor:
    """
    Computes the L-2 norm of the model's gradients
    """
    total_norm = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    return total_norm


def check_nan_in_model(model: Regressor) -> bool:
    """
    Checks for NaN in model
    """
    for param in model.parameters():
        if torch.any(torch.isnan(param)):
            return True
    return False


# Matrix Operations
def get_positive_definite_matrix(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Multiplies matrix with its transpose to get positive semidefinite matrix
    """
    tensor = tensor[:, :dim ** 2]
    tensor = tensor.view(-1, dim, dim)
    return torch.matmul(tensor, tensor.mT)


def pairwise_matmul_with_trace(tensor: torch.Tensor) -> torch.Tensor:
    """ 
    Uses broadcasting to multiply batches of matrices pairwise
    Let Tensor.shape be BS x K x K
    Then this function makes two matrices BS x 1 x K x K and 1 x BS x K x K
    The goal is to use broadcasting to parallelize the pairwise computation between matrices to give:
    BS x BS x K x K

    We return the trace for use in the KL divergence objective
    """
    a = tensor.unsqueeze(1)
    b = tensor.unsqueeze(0)

    ab = torch.matmul(a, b)
    ab_diag = torch.diagonal(ab, dim1=-2, dim2=-1)
    return torch.sum(ab_diag, dim=-1)


def calculate_tac_per_sample(y_pred: torch.Tensor, covariance_hat: torch.Tensor,
                             y_gt: torch.Tensor, loss_placeholder: torch.Tensor) -> torch.Tensor:
    """
    Compute TAC by observing how i-th dimension by observing other dimensions
    """
    dim = y_pred.shape[-1]

    cov_hat = covariance_hat.clone()
    y_hat = y_pred.clone()
    y_obs = y_gt.clone()
    
    for i in range(dim):
        # Swap the i'th variable to the front
        cov_hat[[0, i]] = cov_hat[[i, 0]]
        cov_hat[:, [0, i]] = cov_hat[:, [i, 0]]
        y_hat[[0, i]] = y_hat[[i, 0]]
        y_obs[[0, i]] = y_obs[[i, 0]]

        sigma_12 = cov_hat[0, 1:].unsqueeze(0)
        sigma_22 = cov_hat[1:, 1:]

        y_cond = y_hat[0] + torch.matmul(
            torch.matmul(sigma_12, torch.linalg.inv(sigma_22)),
            (y_obs[1:] - y_hat[1:]).unsqueeze(1)).squeeze()
        loss_placeholder[i] = torch.abs(y_obs[0] - y_cond)

        # Swap the i'th variable to the back
        cov_hat[[0, i]] = cov_hat[[i, 0]]
        cov_hat[:, [0, i]] = cov_hat[:, [i, 0]]
        y_hat[[0, i]] = y_hat[[i, 0]]
        y_obs[[0, i]] = y_obs[[i, 0]]

    return loss_placeholder


def calculate_ll_per_sample(y_pred: torch.Tensor, precision_hat: torch.Tensor,
                            y_gt: torch.Tensor, loss_placeholder: torch.Tensor) -> torch.Tensor:
    """
    Compute LL for each sample
    """

    out_dim = y_gt.shape[-1]

    ll = -0.5 * (
            (out_dim * log(2 * pi))
          - (torch.logdet(precision_hat))
          + (torch.matmul(torch.matmul((y_gt - y_pred).unsqueeze(0), precision_hat),
                          (y_gt - y_pred).unsqueeze(1)).squeeze()))
    
    return ll


def _predictions(model: Regressor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Obtains the model's target predictions for an input.
    This function is used in conjunction with vmap.
    """
    def pred(x):
        return model(x)[0].squeeze() # y_hat, var_hat
    return pred


def _get_derivatives(x: torch.Tensor, model: Regressor) -> (torch.Tensor, torch.Tensor):
    """
    Compute the gradient and hessian wrt the input x for a model
    """
    model_is_train = model.training

    model.train(False)
                    
    grads = vmap(jacrev(_predictions(model)))(x.unsqueeze(1)).squeeze()
    hessians = vmap(hessian(_predictions(model)))(x.unsqueeze(1)).squeeze()

    grads = grads @ grads.mT
    hessians = batched_hessian_var(hessians)

    model.train(model_is_train)
    return grads.detach(), hessians.detach()


def get_tic_covariance(x: torch.Tensor, model: Regressor,
                       cov_hat: torch.Tensor, psd_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the Taylor Induced Covariance
    """
    with torch.no_grad():
        grads, hessians = _get_derivatives(x, model)

    epsilon = cov_hat[:, -2:] ** 2

    covariance = (epsilon[:, 0].view(-1, 1, 1) * grads) \
               + (epsilon[:, 1].view(-1, 1, 1) * hessians) \
               + psd_matrix

    return covariance


# Plotting functions
def plot_comparison(training_pkg: dict, dimensions: range,
                    experiment_name: str, metric: str) -> None:
    """
    Compare the metric vs dimensionality for various observations
    """
    training_methods = training_pkg['training_methods']

    color = ['purple', 'green', 'crimson', 'steelblue', 'goldenrod', 'coral']
    marker = ['s', '*', 'o', '.', 'x', 'D']

    for method in training_methods:

        c = color.pop()
        m = marker.pop()
        
        mean = training_pkg[method][metric]['mean'].cpu().numpy()
        std = training_pkg[method][metric]['std'].cpu().numpy()

        plt.plot(list(dimensions), mean, label=method, color=c, marker=m)
        plt.fill_between(list(dimensions), mean + std, mean - std, alpha=0.25, color=c)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.xlabel('Dimensions',fontsize=12)
    plt.ylabel('{}'.format(metric.upper()), fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(experiment_name, "{}.pdf".format(metric.upper())),
                format='pdf', bbox_inches="tight", dpi=300)
    plt.close()


# Batched versions
calculate_tac = vmap(calculate_tac_per_sample, in_dims=(0, 0, 0, 0))

calculate_ll = vmap(calculate_ll_per_sample, in_dims=(0, 0, 0, 0))

batched_hessian_var = vmap(pairwise_matmul_with_trace)