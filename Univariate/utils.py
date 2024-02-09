from typing import Callable


import torch
import numpy as np

if int(torch.__version__.split('.')[0]) >= 2: from torch.func import vmap, hessian, jacrev
else: from functorch import vmap, hessian, jacrev

import matplotlib as mpl
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


def _predictions(model: Regressor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Obtains the model's target predictions for an input.
    This function is used in conjunction with vmap.
    """
    def pred(x):
        return model(x)[0].squeeze() # y_hat, var_hat (reminder var_hat is 3-D)
    return pred


def _get_derivatives(x: torch.Tensor, model: Regressor) -> (torch.Tensor, torch.Tensor):
    """
    Compute the gradient and hessian wrt the input x for a model
    """
    model_is_train = model.training
    
    model.train(False)
    grads = vmap(jacrev(_predictions(model)))(x.unsqueeze(1))
    hessians = vmap(hessian(_predictions(model)))(x.unsqueeze(1))

    grads = grads.squeeze().unsqueeze(1) ** 2
    hessians = hessians.squeeze().unsqueeze(1) ** 2
                    
    model.train(model_is_train)
    return grads.detach(), hessians.detach()


def get_tic_variance(x: torch.Tensor, model: Regressor, taylor_coeffecients: torch.Tensor) -> torch.Tensor:
    """
    Compute the Taylor Induced Covariance
    """
    with torch.no_grad():
        grads, hessians = _get_derivatives(x, model)
        taylor_expansion = torch.cat(
            (grads, hessians, torch.ones_like(grads, device='cuda')), dim=1)

    variance = taylor_coeffecients * taylor_expansion
    variance = torch.sum(variance, dim=1, keepdim=True)

    return variance


def plot_sine(ax: mpl.axes._axes, x: np.array, sine: np.array, y_hat: np.array,
              std_dev: np.array, color: str, label: str, i: int) -> mpl.axes._axes:
    """
    Plots the ground truth and predictions with standard deviation
    """
    ax[i % 7].plot(x, sine, color='darkgrey')

    ax[i % 7].plot(x, y_hat, c=color, label=label, linewidth=3)
    ax[i % 7].fill_between(x, y_hat + std_dev, y_hat - std_dev, alpha=0.25, color=color)

    ax[i % 7].set_ylim(-1.2 * max(sine), 1.2 * max(sine))
    ax[i % 7].legend(loc='upper right', fontsize=32)

    ax[i % 7].tick_params(axis='both', which='major', labelsize=24)
    ax[i % 7].tick_params(axis='both', which='minor', labelsize=20)
    ax[i % 7].locator_params(tight=True, nbins=8)

    ax[0].set_ylabel('f(x)', fontsize=32)

    return ax