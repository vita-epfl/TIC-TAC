import torch

from regressor import Regressor
from utils import get_differential_variance


def mse_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, _ = model(x)
    loss = (y - y_hat) ** 2
    return loss.sum()


def nll_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, var_hat = model(x)
    var_hat = var_hat[:, 0].unsqueeze(1)
    
    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    return loss.sum()


def beta_nll_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor, beta_nll: float) -> torch.Tensor:
    y_hat, var_hat = model(x)
    var_hat = var_hat[:, 0].unsqueeze(1)

    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    scaling = torch.clone(var_hat).detach() ** beta_nll
    loss *= scaling
    
    return loss.sum()


def faithful_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, var_hat = model(x)
    var_hat = var_hat[:, 0].unsqueeze(1)
    
    # This trains the mean square error module independent of NLL
    mse_loss = torch.pow(y - y_hat, 2)

    # Ensure NLL gradients don't train the MSE module
    nll_loss = torch.log(var_hat) + (mse_loss.detach() / var_hat)
    
    loss = mse_loss + nll_loss
    
    return loss.sum()


def tic_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, var_hat = model(x)
    
    var_hat = get_differential_variance(x, model, var_hat)
    
    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    return loss.sum()