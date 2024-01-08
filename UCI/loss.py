import torch

from regressor import Regressor
from utils import get_positive_definite_matrix, get_tic_covariance


def mse_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, _ = model(x)
    loss = (y - y_hat) ** 2
    return loss.sum()


def nll_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out_dim = y.shape[1]

    y_hat, precision_hat = model(x)
    precision_hat = get_positive_definite_matrix(precision_hat, out_dim)

    loss = -torch.logdet(precision_hat) + torch.matmul(
        torch.matmul((y - y_hat).unsqueeze(1), precision_hat), (y - y_hat).unsqueeze(2)).squeeze()
    
    return loss.sum()


def diagonal_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out_dim = y.shape[1]
    
    y_hat, var_hat = model(x)
    var_hat = var_hat[:, :out_dim] ** 2

    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    return loss.sum()


def beta_nll_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor, beta_nll: float) -> torch.Tensor:
    out_dim = y.shape[1]

    y_hat, var_hat = model(x)
    var_hat = var_hat[:, :out_dim] ** 2

    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    scaling = torch.clone(var_hat).detach() ** beta_nll
    loss *= scaling
    
    return loss.sum()


def faithful_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out_dim = y.shape[1]

    y_hat, precision_hat = model(x)
    precision_hat = get_positive_definite_matrix(precision_hat, out_dim)

    # This trains the mean square error module independent of NLL
    mse_loss = torch.pow(y - y_hat, 2).sum(dim=1)

    # Ensure NLL gradients don't train the MSE module
    detached_ = (y - y_hat).detach()
    nll_loss = -torch.logdet(precision_hat) + torch.matmul(
        torch.matmul(detached_.unsqueeze(1), precision_hat), detached_.unsqueeze(2)).squeeze()

    loss = mse_loss + nll_loss

    return loss.sum()


def tic_gradient(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out_dim = y.shape[1]

    y_hat, cov_hat = model(x)

    psd_matrix = get_positive_definite_matrix(cov_hat, out_dim)
    covariance_hat = get_tic_covariance(x, model, cov_hat, psd_matrix)
    precision_hat = torch.linalg.inv(covariance_hat)

    loss = -torch.logdet(precision_hat) + torch.matmul(
        torch.matmul((y - y_hat).unsqueeze(1), precision_hat), (y - y_hat).unsqueeze(2)).squeeze()
    
    return loss.sum()