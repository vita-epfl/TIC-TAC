import torch


from regressor import Regressor
from utils import get_tic_variance


from torch.nn.utils import parameters_to_vector
from utils_natural_laplace.utilities import Regressor_NaturalHead
from utils_natural_laplace.utilities import NaturalLaplace, expand_prior_precision


def mse_loss(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, _ = model(x)
    loss = (y - y_hat) ** 2
    return loss.sum()


def nll_loss(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, var_hat = model(x)
    var_hat = var_hat[:, 0].unsqueeze(1)
    
    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    return loss.sum()


def beta_nll_loss(model: Regressor, x: torch.Tensor, y: torch.Tensor,
                  beta_nll: float) -> torch.Tensor:
    y_hat, var_hat = model(x)
    var_hat = var_hat[:, 0].unsqueeze(1)

    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    scaling = torch.clone(var_hat).detach() ** beta_nll
    loss *= scaling
    
    return loss.sum()


def faithful_loss(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, var_hat = model(x)
    var_hat = var_hat[:, 0].unsqueeze(1)
    
    # This trains the mean square error module independent of NLL
    mse_loss = torch.pow(y - y_hat, 2)

    # Ensure NLL gradients don't train the MSE module
    nll_loss = torch.log(var_hat) + (mse_loss.detach() / var_hat)
    
    loss = mse_loss + nll_loss
    
    return loss.sum()


def tic_loss(model: Regressor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat, var_hat = model(x)
    
    var_hat = get_tic_variance(x, model, var_hat)
    
    loss = torch.log(var_hat) + (((y - y_hat) ** 2) / var_hat)
    
    return loss.sum()


def natural_laplace_loss(model: Regressor_NaturalHead, x: torch.Tensor, y: torch.Tensor,
                             natural_laplace: NaturalLaplace) -> torch.Tensor:

    natural = model(x)

    prior_prec = torch.exp(natural_laplace.log_prior_prec).detach()
    delta = expand_prior_precision(prior_prec, model)

    theta = parameters_to_vector(model.parameters())
    loss = natural_laplace.heteroscedastic_mse_loss(natural, y) + (0.5 * (delta * theta) @ theta)

    return loss.sum()