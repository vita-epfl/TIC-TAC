import torch

from utils.tic import get_positive_definite_matrix, get_tic_covariance
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass


def mse_gradient(means: torch.Tensor) -> torch.Tensor:
    loss = (means ** 2).sum(dim=1)
    return loss.mean()


def nll_gradient(means: torch.Tensor, matrix: torch.Tensor, dim: int) -> torch.Tensor:
    precision_hat = get_positive_definite_matrix(matrix, dim)

    loss = -torch.logdet(precision_hat) + torch.matmul(
        torch.matmul(means.unsqueeze(1), precision_hat),
        means.unsqueeze(2)).squeeze()
    
    return loss.mean()


def diagonal_gradient(means: torch.Tensor, matrix: torch.Tensor, dim: int) -> torch.Tensor:
    var_hat = matrix[:, :dim] ** 2
    loss = torch.log(var_hat) + ((means ** 2) / var_hat)
    return loss.mean()


def beta_nll_gradient(means: torch.Tensor, matrix: torch.Tensor, dim: int) -> torch.Tensor:
    var_hat = matrix[:, :dim] ** 2
    loss = torch.log(var_hat) + ((means ** 2) / var_hat)
    scaling = torch.clone(var_hat).detach() ** 0.5
    loss *= scaling

    return loss.mean()


def faithful_gradient(means: torch.Tensor, matrix: torch.Tensor, dim: int) -> torch.Tensor:
    precision_hat = get_positive_definite_matrix(matrix, dim)

    # This trains the mean square error module independent of NLL
    mse_loss = (means ** 2).sum(dim=1)

    # Ensure NLL gradients don't train the MSE module
    detached_ = means.detach()
            
    nll_loss = -torch.logdet(precision_hat) + torch.matmul(
        torch.matmul(detached_.unsqueeze(1), precision_hat),
        detached_.unsqueeze(2)).squeeze()

    loss = mse_loss + nll_loss
    return loss.mean()


def tic_gradient(means: torch.Tensor, matrix: torch.Tensor, dim: int, pose_net: Hourglass,
                 pose_encodings: dict, use_hessian: bool, model_name: str, imgs: torch.Tensor) -> torch.Tensor:
    
    psd_matrix = get_positive_definite_matrix(matrix, dim)
    covariance_hat = get_tic_covariance(
        pose_net, pose_encodings, matrix, psd_matrix, use_hessian, model_name, imgs)
    
    precision_hat = torch.linalg.inv(covariance_hat)
            
    loss = -torch.logdet(precision_hat) + torch.matmul(
        torch.matmul(means.unsqueeze(1), precision_hat),
        means.unsqueeze(2)).squeeze()
    
    return loss.mean()