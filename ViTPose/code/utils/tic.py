import os
from math import log, pi
from typing import Callable

import torch
from matplotlib import pyplot as plt

if int(torch.__version__.split('.')[0]) >= 2: from torch.func import vmap, hessian, jacrev
else: from functorch import vmap, hessian, jacrev

from utils.pose import soft_argmax
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass



# Matrix Operations
def get_positive_definite_matrix(tensor, dim):
    tensor = tensor[:, :dim ** 2]
    tensor = tensor.view(tensor.shape[0], dim, dim)
    return torch.matmul(tensor, tensor.mT)


def pairwise_matmul_with_trace(tensor):
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


def calculate_tac_per_sample(y_pred, covariance_hat,
                             y_gt, loss_placeholder):
    """
    Compute TAC by observing how i-th dimension by observing other dimensions
    """
    dim = y_pred.shape[-1] // 2

    cov_hat = covariance_hat.clone()
    y_hat = y_pred.clone()
    y_obs = y_gt.clone()
        
    for i in range(dim):
        # Swap the i'th variable's U and V to the front
        cov_hat[[0, 2*i]] = cov_hat[[2*i, 0]]
        cov_hat[[1, (2*i) + 1]] = cov_hat[[(2*i) + 1, 1]]
        
        cov_hat[:, [0, 2*i]] = cov_hat[:, [2*i, 0]]
        cov_hat[:, [1, (2*i) + 1]] = cov_hat[:, [(2*i) + 1, 1]]
        
        
        y_hat[[0, 2*i]] = y_hat[[2*i, 0]]
        y_hat[[1, (2*i) + 1]] = y_hat[[(2*i) + 1, 1]]
            
        y_obs[[0, 2*i]] = y_obs[[2*i, 0]]
        y_obs[[1, (2*i) + 1]] = y_obs[[(2*i) + 1, 1]]

        sigma_12 = cov_hat[:2, 2:].unsqueeze(0)
        sigma_22 = cov_hat[2:, 2:]

        y_cond = y_hat[:2] + torch.matmul(
            torch.matmul(sigma_12, torch.linalg.inv(sigma_22)),
            (y_obs[2:] - y_hat[2:]).unsqueeze(1)).squeeze()
        
        loss_placeholder[i] = torch.sqrt(torch.sum(torch.pow(y_obs[:2] - y_cond, 2)))

        # Swap the i'th variable's U and V to the back
        cov_hat[[0, 2*i]] = cov_hat[[2*i, 0]]
        cov_hat[[1, (2*i) + 1]] = cov_hat[[(2*i) + 1, 1]]
        
        cov_hat[:, [0, 2*i]] = cov_hat[:, [2*i, 0]]
        cov_hat[:, [1, (2*i) + 1]] = cov_hat[:, [(2*i) + 1, 1]]
        
        
        y_hat[[0, 2*i]] = y_hat[[2*i, 0]]
        y_hat[[1, (2*i) + 1]] = y_hat[[(2*i) + 1, 1]]
            
        y_obs[[0, 2*i]] = y_obs[[2*i, 0]]
        y_obs[[1, (2*i) + 1]] = y_obs[[(2*i) + 1, 1]]

    return loss_placeholder


def calculate_ll_per_sample(y_pred, precision_hat,
                            y_gt, loss_placeholder):
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


def _predictions_hg(hg_level_6, hg_feat, hg_out):
    """
    Obtains the model's target predictions for an input.
    This function is used in conjunction with vmap.
    """
    hg_level_5 = hg_level_6.low2
    hg_level_4 = hg_level_5.low2
    hg_level_3 = hg_level_4.low2
    hg_level_2 = hg_level_3.low2
    hg_level_1 = hg_level_2.low2 
    
    def pred(x):    
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = hg_level_1.up2(x)
        x = hg_level_2.up2(hg_level_2.low3(x))
        x = hg_level_3.up2(hg_level_3.low3(x))
        x = hg_level_4.up2(hg_level_4.low3(x))
        x = hg_level_5.up2(hg_level_5.low3(x))
        x = hg_level_6.up2(hg_level_6.low3(x))
        x = hg_feat(x)
        x = hg_out(x)

        return soft_argmax(x).view(x.shape[0], -1)
    return pred


def _predictions_vitpose(x, imgs, vitpose):
    """
    Obtains the model's target predictions for an input.
    This function is used in conjunction with vmap.
    """
    x = vitpose.upscale(x) + vitpose.forward_features(imgs)
    x = vitpose.keypoint_head(x)
    
    return soft_argmax(x).view(x.shape[0], -1)


def _get_derivatives_hg(inputs, pose_net, use_hessian):
    """
    Compute the gradient and hessian wrt the input x for a model
    """
    model_is_train = pose_net.training
    pose_net.train(False)

    grads = None
    hessians = None

    _pred_fn = _predictions_hg(pose_net.hgs[-1][0], pose_net.features[-1], pose_net.outs[-1])

    grads = vmap(jacrev(_pred_fn))(inputs.unsqueeze(1)).squeeze()
    grads = grads @ grads.mT

    if use_hessian:
        inputs = inputs.chunk(8)

        hessians = torch.cat(
            [vmap(hessian(_pred_fn))(input_.unsqueeze(1)).squeeze() for input_ in inputs], dim=0)
        hessians = batched_hessian_var(hessians)

    pose_net.train(model_is_train)
    pose_net.zero_grad()    # As a precaution
    
    if use_hessian:
        return grads.detach(), hessians.detach()
    else:
        return grads.detach(), None
    

def _get_derivatives_vitpose(inputs, pose_net, use_hessian, imgs):
    """
    Compute the gradient and hessian wrt the input x for a model
    """
    model_is_train = pose_net.training
    pose_net.train(False)

    grads = None
    hessians = None
    
    grads = vmap(jacrev(_predictions_vitpose, argnums=0), in_dims=(0, 0, None))(
        inputs.unsqueeze(1), imgs.unsqueeze(1), pose_net).squeeze()
    
    grads = grads @ grads.mT

    if use_hessian:
        hessians = vmap(hessian(_predictions_vitpose, argnums=0), in_dims=(0, 0, None), chunk_size=4)(
            inputs.unsqueeze(1), imgs.unsqueeze(1), pose_net).squeeze()

        hessians = batched_hessian_var(hessians)

    pose_net.train(model_is_train)
    pose_net.zero_grad()    # As a precaution
    
    if use_hessian:
        return grads.detach(), hessians.detach()
    else:
        return grads.detach(), None


def get_tic_covariance(pose_net, pose_encodings, matrix,
                       psd_matrix, use_hessian, model_name, imgs):
    """
    Compute the Taylor Induced Covariance
    """
    with torch.no_grad():
        if model_name == 'Hourglass':
            grads, hessians = _get_derivatives_hg(pose_encodings['vector'], pose_net, use_hessian)
        else:
            assert model_name == 'ViTPose', "Model name: {} not recognized.".format(model_name)
            grads, hessians = _get_derivatives_vitpose(pose_encodings, pose_net, use_hessian, imgs)

    epsilon = matrix[:, -2:] ** 2

    if use_hessian:
        covariance = (epsilon[:, 0].view(-1, 1, 1) * grads) \
            + (epsilon[:, 1].view(-1, 1, 1) * hessians) \
            + psd_matrix
    else:
        covariance = (epsilon[:, 0].view(-1, 1, 1) * grads) + psd_matrix

    return covariance


# Batched functions
calculate_tac = vmap(calculate_tac_per_sample, in_dims=(0, 0, 0, 0))
calculate_ll = vmap(calculate_ll_per_sample, in_dims=(0, 0, 0, 0))
batched_hessian_var = vmap(pairwise_matmul_with_trace)