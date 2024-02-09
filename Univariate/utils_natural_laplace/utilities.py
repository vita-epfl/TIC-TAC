import sys
from pathlib import Path
from math import log, pi


import torch
import numpy as np
from torch.optim import Adam
from torch.nn.utils import parameters_to_vector


from sampler import Sampling
from regressor import Regressor


sys.path.append(str(Path(__file__).parent))
from laplace import FullLaplace
from laplace.curvature.asdl import AsdlGGN


class Regressor_NaturalHead(torch.nn.Module):
    """
    Computes the natural parameterization of the gaussian
    """
    def __init__(self, regressor: Regressor) -> None:
        super(Regressor_NaturalHead, self).__init__()
        self.model = regressor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_one, n_two = self.model(x)
        natural = torch.stack([n_one.squeeze(), -0.5 * n_two[:, 0]], dim=1)
        return natural


class Sampling_LaplaceWrapper(torch.utils.data.Dataset):
    def __init__(self, sampler: Sampling) -> None:
        self.sampler = sampler

    def __getitem__(self, i: int) -> (float, float):
        return self.sampler[i][0].astype(np.float32), self.sampler[i][1].astype(np.float32)

    def __len__(self) -> int:
        return len(self.sampler)


class NaturalLaplace():
    def __init__(self, model: Regressor, sampler: Sampling) -> None:
        model = Regressor_NaturalHead(model)
        self.full_laplace = FullLaplace
        self.backend = AsdlGGN

        self.H = len(list(model.parameters()))
        self.P = len(parameters_to_vector(model.parameters()))
        self.backend_kwargs = dict(differentiable=False, kron_jac=False)
        self.la_kwargs = dict(sod=False, single_output=False)
        self.prior_prec_init = 1e-3
        self.prior_structure = 'layerwise'
        
        self.log_prior_prec = self.get_prior_hyperparams()
        self.hyper_optimizer = None # Initialized later
        

        sampler = Sampling_LaplaceWrapper(sampler)
        self.dataloader = torch.utils.data.DataLoader(
            sampler, num_workers=0, batch_size=32, shuffle=False)

        self.lap = None


    def get_prior_hyperparams(self) -> None:
        log_prior_prec_init = np.log(self.prior_prec_init)
        log_prior_prec = log_prior_prec_init * torch.ones(self.H, device='cuda')
        log_prior_prec.requires_grad = False
        return log_prior_prec


    def heteroscedastic_mse_loss(self, input, target):
        """Heteroscedastic negative log likelihood Normal.

        Parameters
        ----------
        input : torch.Tensor (n, 2)
            two natural parameters per data point
        target : torch.Tensor (n, 1)
            targets
        """
        C = - 0.5 * log(2 * pi)
        
        assert input.ndim == target.ndim == 2
        assert input.shape[0] == target.shape[0]
        
        n, _ = input.shape
        target = torch.cat([target, target.square()], dim=1)
        inner = torch.einsum('nk,nk->n', target, input)
        log_A = input[:, 0].square() / (4 * input[:, 1]) + 0.5 * torch.log(- 2 * input[:, 1])
        log_lik = n * C + inner.sum() + log_A.sum()
        
        return - log_lik


    def hyperparameter_optimization(self, model: Regressor_NaturalHead):
        """
        Optimizing hyperparameters for Natural Laplace
        """
        self.log_prior_prec.requires_grad = True
        self.hyper_optimizer = Adam([self.log_prior_prec], lr=3e-4)

        for i in range(50):  # Code was with 50 steps
            if i == 0:
                prior_prec = torch.exp(self.log_prior_prec)
                self.lap = self.full_laplace(
                    model, 'heteroscedastic_regression', sigma_noise=1,
                    prior_precision=prior_prec, temperature=1., backend=self.backend,
                    backend_kwargs=self.backend_kwargs, **self.la_kwargs)
                self.lap.fit(self.dataloader)
            
            self.hyper_optimizer.zero_grad()
            prior_prec = torch.exp(self.log_prior_prec)
            marglik = - self.lap.log_marginal_likelihood(prior_prec, None)
            marglik.backward()
            self.hyper_optimizer.step()

        self.hyper_optimizer.zero_grad()
        self.log_prior_prec.requires_grad = False
        del self.lap, self.hyper_optimizer


def expand_prior_precision(prior_prec: torch.Tensor, model: Regressor_NaturalHead) -> torch.Tensor:
    P = len(parameters_to_vector(model.parameters()))
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device='cuda') * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.cuda()
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])