# System imports
import os
import copy
from tqdm import tqdm


# Science-y imports
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau


# File imports
from sampler import UCI_DatasetLoader
from regressor import Regressor

from utils import check_nan_in_model, calculate_tac
from utils import get_positive_definite_matrix, get_tic_covariance

from loss import mse_gradient, nll_gradient, diagonal_gradient
from loss import beta_nll_gradient, faithful_gradient
from loss import tic_gradient


plt.switch_backend('agg')
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 300


# Configuration
trials = 10
dataset_uci = 'concrete'
# Possible options: red_wine, white_wine, energy, concrete, power, air, naval, electrical
# abalone, gas_turbine, appliances, parkinson


experiment_name = 'Results/{}_Trials_{}'.format(dataset_uci, trials)
os.makedirs(experiment_name)


# Training config
batch_size = 32
learning_rate = 3e-4
epochs = 100

# Training variables
Beta_BetaNLL = 0.5

training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']


#To print out the entire matrix
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)


def train(sampler: UCI_DatasetLoader, network: Regressor, training_pkg: dict) -> dict:
    """
    Train and compare various covariance methods.
    :param sampler: Instance of UCI_DatasetLoader, returns samples from the selcted UCI dataset
    :param network: Base network which acts as the initialization for all training methods
    """

    num_samples = sampler.get_num_samples()
    batcher = torch.utils.data.DataLoader(sampler, num_workers=0, batch_size=batch_size, shuffle=True)

    for method in training_methods:
        training_pkg[method]['loss'] = torch.zeros(epochs, device='cuda', requires_grad=False)
        training_pkg[method]['network'] = copy.deepcopy(network).cuda()
        training_pkg[method]['optimizer'] = torch.optim.AdamW(
            training_pkg[method]['network'].parameters(), lr=learning_rate)
        training_pkg[method]['scheduler'] = ReduceLROnPlateau(
            training_pkg[method]['optimizer'],
            factor=0.25, patience=10, cooldown=0, min_lr=1e-6, verbose=True)

    for e in range(epochs):
        print('Epoch: {}/{}'.format(e+1,  epochs))

        # Training Loop
        for x, y, i in tqdm(batcher, ascii=True, position=0, leave=True):
            
            x = x.type(torch.float32).cuda()
            y = y.type(torch.float32).cuda()

            for method in training_methods:

                model = training_pkg[method]['network']
                optimizer = training_pkg[method]['optimizer']
            
                model.train()

                if method == 'MSE':
                    loss = mse_gradient(model, x, y)

                elif method == 'NLL':
                    loss = nll_gradient(model, x, y)

                elif method == 'Diagonal':
                    loss = diagonal_gradient(model, x, y)

                elif method == 'Beta-NLL':
                    loss = beta_nll_gradient(model, x, y, Beta_BetaNLL)

                elif method == 'Faithful':
                    loss = faithful_gradient(model, x, y)

                elif method == 'TIC':
                    loss = tic_gradient(model, x, y)

                else:
                    raise Exception

                training_pkg[method]['loss'][e] += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if check_nan_in_model(model):
                    print('Model: {} has NaN'.format(method))
                    exit()
        
        # Scheduler Step
        for method in training_methods:
            training_pkg[method]['loss'][e] /= num_samples
            training_pkg[method]['scheduler'].step(training_pkg[method]['loss'][e])
    
    return training_pkg


def task_agnostic_correlations(sampler: UCI_DatasetLoader, training_pkg: dict, trial: int) -> dict:
    """
    Evaluates Task Agnostic Correlations error
    :param sampler: Instance of UCI_DatasetLoader, returns samples from the selected UCI dataset
    :param training_pkg: Dictionary which stores the network for various methods
    :param trial: Which trial is ongoing
    """

    batcher = torch.utils.data.DataLoader(sampler, num_workers=0, batch_size=batch_size, shuffle=False)
    
    for x, y, _ in tqdm(batcher, ascii=True, position=0, leave=True):
        x = x.type(torch.float32).cuda()
        y = y.type(torch.float32).cuda()

        for method in training_methods:
            
            model = training_pkg[method]['network']
            model.eval()

            if method in ['NLL', 'Faithful']:
                y_hat, precision_hat = model(x)
                precision_hat = get_positive_definite_matrix(precision_hat, out_dim)
                covariance_hat = torch.linalg.inv(precision_hat)

            elif method in ['Diagonal', 'Beta-NLL']:
                y_hat, var_hat = model(x)
                var_hat = var_hat[:, :out_dim] ** 2
                covariance_hat = torch.diag_embed(var_hat)

            elif method in ['TIC']:
                y_hat, cov_hat = model(x)
                psd_matrix = get_positive_definite_matrix(cov_hat, out_dim)
                covariance_hat = get_tic_covariance(x, model, cov_hat, psd_matrix)

            else:
                assert method == 'MSE'
                y_hat, cov_hat = model(x)
                covariance_hat = torch.eye(y_hat.shape[1], device=y_hat.device).expand(
                    y_hat.shape[0], y_hat.shape[1], y_hat.shape[1])

            with torch.no_grad():
                loss_placeholder = torch.zeros((y_hat.shape[0], y_hat.shape[1]), device=y_hat.device)
                training_pkg[method]['tac'][trial] += calculate_tac(
                    y_hat, covariance_hat, y, loss_placeholder).sum(dim=0)
                
    return training_pkg


############### SCRIPT ###############


sampler = UCI_DatasetLoader(dataset_uci)
out_dim = sampler.get_out_dim()
num_samples = sampler.get_num_samples()

training_pkg = dict()
for method in training_methods:
    training_pkg[method] = dict()
    training_pkg[method]['tac'] = torch.zeros((trials, out_dim), dtype=torch.float32, device='cuda')
training_pkg['training_methods'] = training_methods


# Compute the TAC metric for each method across trials
for trial in range(trials):
    print(' \n\n\n\n######## Evaluating Dataset: {}\tTrial: {}/{} ########\n\n\n\n'.format(dataset_uci, trial + 1, trials))

    print('\nPART 1: In-Progress: Initializing Sampler and Regressor')
    sampler = UCI_DatasetLoader(dataset_uci)
    network = sampler.get_network()
    print('PART 1: Completed: Initializing Sampler and Regressor\n')

    print('\nPART 2: In-Progress: Training Comparison')
    training_pkg = train(sampler, network, training_pkg)
    print('PART 2: Completed: Training Comparison\n')

    with torch.no_grad():
        print('\nPART 3: In-Progress: Calculating TAC')
        training_pkg = task_agnostic_correlations(sampler, training_pkg, trial)
        print('PART 3: Completed: Calculating TAC\n')

    with open(os.path.join(experiment_name, "output.txt"), "a+") as f:
        print('Completed Trial: {}'.format(trial+1), file=f)


with open(os.path.join(experiment_name, "output.txt"), "a+") as f:
        print('\nTAC metrics:\n', file=f)


for method in training_methods:
    with open(os.path.join(experiment_name, "output.txt"), "a+") as f:
        print('Name: {}\tTAC error: {}'.format(
            method, training_pkg[method]['tac'].mean().cpu().numpy() / num_samples), file=f)


torch.save(training_pkg, os.path.join(experiment_name, "training_pkg.pt"))