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
from sampler import Sampling
from regressor import Regressor

from utils import check_nan_in_model, calculate_tac, calculate_ll
from utils import get_positive_definite_matrix, get_tic_covariance
from utils import plot_comparison

from loss import mse_loss, nll_loss, diagonal_loss
from loss import beta_nll_loss, faithful_loss
from loss import tic_loss


plt.switch_backend('agg')
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 300


###### Configuration ######
trials = 10
max_out_dim = 20
min_out_dim = 4

experiment_name = 'Results/Trials_{}_Dims_{}_to_{}'.format(trials, min_out_dim, max_out_dim)
os.makedirs(experiment_name)


# Training config
batch_size = 32
learning_rate = 3e-4


# Number of training samples
num_samples = 1000
epochs = 100


# Training variables
Beta_BetaNLL = 0.5


training_methods = ['MSE', 'NLL: Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'NLL: TIC']


#To print out the entire matrix
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)


def train(sampler: Sampling, network: Regressor, training_pkg: dict) -> dict:
    """
    Train and compare various covariance methods.
    :param sampler: Instance of Sampling, returns samples from the multivariate distribution Q = Y + Z
    :param network: Base network which acts as the initialization for all training methods
    """

    batcher = torch.utils.data.DataLoader(sampler, num_workers=0, batch_size=batch_size, shuffle=True)

    # A Dictionary to hold the networks, loss, optimizer and scheduler for various covariance methods
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
        
        # Part 1: Training Loop
        for x, q, _, _, _, _, i in tqdm(batcher, ascii=True, position=0, leave=True):
            
            x = x.type(torch.float32).cuda()
            q = q.type(torch.float32).cuda()

            for method in training_methods:

                model = training_pkg[method]['network']
                optimizer = training_pkg[method]['optimizer']

                model.train()

                if method == 'MSE':
                    loss = mse_loss(model, x, q)

                elif method == 'NLL':
                    loss = nll_loss(model, x, q)

                elif method == 'NLL: Diagonal':
                    loss = diagonal_loss(model, x, q)

                elif method == 'Beta-NLL':
                    loss = beta_nll_loss(model, x, q, Beta_BetaNLL)

                elif method == 'Faithful':
                    loss = faithful_loss(model, x, q)

                elif method == 'NLL: TIC':
                    loss = tic_loss(model, x, q)

                else:
                    raise Exception

                training_pkg[method]['loss'][e] += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if check_nan_in_model(model):
                    print('Model: {} has NaN'.format(method))
                    exit()

        # Scheduler step
        for method in training_methods:
            training_pkg[method]['loss'][e] /= num_samples
            training_pkg[method]['scheduler'].step(training_pkg[method]['loss'][e])

    return training_pkg


def metric_calculator(sampler: Sampling, training_pkg: dict,
                      trial: int, dim: int, metric: str) -> dict:
    """
    Evaluates Task Agnostic Correlations error. Lower is better.
    Alternatively evaluates Log Likelihood. Higher is better.
    :param sampler: Instance of Sampling, returns samples from the multivariate distribution
    :param training_pkg: Corresponds to different training methods
    :param trial: Which trial (on run) is ongoing
    :param dim: Which dimension is ongoing
    """

    batcher = torch.utils.data.DataLoader(
        sampler, num_workers=0, batch_size=batch_size, shuffle=False)

    for x, q, _, _, _, _, _ in tqdm(batcher, ascii=True, position=0, leave=True):
        x = x.type(torch.float32).cuda()
        q = q.type(torch.float32).cuda()

        for method in training_methods:
            
            model = training_pkg[method]['network']
            model.eval()

            if method in ['NLL', 'Faithful']:
                y_hat, precision_hat = model(x)
                precision_hat = get_positive_definite_matrix(precision_hat, out_dim)
                
                # Stabilize inversion and avoid underflow
                precision_hat += (1e-2 * torch.eye(out_dim, device='cuda').unsqueeze(0))
                covariance_hat = torch.linalg.inv(precision_hat)

            elif method in ['NLL: Diagonal', 'Beta-NLL']:
                y_hat, var_hat = model(x)
                var_hat = var_hat[:, :out_dim] ** 2
                covariance_hat = torch.diag_embed(var_hat)
                precision_hat = torch.linalg.inv(covariance_hat)

            elif method in ['NLL: TIC']:
                y_hat, cov_hat = model(x)
                psd_matrix = get_positive_definite_matrix(cov_hat, out_dim)
                covariance_hat = get_tic_covariance(x, model, cov_hat, psd_matrix)
                precision_hat = torch.linalg.inv(covariance_hat)

            else:
                assert method == 'MSE'
                y_hat, cov_hat = model(x)
                covariance_hat = torch.eye(y_hat.shape[1], device=y_hat.device).expand(
                    y_hat.shape[0], y_hat.shape[1], y_hat.shape[1])
                precision_hat = torch.linalg.inv(covariance_hat)

            with torch.no_grad():
                if metric == 'tac':
                    loss_placeholder = torch.zeros((y_hat.shape[0], y_hat.shape[1]),
                                                   device=y_hat.device)
                    training_pkg[method]['tac']['{}'.format(dim)][trial] += calculate_tac(
                        y_hat, covariance_hat, q, loss_placeholder).sum(dim=0)

                else:
                    assert metric == 'll'
                    loss_placeholder = torch.zeros(y_hat.shape[0], device=y_hat.device)
                    training_pkg[method]['ll']['{}'.format(dim)][trial] += calculate_ll(
                        y_hat, precision_hat, q, loss_placeholder).sum()

    return training_pkg


############### SCRIPT ###############


dimensions = range(min_out_dim, max_out_dim + 1, 2)

# Placeholders to store the Task Agnostic Correlations Error
training_pkg = dict()
for method in training_methods:
    training_pkg[method] = dict()
    training_pkg[method]['tac'] = {
        'mean': torch.inf * torch.ones(len(dimensions), device='cuda'),
        'std': torch.inf * torch.ones(len(dimensions), device='cuda')}
    training_pkg[method]['ll'] = {
        'mean': torch.inf * torch.ones(len(dimensions), device='cuda'),
        'std': torch.inf * torch.ones(len(dimensions), device='cuda')}
training_pkg['training_methods'] = training_methods


for k, out_dim in enumerate(dimensions):
    num_samples *= out_dim
    in_dim = out_dim
    latent_dim = (out_dim ** 2) + 2

    for method in training_methods:
        training_pkg[method]['tac']['{}'.format(out_dim)] = torch.zeros(
            (trials, out_dim), dtype=torch.float32, device='cuda')

        training_pkg[method]['ll']['{}'.format(out_dim)] = torch.zeros(
            trials, dtype=torch.float32, device='cuda')

    # Compute the TAC for each method across trials
    for trial in range(trials):
        print('\n\n\n\n######## Evaluating Out_Dim: {}/{}\tTrial: {}/{} ########\n\n\n\n'.format(
            out_dim, max_out_dim, trial + 1, trials))

        print('\nPART 1: In-Progress: Initializing Regressor')
        network = Regressor(in_dim, out_dim, latent_dim).cuda()
        print('PART 1: Completed: Initializing Regressor\n')

        print('\nPART 2: In-Progress: Initializing Sampler')
        sampler = Sampling(in_dim, out_dim, num_samples)
        print('PART 2: Completed: Initializing Sampler\n')

        print('\nPART 3: In-Progress: Training Comparison')
        network = Regressor(in_dim, out_dim, latent_dim).cuda()
        training_pkg = train(sampler, network, training_pkg)
        print('PART 3: Completed: Training Comparison\n')

        with torch.no_grad():
            print('\nPART 4: In-Progress: Calculating TAC')
            training_pkg = metric_calculator(sampler, training_pkg, trial, out_dim, metric='tac')
            print('PART 4: Completed: Calculating TAC\n')

            print('\nPART 5: In-Progress: Calculating LL')
            training_pkg = metric_calculator(sampler, training_pkg, trial, out_dim, metric='ll')
            print('PART 5: Completed: Calculating LL\n')
    
        with open(os.path.join(experiment_name, "output.txt"), "a+") as f:
            print('Completed Out Dim: {}\tTrial: {}'.format(out_dim, trial+1), file=f)

    for metric in ['tac', 'll']:
        with open(os.path.join(experiment_name, "output.txt"), "a+") as f:
            print('\n{} metrics:\n'.format(metric.upper()), file=f)
        
        for method in training_methods:
            training_pkg[method]['{}'.format(metric)]['mean'][k] = training_pkg[
                method]['{}'.format(metric)]['{}'.format(out_dim)].mean() / num_samples

            avg_across_N = training_pkg[method]['tac']['{}'.format(out_dim)] / num_samples
            
            if metric == 'tac':
                avg_across_N_and_dim = avg_across_N.mean(dim=1)
                std_for_method_at_dim = avg_across_N_and_dim.std()
                training_pkg[method]['tac']['std'][k] = std_for_method_at_dim

            else:
                assert metric == 'll'
                std_for_method_at_dim = avg_across_N.std()
                training_pkg[method]['ll']['std'][k] = std_for_method_at_dim

            with open(os.path.join(experiment_name, "output.txt"), "a+") as f:
                print('Out Dim: {}\tName: {}\t{} Mean: {}\tStd. Dev: {}'.format(
                    out_dim, method, metric.upper(),
                    training_pkg[method]['{}'.format(metric)]['mean'][k],
                    training_pkg[method]['{}'.format(metric)]['std'][k]),
                    file=f)
        
    with open(os.path.join(experiment_name, "output.txt"), "a+") as f:
            print('\n\n', file=f)

    num_samples = num_samples // out_dim

    # Save metrics after each epoch
    torch.save(training_pkg, os.path.join(experiment_name, "training_pkg.pt"))

for metric in ['tac', 'll']:
    plot_comparison(training_pkg, dimensions, experiment_name, metric=metric)

torch.save(training_pkg, os.path.join(experiment_name, "training_pkg.pt"))