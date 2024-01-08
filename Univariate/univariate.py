# System imports
import os
import copy
import subprocess
from tqdm import tqdm


# Science-y imports
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau


# File imports
from sampler import Sampling
from regressor import Regressor

from utils import check_nan_in_model, get_tic_variance, plot_sine

from loss import mse_gradient, nll_gradient
from loss import beta_nll_gradient, faithful_gradient
from loss import tic_gradient


# Matplotlilb settings
plt.switch_backend('agg')
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (45,6)


###### Configuration ######
# Sinusoidal Configuration
varying_amplitude = False
invert_varying = None
frequency = 2 * np.pi * 1

# Create experiment folder
experiment_name = 'Results/VaryingAmplitude_{}_and_InvertVarying_{}'.format(
    varying_amplitude, invert_varying)


os.makedirs(experiment_name)
os.mkdir(os.path.join(experiment_name, 'Sine'))
os.mkdir(os.path.join(experiment_name, 'PDF'))

# Training config
batch_size = 32
learning_rate = 3e-4

# Architecture
latent_dim = 50
mu_blocks = 5
var_blocks = 5

sample_min = -5.0
sample_max = 5.0

# Number of training samples
num_train_samples =  50_000
epochs = 100

# Plotting sine wave
N = 1000

# Training variables: Beta-NLL
Beta_BetaNLL = 0.5

training_methods = ['MSE', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']

#To print out the entire matrix
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)


def train(sampler: Sampling, network: Regressor) -> None:
    """
    Train and compare various covariance methods.
    :param sampler: Instance of Sampling, returns samples from the simulated sinusoidal
    :param network: Base network which acts as the initialization for all training methods
    """

    batcher = torch.utils.data.DataLoader(sampler, num_workers=0, batch_size=batch_size, shuffle=True)

    # A Dictionary to hold the networks, loss, optimizer and scheduler for various covariance methods
    training_pkg = dict()
    for method in training_methods:
        training_pkg[method] = dict()
        training_pkg[method]['network'] = copy.deepcopy(network).cuda()
        training_pkg[method]['loss'] = torch.tensor(0., device='cuda', requires_grad=False)
        training_pkg[method]['optimizer'] = torch.optim.AdamW(
            training_pkg[method]['network'].parameters(), lr=learning_rate)
        training_pkg[method]['scheduler'] = ReduceLROnPlateau(
            training_pkg[method]['optimizer'],
            factor=0.25, patience=10, cooldown=0, min_lr=1e-6, verbose=True)

    for e in range(epochs):
        print('Epoch: {}/{}'.format(e+1,  epochs))
        
        for x, y, i in tqdm(batcher, ascii=True, position=0, leave=True):
            
            x = x.unsqueeze(1).type(torch.float32).cuda()
            y = y.unsqueeze(1).type(torch.float32).cuda()

            for method in training_methods:

                model = training_pkg[method]['network']
                optimizer = training_pkg[method]['optimizer']

                model.train()

                if method == 'MSE':
                    loss = mse_gradient(model, x, y)

                elif method == 'NLL':
                    loss = nll_gradient(model, x, y)

                elif method == 'Beta-NLL':
                    loss = beta_nll_gradient(model, x, y, Beta_BetaNLL)

                elif method == 'Faithful':
                    loss = faithful_gradient(model, x, y)

                elif method == 'TIC':
                    loss = tic_gradient(model, x, y)

                else:
                    raise Exception

                training_pkg[method]['loss'] += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if check_nan_in_model(model): 
                    print('Model: {} has NaN'.format(method))
                    exit()

        for method in training_methods:
            training_pkg[method]['scheduler'].step(training_pkg[method]['loss'] / num_train_samples)
            training_pkg[method]['loss'] = torch.tensor(0., device='cuda', requires_grad=False)
        
        # Plot the predictions
        with torch.no_grad():
            fig, ax = plt.subplots(nrows=1, ncols=6, sharey='row', sharex='col')
            plt.subplots_adjust(wspace=0, hspace=0)

            x_axis_plot = torch.linspace(sample_min, sample_max, steps=N).type(torch.float32).numpy()
            x_axis = torch.from_numpy(x_axis_plot).unsqueeze(1).cuda()
            
            # First plot the ground truth mean and standard deviation
            sine_wave = sampler.get_amplitude(x_axis_plot) * np.sin(frequency * x_axis_plot)
            std_dev = sampler.get_std_dev(x_axis_plot)

            ax = plot_sine(ax, x_axis_plot, sine_wave, sine_wave, std_dev, 'black', 'Ground Truth', 0)

            # Plot the predicted mean and variance for all methods
            color = ['purple', 'green', 'crimson', 'lightseagreen', 'coral']
            for i, method in enumerate(training_methods, start=1):
                
                model = training_pkg[method]['network']
                model.eval()

                y_hat, vars_hat = model(x_axis)
                var_hat = vars_hat[:, 0].unsqueeze(1)

                if method == 'MSE':
                    std_dev = torch.zeros_like(var_hat)
                    
                elif method in ['NLL', 'Beta-NLL', 'Faithful']:
                    std_dev = var_hat ** 0.5
                        
                elif method in ['TIC']:
                    variance_hat = get_tic_variance(x_axis, model, vars_hat)    
                    std_dev = variance_hat ** 0.5
                
                else:
                    raise Exception

                std_dev = std_dev.squeeze().cpu().numpy()
                y_hat = y_hat.squeeze().cpu().numpy()
                c = color.pop()

                # Plot Ground Truth
                ax = plot_sine(ax, x_axis_plot, sine_wave, y_hat, std_dev, c, method, i)

            plt.savefig(os.path.join(experiment_name, "Sine/file%07d.png" % (e+1)), format='png', bbox_inches="tight", dpi=150)
            plt.savefig(os.path.join(experiment_name, "PDF/sine%07d.pdf" % (e+1)), format='pdf', bbox_inches="tight")
            plt.close()

    # Save the models
    torch.save(training_pkg, os.path.join(experiment_name, "training_pkg.pt"))
    

# Print the configuration
print('Config:')
print('Is Varying Amplitude? {}'.format(varying_amplitude))
print('Is Amplitude Inverted? {}\n'.format(invert_varying))


print('\nPART 1: In-Progress: Initializing Regressor')
network = Regressor(latent_dim, mu_blocks, var_blocks)
print('PART 1: Completed: Initializing Regressor\n')

print('\nPART 2: In-Progress: Initializing Sampler')
sampler = Sampling(sample_min, sample_max, num_train_samples, frequency, varying_amplitude, invert_varying)
print('PART 2: Completed: Initializing Sampler\n')

print('\nPART 3: In-Progress: Training Comparison')
network = Regressor(latent_dim, mu_blocks, var_blocks)
train(sampler, network)
print('PART 3: Completed: Training Comparison\n')

print('\nPART 4: In-Progress: Creating Video')
subprocess.call([
    'ffmpeg',
    '-framerate', '18',
    '-i', os.path.join(os.getcwd(), experiment_name, 'Sine/file%07d.png'),
    '-r', '18',
    '-pix_fmt', 'yuv420p',
    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
    os.path.join(os.getcwd(), experiment_name, 'sine_regression.mp4')])
print('PART 4: Completed: Creating Video\n')


# Print the configuration
print('\nConfig:')
print('Is Varying Amplitude? {}'.format(varying_amplitude))
print('Is Amplitude Inverted? {}'.format(invert_varying))