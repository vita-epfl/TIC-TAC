# System imports
import os
import copy
import logging
from tqdm import tqdm
from typing import Union


# Science-y imports
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# File imports
from config import ParseConfig
from dataloader import load_hp_dataset, HumanPoseDataLoader

from models.auxiliary.AuxiliaryNet import AuxNet_HG, AuxNet_ViTPose
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
from models.vit_pose import vitpose_config
from models.vit_pose.ViTPose import ViTPose

from utils.pose import fast_argmax, soft_argmax
from utils.pose import heatmap_loss, count_parameters

from utils.tic import get_positive_definite_matrix, get_tic_covariance
from utils.tic import calculate_tac, calculate_ll

from loss import mse_loss, nll_loss, diagonal_loss
from loss import beta_nll_loss, faithful_loss
from loss import tic_loss

# Global declarations
logging.getLogger().setLevel(logging.INFO)
os.chdir(os.path.dirname(os.path.realpath(__file__)))


training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']


class Train(object):
    def __init__(self, sampler: HumanPoseDataLoader, models: tuple, conf: ParseConfig,
                 training_pkg: dict, trial: int) -> None:
        """
        Train and compare various covariance methods.
        :param sampler: Instance of HumanPoseDataLoader, samples from MPII + LSP
        :param models: Contains (Hourglass or ViTPose, AuxNet) models
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        """

        self.conf = conf
        self.sampler = sampler
        self.training_pkg = training_pkg
        self.trial = trial

        # Experiment Settings
        self.batch_size = conf.experiment_settings['batch_size']
        self.epoch = conf.experiment_settings['epochs']
        self.learning_rate = conf.experiment_settings['lr']
        
        self.loss_fn = torch.nn.MSELoss()  # MSE
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        self.joint_names = self.sampler.ind_to_jnt
        self.model_save_path = conf.save_path

        self.torch_dataloader = DataLoader(
            self.sampler, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)

        for method in training_methods:
            self.training_pkg[method]['networks'] = (
                copy.deepcopy(models[0]).cuda(), copy.deepcopy(models[1]).cuda())
            
            params = ({'params': self.training_pkg[method]['networks'][0].parameters()},
                      {'params': self.training_pkg[method]['networks'][1].parameters()})
            self.training_pkg[method]['optimizer'] = torch.optim.Adam(params, lr=self.learning_rate)
            
            self.training_pkg[method]['scheduler'] = ReduceLROnPlateau(
                self.training_pkg[method]['optimizer'],
                factor=0.25, patience=10, cooldown=0, min_lr=1e-6, verbose=True)


    def train_model(self) -> dict:
        """
        Training loop
        """
        print("Covariance Estimation: training: Epochs - {}\tBatch Size - {}".format(
            self.epoch, self.batch_size))

        for e in range(self.epoch):

            self.sampler.set_augmentation(augment=True)

            # Training loop
            logging.info('Training for epoch: {}'.format(e+1))
            for images, heatmaps in tqdm(self.torch_dataloader, ascii=True):
                
                for method in training_methods:
                    net = self.training_pkg[method]['networks'][0]
                    aux_net = self.training_pkg[method]['networks'][1]
                    optimizer = self.training_pkg[method]['optimizer']

                    net.train()
                    aux_net.train()
                    
                    outputs, pose_features = net(images)        # images.cuda() done internally within the model
                    hm_loss = heatmap_loss(outputs, heatmaps)   # heatmaps transferred to GPU within the function
                    
                    # Flattened predictions
                    pred_uv = soft_argmax(outputs[:, -1]).view(
                        outputs.shape[0], self.num_hm * 2)
                    
                    # Flattened ground truths
                    gt_uv = fast_argmax(heatmaps.to(pred_uv.device)).view(
                        outputs.shape[0], self.num_hm * 2)

                    loss_covariance = self.covariance_estimation(
                        aux_net=aux_net, pose_net=net, pose_encodings=pose_features,
                        pred=pred_uv, gt=gt_uv, name=method, imgs=images)
                    
                    # Weight update
                    (loss_covariance + torch.mean(hm_loss)).backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
            self.validation(e)

        return self.training_pkg


    def validation(self, e: int) -> None:
        """
        Runs the validation loop
        """
        
        with torch.no_grad():

            self.sampler.set_augmentation(augment=False)

            for images, heatmaps in tqdm(self.torch_dataloader, ascii=True):
                for method in training_methods:
                    net = self.training_pkg[method]['networks'][0]
                    aux_net = self.training_pkg[method]['networks'][1]

                    net.eval()
                    aux_net.eval()

                    outputs, pose_features = net(images)
                    
                    # Flattened predictions
                    pred_uv = soft_argmax(outputs[:, -1]).view(
                        outputs.shape[0], self.num_hm * 2)
                        
                    gt_uv = fast_argmax(heatmaps.to(pred_uv.device)).view(
                        outputs.shape[0], self.num_hm * 2)

                    regression_loss = torch.sqrt(torch.sum((pred_uv - gt_uv) ** 2, dim=-1))

                    self.training_pkg[method]['loss'][self.trial][e] += regression_loss.sum()

            # Print the covariance loss
            for method in training_methods:
                self.training_pkg[method]['loss'][self.trial][e] /= len(self.sampler)
                print('Name: {}\t\tLoss: {}'.format(
                    method, self.training_pkg[method]['loss'][self.trial][e]))
            
            print()

            # Scheduler step
            for method in training_methods:
                self.training_pkg[method]['scheduler'].step(
                    self.training_pkg[method]['loss'][self.trial][e])

    
    def covariance_estimation(self, aux_net: Union[AuxNet_HG, AuxNet_ViTPose],
                              pose_net: Union[Hourglass, ViTPose], pose_encodings: dict, 
                              pred: torch.Tensor, gt: torch.Tensor, name: str, imgs: torch.Tensor) -> torch.Tensor:
        """
        Computing the full covariance matrix

        :param aux_net: The covariance estimator
        :param pose_net: Hourglass model
        :param pose_encodings: (Dict of tensors) Intermediate and penultimate layer outputs of the pose model
        :param pred: Tensor of size (batch_size, num_joints * 2) Hourglass prediction
        :param gt: Tensor of size (batch_size, num_joints * 2) Pose ground truth
        :param name: Name of the covariance estimation method
        :return: Negative loglikelihood for the corresponding method
        """
        out_dim = 2 * self.num_hm

        # Get aux_net outputs -----------------------------------------------------------------------
        matrix = self._aux_net_inference(pose_encodings, aux_net)

        # Compute y - y_hat ---------------------------------------------------------------------------------------------
        means =  pred - gt 

        # Various covariance implentations ------------------------------------------------------------
        if name == 'MSE':
            loss = mse_loss(means)

        elif name == 'NLL':
            loss = nll_loss(means, matrix, out_dim)
        
        elif name == 'Diagonal':
            loss = diagonal_loss(means, matrix, out_dim)

        elif name == 'Beta-NLL':
            loss = beta_nll_loss(means, matrix, out_dim)

        elif name == 'Faithful':
            loss = faithful_loss(means, matrix, out_dim)

        elif name == 'TIC':
            loss = tic_loss(means, matrix, out_dim, pose_net, pose_encodings,
                            self.conf.use_hessian, self.conf.model_name, imgs)
            
        else:
            raise NotImplementedError

        return loss


    def _aux_net_inference(self, pose_features: dict, aux_net: Union[AuxNet_HG, AuxNet_ViTPose]) -> torch.Tensor:
        """
        Obtaining the flattened matrix from the aux net inference module
        """
        if self.conf.model_name == 'Hourglass':
            with torch.no_grad():
                depth = len(self.conf.architecture['aux_net']['spatial_dim'])
                encodings = torch.cat(
                    [pose_features['feature_{}'.format(i)].reshape(
                        self.batch_size, pose_features['feature_{}'.format(i)].shape[1], -1) \
                        for i in range(depth, 0, -1)],
                    dim=2)
        else:
            encodings = pose_features

        aux_out = aux_net(encodings)
        return aux_out


class Evaluate(object):
    def __init__(self, sampler: HumanPoseDataLoader, conf: ParseConfig,
                 training_pkg: dict, trial: int) -> None:
        """
        Compare the TAC error or likelihood for various covariance methods.
        :param sampler: Instance of HumanPoseDataLoader, samples from MPII + LSP
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        """

        self.sampler = sampler
        self.conf = conf
        self.training_pkg = training_pkg
        self.trial = trial
        
        self.batch_size = conf.experiment_settings['batch_size']
        self.ind_to_jnt = self.sampler.ind_to_jnt
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps

        self.torch_dataloader = DataLoader(
            self.sampler, batch_size=self.batch_size, shuffle=False, num_workers=1, drop_last=True)


    def calculate_metric(self, metric: str) -> None:
        """
        Calculate and store TAC or LL for all methods
        """
        print("Covariance Estimation: {} Evaluation".format(metric.upper()))

        self.sampler.set_augmentation(augment=False)

        with torch.no_grad():
            for images, heatmaps in tqdm(self.torch_dataloader, ascii=True):
                for method in training_methods:
                    
                    net = self.training_pkg[method]['networks'][0]
                    aux_net = self.training_pkg[method]['networks'][1]

                    net.eval()
                    aux_net.eval()

                    outputs, pose_features = net(images)

                    # At 64 x 64 level
                    pred_uv = soft_argmax(outputs[:, -1]).view(
                        outputs.shape[0], self.num_hm * 2)
                    gt_uv = fast_argmax(heatmaps.to(pred_uv.device)).view(
                        outputs.shape[0], self.num_hm * 2)

                    matrix = self._aux_net_inference(pose_features, aux_net)
                    covariance = self._get_covariance(method, matrix, net, pose_features, images)
                    precision = torch.linalg.inv(covariance)

                    if metric == 'tac':
                        loss_placeholder = torch.zeros((pred_uv.shape[0], self.num_hm),
                                                       device=pred_uv.device)
                        self.training_pkg[method]['tac'][self.trial] += calculate_tac(
                            pred_uv, covariance, gt_uv, loss_placeholder).sum(dim=0)

                    else:
                        assert metric == 'll'
                        loss_placeholder = torch.zeros(pred_uv.shape[0],
                                                       device=pred_uv.device)
                        self.training_pkg[method]['ll'][self.trial] += calculate_ll(
                            pred_uv, precision, gt_uv, loss_placeholder).sum()

            # Save TAC
            with open(os.path.join(self.conf.save_path, "output_{}.txt".format(self.trial)), "a+") as f:
                for method in training_methods:
                    cumulative_error = self.training_pkg[method]['{}'.format(metric)][
                        self.trial].mean().cpu().numpy()
                    print('Method: {}\t{} (Mean): '.format(method, metric.upper()),
                          cumulative_error / len(self.sampler), file=f)

                    print('\n', file=f)

                print('\n\n', file=f)
                if metric == 'tac':
                    for method in training_methods:
                        print('Method: {}\tTAC (joint): '.format(method), self.training_pkg[method]['tac'][
                            self.trial].cpu().numpy() / len(self.sampler), file=f)
                    print('\n\n', file=f)    

            torch.save(self.training_pkg,
                os.path.join(self.conf.save_path, "training_pkg_{}.pt".format(self.trial)))


    def _get_covariance(self, name: str, matrix: torch.Tensor, pose_net: Union[Hourglass, ViTPose],
                        pose_encodings: dict, imgs: torch.Tensor) -> torch.Tensor:
        
        out_dim = 2 * self.num_hm

        if name == 'MSE':
            return torch.eye(out_dim).expand(matrix.shape[0], out_dim, out_dim).cuda()

        # Various covariance implentations ------------------------------------------------------------
        elif name in ['NLL', 'Faithful']:
            precision_hat = get_positive_definite_matrix(matrix, out_dim)
            return torch.linalg.inv(precision_hat)
        
        elif name in ['Diagonal', 'Beta-NLL']:
            var_hat = matrix[:, :out_dim] ** 2
            return torch.diag_embed(var_hat)

        elif name in ['TIC']:
            psd_matrix = get_positive_definite_matrix(matrix, out_dim)
            covariance_hat = get_tic_covariance(
                pose_net, pose_encodings, matrix, psd_matrix, self.conf.use_hessian, self.conf.model_name, imgs)
            
            return covariance_hat

        else:
            raise NotImplementedError


    def _aux_net_inference(self, pose_features: dict,
                           aux_net: Union[AuxNet_HG, AuxNet_ViTPose]) -> torch.Tensor:
        """
        Obtaining the flattened matrix from the aux net inference module
        """
        if self.conf.model_name == 'Hourglass':
            with torch.no_grad():
                depth = len(self.conf.architecture['aux_net']['spatial_dim'])
                encodings = torch.cat(
                    [pose_features['feature_{}'.format(i)].reshape(
                        self.batch_size, pose_features['feature_{}'.format(i)].shape[1], -1) \
                        for i in range(depth, 0, -1)],
                    dim=2)
        else:
            encodings = pose_features

        aux_out = aux_net(encodings)
        return aux_out


def init_models(conf: ParseConfig) -> tuple:
    """
    Initializes and returns Hourglass and AuxNet models
    """

    logging.info('Initializing Auxiliary Network')
    

    if conf.model_name == 'ViTPose':
        logging.info('Initializing ViTPose Network')
        pose_net = ViTPose(vitpose_config.model).cuda()
        aux_net = AuxNet_ViTPose(arch=conf.architecture['aux_net'])
        aux_net.cuda(torch.device('cuda:{}'.format(torch.cuda.device_count()-1)))
        print('Number of parameters (ViTPose): {}\n'.format(count_parameters(pose_net)))
    
    else:
        logging.info('Initializing Hourglass Network')
        pose_net = Hourglass(arch=conf.architecture['hourglass'])
        aux_net = AuxNet_HG(arch=conf.architecture['aux_net'])
        aux_net.cuda(torch.device('cuda:{}'.format(torch.cuda.device_count()-1)))
        print('Number of parameters (Hourglass): {}\n'.format(count_parameters(pose_net)))

    logging.info('Successful: Model transferred to GPUs.\n')

    return pose_net, aux_net


def main() -> None:
    """
    Control flow for the code
    """

    # 1. Load configuration file ----------------------------------------------------------------------------
    logging.info('Loading configurations.\n')
    conf  = ParseConfig()

    num_hm = conf.architecture['aux_net']['num_hm']
    epochs = conf.experiment_settings['epochs']
    trials = conf.trials

    training_pkg = dict()
    for method in training_methods:
        training_pkg[method] = dict()
        training_pkg[method]['tac'] = torch.zeros((trials, num_hm), dtype=torch.float32, device='cuda')
        training_pkg[method]['ll'] = torch.zeros(trials, dtype=torch.float32, device='cuda')
        training_pkg[method]['loss'] = torch.zeros((trials, epochs), device='cuda')
    training_pkg['training_methods'] = training_methods 
    

    # 2. Loading datasets -----------------------------------------------------------------------------------
    logging.info('Loading pose dataset(s)\n')
    dataset_dict = load_hp_dataset(dataset_conf=conf.dataset, load_images=conf.load_images)


    # 3. Defining DataLoader --------------------------------------------------------------------------------
    logging.info('Defining DataLoader.\n')
    dataset = HumanPoseDataLoader(dataset_dict=dataset_dict, conf=conf)



    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Defining the network -------------------------------------------------------------------------
        logging.info('Initializing human pose network and auxiliary network.\n')
        pose_model, aux_net = init_models(conf=conf)


        # 4.b: Train the covariance approximation model
        train_obj = Train(dataset, (pose_model, aux_net), conf, training_pkg, trial)
        training_pkg = train_obj.train_model()


        with torch.no_grad():
            eval_obj = Evaluate(
                sampler=dataset, conf=conf, training_pkg=training_pkg, trial=trial)
            
            eval_obj.calculate_metric(metric='tac')
            eval_obj.calculate_metric(metric='ll')


main()