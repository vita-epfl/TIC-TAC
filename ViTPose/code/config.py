import os
import yaml
import shutil
import logging
from pathlib import Path


class ParseConfig(object):
    """
    Loads and returns the configuration specified in configuration.yml
    """
    def __init__(self) -> None:

        # 1. Load the configuration file ------------------------------------------------------------------------------
        try:
            f = open('configuration.yml', 'r')
            conf_yml = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        except FileNotFoundError:
            logging.warning('Could not find configuration.yml')
            exit()


        # 2. Initializing ParseConfig object --------------------------------------------------------------------------
        self.trials = conf_yml['trials']
        self.model_name = conf_yml['model']
        self.dataset = conf_yml['dataset']
        self.experiment_settings = conf_yml['experiment_settings']
        self.architecture = conf_yml['architecture']
        self.use_hessian = conf_yml['use_hessian']
        self.load_images = conf_yml['load_images']


        # 3. Extra initializations based on configuration chosen ------------------------------------------------------
        # Number of convolutional channels for AuxNet
        self.architecture['aux_net']['channels'] = [self.architecture['hourglass']['channels']] * 7
        self.architecture['aux_net']['spatial_dim'] = [64, 32, 16, 8, 4, 2, 1]

        # Number of heatmaps (or joints)
        self.experiment_settings['num_hm'] = 14
        self.architecture['hourglass']['num_hm'] = 14
        self.architecture['aux_net']['num_hm'] = 14

        # Number of output nodes for the aux_network
        self.architecture['aux_net']['fc'].append(((2 * self.architecture['aux_net']['num_hm']) ** 2) + 2)


        # 4. Create directory for model save path ----------------------------------------------------------------------
        self.experiment_name = conf_yml['experiment_name']
        i = 1
        model_save_path = os.path.join(conf_yml['save_path'], self.experiment_name + '_' + str(i))
        while os.path.exists(model_save_path):
            i += 1
            model_save_path = os.path.join(conf_yml['save_path'], self.experiment_name + '_' + str(i))

        logging.info('Saving the model at: ' + model_save_path)

        # Copy the configuration file into the model dump path
        code_directory = Path(os.path.abspath(__file__)).parent
        shutil.copytree(src=str(code_directory),
                        dst=os.path.join(model_save_path, code_directory.parts[-1]))

        self.save_path = model_save_path