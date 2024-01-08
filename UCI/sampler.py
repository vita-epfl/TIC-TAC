import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


from regressor import Regressor


class UCI_DatasetLoader(torch.utils.data.Dataset):
    """
    Draw samples from various UCI Regression datasets
    """
    def __init__(self, dataset: str) -> None:
        
        self.fraction_variables_training = 0.25

        if dataset == 'red_wine':
            self.train_dataset, self.architecture = self.red_wine()
        
        elif dataset == 'white_wine':
            self.train_dataset, self.architecture = self.white_wine()

        elif dataset == 'superconductivity':
            self.train_dataset, self.architecture = self.superconductivity()

        elif dataset == 'energy':
            self.train_dataset, self.architecture = self.energy()

        elif dataset == 'concrete':
            self.train_dataset, self.architecture = self.concrete()

        elif dataset == 'power':
            self.train_dataset, self.architecture = self.power()

        elif dataset == 'air':
            self.train_dataset, self.architecture = self.air()

        elif dataset == 'naval':
            self.train_dataset, self.architecture = self.naval()

        elif dataset == 'electrical':
            self.train_dataset, self.architecture = self.electrical()

        elif dataset == 'abalone':
            self.train_dataset, self.architecture = self.abalone()

        elif dataset == 'gas_turbine':
            self.train_dataset, self.architecture = self.gas_turbine()

        elif dataset == 'appliances':
            self.train_dataset, self.architecture = self.appliances()

        elif dataset == 'parkinson':
            self.train_dataset, self.architecture = self.parkinson()

        else:
            raise NotImplementedError

        [self.train_x, self.train_y] = self.train_dataset

        self.network = Regressor(*self.architecture).cuda()

    
    def __len__(self) -> int:
        return self.train_x.shape[0]


    def __getitem__(self, i: int) -> (float, float, int):
        return self.train_x[i], self.train_y[i], i

    
    def red_wine(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Red Wine')
        dataset = pd.read_csv('UCI_Datasets/Wine Quality/winequality-red.csv', header=0, sep=';').dropna()
        
        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]

    
    def white_wine(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: White Wine')
        dataset = pd.read_csv('UCI_Datasets/Wine Quality/winequality-white.csv', header=0, sep=';').dropna()
        
        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))
        

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]

    
    def superconductivity(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Superconductivity')
        dataset = pd.read_csv('UCI_Datasets/Superconductivity/train.csv', header=0)
        
        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]


    def energy(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Energy')
        dataset = pd.read_csv('UCI_Datasets/Energy/ENB2012_data.csv', header=0).dropna()
        
        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]

    
    def concrete(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Concrete')
        dataset = pd.read_csv('UCI_Datasets/Concrete/Concrete_Data.csv', header=0).dropna()
        
        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]

    
    def power(self):
        print('Loading Dataset: Power')
        dataset = pd.read_csv('UCI_Datasets/Power/power.csv', header=0).dropna()

        dataset[['Date', 'Time']] = dataset['DateTime'].str.split(expand=True)
        del dataset['DateTime']

        dataset[['Month', 'Day', 'Year']] = dataset['Date'].str.split('/', expand=True)
        del dataset['Date'], dataset['Year']

        dataset[['Hour', 'Minute']] = dataset['Time'].str.split(':', expand=True)
        del dataset['Time']

        dataset = dataset[::2]

        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]

    
    def air(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Air')
        dataset = pd.read_csv('UCI_Datasets/Air Quality/AirQualityUCI.csv', header=0).dropna()
        del dataset['NMHC(GT)']


        dataset= dataset[~dataset.eq(-200).any(axis='columns')]
        
        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset[['Day', 'Month', 'Year']] = dataset['Date'].str.split('/', expand=True)
        del dataset['Date'], dataset['Year']

        dataset[['Hour', 'Minute', 'Second']] = dataset['Time'].str.split('.', expand=True)
        del dataset['Time'], dataset['Minute'], dataset['Second']

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]


    def naval(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Condition Naval Propulsion')
        dataset = np.loadtxt('UCI_Datasets/Condition Naval Propulsion/data.txt', usecols=range(18))

        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]

    
    def electrical(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Electrical')
        dataset = pd.read_csv('UCI_Datasets/Electrical/ElectricalData.csv', header=0).dropna()

        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]


    def abalone(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Abalone')
        dataset = pd.read_csv('UCI_Datasets/Abalone/abalone.csv', header=None).dropna()

        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]
    

    def gas_turbine(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Gas Turbine')
        dataset_2011 = pd.read_csv('UCI_Datasets/Gas Turbine/gt_2011.csv', header=0).dropna()
        dataset_2012 = pd.read_csv('UCI_Datasets/Gas Turbine/gt_2012.csv', header=0).dropna()
        dataset_2013 = pd.read_csv('UCI_Datasets/Gas Turbine/gt_2013.csv', header=0).dropna()
        dataset_2014 = pd.read_csv('UCI_Datasets/Gas Turbine/gt_2014.csv', header=0).dropna()
        dataset_2015 = pd.read_csv('UCI_Datasets/Gas Turbine/gt_2015.csv', header=0).dropna()

        dataset = pd.concat([dataset_2011, dataset_2012, dataset_2013, dataset_2014, dataset_2015])

        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]
    

    def appliances(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Appliances')
        dataset = pd.read_csv('UCI_Datasets/Appliances/energydata_complete.csv', header=0).dropna()
        del dataset['date']
        
        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]


    def parkinson(self) -> ([float, float], [int, int, int]):
        print('Loading Dataset: Parkinson')
        dataset_train = pd.read_csv('UCI_Datasets/Parkinson/train_data.csv', header=None).dropna()
        dataset_test = pd.read_csv('UCI_Datasets/Parkinson/test_data.csv', header=None).dropna()

        dataset_train = dataset_train.iloc[:, 1: -2]
        dataset_test = dataset_test.iloc[:, 1: -1]

        dataset = pd.concat([dataset_train, dataset_test])

        num_samples = len(dataset)
        print('Number of samples: {}'.format(num_samples))

        dataset = 10 * StandardScaler().fit_transform(dataset)
        dataset = np.random.permutation(dataset.T).T
        
        train_x = torch.from_numpy(dataset[:, :int(dataset.shape[1] * self.fraction_variables_training)])
        train_y = torch.from_numpy(dataset[:, int(dataset.shape[1] * self.fraction_variables_training):])

        in_dim = train_x.shape[1]
        out_dim = train_y.shape[1]

        print('Out Dim: ', out_dim)

        # Pass keyword arguments through dictionary
        return [train_x, train_y], [in_dim, out_dim, (out_dim ** 2) + 2]
    

    def get_network(self) -> Regressor:
        return self.network

    
    def get_num_samples(self) -> int:
        return self.train_x.shape[0]

    
    def get_out_dim(self) -> int:
        return self.train_y.shape[1]
