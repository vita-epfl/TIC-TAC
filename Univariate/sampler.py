import torch
import numpy as np


class Sampling(torch.utils.data.Dataset):
    """
    Draw samples from the sinusoidal with heteroscedastic variance
    """
    def __init__(self, sample_min: float, sample_max: float, num_train_samples: int,
                 frequency: float, varying_amplitude: bool, invert_varying: bool) -> None:
        
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.max_abs_val = max(abs(sample_min), abs(sample_max))

        self.amplitude = 5.0
        self.varying_amplitude = varying_amplitude
        self.invert_varying = invert_varying
        
        self.num_train_samples = num_train_samples
        
        self.x = np.random.uniform(low=sample_min, high=sample_max, size=num_train_samples)
        
        amplitude_ = self.get_amplitude(self.x)
        std_dev = self.get_std_dev(self.x)
        
        self.y = (amplitude_ * np.sin(frequency * self.x)) + np.random.normal(scale=std_dev)

        print('Shape of X: ', self.x.shape)
        print('Shape of Y: ', self.y.shape)
    

    def __len__(self) -> int:
        return self.num_train_samples


    def __getitem__(self, i: int) -> (float, float, int):
        return self.x[i].reshape(1), self.y[i].reshape(1), i

    
    def get_amplitude(self, x: np.array) -> np.array:
        if self.varying_amplitude:
            if self.invert_varying:
                # Max value at the centre
                return (self.max_abs_val - abs(x)) * (self.amplitude / self.max_abs_val)
            else:
                # Max value at the edge
                return abs(x) * (self.amplitude / self.max_abs_val)
        else:
            # Constant amplitude
            return self.amplitude * np.ones_like(x)


    def get_std_dev(self, x: np.array, eps: float = 1e-5) -> np.array:
        # Always heteroscedastic
        std_dev = (self.amplitude / self.max_abs_val) * abs(x)

        return std_dev + eps