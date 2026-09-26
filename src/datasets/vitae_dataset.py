import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
from pytorch_lightning import seed_everything

from src.datasets.utils import (
    generate_random_mask, 
    train_val_test_split, 
    read_real_observation_files,
    read_realistic_sensor_mask,
    get_noise
)

def scale(
    data: np.ndarray,
    scaling_type: str,
    data_min: np.ndarray = None,
    data_max: np.ndarray = None,
    data_mean: np.ndarray = None,
    data_std: np.ndarray = None
) -> np.ndarray:
    """
    Scales the input data based on the specified scaling type. The scaling can be, min-max scaling, standard scaling or none.

    Args:
        data (np.ndarray): The input data to be scaled.
        scaling_type (str): The type of scaling to apply. Options are 'min-max', 'standard', or 'none'.
        data_min (np.ndarray, optional): Minimum values for min-max scaling (used for min-max scaling).
        data_max (np.ndarray, optional): Maximum values for min-max scaling (used for min-max scaling).
        data_mean (np.ndarray, optional): Mean values for standard scaling (used for standard scaling).
        data_std (np.ndarray, optional): Standard deviation values for standard scaling (used for standard scaling).
    Returns:
        np.ndarray: The scaled data.
    Raises:
        ValueError: If scaling_type is not one of 'min-max', 'standard', or 'none'.
    """

    assert scaling_type in ('min-max', 'standard', 'none'), f"scaling-type {scaling_type} must be either min-max or standard"

    if scaling_type == 'none':
        return data

    if scaling_type == 'min-max':
        data = (data - data_min) / (data_max - data_min + 1e-8)
    else:
        data = (data - data_mean)  / data_std

    return data


def unscale(
    data: np.ndarray,
    scaling_type: str,  
    data_min: np.ndarray = None,
    data_max: np.ndarray = None,
    data_mean: np.ndarray = None,
    data_std: np.ndarray = None
) -> np.ndarray:
    """
    Unscales the input data based on the specified scaling type. The unscaling can be, min-max scaling, standard scaling or none.

    Args:
        data (np.ndarray): The input data to be unscaled.
        scaling_type (str): The type of scaling to apply. Options are 'min-max', 'standard', or 'none'.
        data_min (np.ndarray, optional): Minimum values for min-max scaling (used for min-max scaling).
        data_max (np.ndarray, optional): Maximum values for min-max scaling (used for min-max scaling).
        data_mean (np.ndarray, optional): Mean values for standard scaling (used for standard scaling).
        data_std (np.ndarray, optional): Standard deviation values for standard scaling (used for standard scaling).
    Returns:
        np.ndarray: The unscaled data.
    Raises:
        ValueError: If scaling_type is not one of 'min-max', 'standard', or 'none'.
    """

    assert scaling_type in ('min-max', 'standard', 'none'), f"scaling-type {scaling_type} must be either min-max or standard"

    if scaling_type == 'none':
        return data

    if scaling_type == 'min-max':
        data = data * (data_max - data_min + 1e-8) + data_min
    else:
        data = data * data_std + data_mean

    return data


class VitaeDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        mask: torch.Tensor,
        sensor_type: str,
        sensor_number: int,
        scaling_type: str = 'min-max',
        timesteps: int  = 1,
        timesteps_jump: int = 1,
        seed: int = 42,
        noise: str = 'none',
        full_noise: bool = True,
        **stats
    ):
        
        self.sensor_type = sensor_type
        self.sensor_number = sensor_number
        self.timesteps = timesteps
        self.jump = timesteps_jump
        self.noise = noise
        self.full_noise = full_noise

        self.seed = seed
        seed_everything(seed=seed, verbose=False)

        data = scale(data, scaling_type, **stats)

        if self.noise != 'none' and sensor_type in ['real', 'real-random']:
            self.generated_noise = get_noise(data.shape, noise_type=self.noise, seed=self.seed, device='cpu').cpu().float().numpy()

        self.get_count = 0

        self.mask = mask
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.timesteps + 1

    def __getitem__(self, idx):
        
        img = torch.from_numpy(self.data[idx: idx + self.timesteps: self.jump])

        if self.sensor_type == 'random':
            mask = generate_random_mask(img.shape[1:], self.sensor_number).unsqueeze(0)
        elif self.sensor_type in ['real', 'real-random']:
            mask = self.mask[idx: idx + self.timesteps: self.jump]
        else:
            mask = self.mask.unsqueeze(0)

        if self.noise == "none":
            obs = img * mask
        else:
            noise = torch.from_numpy(self.generated_noise[idx: idx + self.timesteps: self.jump])

            if self.full_noise:
                img = img + noise
                obs = img * mask
            else:
                obs = (img + noise) * mask

        obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
        img = img[-1]

        self.get_count += 1

        return obs, img, mask

    def should_reset_noise(self) -> bool:
        return self.get_count >= len(self) and self.noise != 'none' and self.sensor_type in ['real', 'real-random']

    def reset_noise(self) -> None:
        if self.noise == 'none':
            return

        self.get_count = 0

        self.generated_noise = get_noise(
            self.data.shape, noise_type=self.noise, seed=self.seed, device='cpu'
        ).cpu().float().numpy()


def obtain_mask(X, sensor_type: str, sensor_number: int, seed: int) -> torch.Tensor:
    """
    Obtain the sensor mask based on the sensor type and number.

    Args:
        X (np.ndarray): The input data array.
        sensor_type (str): The type of sensor ('real', 'real-random', 'realistic', 'random', 'fixed-random', 'fixed').
        sensor_number (int): The number of sensors to be used.
        seed (int): Random seed for reproducibility.
    Returns:
        torch.Tensor: The mask tensor indicating the sensor positions.
    """

    if sensor_type in ['real', 'realistic']:
        mask = read_realistic_sensor_mask(X[0].shape)

        if sensor_type == 'real':
            real_data = torch.from_numpy(read_real_observation_files())
            real_sensor_positions = real_data != 0

            mask = real_sensor_positions * mask.unsqueeze(0)

    elif sensor_type == 'real-random':
        mask = torch.from_numpy(np.load('data/real/real_random_obs_mask.npy'))
    elif sensor_type == 'fixed':

        x_coord = np.load(f'data/x_coord_{sensor_number}.npy')
        y_coord = np.load(f'data/y_coord_{sensor_number}.npy')

        mask = np.zeros_like(X[0, 0])
        mask[x_coord, mask.shape[1] - y_coord] = True
        mask = torch.Tensor(mask).unsqueeze(0)
    elif sensor_type == 'fixed-random':
        torch.manual_seed(seed)
        mask = generate_random_mask(X[0].shape, sensor_number)
    else:
        mask = None

    return mask

    
def load_data(
    val_size: float = 0.1, 
    test_size: float = 0.1,
    split_mode: str = "monthly",
    combine_train_val: bool = False,
    sensor_type: str = 'realistic',
    sensor_number: int = 30,
    scaling_type: str = 'min-max',
    timesteps: int = 1,
    timesteps_jump: int = 1,
    seed: int = 42,
    noise: str = 'none',
    full_noise: bool = True,
    pollutant_type: str = 'all'
) -> tuple[VitaeDataset, VitaeDataset, VitaeDataset, dict[str, np.ndarray]]:
    """
    Load the Vitae dataset and return train, validation, and test datasets.
    Args:
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        split_mode (str): Mode for splitting the dataset. Options are 'monthly' or 'random'.
        combine_train_val (bool): Whether to combine training and validation sets.
        sensor_type (str): Type of sensor data to use ('real', 'real-random', 'realistic', 'random', 'fixed-random', 'fixed').
        sensor_number (int): Number of sensors to use.
        scaling_type (str): Type of scaling to apply ('min-max', 'standard', or 'none').
        timesteps (int): Number of timesteps for each sample.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing the train, validation, and test datasets, and a dictionary of dataset statistics.
    Raises:
        ValueError: If the split_mode is not one of 'monthly' or 'random'.
        ValueError: If the sensor_type is not one of 'real', 'random', 'fixed-random', or 'fixed'.
    """
    
    if sensor_type not in ['real', 'real-random', 'realistic', 'random', 'fixed-random', 'fixed']:
        raise ValueError(
            f"Invalid sensor_type '{sensor_type}'. Expected one of: 'real', 'random', 'fixed-random', 'fixed'."
        )
    
    seed_everything(seed=42, verbose=False)

    # Load the dataset
    if pollutant_type == 'o3':
        all_modalities = np.load('data/d_polair_O3.npy')

    elif pollutant_type == 'pm10':
        all_modalities = np.load('data/d_polair_PM10.npy')

    elif pollutant_type == 'pm25':
        all_modalities = np.load('data/d_polair_PM25.npy')

    elif pollutant_type == 'no2':
        all_modalities = np.load('data/d_polair_NO2.npy')

    elif pollutant_type == 'all':
        d_polair_o3 = np.load('data/d_polair_O3.npy')
        d_polair_pm10 = np.load('data/d_polair_PM10.npy')
        d_polair_pm25 = np.load('data/d_polair_PM25.npy')
        d_polair_no2 = np.load('data/d_polair_NO2.npy')
        all_modalities = np.concatenate((d_polair_o3, d_polair_pm10, d_polair_pm25, d_polair_no2), axis=1)

    # Splitting the dataset into train, val, and test
    n_samples = all_modalities.shape[0]
    
    train_indices, val_indices, test_indices = train_val_test_split(split_mode, n_samples, test_size, val_size, combine_train_val)
    
    X_test = all_modalities[test_indices]
    X_val = all_modalities[val_indices]
    X_train = all_modalities[train_indices]

    # Obtain the sensor mask
    mask = obtain_mask(X_train, sensor_type, sensor_number, seed)
    if pollutant_type == 'o3':
        mask = mask[:, 0, :, :].unsqueeze(1)
    elif pollutant_type == 'pm10':
        mask = mask[:, 1, :, :].unsqueeze(1)
    elif pollutant_type == 'pm25':
        mask = mask[:, 2, :, :].unsqueeze(1)
    elif pollutant_type == 'no2':
        mask = mask[:, 3, :, :].unsqueeze(1)

    # Compute dataset statistics for standardization
    stats = {
        'data_min': X_train.min(axis=(0, 2, 3), keepdims=True),
        'data_max': X_train.max(axis=(0, 2, 3), keepdims=True),
        'data_mean': X_train.mean(axis=(0, 2, 3), keepdims=True),
        'data_std': X_train.std(axis=(0, 2, 3), keepdims=True),
    }

    train_dataset = VitaeDataset(
        X_train, mask, sensor_type, sensor_number, 
        scaling_type, timesteps=timesteps, 
        timesteps_jump=timesteps_jump, seed=seed, 
        noise=noise, full_noise=full_noise,
        **stats)
    
    val_dataset = VitaeDataset(
        X_val, mask, sensor_type, sensor_number, 
        scaling_type, timesteps=timesteps, 
        timesteps_jump=timesteps_jump, seed=seed, 
        noise='none',
        **stats)
    
    test_dataset = VitaeDataset(
        X_test, mask, sensor_type, sensor_number, 
        scaling_type, timesteps=timesteps, 
        timesteps_jump=timesteps_jump, seed=seed,
        noise='none',
        **stats)
    
    return train_dataset, val_dataset, test_dataset, stats
