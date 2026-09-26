import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
from pytorch_lightning import seed_everything

from src.utils.voronoi import voronoi_tessellation, batched_voronoi_tessellation, build_voronoi_assign, apply_voronoi_gather
from src.datasets.utils import (
    generate_random_mask, 
    train_val_test_split,
    read_voronoi_data,
    get_noise
)


def scale(
    data: np.ndarray,
    scaling_type: str,
    stats
) -> np.ndarray:
    """
    Scales the input data based on the specified scaling type. The scaling can be, min-max scaling, standard scaling or none.
    Args:
        data (np.ndarray): The input data to be scaled.
        scaling_type (str): The type of scaling to apply. Options are 'min-max', 'standard', or 'none'.
        stats (dict): A dictionary containing the statistics for scaling.
            - For 'min-max': {'Y_max': np.ndarray, 'Y_min': np.ndarray}
            - For 'standard': {'Y_mean': np.ndarray, 'Y_std': np.ndarray}
    Returns:
        np.ndarray: The scaled data.
    Raises:
        AssertionError: If scaling_type is not one of 'min-max', 'standard', or 'none'.
    """

    assert scaling_type in ('min-max', 'standard', 'none'), f"scaling-type {scaling_type} must be either min-max or standard"

    if scaling_type == 'none':
        return data

    if scaling_type == 'min-max':
        Y_max = stats['Y_max']
        Y_min = stats['Y_min']

        data = (data - Y_min) / (Y_max - Y_min + 1e-8)
    else:
        Y_mean = stats['Y_mean']
        Y_std = stats['Y_std']

        data = (data - Y_mean) / Y_std

    return data


def unscale(
    data: np.ndarray,
    scaling_type: str,
    Y_min: np.ndarray = None,
    Y_max: np.ndarray = None,
    Y_mean: np.ndarray = None,
    Y_std: np.ndarray = None
) -> np.ndarray:
    """
    Unscales the input data based on the specified scaling type. The unscaling can be, min-max scaling, standard scaling or none.
    Args:
        data (np.ndarray): The input data to be unscaled.
        scaling_type (str): The type of scaling to apply. Options are 'min-max', 'standard', or 'none'.
        Y_min (np.ndarray, optional): Minimum values for min-max scaling (used for min-max scaling).
        Y_max (np.ndarray, optional): Maximum values for min-max scaling (used for min-max scaling).
        Y_mean (np.ndarray, optional): Mean values for standard scaling (used for standard scaling).
        Y_std (np.ndarray, optional): Standard deviation values for standard scaling (used for standard scaling).
    Returns:
        np.ndarray: The unscaled data.
    Raises:
        AssertionError: If scaling_type is not one of 'min-max', 'standard', or 'none'.
    """
    
    assert scaling_type in ('min-max', 'standard', 'none'), f"scaling-type {scaling_type} must be either min-max or standard"

    if scaling_type == 'none':
        return data

    if scaling_type == 'min-max':
        assert Y_min is not None and Y_max is not None, "Y_min and Y_max must be provided for min-max scaling"
        data = data * (Y_max - Y_min + 1e-8) + Y_min
    else:
        assert Y_mean is not None and Y_std is not None, "Y_mean and Y_std must be provided for standard scaling"
        data = data * Y_std + Y_mean

    return data

@torch.no_grad()
def generate_noisy_voronoi_fast(
    assignment: torch.Tensor,
    Y: torch.Tensor,
    noise_type: str = "gaussian",
    full_noise: bool = True,
    noise_params: dict = None,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_Y = Y + get_noise(Y.shape, noise_type=noise_type, noise_params=noise_params, seed=seed, device='cuda').cpu()
    
    len_Y = Y.shape[0]
    noisy_X = apply_voronoi_gather(noisy_Y, assignment[:len_Y])

    if full_noise:
        return noisy_X, noisy_Y
    else:
        return noisy_X, Y


def generate_noisy_voronoi(
    mask: torch.Tensor,
    Y: torch.Tensor,
    noise_type: str = 'gaussian',
    full_noise: bool = False,
    noise_params: dict = None,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    len_Y = Y.shape[0]

    mask = mask[:len_Y]
    noise = get_noise(Y.shape, noise_type=noise_type, noise_params=noise_params, seed=seed, device='cuda').cpu()

    noisy_Y = (Y + noise).to('cuda')

    noisy_X = batched_voronoi_tessellation(
        mask.to('cuda'),
        noisy_Y
    ).cpu()

    if full_noise:
        return noisy_X, noisy_Y.cpu()
    else:
        return noisy_X, Y.cpu()

class VCNNDataset(Dataset):
    def __init__(
            self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            mask: np.ndarray, 
            sensor_type: str,
            sensor_number: int,
            scaling_type: str = 'min-max',
            timesteps: int = 1,
            timesteps_jump: int = 1,
            seed: int = 42,
            noise: str = 'none',
            full_noise: bool = False,
            noise_params: dict = None,
            **stats
            ):

        self.sensor_type = sensor_type
        self.sensor_number = sensor_number
        self.timesteps = timesteps
        self.jump = timesteps_jump
        self.noise = noise
        self.full_noise = full_noise
        self.noise_params = noise_params

        self.seed = seed
        seed_everything(seed=seed, verbose=False)

        if sensor_type != 'random':
            X = torch.Tensor(scale(X, scaling_type, stats))
            Y = torch.Tensor(scale(Y, scaling_type, stats))
            
            mask = torch.Tensor(mask)
        else:
            X = torch.empty(0)
            Y = torch.Tensor(scale(Y, scaling_type, stats))

        if self.noise != 'none' and sensor_type in ['real', 'real-random']:
            X, Y = generate_noisy_voronoi(mask, Y, noise_type=self.noise, full_noise=self.full_noise, noise_params=self.noise_params, seed=self.seed)

        self.get_count = 0

        # Store the data
        self.mask = mask
        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.Y) - self.timesteps + 1

    def __getitem__(self, idx):
        Y = self.Y[idx: idx + self.timesteps: self.jump]

        if self.sensor_type == 'random':
            mask = generate_random_mask(Y.shape[1:], self.sensor_number)

            # TODO: add noise

            X = voronoi_tessellation(mask, Y)
        elif self.sensor_type in ['real', 'real-random']:
            X = self.X[idx: idx + self.timesteps: self.jump]
            mask = self.mask[idx: idx + self.timesteps: self.jump]
            # Repeat the mask for timesteps
            #if self.timesteps>1:
            #    mask[idx+1:] = mask[idx]
            mask = mask.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])

            self.get_count += 1

            if self.should_reset_noise():
                self.reset_noise()
        else:
            #  TODO: add noise
            
            X = self.X[idx: idx + self.timesteps: self.jump]
            mask = self.mask

        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])
        X = torch.cat((X, mask), dim=0)

        Y = Y[-1]

        return X, Y

    def should_reset_noise(self) -> bool:
        return self.get_count >= len(self) and self.noise != 'none' and self.sensor_type in ['real', 'real-random']

    def reset_noise(self) -> None:
        if self.noise == 'none':
            return

        self.get_count = 0

        self.X, self.Y = generate_noisy_voronoi(
            self.mask, self.Y,
            noise_type=self.noise, full_noise=self.full_noise, noise_params=self.noise_params,
            seed=self.seed
        )

class VCLSTMDataset(Dataset):
    def __init__(
            self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            mask: np.ndarray, 
            sensor_type: str,
            sensor_number: int,
            scaling_type: str = 'min-max',
            timesteps: int = 1,
            timesteps_jump: int = 1,
            seed: int = 42,
            noise: str = 'none',
            full_noise: bool = False,
            noise_params: dict = None,
            **stats
            ):

        self.sensor_type = sensor_type
        self.sensor_number = sensor_number
        self.timesteps = timesteps
        self.jump = timesteps_jump
        self.noise = noise
        self.full_noise = full_noise
        self.noise_params = noise_params

        self.seed = seed
        seed_everything(seed=seed, verbose=False)

        if sensor_type != 'random':
            X = torch.Tensor(scale(X, scaling_type, stats))
            Y = torch.Tensor(scale(Y, scaling_type, stats))
            
            mask = torch.Tensor(mask)
        else:
            X = torch.empty(0)
            Y = torch.Tensor(scale(Y, scaling_type, stats))

        if self.noise != 'none' and sensor_type in ['real', 'real-random']:
            assignment = build_voronoi_assign(mask)

            X, Y = generate_noisy_voronoi_fast(assignment, Y, noise_type=self.noise, full_noise=self.full_noise, noise_params=self.noise_params, seed=self.seed)

            self.assignment = assignment.numpy()

        self.get_count = 0

        # Store the data
        self.mask = mask.float().numpy() if mask is not None else mask
        self.X = X.float().numpy()
        self.Y = Y.float().numpy()


    def __len__(self):
        return len(self.Y) - self.timesteps + 1

    def __getitem__(self, idx):
        Y = torch.from_numpy(self.Y[idx: idx + self.timesteps: self.jump])
        self.get_count += 1

        if self.sensor_type == 'random':
            mask = generate_random_mask(Y.shape[1:], self.sensor_number)
            expanded_mask = mask.unsqueeze(0).expand(X.shape[0], -1, -1, -1)

            X = voronoi_tessellation(mask, Y)
        elif self.sensor_type in ['real', 'real-random']:
            expanded_mask = self.mask[idx: idx + self.timesteps: self.jump]
            # Same mask
            #expanded_mask[idx+1:] = expanded_mask[idx]
            expanded_mask = torch.from_numpy(expanded_mask)
            X = torch.from_numpy(self.X[idx: idx + self.timesteps: self.jump])

            if self.should_reset_noise():
                self.reset_noise()
        else:
            X = torch.from_numpy(self.X[idx: idx + self.timesteps: self.jump])

            mask = torch.from_numpy(self.mask)
            expanded_mask = mask.unsqueeze(0).expand(X.shape[0], -1, -1, -1)

        X = torch.cat((X, expanded_mask), dim=1)

        return X, Y

    def should_reset_noise(self) -> bool:
        return self.get_count >= len(self) and self.noise != 'none'

    def reset_noise(self) -> None:
        if self.noise == 'none':
            return

        self.get_count = 0

        X, Y = generate_noisy_voronoi_fast(
            torch.from_numpy(self.assignment), torch.from_numpy(self.Y),
            noise_type=self.noise, full_noise=self.full_noise, noise_params=self.noise_params,
            seed=self.seed
        )

        self.X = X.float().numpy()
        self.Y = Y.float().numpy()

class DiffusionDataset(Dataset):
    def __init__(
            self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            mask: np.ndarray, 
            sensor_type: str,
            sensor_number: int,
            scaling_type: str = 'min-max',
            timesteps: int = 1,
            timesteps_jump: int = 1,
            seed: int = 42,
            noise: str = 'none',
            full_noise: bool = False,
            noise_params: dict = None,
            **stats
            ):

        self.sensor_type = sensor_type
        self.sensor_number = sensor_number
        self.timesteps = timesteps
        self.jump = timesteps_jump
        self.noise = noise
        self.full_noise = full_noise
        self.noise_params = noise_params

        self.seed = seed
        seed_everything(seed=seed, verbose=False)

        if sensor_type != 'random':
            X = torch.Tensor(scale(X, scaling_type, stats))
            Y = torch.Tensor(scale(Y, scaling_type, stats))
            
            mask = torch.Tensor(mask)
        else:
            X = torch.empty(0)
            Y = torch.Tensor(scale(Y, scaling_type, stats))

        if self.noise != 'none' and sensor_type in ['real', 'real-random']:
            X, Y = generate_noisy_voronoi(mask, Y, noise_type=self.noise, full_noise=self.full_noise, noise_params=self.noise_params, seed=self.seed)

        self.get_count = 0

        # Store the data
        self.mask = mask
        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.Y) - self.timesteps + 1

    def __getitem__(self, idx):
        Y = self.Y[idx: idx + self.timesteps: self.jump]

        if self.sensor_type == 'random':
            mask = generate_random_mask(Y.shape[1:], self.sensor_number)

            # TODO: add noise

            X = voronoi_tessellation(mask, Y)
        elif self.sensor_type in ['real', 'real-random']:
            X = self.X[idx: idx + self.timesteps: self.jump]
            mask = self.mask[idx: idx + self.timesteps: self.jump]
            mask = mask.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])

            self.get_count += 1

            if self.should_reset_noise():
                self.reset_noise()
        else:
            #  TODO: add noise
            
            X = self.X[idx: idx + self.timesteps: self.jump]
            mask = self.mask

        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])

        Y = Y[-1]

        return X, Y, mask

    def should_reset_noise(self) -> bool:
        return self.get_count >= len(self) and self.noise != 'none' and self.sensor_type in ['real', 'real-random']

    def reset_noise(self) -> None:
        if self.noise == 'none':
            return

        self.get_count = 0

        self.X, self.Y = generate_noisy_voronoi(
            self.mask, self.Y,
            noise_type=self.noise, full_noise=self.full_noise, noise_params=self.noise_params,
            seed=self.seed
        )
    
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
    channel_timesteps: bool = True,
    noise: str = 'none',
    full_noise: bool = False,
    noise_params: dict = None,
    seed: int = 42,
    diffusion: bool = False,
    pollutant_type: str = 'all'
) -> tuple[VCNNDataset, VCNNDataset, VCNNDataset, dict[str, np.ndarray]]:
    """
    Loads the dataset based on the specified parameters and returns train, validation, and test datasets along with scaling statistics.
    Args:
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        split_mode (str): Mode for splitting the dataset. Options are 'monthly', 'random', or 'sequential'.
        combine_train_val (bool): Whether to combine training and validation datasets.
        sensor_type (str): Type of sensor data to use ('real', 'real-random', 'realistic', 'random', 'fixed-random', 'fixed').
        sensor_number (int): The number of sensors to consider for fixed or fixed-random types.
        scaling_type (str): Type of scaling to apply. Options are 'min-max', 'standard', or 'none'.
        timesteps (int): Number of timesteps to consider for each sample.
        channel_timesteps (bool): Whether to use channel-wise timesteps.
        seed (int): Random seed for reproducibility.
        diffusion (bool) : Whether to use for diffusion model or not
    Returns:
        tuple: A tuple containing:
            - train_dataset (VCNNDataset or VCLSTMDataset): The training dataset.
            - val_dataset (VCNNDataset or VCLSTMDataset): The validation dataset.
            - test_dataset (VCNNDataset or VCLSTMDataset): The test dataset.
            - stats (dict): A dictionary containing scaling statistics.
    Raises:
        ValueError: If the sensor_type is not one of 'real', 'random', 'fixed-random', or 'fixed'.
    """
    
    if sensor_type not in ['real', 'real-random', 'realistic', 'random', 'fixed-random', 'fixed']:
        raise ValueError(
            f"Invalid sensor_type '{sensor_type}'. Expected one of: 'real', 'random', 'fixed-random', 'fixed'."
        )
    
    seed_everything(seed=42, verbose=False)

    all_modalities, all_modalities_Y, mask = read_voronoi_data(sensor_type, sensor_number, seed, pollutant_type)

    n_samples = all_modalities_Y.shape[0]

    train_indices, val_indices, test_indices = train_val_test_split(split_mode, n_samples, test_size, val_size, combine_train_val)
    
    if sensor_type == 'random':
        X_test = None
        X_val = None
        X_train = None
    else:
        X_test = all_modalities[test_indices]
        X_val = all_modalities[val_indices]
        X_train = all_modalities[train_indices]

    Y_test = all_modalities_Y[test_indices]
    Y_val = all_modalities_Y[val_indices]
    Y_train = all_modalities_Y[train_indices]

    stats = {
        'Y_min': X_train.min(axis=(0, 2, 3), keepdims=True),           # This is the Voronoi tesselations
        'Y_max': X_train.max(axis=(0, 2, 3), keepdims=True),
        'Y_mean': Y_train.mean(axis=(0, 2, 3), keepdims=True),         # This is the simulation ground truth
        'Y_std': Y_train.std(axis=(0, 2, 3), keepdims=True)*2,
    }

    if diffusion:
        train_dataset = DiffusionDataset(
            X_train, Y_train, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise=noise, full_noise=full_noise, noise_params=noise_params,
            **stats)
        
        val_dataset = DiffusionDataset(
            X_val, Y_val, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise="none",
            **stats) if not combine_train_val else None
        
        test_dataset = DiffusionDataset(
            X_test, Y_test, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise="none",
            **stats)
    elif channel_timesteps:
        train_dataset = VCNNDataset(
            X_train, Y_train, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise=noise, full_noise=full_noise, noise_params=noise_params,
            **stats)
        
        val_dataset = VCNNDataset(
            X_val, Y_val, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise="none",
            **stats) if not combine_train_val else None
        
        test_dataset = VCNNDataset(
            X_test, Y_test, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise="none",
            **stats)
    else:
        train_dataset = VCLSTMDataset(
            X_train, Y_train, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise=noise, full_noise=full_noise, noise_params=noise_params,
            **stats)
        
        val_dataset = VCLSTMDataset(
            X_val, Y_val, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise="none",
            **stats) if not combine_train_val else None
        
        test_dataset = VCLSTMDataset(
            X_test, Y_test, mask, sensor_type, 
            sensor_number, scaling_type, timesteps=timesteps, 
            timesteps_jump=timesteps_jump, seed=seed,
            noise="none",
            **stats)
    
    return train_dataset, val_dataset, test_dataset, stats
