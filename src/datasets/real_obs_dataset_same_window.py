import numpy as np
import torch
import pickle

from torch.utils.data import Dataset
from pytorch_lightning import seed_everything

from src.datasets.vitae_dataset import load_data as load_vitae
from src.datasets.voronoi_datasets import load_data as load_voronoi
from src.utils.voronoi import batched_voronoi_tessellation
from src.datasets.utils import train_val_test_split, read_real_observation_files


class RealObsDataset(Dataset):
    def __init__(
        self,
        all_observations: np.ndarray,
        observations: np.ndarray,
        targets: np.ndarray,
        total_mask: torch.Tensor,
        obs_mask: torch.Tensor,
        target_mask: torch.Tensor,
        model_type: str,
        timesteps: int = 1,
        scale = None,
        stats = None
    ):
        self.all_observations = all_observations
        self.observations = observations
        self.targets = targets
        self.total_mask = total_mask
        self.obs_mask = obs_mask
        self.target_mask = target_mask
        self.scale = scale
        self.stats = stats

        assert model_type in ["vitae", "vunet", "vcnn", "vunet", "clstm", "diffusion"], f"Model type {model_type} must be one of 'vitae', 'vcnn', 'vunet', or 'clstm'."
        self.model_type = model_type

        self.timesteps = timesteps

    def __len__(self):
        return len(self.observations) - self.timesteps + 1



    def __getitem__(self, idx):
        target_mask = self.target_mask[idx + self.timesteps - 1]    # These are the sensors that should be excluded from previous timesteps
        obs_mask = self.total_mask[idx: idx + self.timesteps] * (~ target_mask)

        obs = self.all_observations[idx: idx + self.timesteps] * obs_mask
        target = self.targets[idx + self.timesteps - 1]

        if self.model_type in ["vcnn", "vunet", "clstm", "diffusion"]:
            obs = batched_voronoi_tessellation(obs_mask, obs)

        if self.scale:
            data_min = self.stats['data_min']
            data_max = self.stats['data_max']
            obs = (obs - data_min) / (data_max - data_min + 1e-8)

        # Concatenate the timesteps as extra channels
        if self.model_type != "clstm":
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])

        if self.model_type in ["vcnn", "vunet"]:
            obs_mask = obs_mask.reshape(obs_mask.shape[0] * obs_mask.shape[1], obs_mask.shape[2], obs_mask.shape[3])

            obs = torch.cat((obs, obs_mask), dim=0)

        if self.model_type == "diffusion":
            obs_mask = obs_mask.reshape(obs_mask.shape[0] * obs_mask.shape[1], obs_mask.shape[2], obs_mask.shape[3])
            obs_list = [obs.float(), obs_mask.float()]
            return obs_list, target.float(), target_mask.float()

        elif self.model_type == "clstm":
            obs_mask = obs_mask.expand(obs.size(0), -1, -1, -1)
            obs = torch.cat((obs, obs_mask), dim=1)

        return obs, target, target_mask

def choose_pollutant(obs, target, obs_mask, target_mask, pollutant_no):

    obs = obs[:,pollutant_no, :, :].unsqueeze(1)
    target = target[:,pollutant_no, :, :].unsqueeze(1)
    obs_mask = obs_mask[:,pollutant_no, :, :].unsqueeze(1)
    target_mask = target_mask[:,pollutant_no, :, :].unsqueeze(1)

    return obs, target, obs_mask, target_mask

def load_data(
    model_type: str,
    sensor_type: str,
    timesteps: int = 1,
    scale: bool = True,
    test_set: bool = True,
    val_set: bool = True,
    pollutant_type: str = 'all',
    inner_city:bool = True
) -> tuple[RealObsDataset, dict[str, np.ndarray]]:
    """
    
    Loads the data required to evaluate models on the real station observations.

    The function returns the training, validation, and test datasets. The training set
    contains only the sensors that have been present in the training set while the
    validation and test sets contain the sensors that have not been present.

    Args:
        model_type (str): The type of model to be evaluated. This determines the dataset
                          and statistics used for scaling.
    Returns:
        tuple: A tuple containing the dataset and the statistics used to scale the data.

    """

    if sensor_type not in ['real', 'real-random']:
        raise ValueError(
            f"Invalid sensor_type '{sensor_type}'. Expected one of: 'real', 'real-random'."
        )

    seed_everything(seed=42, verbose=False)

    if sensor_type == "real":
        vitae_training_dataset, _, _, stats = load_vitae(scaling_type="none", sensor_type="real", pollutant_type=pollutant_type)
    else:
        vitae_training_dataset, _, _, stats = load_vitae(scaling_type="none", sensor_type="real-random", pollutant_type=pollutant_type)

    if model_type != "vitae":
        stats = load_voronoi(pollutant_type=pollutant_type)[-1]
        
    stats = {
        'data_min': stats['Y_min'] if model_type != "vitae" else stats['data_min'],
        'data_max': stats['Y_max'] if model_type != "vitae" else stats['data_max'],
        'data_mean': stats['Y_mean'] if model_type != "vitae" else stats['data_mean'],
        'data_std': stats['Y_std'] if model_type != "vitae" else stats['data_std'],
    }

    real_data = torch.from_numpy(read_real_observation_files())

    # Apply the masks to the real data to create observations and targets.
    if sensor_type == "real":
        real_sensor_positions = real_data != 0

        obs_mask = vitae_training_dataset[0][-1] != 0
        target_mask = real_sensor_positions * ~obs_mask
    else:
        obs_mask = torch.from_numpy(np.load('data/real/real_random_obs_mask.npy'))
        target_mask = torch.from_numpy(np.load('data/real/real_random_target_mask.npy'))

    H, W = target_mask.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(H, device=target_mask.device),
        torch.arange(W, device=target_mask.device),
        indexing="ij"
    )
    inside_city = (
        (x >= 45) & (x <= 65) &
        (y >= 30) & (y <= 45)
    )
    outside_city = ~inside_city
    if inner_city:
        # Keep ONLY the inner-city region
        target_mask &= inside_city[None, None, :, :]
    else:
        # Remove the inner-city region
        target_mask &= ~inside_city[None, None, :, :]
    

    total_mask = obs_mask + target_mask

    all_observations = real_data * total_mask

    observations = real_data * obs_mask
    targets = real_data * target_mask

    # Split the data based on the training, val, and test sets
    if test_set:
        indices = train_val_test_split(
            split_mode="monthly",
            n_samples=observations.shape[0],
            test_size=0.8, val_size=0.1,
            combine_train_val=True,
        )[-1]
    elif val_set:
        indices = train_val_test_split(
            split_mode="monthly",
            n_samples=observations.shape[0],
            test_size=0.9, val_size=0.1,
            combine_train_val=False,
        )[1]
    else:
        indices = train_val_test_split(
            split_mode="monthly",
            n_samples=observations.shape[0],
            test_size=0.8, val_size=0.1,
            combine_train_val=True,
        )[0]

    observations = observations[indices]
    all_observations = all_observations[indices]
    targets = targets[indices]

    total_mask = total_mask[indices]
    obs_mask = obs_mask[indices]
    target_mask = target_mask[indices]

    if pollutant_type == 'o3':
        observations, targets, obs_mask, target_mask = choose_pollutant(observations, targets, obs_mask, target_mask, 0)

    elif pollutant_type == 'pm10':
        observations, targets, obs_mask, target_mask = choose_pollutant(observations, targets, obs_mask, target_mask, 1)

    elif pollutant_type == 'pm25':
        observations, targets, obs_mask, target_mask = choose_pollutant(observations, targets, obs_mask, target_mask, 2)

    elif pollutant_type == 'no2':
        observations, targets, obs_mask, target_mask = choose_pollutant(observations, targets, obs_mask, target_mask, 3)

    if scale:
        if model_type == "diffusion":
            data_mean = stats['data_mean']
            data_std = stats['data_std']
            observations = (observations - data_mean) / (data_std + 1e-8)
            targets = (targets - data_mean) / (data_std + 1e-8)
        else:
            data_min = stats['data_min']
            data_max = stats['data_max']
            targets = (targets - data_min) / (data_max - data_min + 1e-8)

    dataset = RealObsDataset(
        all_observations=all_observations,
        observations=observations,
        targets=targets,
        total_mask=total_mask,
        obs_mask=obs_mask,
        target_mask=target_mask,
        model_type=model_type,
        timesteps=timesteps,
        scale=scale,
        stats=stats
    )

    return dataset, stats


