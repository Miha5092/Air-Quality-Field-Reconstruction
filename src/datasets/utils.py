import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
import math

from pytorch_lightning import seed_everything
from src.utils.voronoi import voronoi_tessellation, batched_voronoi_tessellation


def read_voronoi_data(sensor_type: str, sensor_number: int, seed: int, pollutant_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the data based on the sensor type and number, and returns the modalities and their corresponding labels.
    Args:
        sensor_type (str): Type of sensor data to use ('real', 'real-random', 'realistic', 'random', 'fixed-random', 'fixed').
        sensor_number (int): The number of sensors to consider for fixed or fixed-random types.
        seed (int): Random seed for reproducibility when generating random masks.
        pollutant_type (str): 'all' for all pollutants. 'o3','pm10','pm25','no2' for individual pollutants
    Returns:
        tuple: A tuple containing:
            - all_modalities (np.ndarray): The modalities data.
            - all_modalities_Y (np.ndarray): The labels corresponding to the modalities.
            - mask (np.ndarray): The sensor mask.
    """
    if pollutant_type == 'o3':
        all_modalities_Y = np.load('data/d_polair_O3.npy')

    elif pollutant_type == 'pm10':
        all_modalities_Y = np.load('data/d_polair_PM10.npy')

    elif pollutant_type == 'pm25':
        all_modalities_Y = np.load('data/d_polair_PM25.npy')

    elif pollutant_type == 'no2':
        all_modalities_Y = np.load('data/d_polair_NO2.npy')

    elif pollutant_type == 'all':
        d_polair_o3 = np.load('data/d_polair_O3.npy')
        d_polair_pm10 = np.load('data/d_polair_PM10.npy')
        d_polair_pm25 = np.load('data/d_polair_PM25.npy')
        d_polair_no2 = np.load('data/d_polair_NO2.npy')
        all_modalities_Y = np.concatenate((d_polair_o3, d_polair_pm10, d_polair_pm25, d_polair_no2), axis=1)

    if sensor_type in ['real', 'realistic']:

        all_modalities = np.load('data/voronoi_real.npy')
        mask = read_realistic_sensor_mask(all_modalities[0].shape)

        if pollutant_type == 'o3':
            all_modalities = np.expand_dims(all_modalities[:,0,:,:], axis=1)
            mask = mask[0, :, :].unsqueeze(0)
        elif pollutant_type == 'pm10':
            all_modalities = np.expand_dims(all_modalities[:,1,:,:], axis=1)
            mask = mask[1, :, :].unsqueeze(0)
        elif pollutant_type == 'pm25':
            all_modalities = np.expand_dims(all_modalities[:,2,:,:], axis=1)
            mask = mask[2, :, :].unsqueeze(0)
        elif pollutant_type == 'no2':
            all_modalities = np.expand_dims(all_modalities[:,3,:,:], axis=1)
            mask = mask[3, :, :].unsqueeze(0)

        if sensor_type == 'real':
            real_data = torch.from_numpy(read_real_observation_files())
            real_sensor_positions = real_data != 0

            mask = real_sensor_positions * mask.unsqueeze(0)

    elif sensor_type == 'real-random':
        mask = torch.from_numpy(np.load('data/real/real_random_obs_mask.npy'))
        if pollutant_type == 'o3':
            mask = mask[:,0,:,:].unsqueeze(1)
        elif pollutant_type == 'pm10':
            mask = mask[:,1,:,:].unsqueeze(1)
        elif pollutant_type == 'pm25':
            mask = mask[:,2,:,:].unsqueeze(1)
        elif pollutant_type == 'no2':
            mask = mask[:,3,:,:].unsqueeze(1)

        all_modalities_Y = torch.from_numpy(all_modalities_Y)
        all_modalities = batched_voronoi_tessellation(mask[:len(all_modalities_Y)], all_modalities_Y)

        mask = mask.cpu().numpy()
        all_modalities = all_modalities.cpu().numpy()
        all_modalities_Y = all_modalities_Y.cpu().numpy()
    elif sensor_type == 'fixed':
        voronoi_o3 = np.load(f'data/voronoi_{sensor_number}_O3.npy')
        voronoi_pm10 = np.load(f'data/voronoi_{sensor_number}_PM10.npy')
        voronoi_pm25 = np.load(f'data/voronoi_{sensor_number}_PM25.npy')
        voronoi_no2 = np.load(f'data/voronoi_{sensor_number}_NO2.npy')

        all_modalities = np.concatenate((voronoi_o3, voronoi_pm10, voronoi_pm25, voronoi_no2), axis=1)

        mask = np.load(f'data/sensor_mask_{sensor_number}.npy')
        mask = np.expand_dims(mask, axis=0)
    elif sensor_type == 'fixed-random':
        torch.manual_seed(seed)
        mask = generate_random_mask(all_modalities_Y[0].shape, sensor_number)

        all_modalities = voronoi_tessellation(torch.Tensor(mask), torch.Tensor(all_modalities_Y)).numpy()
    else:
        mask = None
        all_modalities = None

    return all_modalities, all_modalities_Y, mask


def generate_random_mask(image_shape: tuple[int,int,int], sensor_number: int):
    c, h, w = image_shape

    flat_mask = torch.zeros((c, h * w), dtype=torch.bool)

    for i in range(c):
        perm = torch.randperm(h * w)
        flat_mask[i, perm[:sensor_number]] = True

    return flat_mask.view(c, h, w)


def train_val_test_split(split_mode: str, n_samples: int, test_size: float, val_size: float, combine_train_val: bool):
    indices = np.arange(n_samples)

    if split_mode == 'monthly':
        months = pd.date_range(f'2014-01-01', periods=n_samples, freq='h').month
        train_indices = indices[months <= 10]
        val_indices = indices[months == 11]
        test_indices = indices[months == 12]
    elif split_mode == 'percent':
        test_count = int(test_size * n_samples)
        val_count = int(val_size  * n_samples)

        test_indices = indices[-test_count:]
        val_indices = indices[-test_count - val_count:-test_count]
        train_indices = indices[: -test_count - val_count]
    else:
        raise ValueError("split_mode must be 'percent' or 'monthly'")
    
    if combine_train_val and split_mode == 'monthly':
        train_indices = np.concatenate((train_indices, val_indices))
        val_indices = np.array([], dtype=int)

    return train_indices, val_indices, test_indices


def read_realistic_sensor_mask(mask_shape: tuple):
    o3_position = np.load("data/convert_coord_O3.npy")
    pm10_position = np.load("data/convert_coord_pm10.npy")
    pm25_position = np.load("data/convert_coord_pm25.npy")
    no2_position = np.load("data/convert_coord_NO2.npy")

    mask = np.zeros(mask_shape)
    _, h, _ = mask.shape

    mask[0][h - o3_position[:, 1], o3_position[:, 0]] = True
    mask[1][h - pm10_position[:, 1], pm10_position[:, 0]] = True
    mask[2][h - pm25_position[:, 1], pm25_position[:,0]] = True
    mask[3][h - no2_position[:, 1], no2_position[:, 0]] = True
    mask = torch.Tensor(mask).float()

    return mask


def read_real_observation_files() -> np.ndarray:
    all_data = np.zeros((8759, 4, 75, 110))

    for pollutant_idx, pollutant in enumerate(["o3", "pm10", "pm25", "no2"]):
        pollutant_file = pollutant.upper()

        file_name = f"data/real/station_data_{pollutant_file}.pkl"

        with open(file_name, "rb") as f:
            pollutant_data = pickle.load(f)
            
            for sensor_data in pollutant_data.values():
                x = sensor_data['x_index']
                y = sensor_data['y_index']
                obs = sensor_data['obs']

                for t, ob in enumerate(obs):
                    all_data[t, pollutant_idx, 75 - y, x] = 0 if ob < 0 else ob

    return all_data


def pad_tensor(observations: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current_len = observations.shape[0]
    target_len = target.shape[0]

    repeat_count = target_len - current_len

    last_row = observations[-1:].clone()
    padding = last_row.repeat(repeat_count, *[1 for _ in range(last_row.dim() - 1)])

    padded_observations = torch.cat([observations, padding], dim=0)

    return padded_observations


def split_sensor_mask_into_two(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T, C, _, _ = mask.shape

    with_two_points = torch.zeros_like(mask, dtype=torch.bool)
    without_two_points = mask.clone()

    # Define center bounding box
    center_i_min, center_i_max = 30, 45
    center_j_min, center_j_max = 45, 65

    for t in range(T):
        for c in range(C):
            # Get all valid positions
            valid_positions = torch.nonzero(mask[t, c], as_tuple=False)
            if valid_positions.size(0) < 2:
                continue

            # Split into center and outer positions
            i_vals, j_vals = valid_positions[:, 0], valid_positions[:, 1]
            is_center = (
                (i_vals >= center_i_min) & (i_vals < center_i_max) &
                (j_vals >= center_j_min) & (j_vals < center_j_max)
            )

            center_positions = valid_positions[is_center]
            outer_positions = valid_positions[~is_center]

            # Require at least one valid point in each region
            if center_positions.size(0) == 0 or outer_positions.size(0) == 0:
                continue

            # Sample one from each region
            center_idx = torch.randint(0, center_positions.size(0), (1,))
            outer_idx = torch.randint(0, outer_positions.size(0), (1,))
            center_pos = center_positions[center_idx].squeeze(0)
            outer_pos = outer_positions[outer_idx].squeeze(0)

            # Set the two points in the new mask
            with_two_points[t, c, center_pos[0], center_pos[1]] = True
            with_two_points[t, c, outer_pos[0], outer_pos[1]] = True

            # Remove them from the original mask
            without_two_points[t, c, center_pos[0], center_pos[1]] = False
            without_two_points[t, c, outer_pos[0], outer_pos[1]] = False

    return without_two_points, with_two_points


def get_data_splits(sensor_type: str = 'real-random', seed: int = 42, pollutant_type: str = 'all') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    _, all_modalities_Y, _ = read_voronoi_data(sensor_type, 30, seed, pollutant_type)

    n_samples = all_modalities_Y.shape[0]
    train_indices, val_indices, test_indices = train_val_test_split("monthly", n_samples, 0.1, 0.1, False)

    Y_test = all_modalities_Y[test_indices]
    Y_val = all_modalities_Y[val_indices]
    Y_train = all_modalities_Y[train_indices]

    return Y_train, Y_val, Y_test 


def create_gaussian_kernel(size: int, sigma: float, device=None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel using the specified size and standard deviation (sigma).
    """
    coords = torch.arange(size, device=device) - size // 2
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

@torch.no_grad()
def gaussian_noise(
    target_shape: tuple[int], 
    noise_mean: torch.Tensor, 
    noise_std: torch.Tensor, 
    correlation_scale: float = 10.0, 
    seed: int = 42, 
    device='cpu'
) -> torch.Tensor:
    """
    Creates spatially correlated noise using Gaussian filtering in the target shape.

    Args:
        target_shape: Tuple[int] of (timesteps, channels, height, width)
        correlation_scale: Determines the spatial correlation. Higher means smoother noise.
        device: Device to run the computation on ('cuda' or 'cpu').

    Returns:
        Tensor of shape `target_shape` with spatially correlated noise.
    """
    seed_everything(seed=seed, verbose=False)

    timesteps, channels, height, width = target_shape

    # Create Gaussian kernel on the correct device
    kernel_size = int(6 * correlation_scale) | 1  # ensure it's odd
    kernel = create_gaussian_kernel(kernel_size, correlation_scale, device=device)
    kernel = kernel.to(dtype=torch.float32)

    # Reshape for depthwise conv2d
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)  # shape (channels, 1, k, k)

    # Reshape the noise parameters
    noise_mean = noise_mean.view(1, channels, 1, 1).to(device)
    noise_std = noise_std.view(1, channels, 1, 1).to(device)

    # Create white noise on device
    noise = torch.randn((timesteps, channels, height, width), dtype=torch.float32, device=device) * noise_std + noise_mean

    # Pad input to maintain shape after conv
    padding = kernel_size // 2
    padded_noise = F.pad(noise, (padding, padding, padding, padding), mode='reflect')

    padded_noise = padded_noise.float()
    kernel = kernel.float()

    noise_filtered = F.conv2d(padded_noise, kernel, padding=0, groups=channels)

    return noise_filtered

@torch.no_grad()
def gaussian_noise_3d(
    target_shape: tuple[int], 
    noise_mean: torch.Tensor, 
    noise_std: torch.Tensor, 
    correlation_scale: float = 10.0, 
    seed: int = 42, 
    device='cpu'
) -> torch.Tensor:
    """
    Creates temporally and spatially correlated noise using separable filtering.
    Spatial filtering is 2D, followed by 1D filtering over time.
    """
    import torch.nn.functional as F
    from torch import nn
    
    seed_everything(seed=seed, verbose=False)

    # (T, C, H, W) with spatially correlated noise
    noise_2d = gaussian_noise(target_shape, noise_mean, noise_std, correlation_scale, seed=seed, device=device)

    kernel_scale = 2.0
    k_temporal = int(6 * kernel_scale) | 1  # ensure odd size
    t_coords = torch.arange(k_temporal, device=device) - k_temporal // 2
    t_kernel = torch.exp(-(t_coords**2) / (2 * kernel_scale**2))
    t_kernel = t_kernel / t_kernel.sum()
    t_kernel = t_kernel.view(1, 1, k_temporal)  # (1, 1, K)

    T, C, H, W = target_shape
    noise = noise_2d.permute(1, 2, 3, 0).contiguous().view(C * H * W, 1, T)

    # Pad and convolve over time
    pad = k_temporal // 2
    noise = F.conv1d(F.pad(noise, (pad, pad), mode='reflect'), t_kernel)

    # Reshape back to (T, C, H, W)
    noise = noise.view(C, H, W, T).permute(3, 0, 1, 2).contiguous()

    return noise

@torch.no_grad()
def perlin_noise(
    target_shape: tuple[int, int, int, int],  # (B, C, H, W)
    noise_mean: torch.Tensor,
    noise_std: torch.Tensor,
    perlin_res: tuple[int, int] = (8, 8),
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Memory-efficient multi-octave Perlin noise.
    Returns (B,C,H,W); each (b,c) map is zero-mean/unit-std then scaled
    by noise_mean[c], noise_std[c].
    """
    B, C, H, W = target_shape
    N = B * C
    dev = torch.device(device)

    # Heuristic chunk size to cap VRAM; reduce base_chunk if still OOM.
    base_chunk = 16
    scale = max(1, (256 * 256) // max(1, H * W))
    chunk_size = max(1, min(N, base_chunk * scale))

    # Use lower precision on CUDA to save VRAM
    out_dtype = torch.float16 if dev.type == 'cuda' else torch.float32
    work_dtype = out_dtype

    out = torch.empty((B, C, H, W), device=dev, dtype=out_dtype)

    # Deterministic RNG for gradients
    gen = torch.Generator(device=dev).manual_seed(seed)

    # Per-channel stats on device
    noise_mean = noise_mean.to(device=dev, dtype=work_dtype).view(1, C, 1, 1)
    noise_std  = noise_std.to(device=dev, dtype=work_dtype).view(1, C, 1, 1)

    def fade(t: torch.Tensor) -> torch.Tensor:
        return ((6*t - 15)*t + 10)*t*t*t  # 6t^5 - 15t^4 + 10t^3

    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        n = e - s  # chunk size

        total = torch.zeros((n, H, W), device=dev, dtype=work_dtype)
        amp = 1.0
        freq = 1.0
        amp_sum = 0.0

        for _ in range(octaves):
            rx = max(int(perlin_res[0] * freq), 1)
            ry = max(int(perlin_res[1] * freq), 1)

            # Spatial grid in lattice coordinates
            gy, gx = torch.meshgrid(
                torch.linspace(0, rx, H, device=dev, dtype=work_dtype),
                torch.linspace(0, ry, W, device=dev, dtype=work_dtype),
                indexing='ij'
            )
            x0 = gx.floor().clamp_(0, rx).to(torch.int64)
            y0 = gy.floor().clamp_(0, ry).to(torch.int64)
            x1 = (x0 + 1).clamp_(0, rx)
            y1 = (y0 + 1).clamp_(0, ry)

            fx = gx - x0.to(work_dtype)
            fy = gy - y0.to(work_dtype)
            sx = fade(fx)
            sy = fade(fy)

            # One gradient lattice per chunk; expand to (n, rx+1, ry+1, 2)
            theta = 2 * math.pi * torch.rand((1, rx + 1, ry + 1), device=dev, generator=gen, dtype=work_dtype)
            grads = torch.stack((theta.cos(), theta.sin()), dim=-1).expand(n, rx + 1, ry + 1, 2)

            # Per-sample random phase offsets (decorrelate samples)
            offx = torch.randint(0, rx + 1, (n, 1, 1), device=dev)
            offy = torch.randint(0, ry + 1, (n, 1, 1), device=dev)

            # Batch index to avoid advanced-indexing shape explosion
            batch_idx = torch.arange(n, device=dev).view(n, 1, 1)  # (n,1,1)

            def gcorner(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
                ixn = (ix[None, ...] + offx).remainder(rx + 1)   # (n,H,W)
                iyn = (iy[None, ...] + offy).remainder(ry + 1)   # (n,H,W)
                return grads[batch_idx, ixn, iyn, :]              # (n,H,W,2)

            # Displacements to corners (broadcast to (n,H,W))
            dx00x, dx00y =  fx[None, ...],  fy[None, ...]
            dx10x, dx10y = (fx-1)[None, ...],  fy[None, ...]
            dx01x, dx01y =  fx[None, ...], (fy-1)[None, ...]
            dx11x, dx11y = (fx-1)[None, ...], (fy-1)[None, ...]

            # Dot products -> (n,H,W)
            g00 = gcorner(x0, y0); n00 = g00[..., 0]*dx00x + g00[..., 1]*dx00y
            g10 = gcorner(x1, y0); n10 = g10[..., 0]*dx10x + g10[..., 1]*dx10y
            g01 = gcorner(x0, y1); n01 = g01[..., 0]*dx01x + g01[..., 1]*dx01y
            g11 = gcorner(x1, y1); n11 = g11[..., 0]*dx11x + g11[..., 1]*dx11y

            # Interpolate -> (n,H,W)
            nx0 = n00 * (1 - sx)[None, ...] + n10 * sx[None, ...]
            nx1 = n01 * (1 - sx)[None, ...] + n11 * sx[None, ...]
            octave = nx0 * (1 - sy)[None, ...] + nx1 * sy[None, ...]

            total.add_(amp * octave)
            amp_sum += amp
            amp *= persistence
            freq *= lacunarity

        total = total / max(amp_sum, 1e-12)      # (n,H,W)

        # Per-(b,c) normalization
        total = total.view(n, 1, H, W)
        mu = total.mean(dim=(2, 3), keepdim=True)
        sd = total.std(dim=(2, 3), keepdim=True).clamp_min_(1e-6)
        total = (total - mu) / sd                # (n,1,H,W)

        # Map flat [s:e) back to (b,c) and assign properly
        for i, idx in enumerate(range(s, e)):
            b = idx // C
            c = idx % C
            # Scale and assign individual tensor
            scaled = total[i, 0] * noise_std[0, c, 0, 0] + noise_mean[0, c, 0, 0]
            out[b, c] = scaled.to(out_dtype)

    return out

@torch.no_grad()
def compute_pixelwise_correlation(seed: int = 42) -> torch.Tensor:
    """
    Check if the matrix has been computed before, if not compute it.

    Compute per-channel empirical spatial correlation matrix from a batch of images.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        corr: Tensor of shape (C, N, N), where N = H * W
    """

    if os.path.exists('results/misc/correlation_matrix.pth'):
        corr = torch.load('results/misc/correlation_matrix.pth', weights_only=False)
        return corr
    
    train_examples = torch.from_numpy(get_data_splits(sensor_type='real-random', seed=seed)[0])

    B, C, H, W = train_examples.shape
    N = H * W

    x = train_examples.view(B, C, N)
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True) + 1e-6)
    x = x.permute(1, 0, 2).contiguous()  # (C, B, N)

    corr = torch.bmm(x.transpose(1, 2), x) / (B - 1)  # (C, N, N)

    torch.save(corr, 'results/misc/correlation_matrix.pth')

    return corr

@torch.no_grad()
def apply_pixelwise_correlation(white_noise, corr_matrix):
    """
    Apply pixelwise correlation to white noise using the given correlation matrix.

    Args:
        white_noise: Tensor of shape (B, C, H, W) - i.i.d. Gaussian noise
        corr_matrix: Tensor of shape (B, C, H*W, H*W) - Pearson correlation matrices

    Returns:
        correlated_noise: Tensor of shape (B, C, H, W)
    """
    B, C, H, W = white_noise.shape
    N = H * W
    device = white_noise.device

    # Flatten the noise to (B, C, N)
    z = white_noise.view(B, C, N)

    # Cholesky decomposition of corr_matrix: shape (B, C, N, N)
    eps = 1e-4
    L = torch.linalg.cholesky(corr_matrix + eps * torch.eye(N, device=device).unsqueeze(0))  # (C, N, N)

    z_corr = torch.einsum('cnm,bcm->bcn', L.to(device=device, dtype=torch.float32), z.to(device=device, dtype=torch.float32))  # (B, C, N)

    # Reshape back to (B, C, H, W)
    return z_corr.view((B, C, H, W))

@torch.no_grad()
def generate_correlated_noise(
    target_shape: tuple[int],
    noise_mean: torch.Tensor,
    noise_std: torch.Tensor,
    seed: int = 42,
    device='cpu'
) -> torch.Tensor:
    
    seed_everything(seed=seed, verbose=False)

    noise = torch.randn(target_shape, device=device)
    corr = compute_pixelwise_correlation().to(device=device, dtype=torch.float32)

    correlated_noise = apply_pixelwise_correlation(noise, corr)
    correlated_noise -= correlated_noise.mean(dim=(0, 2, 3), keepdim=True)
    correlated_noise /= correlated_noise.std(dim=(0, 2, 3), keepdim=True) + 1e-6

    return (correlated_noise * noise_std.view(1, -1, 1, 1) + noise_mean.view(1, -1, 1, 1)).to(device=device, dtype=torch.float32)

@torch.no_grad()
def generate_white_noise(
    target_shape: tuple[int],
    noise_mean: torch.Tensor,
    noise_std: torch.Tensor,
    seed: int = 42,
    device='cpu'
) -> torch.Tensor:
    seed_everything(seed=seed, verbose=False)
    
    noise = torch.randn(target_shape, device=device)
    return (noise * noise_std.view(1, -1, 1, 1) + noise_mean.view(1, -1, 1, 1)).to(device=device, dtype=torch.float32)
    
supported_noise_types = ['gaussian', 'time_gaussian', 'perlin', 'correlated', 'white']

@torch.no_grad()
def get_noise(
    target_shape: tuple[int],
    noise_type: str = 'gaussian',
    noise_params: dict = None,
    seed: int = 42,
    device='cpu'
) -> torch.Tensor:
    """
    Generate noise based on the specified type.
    
    Args:
        target_shape: Shape of the output tensor.
        noise_mean: Mean values for each channel.
        noise_std: Standard deviation for each channel.
        noise_type: Type of noise ('gaussian' or 'perlin').
        correlation_scale: Scale for spatial correlation in Gaussian noise.
        perlin_res: Resolution for Perlin noise.
        octaves, persistence, lacunarity: Parameters for Perlin noise.
        device: Device to run the computation on ('cuda' or 'cpu').
    
    Returns:
        Tensor of shape `target_shape` with generated noise.
    """
    if noise_params is None:
        params = torch.load(f'results/misc/noise_params_{noise_type}.pth', weights_only=False)
    else:
        params = noise_params

    noise_mean = params["noise_mean"].to(device)
    noise_std = params["noise_std"].to(device)
    
    if noise_type == 'gaussian':
        correlation_scale = params['correlation_scale']

        return gaussian_noise(target_shape, noise_mean, noise_std, correlation_scale, seed=seed, device=device)
    elif noise_type == 'time_gaussian':
        correlation_scale = params['correlation_scale']

        return gaussian_noise_3d(target_shape, noise_mean, noise_std, correlation_scale, seed=seed, device=device)
    elif noise_type == 'perlin':
        perlin_res = (params['perlin_res_x'], params['perlin_res_y'])
        octaves, persistence, lacunarity = params['octaves'], params['persistence'], params['lacunarity']
        
        return perlin_noise(target_shape, noise_mean, noise_std, perlin_res, octaves, persistence, lacunarity, seed=seed, device=device)
    elif noise_type == 'correlated':
        return generate_correlated_noise(target_shape, noise_mean, noise_std, seed=seed, device=device)
    elif noise_type == 'white':
        return generate_white_noise(target_shape, noise_mean, noise_std, seed=seed, device=device)
    else:
        raise ValueError(f"Unsupported noise type. Use {supported_noise_types}.")
    
def get_custom_noise(
    target_shape: tuple[int],
    noise_mean: torch.Tensor,
    noise_std: torch.Tensor,
    correlation_scale: float = 10.0,
    perlin_res: tuple[int, int] = (8, 8),
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    noise_type: str = 'gaussian',
    seed: int = 42,
    device='cpu'
) -> torch.Tensor:
    """
    Generate noise based on the specified type.
    
    Args:
        target_shape: Shape of the output tensor.
        noise_mean: Mean values for each channel.
        noise_std: Standard deviation for each channel.
        noise_type: Type of noise ('gaussian' or 'perlin').
        correlation_scale: Scale for spatial correlation in Gaussian noise.
        perlin_res: Resolution for Perlin noise.
        octaves, persistence, lacunarity: Parameters for Perlin noise.
        device: Device to run the computation on ('cuda' or 'cpu').
    
    Returns:
        Tensor of shape `target_shape` with generated noise.
    """
    
    if noise_type == 'gaussian':
        return gaussian_noise(target_shape, noise_mean, noise_std, correlation_scale, seed=seed, device=device)
    elif noise_type == "time_gaussian":
        return gaussian_noise_3d(target_shape, noise_mean, noise_std, correlation_scale, seed=seed, device=device)
    elif noise_type == 'perlin':
        return perlin_noise(target_shape, noise_mean, noise_std, perlin_res, octaves, persistence, lacunarity, seed=seed, device=device)
    elif noise_type == 'correlated':
        return generate_correlated_noise(target_shape, noise_mean, noise_std, seed=seed, device=device)
    elif noise_type == 'white':
        return generate_white_noise(target_shape, noise_mean, noise_std, seed=seed, device=device)
    else:
        raise ValueError(f"Unsupported noise type. Use {supported_noise_types}.")