import torch
import numpy as np
import torch.nn as nn
import os
import logging

from torch.utils.data import DataLoader
from typing import Tuple, List, Callable
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from src.datasets.real_obs_dataset import load_data as load_real_data
from src.datasets.vitae_dataset import unscale


EPSILON = 1e-8


def l2_norm(x: torch.Tensor):
    return torch.sqrt(torch.sum(x ** 2))


def compute_local_error(reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return reference - prediction


def compute_local_relative_error(reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((reference - prediction) ** 2) / (torch.sqrt(reference ** 2) + EPSILON)

@torch.no_grad()
def compute_rmse(reference: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor = None, chunk_size=8) -> float:
    total_sq = 0.0
    total_n = 0.0
    for i in range(0, reference.shape[0], chunk_size):
        ref = reference[i:i+chunk_size]
        pred = prediction[i:i+chunk_size]
        if mask is not None:
            m = mask[i:i+chunk_size]
            diff = (pred - ref) * m
            total_sq += torch.sum(diff * diff).item()
            total_n += torch.sum(m).item()
        else:
            diff = pred - ref
            total_sq += torch.sum(diff * diff).item()
            total_n += diff.numel()
        del ref, pred, diff  # free memory each iteration
    denom = max(total_n, 1.0)
    return (total_sq / denom) ** 0.5


@torch.no_grad()
def compute_rrmse(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_size: int = 8
) -> float:
    total_sq_error = 0.0
    total_ref_sq = 0.0
    total_count = 0.0

    for i in range(0, reference.shape[0], chunk_size):
        ref = reference[i:i+chunk_size]
        pred = prediction[i:i+chunk_size]
        m = mask[i:i+chunk_size] if mask is not None else None

        # squared error
        diff = (pred - ref) ** 2
        if m is not None:
            diff *= m
            total_sq_error += diff.sum().item()
            total_ref_sq += ((ref ** 2) * m).sum().item()
            total_count += m.sum().item()
        else:
            total_sq_error += diff.sum().item()
            total_ref_sq += (ref ** 2).sum().item()
            total_count += diff.numel()

        # free memory per batch
        del ref, pred, m, diff
        torch.cuda.empty_cache()

    # Avoid divide-by-zero
    denom = total_ref_sq + EPSILON
    return (total_sq_error / denom) ** 0.5

@torch.no_grad()
def compute_mean_fractional_bias(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor | None, chunk_size: int = 8) -> float:
    total = 0.0
    count = 0.0

    for i in range(0, target.shape[0], chunk_size):
        t = target[i:i+chunk_size]
        p = pred[i:i+chunk_size]
        m = mask[i:i+chunk_size] if mask is not None else None

        # Standard mean fractional bias: bounded to [-2, 2] for non-negative values.
        denom = t + p + EPSILON
        frac_bias = (2.0 * (t - p)) / denom

        if m is not None:
            frac_bias *= m
            total += frac_bias.sum().item()
            count += m.sum().item()
        else:
            total += frac_bias.sum().item()
            count += frac_bias.numel()

        # free memory per batch
        del t, p, m, denom, frac_bias

    denom = max(count, 1.0)
    return total / denom

@torch.no_grad()
def compute_mean_fractional_error(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor | None, chunk_size: int = 8) -> float:
    total = 0.0
    count = 0.0

    for i in range(0, target.shape[0], chunk_size):
        t = target[i:i+chunk_size]
        p = pred[i:i+chunk_size]
        m = mask[i:i+chunk_size] if mask is not None else None

        diff = (t - p).abs()
        # Standard mean fractional error: bounded to [0, 2] for non-negative values.
        denom = t + p + EPSILON
        frac_err = (2.0 * diff) / denom

        if m is not None:
            frac_err *= m
            total += frac_err.sum().item()
            count += m.sum().item()
        else:
            total += frac_err.sum().item()
            count += frac_err.numel()

        del t, p, m, diff, denom, frac_err

    denom = max(count, 1.0)
    return total / denom


def compute_relative_error(reference: torch.Tensor, prediction: torch.Tensor) -> List[float]:
    errors = []
    for i in range(reference.size(0)):
        error = l2_norm(reference[i] - prediction[i]) / (l2_norm(reference[i]) + EPSILON)
        errors.append(error.item())
    return errors


def compute_SSIM(reference: np.ndarray, prediction: np.ndarray) -> List[float]:
    scores = []
    for i in range(reference.shape[0]):
        ref = np.transpose(reference[i], (1, 2, 0))     # (C, H, W) -> (H, W, C)
        pred = np.transpose(prediction[i], (1, 2, 0))   # (C, H, W) -> (H, W, C)

        data_range = ref.max() - ref.min()        # <---------------- This is a problem for global
        score = structural_similarity(pred, ref, data_range=data_range, channel_axis=-1)
        scores.append(score)
    return scores


def compute_PSNR(reference: np.ndarray, prediction: np.ndarray) -> List[float]:
    scores = []

    for i in range(reference.shape[0]):
        data_range = reference[i].max() - reference[i].min()
        score = peak_signal_noise_ratio(prediction[i], reference[i], data_range=data_range.item())
        scores.append(score.item())

    return scores


def compute_all_metrics(reference: torch.Tensor, prediction: torch.Tensor) -> Tuple[List[float], List[float], List[float], torch.Tensor]:
    if len(prediction.shape) > 4:
        reference = reference[:, -1]
        prediction = prediction[:, -1]

    reference = reference.cpu()
    prediction = prediction.cpu()

    local_error = compute_local_error(reference, prediction)
    error = compute_relative_error(reference, prediction)

    reference_np = reference.numpy()
    prediction_np = prediction.numpy()

    ssim = compute_SSIM(reference_np, prediction_np)
    psnr = compute_PSNR(reference_np, prediction_np)

    return error, ssim, psnr, local_error


def compute_all_metrics_sequence(reference: torch.Tensor, prediction: torch.Tensor) -> Tuple[List[float], List[float], List[float], torch.Tensor]:
    _, T, _, _, _ = prediction.shape

    reference = reference.cpu()
    prediction = prediction.cpu()

    local_error = compute_local_error(reference, prediction)
    error = compute_relative_error(reference, prediction)

    all_relative_errors = []
    all_ssim = []
    all_psnr = []

    for t in range(T):
        ref_t = reference[:, t]
        pred_t = prediction[:, t]

        error_t = compute_relative_error(ref_t, pred_t)

        ref_np = ref_t.numpy()
        pred_np = pred_t.numpy()

        ssim_t = compute_SSIM(ref_np, pred_np)
        psnr_t = compute_PSNR(ref_np, pred_np)

        all_relative_errors.append(torch.tensor(error_t))
        all_ssim.append(torch.tensor(ssim_t))
        all_psnr.append(torch.tensor(psnr_t))

    # rel_err_tensor = torch.stack(all_relative_errors)
    ssim_tensor = torch.stack(all_ssim)
    psnr_tensor = torch.stack(all_psnr)

    mean_ssim = ssim_tensor.mean(dim=0).tolist()
    mean_psnr = psnr_tensor.mean(dim=0).tolist()

    return error, mean_ssim, mean_psnr, local_error


def save_metrics_vit(
    experiment_name: str,
    model: nn.Module,
    loader: DataLoader,
    evaluation_fn: Callable[
        [nn.Module, DataLoader, Callable, torch.device, bool],
        tuple[float, list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]]
    ],
    unscale_fn: Callable[[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    scaling_type: str = "min-max",
    stats: dict[str, np.ndarray] = None,
) -> tuple[float, float, float, np.ndarray]:
    model.to(device)

    _, relative_errors, ssims, psnrs, local_errors, test_obs, test_gt, enc_preds, dec_preds = evaluation_fn(model, loader, None, device, True)

    avg_relative_error = float(np.mean(relative_errors))
    avg_ssim = float(np.mean(ssims))
    avg_psnr = float(np.mean(psnrs))

    logging.info(
        f"Evaluation Results - {model.__class__.__name__} - {experiment_name}: "
        f"Average Relative Error: {avg_relative_error:.4f}, "
        f"Average SSIM: {avg_ssim:.4f}, "
        f"Average PSNR: {avg_psnr:.4f}"
    )

    preds_dir = "results/predictions/vitae"
    os.makedirs(preds_dir, exist_ok=True)

    # Scale back the data
    if stats:
        chanel_count = stats["data_min"].shape[1] if scaling_type == "min-max" else stats["data_mean"].shape[1]
        timestep_chanels = test_obs.shape[1]
        
        test_obs = unscale_fn(test_obs.reshape(-1, chanel_count, test_obs.shape[-2], test_obs.shape[-1]), scaling_type, **stats).reshape(-1, timestep_chanels, test_obs.shape[-2], test_obs.shape[-1])
        test_gt = unscale_fn(test_gt, scaling_type, **stats)
        enc_preds = unscale_fn(enc_preds, scaling_type, **stats)
        dec_preds = unscale_fn(dec_preds, scaling_type, **stats)

        local_errors = compute_local_error(test_gt, dec_preds)

    pred_file = os.path.join(preds_dir, f"{experiment_name}_predictions.npz")
    np.savez_compressed(
        pred_file,
        observations=test_obs,
        ground_truth=test_gt,
        encoder_predictions=enc_preds,
        predictions=dec_preds,
        errors=relative_errors,
        ssim=ssims,
        psnr=psnrs,
        local_errors=local_errors,
    )

    return avg_relative_error, avg_ssim, avg_psnr, local_errors

def save_metrics_voronoi(
    experiment_name: str,
    model: nn.Module,
    loader: DataLoader,
    evaluation_fn: Callable[
        [nn.Module, DataLoader, Callable, torch.device, bool],
        tuple[float, list[float], list[float], list[float], list[float], list[float], list[float], list[float]]
    ],
    unscale_fn: Callable[[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    scaling_type: str = "min-max",
    stats: dict[str, np.ndarray] = None,
) -> tuple[float, float, float, np.ndarray]:

    model.to(device)

    _, relative_errors, ssims, psnrs, local_errors, test_obs, test_gt, preds = evaluation_fn(model, loader, None, device, True)

    avg_relative_error = float(np.mean(relative_errors))
    avg_ssim = float(np.mean(ssims))
    avg_psnr = float(np.mean(psnrs))

    logging.info(
        f"Evaluation Results - {model.__class__.__name__} - {experiment_name}: "
        f"Average Relative Error: {avg_relative_error:.4f}, "
        f"Average SSIM: {avg_ssim:.4f}, "
        f"Average PSNR: {avg_psnr:.4f}"
    )

    preds_dirs_map = {
        "VCNN": "results/predictions/vunet",
        "VCNN_classic": "results/predictions/vcnn",
        "ConvLSTM": "results/predictions/clstm",
        "OptimizedModule": "results/predictions/clstm",
    }

    preds_dir = preds_dirs_map.get(model.__class__.__name__, "results/predictions/unknown_model")
    os.makedirs(preds_dir, exist_ok=True)

    # Scale back the data
    if stats:
        channel_count = stats["Y_min"].shape[1] if scaling_type == "min-max" else stats["Y_mean"].shape[1]
        obs_seq_channels = test_obs.shape[-3]
        gt_seq_channels = test_gt.shape[-3]
        
        test_obs = unscale_fn(test_obs.reshape(-1, channel_count, test_obs.shape[-2], test_obs.shape[-1]), scaling_type, **stats).reshape(-1, obs_seq_channels, test_obs.shape[-2], test_obs.shape[-1])
        test_gt = unscale_fn(test_gt.reshape(-1, channel_count, test_gt.shape[-2], test_gt.shape[-1]), scaling_type, **stats).reshape(-1, gt_seq_channels, test_gt.shape[-2], test_gt.shape[-1])
        preds = unscale_fn(preds.reshape(-1, channel_count, preds.shape[-2], preds.shape[-1]), scaling_type, **stats).reshape(-1, gt_seq_channels, preds.shape[-2], preds.shape[-1])
        
        local_errors = compute_local_error(test_gt, preds)

    pred_file = os.path.join(preds_dir, f"{experiment_name}_predictions.npz")
    np.savez_compressed(
        pred_file,
        observations=test_obs,
        ground_truth=test_gt,
        predictions=preds,
        errors=relative_errors,
        ssim=ssims,
        psnr=psnrs,
        local_errors=local_errors,
    )

    return avg_relative_error, avg_ssim, avg_psnr, local_errors

def evaluate_on_real(
    model: nn.Module,
    timesteps: int,
    experiment_name: str = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_predictions: bool = True,
    validation_set: bool = True,
    verbose: bool = True,
    pollutant_type: str = 'all'
) -> float:
    
    model_names_map = {
        "VCNN": "vunet",
        "VCNN_classic": "vcnn",
        "ConvLSTM": "clstm",
        "OptimizedModule": "clstm",
        "ViTAE": "vitae"
    }
    model_type = model_names_map.get(model.__class__.__name__)
    
    dataset, stats = load_real_data(model_type=model_type, sensor_type="real-random", timesteps=timesteps, val_set=validation_set, pollutant_type=pollutant_type)
    dataloader = DataLoader(
        dataset,
        batch_size=64 if model_type != "clstm" else 32,
        shuffle=False, num_workers=3, pin_memory=True
    )

    model.to(device)

    observations, ground_truth, predictions = [], [], []
    errors = []

    for obs, target, target_mask in dataloader:
        obs = obs.to(device).float()
        target = target.to(device).float()
        target_mask = target_mask.to(device).float()

        preds = model(obs)

        # If we are working with the ConvLSTM model, we need to take the last timestep
        if model_type == "clstm":
            preds = preds[:, -1]

        # The ViTAE model outputs both the encoder and decoder predictions, we only need the decoder predictions
        if model_type == "vitae":
            preds = preds[-1]

        # We apply the mask to the target and predictions and compute the relative error
        error = compute_relative_error(target * target_mask, preds * target_mask)

        # Store the observations, ground truth, and predictions
        observations.append(obs.cpu().detach().numpy())
        ground_truth.append(target.cpu().detach().numpy())
        predictions.append(preds.cpu().detach().numpy())
        errors.append(error)

    observations = np.concatenate(observations)
    ground_truth = np.concatenate(ground_truth)
    predictions = np.concatenate(predictions)

    # Unscale the data
    channel_count = stats["data_min"].shape[1]
    obs_shape = observations.shape

    observations = unscale(observations.reshape(-1, channel_count, observations.shape[-2], observations.shape[-1]), "min-max", **stats).reshape(*obs_shape)
    ground_truth = unscale(ground_truth, "min-max", **stats)
    predictions = unscale(predictions, "min-max", **stats)

    errors = np.concatenate(errors)
    avg_relative_error = np.mean(errors)

    if verbose:
        logging.info(
            f"Evaluation Results - {model.__class__.__name__} - Real Dataset: "
            f"Average Relative Error: {avg_relative_error:.4f}"
        )

    if not save_predictions:
        return avg_relative_error

    preds_dirs_map = {
        "VCNN": "results/predictions/vunet",
        "VCNN_classic": "results/predictions/vcnn",
        "ConvLSTM": "results/predictions/clstm",
        "OptimizedModule": "results/predictions/clstm",
        "ViTAE": "results/predictions/vitae"
    }

    preds_dir = preds_dirs_map.get(model.__class__.__name__, "results/predictions/unknown_model")
    os.makedirs(preds_dir, exist_ok=True)

    preds_file = os.path.join(preds_dir, f"sparse_{experiment_name}_predictions.npz")

    np.savez_compressed(
        preds_file,
        observations=observations,
        ground_truth=ground_truth,
        predictions=predictions,
        errors=errors,
    )

    return avg_relative_error

def compute_extra_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None
) -> Tuple[float, float, float]:
    rmse = compute_rmse(targets, preds, mask)
    
    fractional_error = compute_mean_fractional_error(targets, preds, mask)
    fractional_bias = compute_mean_fractional_bias(targets, preds, mask)

    return rmse.item(), fractional_bias.item(), fractional_error.item()
