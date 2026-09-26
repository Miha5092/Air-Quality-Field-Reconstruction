import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import logging
import warnings

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pytorch_lightning import seed_everything
from typing import Union, Callable

from src.models.vcnn import VCNN, VCNN_classic, VUnet
from src.models.vclstm import ConvLSTM
from src.models.vitae import ViTAE
from src.utils.early_stopper import EarlyStopper
from src.utils.visualization import plot_training


logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_vitae_loss_fn(lamda1: float, use_mask: bool = False) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns a custom loss function that combines encoder and decoder MSE losses, optionally masked.

    Args:
        lamda1 (float): Weight for the decoder loss (between 0 and 1).
        use_mask (bool, optional): Whether to apply element-wise mask to the loss. Defaults to False.

    Returns:
        Callable: A function accepting (pred_enc, pred_dec, mask, ground_truth) and returning scalar loss.
    """

    def loss_fn(pred_enc: torch.Tensor, pred_dec: torch.Tensor, mask: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        reduction = 'none' if use_mask else 'mean'
        loss_enc = F.mse_loss(pred_enc, gt, reduction=reduction)
        loss_dec = F.mse_loss(pred_dec, gt, reduction=reduction)

        if use_mask:
            loss_enc = loss_enc * mask
            loss_dec = loss_dec * mask
            return (lamda1 * loss_dec + (1 - lamda1) * loss_enc).mean()
        else:
            return lamda1 * loss_dec + (1 - lamda1) * loss_enc
    return loss_fn


def get_voronoi_loss_fn() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns a loss function that computes the mean squared error between predicted and ground truth tensors.

    Args:
        None
        
    Returns:
        Callable: A function that takes two tensors (pred and gt) and returns the MSE loss.
    """

    def loss_fn(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, gt)
        return loss
    return loss_fn


def get_scheduler(optimizer, epochs) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Creates a learning rate scheduler that first performs a linear warmup, followed by cosine annealing.

    Args:
        optimizer (Optimizer): The optimizer whose learning rate should be scheduled.
        epochs (int): Total number of training epochs.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: A composed scheduler applying warmup followed by cosine decay.
    """

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs // 2, eta_min=1e-6)

    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[5]
    )

    return scheduler


def safe_scheduler_step(scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    """
    A function defined to deal with the legacy implementation of PyTorch's learning rate schedulers.

    It avoids showing warnings regarding the usage of scheduler.step(epoch) in the training loop.

    Args:
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to step.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*The epoch parameter in `scheduler.step.*",
            category=UserWarning
        )
        scheduler.step()


def get_dirs(model: nn.Module, pollutant_type: str) -> tuple[str, str]:

    model_dir_map = {
        VUnet: "vunet",
        VCNN_classic: "vcnn",
        VCNN: "vunet",
        ConvLSTM: "clstm",
        ViTAE: "vitae"
    }

    model_dir = f"paper_results/trained_models/{model_dir_map.get(type(model), 'unknown_model')}/{pollutant_type}"
    fig_dir = f"paper_results/images/training/{model_dir_map.get(type(model), 'unknown_model')}/{pollutant_type}"

    return model_dir, fig_dir


def train(
    experiment_name: str,
    model: Union[VCNN, VCNN_classic, ConvLSTM, ViTAE],
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    evaluation_fn: Callable[
        [nn.Module, DataLoader, Callable, torch.device, bool],
        tuple[float, list[float], list[int], list[int], list[int], list[int], list[int], list[int], list[int]]
    ],
    loss_fn: Callable,
    verbose: bool,
    save: bool,
    seed: int,
    device: torch.device,
    early_stopping: bool = False,
    pollutant_type: str = 'all'
) -> tuple[nn.Module, list[float], list[float]]:
    """
    Trains a model for a specified number of epochs, evaluates it on a validation set, and saves the best model.

    Args:

        experiment_name (str): Name of the experiment for logging and saving.
        model (nn.Module): The model to be trained.
        epochs (int): Number of training epochs.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        evaluation_fn (Callable): Function to evaluate the model on the validation set.
        loss_fn (Callable): Loss function to be used during training.
        verbose (bool): Whether to print detailed logs.
        save (bool): Whether to save the model after training.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device to perform training on (CPU or GPU).
        early_stopping (bool, optional): Whether to apply early stopping based on validation loss. Defaults to False.
        
    Returns:
        tuple[nn.Module, list[float], list[float]]: The trained model, training losses, and relative errors history.
    """

    seed_everything(seed=seed, verbose=verbose)

    # Basically 3 days and 12 hours
    max_allowed_training_time = 84 * 60 * 60  # 84 hours in seconds

    # Initializing training utils
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, epochs)
    early_stopper = EarlyStopper() if early_stopping else None

    # For saving the model during and after training.

    model_dir, fig_dir = get_dirs(model, pollutant_type)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{experiment_name}.pth")

    # Logging
    training_start_time = time.time()
    training_losses = []
    relative_errors_history = []
    best_val_loss = np.inf

    logging.info("-" * 50)

    for epoch in range(epochs):
        epoch_train_loss = model.train_one_epoch(train_loader, optimizer, loss_fn, device)
        safe_scheduler_step(scheduler)
        training_losses.append(epoch_train_loss)

        # logging.info(f"Epoch {epoch + 1} - {int(time.time() - training_start_time)}s")

        if (epoch + 1) % 5 == 0 and (verbose or early_stopper):

            val_loss, relative_errors, ssims, psnrs, *_ = evaluation_fn(model, val_loader, loss_fn, device, verbose)

            if verbose:
                logging.info(
                    f"Epoch {epoch + 1}, Validation Loss: {val_loss:.6f}, "
                    f"Relative Error: {np.mean(relative_errors):.4f}, "
                    f"SSIM: {np.mean(ssims):.4f}, PSNR: {np.mean(psnrs):.4f}"
                )

                relative_errors_history.append(np.mean(relative_errors))

            if save and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

            if early_stopper and early_stopper.early_stop(val_loss):
                logging.info(
                    f"Early stopping triggered at epoch {epoch + 1} after "
                    f"{early_stopper.patience * 5} epochs with no improvement."
                )
                break
        elif (epoch + 1) % 50 == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs}")

        if int(time.time() - training_start_time) > max_allowed_training_time:
            logging.info(f"Max training time exceeded ({max_allowed_training_time} seconds). Stopping training after {epoch + 1} epochs.")
            break

    training_end_time = time.time()

    if epochs > 0:
        total_seconds = int(training_end_time - training_start_time)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"Training completed in {hours}h:{minutes}m:{seconds}s")

        epoch_seconds = int(total_seconds / epochs)
        epoch_hours, epoch_remainder = divmod(epoch_seconds, 3600)
        epoch_minutes, epoch_seconds = divmod(epoch_remainder, 60)
        logging.info(f"Average time per epoch {epoch_hours}h:{epoch_minutes}m:{epoch_seconds}s")
        logging.info("-" * 50)

    if save and verbose:
        
        os.makedirs(fig_dir, exist_ok=True)
        image_path = os.path.join(fig_dir, f"{experiment_name}.png")
        plot_training(training_losses, relative_errors_history, image_path)

    if model_path is not None and save:
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            pass
    elif verbose and save:
        logging.warning("No model was saved during training. Cannot load model.")

    return model, training_losses, relative_errors_history