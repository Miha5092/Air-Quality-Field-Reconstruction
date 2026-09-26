import argparse
import torch

from torch.utils.data import DataLoader

from src.models.vitae import get_model, evaluate_loader
from src.datasets.vitae_dataset import load_data, unscale
from src.utils.training import get_vitae_loss_fn, train
from src.utils.evaluation_pipeline import evaluate

def get_free_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    free_memories = []

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_memories.append(free)

    gpu_id = max(range(len(free_memories)), key=lambda i: free_memories[i])

    return torch.device(f"cuda:{gpu_id}")

def main(
    experiment_name: str,
    model_type: str,
    sensor_type: str,
    sensor_number: int,
    patch_size: tuple[int, int],
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    lamda1: float,
    verbose: bool,
    seed: int,
    save_model: bool,
    early_stopping: bool,
    kernel_size: int,
    scaling_type: str,
    timesteps: int,
    timesteps_jump: int,
    noise: str = 'none',
    full_noise: bool = True,
    split_mode: str = 'monthly',
    pollutant_type: str = 'all',
    model_path:str = 'none'
):
    """
    Train a ViTAE model.

    Args:
        experiment_name (str): Name of the experiment.
        model_type (str): Type of the model to train.
        sensor_type (str): Type of sensors to use.
        sensor_number (int): Number of sensors.
        patch_size (tuple[int, int]): Patch size for the model.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        epochs (int): Number of epochs to train the model.
        lamda1 (float): Weight for the loss function combination.
        verbose (bool): Whether to print verbose output during training.
        seed (int): Random seed for reproducibility.
        save_model (bool): Whether to save the trained model.
        early_stopping (bool): Whether to use early stopping during training.
        kernel_size (int): Kernel size for convolutional layers in UNet.
        scaling_type (str): Type of scaling to apply to the data.
        timesteps (int): Number of consecutive timesteps to be used for a training example.
        split_mode (str, optional): Mode for splitting the dataset. Defaults to 'monthly'.

    Returns:
        None: The model and the results can optionally be saved to disk.
    """

    train_dataset, _, test_dataset, stats = load_data(
    sensor_type=sensor_type,
    sensor_number=sensor_number,
    split_mode=split_mode,
    combine_train_val=True,
    scaling_type=scaling_type,
    timesteps=timesteps,
    timesteps_jump=timesteps_jump,
    noise=noise,
    full_noise=full_noise,
    seed=seed,
    pollutant_type=pollutant_type
    )

    device = get_free_gpu()
    print(device)

    model = get_model(
        model_type=model_type,
        train_dataset=train_dataset,
        patch_size=patch_size,
        kernel_size=kernel_size
    )
    if epochs == 0:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    if epochs>0:
        model, _, _ = train(
            experiment_name=experiment_name,
            model=model,
            epochs=epochs,
            train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4),
            val_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4),
            lr=lr,
            weight_decay=weight_decay,
            evaluation_fn=evaluate_loader,
            loss_fn=get_vitae_loss_fn(lamda1),
            verbose=verbose,
            save=save_model,
            seed=seed,
            device=device,
            early_stopping=early_stopping,
        )

    #if epochs > 0:
    #    evaluate(
    #        model=model,
    #        data_scaling_type=scaling_type,
    #        timesteps=timesteps,
    #        experiment_name=experiment_name,
    #    )

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a VITAE model')
    parser.add_argument("--experiment_name", type=str, default="test", help="Experiment name for saving model and predictions")
    parser.add_argument("--model_type", type=str, default="unet", help="Model type: 'lite', 'base', 'large', or 'unet'")
    parser.add_argument("--sensor_type", type=str, default="real", help="Type of sensors: fixed (30, 48, 108), fixed-random, random, or real")
    parser.add_argument("--sensor_number", type=int, default=30, help="Number of sensors")
    parser.add_argument("--patch_size", nargs=2, type=int, default=[15, 10], help="Patch size (two integers)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lamda1", type=float, default=0.7, help="Lamda1 for loss combination")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-s", "--save_model", action="store_true", default=False, help="Save the trained model")
    parser.add_argument("-es", "--early_stopping", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the UNet model")
    parser.add_argument("--scaling_type", type=str, default="min-max", help="Type of scaling, e.g., 'standard' or 'min-max'")
    parser.add_argument("--timesteps", type=int, default=1, help="How many consecutive timesteps to be used for a training example.")
    parser.add_argument("--timesteps_jump", type=int, default=1, help="The time difference in hours between subsequen timesteps.")
    parser.add_argument("--split_mode", type=str, default="monthly", help="How to split the data. Between months or as percentages: 'monthly' or 'percentage'")
    parser.add_argument("--noise", type=str, default="none", choices=["none", "gaussian", "perlin"])

    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        model_type=args.model_type,
        sensor_type=args.sensor_type,
        sensor_number=args.sensor_number,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        lamda1=args.lamda1,
        verbose=args.verbose,
        seed=args.seed,
        save_model=args.save_model,
        early_stopping=args.early_stopping,
        kernel_size=args.kernel_size,
        scaling_type=args.scaling_type,
        timesteps=args.timesteps,
        timesteps_jump=args.timesteps_jump,
        split_mode=args.split_mode,
        noise=args.noise,
    )