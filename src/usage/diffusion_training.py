import argparse
import torch
import logging

from torch.utils.data import DataLoader

from src.models.diffusion import get_model, EvaluateDiffusionModel
from src.utils.training_pipeline_diffusion import train, get_voronoi_loss_fn
from src.datasets.voronoi_datasets import load_data
from src.utils.evaluation_pipeline import evaluate, ensemble_evaluate


logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(
    experiment_name: str,
    model_type: str,
    seed: int,
    sensor_type: str,
    sensor_number: int,
    scaling_type: str,
    timesteps: int,
    timesteps_jump: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    verbose: bool,
    save_model: bool,
    early_stopping: bool,
    kernel_size: int,
    use_norm: bool,
    noise: str = "none",
    full_noise: bool = True,
    split_mode: str = 'monthly',
    use_val: bool = False,
    load_checkpoint: bool = False,
    pollutant_type: str = 'all',
    model_path=None
):
    """
    Train a Diffusion model.

    Args:
        experiment_name (str): Name of the experiment.
        model_type (str): Type of the model to train.
        seed (int): Random seed for reproducibility.
        sensor_type (str): Type of sensors to use.
        sensor_number (int): Number of sensors.
        scaling_type (str): Type of scaling to apply to the data.
        timesteps (int): Number of consecutive timesteps to be used for a training example.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        epochs (int): Number of epochs to train the model.
        verbose (bool): Whether to print verbose output during training.
        save_model (bool): Whether to save the trained model.
        early_stopping (bool): Whether to use early stopping during training.
        kernel_size (int): Kernel size for convolutional layers.
        use_norm (bool): Whether to use normalization layers in the model.
        hidden_channels (int): Number of hidden channels for the ConvLSTM model.
        n_layers (int): Number of LSTM layers for the ConvLSTM model.
        split_mode (str, optional): Mode for splitting the dataset. Defaults to 'monthly'.

    Returns:
        None: The model and the results can optionally be saved to disk.
    """
    train_dataset, val_dataset, test_dataset, stats = load_data(
    sensor_type=sensor_type,
    sensor_number=sensor_number,
    split_mode=split_mode,
    scaling_type=scaling_type,
    combine_train_val=not use_val,
    timesteps=timesteps,
    timesteps_jump=timesteps_jump,
    channel_timesteps=model_type != "lstm",
    noise=noise,
    full_noise=full_noise,
    seed=seed,
    diffusion=True,
    pollutant_type=pollutant_type
    )

    if use_val:
        test_dataset = val_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(
        train_dataset=train_dataset,
        load_checkpoint=load_checkpoint,
        device = device
    )

    if epochs==0:
        checkpoint = torch.load(model_path, weights_only=False)
        model.denoiser_model.load_state_dict(checkpoint['model_state_dict'])
        model.cond_model.load_state_dict(checkpoint['cond_model_state_dict'])

    if epochs>0:
        evaluate_loader = EvaluateDiffusionModel(model.denoiser_model, model.cond_model, device=device).evaluate_model

        model, _, _, = train(
            experiment_name=experiment_name,
            model=model,
            epochs=epochs,
            train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3, drop_last=True),
            val_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3, drop_last=True),
            lr=lr,
            weight_decay=weight_decay,
            evaluation_fn=evaluate_loader,
            loss_fn=get_voronoi_loss_fn(),
            verbose=verbose,
            save=save_model,
            seed=seed,
            device=device,
            early_stopping=early_stopping,
            load_checkpoint=load_checkpoint,
            pollutant_type=pollutant_type
    )
    '''
    if epochs > 0:
        evaluate(
            model=model,
            data_scaling_type=scaling_type,
            timesteps=timesteps,
            experiment_name=experiment_name,
            pollutant_type=pollutant_type
        )

        ensemble_evaluate(
            model=model,
            data_scaling_type=scaling_type,
            timesteps=timesteps,
            experiment_name=experiment_name,
            pollutant_type=pollutant_type
        )
    '''

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VCNN model")
    parser.add_argument("--experiment_name", type=str, default="test", help="Name of the experiment")
    parser.add_argument("--model_type", type=str, default="diffusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--sensor_type", type=str, default="real-random", help="Type of sensors: fixed (30, 48, 108), fixed-random, random, or real")
    parser.add_argument("--sensor_number", type=int, default=30, help="Number of sensors")
    parser.add_argument("--scaling_type", type=str, default="standard", help="Scaling type, e.g., 'standard' or 'min-max'")
    parser.add_argument("--timesteps", type=int, default=1, help="How many consecutive timesteps to be used for a training example.")
    parser.add_argument("--timesteps_jump", type=int, default=1, help="The time difference in hours between subsequen timesteps.")
    parser.add_argument("--noise", type=str, default="none", choices=["none", "gaussian", "time_gaussian", "perlin", "correlated", "white"])
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=240, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-s", "--save_model", action="store_true", default=False, help="Save the trained model")
    parser.add_argument("-es", "--early_stopping", action="store_true", default=False, help="Enable early stopping")
    
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--use_norm", action="store_true", default=False, help="Use normalization layers")
    parser.add_argument("-lcp", "--load_checkpoint", action="store_true", help="Use this to load a checkpoint")

    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        model_type=args.model_type,
        seed=args.seed,
        sensor_type=args.sensor_type,
        sensor_number=args.sensor_number,
        scaling_type=args.scaling_type,
        timesteps=args.timesteps,
        timesteps_jump=args.timesteps_jump,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        verbose=args.verbose,
        save_model=args.save_model,
        early_stopping=args.early_stopping,
        kernel_size=args.kernel_size,
        use_norm=args.use_norm,
        noise=args.noise,
        load_checkpoint=args.load_checkpoint
    )