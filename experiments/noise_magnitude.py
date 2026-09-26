import torch
import numpy as np
import optuna
import os
import argparse
import logging

from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from src.models.vcnn import get_model, evaluate_loader
from src.datasets.voronoi_datasets import load_data
from src.datasets.vitae_dataset import load_data as load_vitae_data
from src.utils.training import train, get_voronoi_loss_fn
from src.utils.evaluation import evaluate_on_real


def suggest_noise_std(trial):
    o3_std = trial.suggest_float("o3_std", 1.0, 76.0, step=5.0)
    pm10_std = trial.suggest_float("pm10_std", 1.0, 21.0, step=2.0)
    pm25_std = trial.suggest_float("pm25_std", 1.0, 21.0, step=2.0)
    no2_std = trial.suggest_float("no2_std", 1.0, 21.0, step=2.0)

    noise_std = torch.tensor([o3_std, pm10_std, pm25_std, no2_std])

    return noise_std

def objective(
    trial,
    model_type: str,
    noise_type: str,
    magnitude_scale: torch.Tensor,
    epochs: int = 500,
    seed: int = 42,
) -> float:

    noise_params = torch.load(f'results/misc/noise_params_{noise_type}.pth', weights_only=False)
    noise_std = suggest_noise_std(trial)
    noise_params["noise_std"] = noise_std / magnitude_scale

    best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")
    model_type_old = "tiny" if model_type == "vunet" else "lstm"
    batch_size = 64 if model_type == "vunet" else 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, _, _ = load_data(
        sensor_type="real-random",
        sensor_number=30,
        split_mode="monthly",
        scaling_type="min-max",
        combine_train_val=False,
        timesteps=8,
        timesteps_jump=1,
        channel_timesteps=model_type != "lstm",
        noise=noise_type,
        full_noise=True,
        seed=seed,
        noise_params=noise_params,
        pollutant_type=pollutant_type
    )

    model = get_model(
        model_type=model_type_old,
        train_dataset=train_dataset,
        kernel_size=best_params.get("kernel_size", 3),
        use_norm=best_params.get("use_norm", True),
        hidden_channels=best_params.get("hidden_channels", 64),
        n_layers=best_params.get("n_layers", 3)
    )

    model, _, _, = train(
        experiment_name="Tuning Noise Magnitude",
        model=model,
        epochs=epochs,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3),
        lr=best_params.get("lr", 0.001),
        weight_decay= best_params.get("weight_decay", 0.0),
        evaluation_fn=evaluate_loader,
        loss_fn=get_voronoi_loss_fn(),
        verbose=False,
        save=False,
        seed=seed,
        device=device,
        early_stopping=False,
    )

    relative_error = evaluate_on_real(
        model=model,
        timesteps=8,
        experiment_name="Tuning Noise Magnitude",
        device=device,
        save_predictions=False,
        validation_set=True,
        verbose=False,
    )

    return relative_error

def optimize_noise_params(
    model_type: str,
    noise_type: str,
    n_trials: int = 500,
    epochs: int = 500,
    seed: int = 42,
):
    # Initial setup
    
    seed_everything(seed=seed, verbose=False)
    
    sampler = TPESampler(seed=seed)
    study_name = f"{model_type} noise magnitude optimisation ({noise_type})"
    study = optuna.create_study(
        study_name=study_name, direction="minimize", sampler=sampler
    )

    # Load the data to get the scaling factors for the noise std

    _, _, _, stats = load_vitae_data()
    data_min = torch.tensor(stats['data_min'][0, :, 0, 0])
    data_max = torch.tensor(stats['data_max'][0, :, 0, 0])
    magnitude_scale = data_max - data_min

    # Run the optimization
    study.optimize(
        lambda trial: objective(
            trial,
            model_type=model_type,
            noise_type=noise_type,
            magnitude_scale=magnitude_scale,
            epochs=epochs,
            seed=seed
        ),
        n_trials=n_trials,
    )

    # Save the study results
    best_params = study.best_params
    logging.info(f"Best params: {best_params}")

    save_path = f"results/misc/noise_best_{noise_type}.pth"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(best_params, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vunet", choices=["vunet", "clstm"])
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "time_gaussian", "perlin"])
    parser.add_argument("-n", "--n_trials", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    optimize_noise_params(
        model_type=args.model_type, 
        noise_type=args.noise_type, 
        n_trials=args.n_trials, 
        epochs=args.epochs,
        seed=args.seed
    )