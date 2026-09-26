import argparse
import torch
import logging

from src.models.diffusion import get_model as get_diffusion_model
from src.datasets.voronoi_datasets import load_data
from src.utils.evaluation_pipeline import evaluate, ensemble_evaluate

logging.basicConfig(level=logging.INFO, format="%(message)s")


def eval_diffusion(
    experiment_name: str,
    model_path: str,
    sensor_type: str,
    seed: int,
    eval_ensemble: bool,
    pollutant_type: str
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, *_ = load_data(
        sensor_type=sensor_type,
        sensor_number=30,
        split_mode="monthly",
        scaling_type="standard",
        combine_train_val=False,
        timesteps=1,
        timesteps_jump=1,
        channel_timesteps=True,
        noise="none",
        full_noise=True,
        seed=seed,
        diffusion=True,
        pollutant_type=pollutant_type
    )

    model = get_diffusion_model(
        train_dataset=train_dataset,
        load_checkpoint=False,
        device = device,
        weights_path=model_path
    )

    model.best_model()

    logging.info(f"Loaded diffusion model: {model_path}")

    evaluate(
        model=model,
        data_scaling_type="standard",
        timesteps=1,
        experiment_name=experiment_name,
    )
    if eval_ensemble:
        ensemble_evaluate(
            model=model,
            data_scaling_type="standard",
            timesteps=1,
            experiment_name=experiment_name,
            pollutant_type=pollutant_type
        )


def eval_unet(
    experiment_name: str,
    model_path: str,
    sensor_type: str,
    seed: int,
):
    pass  # Placeholder for UNet evaluation logic


def main(
    experiment_name: str,
    model_type: str,
    model_path: str,
    sensor_type: str,
    seed: int,
    noise: str,
    timesteps: int,
    eval_ensemble:bool,
    pollutant_type:str
):
    if model_type == "diffusion":
        eval_diffusion(
            experiment_name=experiment_name,
            model_path=model_path,
            sensor_type=sensor_type,
            seed=seed,
            eval_ensemble=eval_ensemble,
            pollutant_type = pollutant_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VCNN model")
    parser.add_argument("--experiment_name", type=str, default="test", help="Name of the experiment")
    parser.add_argument("--model_type", type=str, default="diffusion")
    parser.add_argument("--model_path", type=str, help="Path to saved model weights")

    parser.add_argument("--sensor_type", type=str, default="real-random", choices=["real", "real-random"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=1, help="How many consecutive timesteps to be used for a training example.")
    parser.add_argument("--noise", type=str, default="none", choices=["none", "gaussian", "perlin"])
    parser.add_argument("--ensemble", action="store_true", help="Use this to evaluate ensemble effect on diffusion model")
    parser.add_argument("--pollutant_type", type=str, default='all', help="Pollutant types: 'o3', 'pm10', 'pm25', 'no2', 'all'")

    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        model_type=args.model_type,
        model_path=args.model_path,
        sensor_type=args.sensor_type,
        seed=args.seed,
        noise=args.noise,
        timesteps=args.timesteps,
        eval_ensemble=args.ensemble,
        pollutant_type=args.pollutant_type
    )