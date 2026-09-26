import argparse
import logging
import torch

from src.utils.evaluation import evaluate_on_real
from src.utils.evaluation_pipeline import evaluate, ensemble_evaluate, uq_for_diffusion
from src.usage.vcnn_training import main as voronoi_main
from src.usage.diffusion_training import main as diffusion_main
from src.usage.vitae_training import main as vitae_main

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(
    experiment_name: str,
    model_types: list[str],
    epochs: int = 500,
    noise: str = 'none',
    full_noise: bool = True,
    seed: int = 42,
    pollutant_type: str = 'all',
    model_path: str = 'none',
    same_window:bool = False,
    timesteps: int = 4,
    inner_city: bool = True
):
    logging.info(f"Experiment: {experiment_name} with models: {model_types}")
    logging.info(f"Using {epochs} epochs, noise type: {noise}, full noise: {full_noise}, seed: {seed}")

    for model_type in model_types:
        if model_type == "vcnn":
            timesteps = 1  # VCNN performed the best on 1 timestep
            best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")

            model = voronoi_main(
                experiment_name=experiment_name,
                model_type="classic",
                seed=seed,
                sensor_number="real-random",
                scaling_type="min-max",
                timesteps=timesteps,
                timesteps_jump=1,
                batch_size=64,
                lr=best_params.get("lr", 1e-4),
                weight_decay=best_params.get("weight_decay", 1e-5),
                epochs=epochs,
                verbose=False,
                save_model=True,
                early_stopping=True,
                kernel_size=best_params.get("kernel_size", 3),
                use_norm=best_params.get("use_norm", True),
                hidden_channels=best_params.get("hidden_channels", 64),
                n_layers=best_params.get("n_layers", 3),
                noise=noise,
                full_noise=full_noise,
                pollutant_type=pollutant_type,
                model_path=model_path
            )

            model.eval()
            with torch.no_grad():
                evaluate_on_real(model=model, timesteps=timesteps)

        elif model_type == "vunet":
            timesteps = timesteps  # VUnet performed the best on 4 timesteps
            #best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")
            best_params = {}

            model = voronoi_main(
                experiment_name=experiment_name,
                model_type="unet",
                seed=seed,
                sensor_type="real-random",
                sensor_number=30,
                scaling_type="min-max",
                timesteps=timesteps,
                timesteps_jump=1,
                batch_size=64,
                lr=best_params.get("lr", 1e-4),
                weight_decay=best_params.get("weight_decay", 1e-3),
                epochs=epochs,
                verbose=False,
                save_model=True,
                early_stopping=False,
                kernel_size=best_params.get("kernel_size", 5),
                use_norm=best_params.get("use_norm", False),
                hidden_channels=best_params.get("hidden_channels", 64),
                n_layers=best_params.get("n_layers", 2),
                noise=noise,
                full_noise=full_noise,
                pollutant_type=pollutant_type,
                model_path=model_path
            )

            model.eval()
            with torch.no_grad():
                # evaluate_on_real(model=model, timesteps=timesteps, experiment_name=experiment_name)
                evaluate(model=model, timesteps=timesteps, experiment_name=experiment_name, data_scaling_type="min-max",
                          pollutant_type=pollutant_type, same_window=same_window, inner_city=inner_city)

        elif model_type == "vitae":
            timesteps = timesteps  # ViTAE performed the best on 6 timesteps
            #best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")
            best_params = {}

            model = vitae_main(
                experiment_name=experiment_name,
                model_type="unet",
                sensor_type="real-random",
                sensor_number=30,
                patch_size=(15, 10),
                batch_size=32,
                lr=best_params.get("lr", 1e-3),
                weight_decay=best_params.get("weight_decay", 1e-5),
                epochs=epochs,
                lamda1=best_params.get("lamda1", 0.1),
                verbose=False,
                seed=seed,
                save_model=True,
                early_stopping=False,
                kernel_size=best_params.get("kernel_size", 5),
                scaling_type="min-max",
                timesteps=timesteps,
                timesteps_jump=1,
                noise=noise,
                full_noise=full_noise,
                pollutant_type=pollutant_type,
                model_path=model_path
            )

            model.eval()

            with torch.no_grad():
                # evaluate_on_real(model=model, timesteps=timesteps, experiment_name=experiment_name)
                evaluate(model=model, timesteps=timesteps, experiment_name=experiment_name, data_scaling_type="min-max",
                          pollutant_type=pollutant_type, same_window=same_window, inner_city=inner_city)

        elif model_type == "clstm":
            timesteps = timesteps  # ConvLSTM performed the best on 8 timesteps
            #best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")
            best_params = {}

            model = voronoi_main(
                experiment_name=experiment_name,
                model_type="lstm",
                seed=seed,
                sensor_type="real-random",
                sensor_number=30,
                scaling_type="min-max",
                timesteps=timesteps,
                timesteps_jump=1,
                batch_size=32,
                lr=best_params.get("lr", 1e-4),
                weight_decay=best_params.get("weight_decay", 1e-5),
                epochs=epochs,
                verbose=False,
                save_model=True,
                early_stopping=False,
                kernel_size=best_params.get("kernel_size", 7),
                use_norm=best_params.get("use_norm", False),
                hidden_channels=best_params.get("hidden_channels", 64),
                n_layers=best_params.get("n_layers", 2),
                noise=noise,
                full_noise=full_noise,
                pollutant_type=pollutant_type,
                model_path=model_path
            )

            model.eval()

            with torch.no_grad():
                # evaluate_on_real(model=model, timesteps=timesteps, experiment_name=experiment_name)
                evaluate(model=model, timesteps=timesteps, experiment_name=experiment_name, data_scaling_type="min-max",
                          pollutant_type=pollutant_type, same_window=same_window, inner_city=inner_city)

        elif model_type == "diffusion":
            timesteps = 1  
            #best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")
            best_params = {}

            model = diffusion_main(
                experiment_name=experiment_name,
                model_type="diffusion",
                seed=seed,
                sensor_type="real-random",
                sensor_number=30,
                scaling_type="standard",
                timesteps=1,
                timesteps_jump=1,
                batch_size=64,
                lr=best_params.get("lr", 1e-4),
                weight_decay=best_params.get("weight_decay", 0),
                epochs=epochs,
                verbose=False,
                save_model=False,
                early_stopping=False,
                kernel_size=best_params.get("kernel_size", 3),
                use_norm=best_params.get("use_norm", True),
                noise=noise,
                full_noise=full_noise,
                pollutant_type=pollutant_type,
                model_path=model_path
            )

            #model.eval()
            with torch.no_grad():
                # evaluate_on_real(model=model, timesteps=timesteps, experiment_name=experiment_name)
                #uq_for_diffusion(model=model, data_scaling_type='standard', timesteps=timesteps, pollutant_type=pollutant_type)
                evaluate(model=model, timesteps=timesteps, experiment_name=experiment_name, data_scaling_type="standard", pollutant_type=pollutant_type)
                #ensemble_evaluate(model=model, data_scaling_type="standard", timesteps=timesteps, experiment_name=experiment_name, pollutant_type=pollutant_type)

        else:
            raise ValueError(f"Model type {model_type} is not recognized. Choose from 'vcnn', 'vunet', 'vitae', or 'clstm'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with specified timesteps")
    parser.add_argument("--experiment_name", type=str, default="real_random", help="Name of the experiment")
    parser.add_argument("--model_types", type=str, nargs='+', default=["vcnn", "vunet", "vitae", "clstm", "diffusion"], help="Model types: 'vcnn', 'vunet', 'vitae', or 'clstm'")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for training")
    parser.add_argument("--noise", type=str, default="none", choices=["none", "gaussian", "time_gaussian", "perlin", "correlated", "white"])
    parser.add_argument("--full_noise", action='store_true', help="Use full noise for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pollutant_type", type=str, default='all', help="Pollutant types: 'o3', 'pm10', 'pm25', 'no2', 'all'")
    parser.add_argument("--model_path", type=str, default='none', help='Path of the model weights')
    parser.add_argument('--same_window', action='store_true', default=False, help='Enable this feature to evaluate on same window')
    parser.add_argument("--timesteps", type=int, default=4, help="Number of previous timesteps")
    parser.add_argument('--inner_city', action='store_true', default=False, help='Enable this feature to evaluate on inner city')


    args = parser.parse_args()
    print(args.inner_city)

    main(
        experiment_name=args.experiment_name,
        model_types=args.model_types,
        epochs=args.epochs,
        noise=args.noise,
        full_noise=args.full_noise,
        seed=args.seed,
        pollutant_type=args.pollutant_type,
        model_path = args.model_path,
        same_window = args.same_window,
        timesteps=args.timesteps,
        inner_city = args.inner_city
    )