import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import optuna
from optuna.samplers import TPESampler

from pykrige.ok import OrdinaryKriging

from src.datasets.vitae_dataset import load_data
from src.utils.evaluation import compute_all_metrics, compute_relative_error
from src.datasets.real_obs_dataset import load_data as load_real
from src.utils.evaluation_pipeline import evaluate, compute_real_metrics, compute_simulated_metrics

def make_variogram_params(variogram_model: str, params: dict) -> dict | None:
    try:
        if variogram_model == "power":
            return {
                "scale": params["scale"],
                "exponent": params["exponent"],
                "nugget": params["nugget"],
            }
        elif variogram_model == "linear":
            return {
                "slope": params["slope"],
                "nugget": params["nugget"],
            }
        else:  # "gaussian", "spherical", or "exponential"
            return {
                "range": params["range"],
                "sill": params["sill"],
                "nugget": params["nugget"],
            }
    except KeyError:
        # If a key is missing (shouldn't happen for best_trial), fall back to PyKrige auto-fit
        return None

def kriging_interpolate_image(
    img: np.ndarray,
    variogram_model: str,
    variogram_parameters: dict,
    nlags: int,
    weight: bool,
    anisotropy_scaling: float,
    anisotropy_angle: float,
    exact_values: bool
) -> np.ndarray:
    h, w = img.shape
    y_idx, x_idx = np.where(img != 0)
    values = img[y_idx, x_idx]

    # if too few points, skip
    if len(values) < 3:
        return img.copy(), np.zeros_like(img, dtype=float)

    # build the Ordinary Kriging model with hyperparameters
    OK = OrdinaryKriging(
        x_idx.astype(float),
        y_idx.astype(float),
        values.astype(float),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nlags=nlags,
        weight=weight,
        anisotropy_scaling=anisotropy_scaling,
        anisotropy_angle=anisotropy_angle,
        exact_values=exact_values,
        verbose=False,
        enable_plotting=False
    )

    grid_x = np.arange(w, dtype=float)
    grid_y = np.arange(h, dtype=float)
    z_interp, krig_variance = OK.execute("grid", grid_x, grid_y)
    return z_interp, krig_variance


def kriging_interpolate_tensor(
    tensor_np: np.ndarray,
    variogram_model: str,
    variogram_parameters: dict,
    nlags: int,
    weight: bool,
    anisotropy_scaling: float,
    anisotropy_angle: float,
    exact_values: bool
) -> np.ndarray:
    out = np.zeros_like(tensor_np)
    variance = np.zeros_like(tensor_np, dtype=float)

    for i in range(tensor_np.shape[0]):      # time
        for c in range(tensor_np.shape[1]):  # channel
            pred, var = kriging_interpolate_image(
                tensor_np[i, c],
                variogram_model,
                variogram_parameters,
                nlags,
                weight,
                anisotropy_scaling,
                anisotropy_angle,
                exact_values
            )

            out[i, c] = pred
            variance[i,c] = var
    return out, variance

def tune_parameters(seed: int = 42, n_trials: int = 100):

    def objective(trial):
        train_dataset, _, _, _ = load_data(sensor_type="realistic", combine_train_val=True, scaling_type="none")

        train_obs = torch.stack([obs for obs, _, _ in train_dataset]).numpy()
        train_gts = torch.stack([gt for _, gt, _ in train_dataset])

        # Define the hyperparameters here

        variogram_model = trial.suggest_categorical(
            "variogram_model", ["linear", "power", "gaussian", "spherical", "exponential"]
        )

        if variogram_model == "power":
            # power: scale, exponent, nugget
            scale   = trial.suggest_float("scale",   1e-3, float(np.nanstd(train_obs)), log=True)
            exponent= trial.suggest_float("exponent", 0.1, 2.0)
            nugget  = trial.suggest_float("nugget",  0.0, scale * 0.5)
            variogram_parameters = {
                "scale": scale,
                "exponent": exponent,
                "nugget": nugget
            }

        elif variogram_model == "linear":
            # linear: slope, nugget
            slope   = trial.suggest_float("slope",   0.1, 10.0, log=True)
            nugget  = trial.suggest_float("nugget",  0.0, slope * 0.5)
            variogram_parameters = {
                "slope": slope,
                "nugget": nugget
            }

        else:
            # gaussian / spherical / exponential: range, sill, nugget
            v_range = trial.suggest_float("range",  1e-1, max(train_obs.shape), log=True)
            sill    = trial.suggest_float("sill",   1e-3, float(np.nanstd(train_obs)), log=True)
            nugget  = trial.suggest_float("nugget", 0.0, sill * 0.5)
            variogram_parameters = {
                "range": v_range,
                "sill": sill,
                "nugget": nugget
            }

        nlags = trial.suggest_int("nlags", 2, 20)
        weight = trial.suggest_categorical("weight", [True, False])
        anisotropy_scaling = trial.suggest_float("anisotropy_scaling", 0.1, 10.0, log=True)
        anisotropy_angle = trial.suggest_float("anisotropy_angle", 0.0, 180.0)
        exact_values = trial.suggest_categorical("exact_values", [True, False])

        pred, var = kriging_interpolate_tensor(
            train_obs,
            variogram_model,
            variogram_parameters,
            nlags,
            weight,
            anisotropy_scaling,
            anisotropy_angle,
            exact_values
        )

        relative_errs = compute_relative_error(train_gts, torch.from_numpy(pred))

        return np.mean(relative_errs)
    
    study = optuna.create_study(study_name="Optimizing Kriging", direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    flat = best.params
    vm = flat["variogram_model"]
    vp = make_variogram_params(vm, flat)

    best_params = {
        "variogram_model": vm,
        "variogram_parameters": vp,
        "nlags": flat["nlags"],
        "weight": flat["weight"],
        "anisotropy_scaling": flat["anisotropy_scaling"],
        "anisotropy_angle": flat["anisotropy_angle"],
        "exact_values": flat["exact_values"],
        "_flat": flat,
    }

    print("Best parameters found (packed):")
    print(best_params)

    os.makedirs("results/trained_models/kriging/params", exist_ok=True)
    params_file = os.path.join("results/trained_models/kriging/params", "best_params.pth")
    torch.save(best_params, params_file)

    _, _, test_dataset, _ = load_data(sensor_type="realistic", combine_train_val=False, scaling_type="none")

    obs = torch.stack([obs for obs, _, _ in test_dataset]).numpy()
    gts = torch.stack([gt for _, gt, _ in test_dataset])

    pred, var = kriging_interpolate_tensor(
            obs,
            best_params.get("variogram_model", None),
            best_params.get("variogram_parameters", None),
            best_params.get("nlags", None),
            best_params.get("weight", None),
            best_params.get("anisotropy_scaling", None),
            best_params.get("anisotropy_angle", None),
            best_params.get("exact_values", None)
        )

    error, ssim, psnr, local_error = compute_all_metrics(gts, torch.from_numpy(pred))

    preds_dir = f"results/predictions/kriging"
    os.makedirs(preds_dir, exist_ok=True)
    pred_file = os.path.join(preds_dir, f"realistic_predictions.npz")

    np.savez_compressed(
        pred_file,
        observations=obs,
        ground_truth=gts,
        predictions=np.array(pred),
        errors=np.array(error),
        ssim=np.array(ssim),
        psnr=np.array(psnr),
        local_errors=local_error,
    )

    return best_params


def predict_real():
    best_params = torch.load("results/trained_models/kriging/params/best_params.pth")

    # Timesteps is kept at 8 for consistency with all the other models.
    dataset, _ = load_real(model_type="vitae", sensor_type="real-random", timesteps=8, scale=False)

    obs = torch.stack([obs[-4:, :, :] for obs, _, _ in dataset])
    gts = torch.stack([gt for _, gt, _ in dataset])
    mask = torch.stack([mask for _, _, mask in dataset])

    pred, var = kriging_interpolate_tensor(
        obs.numpy(),
        best_params.get("variogram_model", None),
        best_params.get("variogram_parameters", None),
        best_params.get("nlags", None),
        best_params.get("weight", None),
        best_params.get("anisotropy_scaling", None),
        best_params.get("anisotropy_angle", None),
        best_params.get("exact_values", None)
        )

    error = compute_relative_error(gts * mask, torch.from_numpy(pred) * mask)

    print(f"Kriging Real Dataset Error: {np.mean(error):.4f}")
    
    preds_dir = f"results/predictions/kriging"
    os.makedirs(preds_dir, exist_ok=True)
    pred_file = os.path.join(preds_dir, f"real_dataset_predictions.npz")

    np.savez_compressed(
        pred_file,
        observations=obs,
        ground_truth=gts,
        predictions=pred,
        errors=error,
    )


def evaluate_paper_simulated():
    best_params = torch.load("results/trained_models/kriging/best_params.pth")

    # Timesteps is kept at 8 for consistency with all the other models.
    _, _, dataset, _ = load_data(sensor_type="real-random", timesteps=1)

    obs = torch.stack([obs[-4:, :, :] for obs, _, _ in dataset])
    gts = torch.stack([gt for _, gt, _ in dataset])

    pred, var = kriging_interpolate_tensor(
        obs.numpy(),
        best_params.get("variogram_model", None),
        best_params.get("variogram_parameters", None),
        best_params.get("nlags", None),
        best_params.get("weight", None),
        best_params.get("anisotropy_scaling", None),
        best_params.get("anisotropy_angle", None),
        best_params.get("exact_values", None)
        )

    return compute_simulated_metrics(obs, gts, torch.from_numpy(pred))


def evaluate_paper_real():
    best_params = torch.load("results/trained_models/kriging/params/best_params.pth")

    # Timesteps is kept at 8 for consistency with all the other models.
    dataset, _ = load_real(model_type="vitae", sensor_type="real-random", timesteps=8, scale=False)

    obs = torch.stack([obs[-4:, :, :] for obs, _, _ in dataset])
    gts = torch.stack([gt for _, gt, _ in dataset])
    mask = torch.stack([mask for _, _, mask in dataset])

    pred, var = kriging_interpolate_tensor(
        obs.numpy(),
        best_params.get("variogram_model", None),
        best_params.get("variogram_parameters", None),
        best_params.get("nlags", None),
        best_params.get("weight", None),
        best_params.get("anisotropy_scaling", None),
        best_params.get("anisotropy_angle", None),
        best_params.get("exact_values", None)
        )

    return compute_real_metrics(obs, gts, torch.from_numpy(pred), mask)


def evaluate_paper():
    results_on_simulated = evaluate_paper_simulated()
    results_on_real = evaluate_paper_real()

        # Decide where to save the results
    preds_dir = "paper_results/predictions/kriging"
    os.makedirs(preds_dir, exist_ok=True)
    preds_file = os.path.join(preds_dir, f"kriging_paper_results.npz")

    np.savez_compressed(
        preds_file,

        # ------ Saving the results on the simulated data ------

        # Save the data used to compute the metrics
        simulated_observations=results_on_simulated["observations"],
        simulated_ground_truths=results_on_simulated["ground_truths"],
        simulated_predictions=results_on_simulated["predictions"],
        # Save the relative error
        simulated_global_re=results_on_simulated["global_re"],
        simulated_pollutants_re=results_on_simulated["pollutants_re"],
        # Save the RMSe
        simulated_global_rmse=results_on_simulated["global_rmse"],
        simulated_pollutants_rmse=results_on_simulated["pollutants_rmse"],
        # Save the RRMSe
        simulated_global_rrmse=results_on_simulated["global_rrmse"],
        simulated_pollutants_rrmse=results_on_simulated["pollutants_rrmse"],
        # Save the MFE
        simulated_global_mfe=results_on_simulated["global_mfe"],
        simulated_pollutants_mfe=results_on_simulated["pollutants_mfe"],
        # Save the MFB
        simulated_global_mfb=results_on_simulated["global_mfb"],
        simulated_pollutants_mfb=results_on_simulated["pollutants_mfb"],
        # Save SSIM
        simulated_global_ssim = results_on_simulated["global_ssim"],
        simulated_pollutants_ssim = results_on_simulated["pollutants_ssim"],


        # ------ Saving the results on the real data ------

        # Save the data used to compute the metrics
        real_observations=results_on_real["observations"],
        real_ground_truths=results_on_real["ground_truths"],
        real_predictions=results_on_real["predictions"],
        real_target_masks=results_on_real["target_masks"],
        # Save the relative error
        real_global_re=results_on_real["global_re"],
        real_pollutants_re=results_on_real["pollutants_re"],
        # Save the RMSe
        real_global_rmse=results_on_real["global_rmse"],
        real_pollutants_rmse=results_on_real["pollutants_rmse"],
        # Save the RRMSe
        real_global_rrmse=results_on_real["global_rrmse"],
        real_pollutants_rrmse=results_on_real["pollutants_rrmse"],
        # Save the MFE
        real_global_mfe=results_on_real["global_mfe"],
        real_pollutants_mfe=results_on_real["pollutants_mfe"],
        # Save the MFB
        real_global_mfb=results_on_real["global_mfb"],
        real_pollutants_mfb=results_on_real["pollutants_mfb"]
    )

    print(f"Saved evaluation results to {preds_file}", flush=True)


def uq_kriging():
    # Load best Kriging parameters
    best_params = torch.load("results/kriging/best_params.pth")
    # Load dataset
    dataset, _ = load_real(model_type="vitae",sensor_type="real-random",timesteps=8,scale=False)
    # Observations used as Kriging input
    obs = torch.stack([obs[-4:, :, :] for obs, _, _ in dataset])
    # Ground truth
    ground_truth = torch.stack([gt for _, gt, _ in dataset])
    # Evaluation mask
    target_mask = torch.stack([mask for _, _, mask in dataset])

    obs_np = obs.numpy()
    ground_truth = ground_truth.numpy()
    target_mask = (
        target_mask.numpy() > 0
    ).astype(np.uint8)

    # Kriging prediction + variance
    kriging_prediction, kriging_variance = (
        kriging_interpolate_tensor(
        obs_np,
        best_params["variogram_model"],
        best_params["variogram_parameters"],
        best_params["nlags"],
        best_params["weight"],
        best_params["anisotropy_scaling"],
        best_params["anisotropy_angle"],
        best_params["exact_values"]
    )
    )

    # Generate Gaussian ensemble
    rng = np.random.default_rng(seed)

    kriging_variance = np.maximum(kriging_variance,0.0)
    kriging_std = np.sqrt(kriging_variance)

    N, P, H, W = kriging_prediction.shape

    noise = rng.standard_normal(size=(N,20,P,H,W))

    prediction_real = (kriging_prediction[:, None, ...]+ noise * kriging_std[:, None, ...])


    # Simulation data
    _, _, dataset, _ = load_data(sensor_type="real-random", timesteps=1, pollutant_type='all')

    obs = torch.stack([obs[-4:, :, :] for obs, _, _ in dataset])
    obs_np = obs.numpy()
    gts = torch.stack([gt for _, gt, _ in dataset])
    ground_truth_sim = gts.numpy()

    # Kriging prediction + variance
    kriging_prediction, kriging_variance = (
        kriging_interpolate_tensor(
        obs_np,
        best_params["variogram_model"],
        best_params["variogram_parameters"],
        best_params["nlags"],
        best_params["weight"],
        best_params["anisotropy_scaling"],
        best_params["anisotropy_angle"],
        best_params["exact_values"]
    )
    )
    # Generate Gaussian ensemble
    rng = np.random.default_rng(seed)
    kriging_variance = np.maximum(kriging_variance,0.0)
    kriging_std = np.sqrt(kriging_variance)
    N, P, H, W = kriging_prediction.shape
    noise = rng.standard_normal(size=(N,20,P,H,W))
    prediction_sim = (kriging_prediction[:, None, ...]+ noise * kriging_std[:, None, ...])

    # Save
    np.savez_compressed(
        "paper_results/predictions/kriging/ensemble_results.npz",
        ground_truth_real=ground_truth,
        prediction_real=prediction_real,
        target_mask=target_mask,
        ground_truth_sim=ground_truth_sim,
        prediction_sim=prediction_sim
    )

if __name__ == "__main__":
    seed = 42
    n_trials = 100

    # best_params = tune_parameters(seed=seed, n_trials=n_trials)
    # predict_real()
    #evaluate_paper()
    uq_kriging()    