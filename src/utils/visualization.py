import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import os
import torch
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from PIL import Image
from matplotlib.legend_handler import HandlerTuple

from IPython.display import HTML
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter1d
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from src.utils.evaluation import compute_relative_error, compute_local_relative_error, compute_local_error
from src.datasets.utils import get_custom_noise, read_real_observation_files
from src.datasets.real_obs_dataset import load_data as load_real_obs_data
from src.datasets.vitae_dataset import load_data as load_vitae_data

background_paris = np.load("data/map_rgb_adjusted.npy")

def print_with_borders(img: np.array, ax, vmax: int, cmap: str):
    img_rgb = background_paris.astype(np.uint8)
    mask_lines = (img_rgb < 250).any(axis=2)
    img_rgb[mask_lines] = [255, 255, 255]
    alpha = (mask_lines * 255 * 0.5).astype(np.uint8)
    paris_rgba = np.dstack([img_rgb, alpha])

    im = ax.imshow(
        np.flipud(img), vmin=0, vmax=vmax, 
        cmap=cmap,
        extent=[0, 110, 0, 75])
    ax.imshow(
        paris_rgba,
        origin="upper",          
        extent=[0, 110, 0, 75],
        zorder=10                
    )

    return im

def _add_aligned_colorbar(im, ax, pad=0.05, size="5%", fontsize=14):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize)


def reset_theme():
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.reset_defaults()
    plt.close()
    return


def animate_predictions(predictions: np.ndarray, num_frames: int, save_dir: str=None, filename: str="predictions_animation.gif"):
    sns.set_theme(style="white", context="talk", font_scale=1)
    
    pollutants = ["O3", "PM10", "PM25", "NO2"]

    v_mins = [np.nanmin(predictions[:, i]) for i in range(len(pollutants))]
    v_maxs = [np.nanmax(predictions[:, i]) for i in range(len(pollutants))]

    fig, axs = plt.subplots(1, 4, figsize=(22, 4))

    # fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    # axs = axs.flatten()

    pred_images = []

    for i, pollutant in enumerate(pollutants):
        im_pred = axs[i].imshow(predictions[0, i], animated=True, cmap='viridis', vmin=0)
        axs[i].set_title(f"{pollutant} Prediction")
        pred_images.append(im_pred)
        _add_aligned_colorbar(im_pred, axs[i])

    def update(frame):
        for i in range(len(pollutants)):
            pred_images[i].set_array(predictions[frame, i])
        return pred_images

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True, interval=200)
    plt.tight_layout()
    plt.close(fig)

    # Save as GIF or return HTML
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        ani.save(save_path, writer='pillow', fps=5)

        reset_theme()
        return None
    else:
        reset_theme()
        return HTML(ani.to_jshtml())


def animate_results(ground_truth, predictions, errors=None, num_frames=100, save_dir=None, filename="results_animation.gif"):
    sns.set_theme(style="white", context="talk", font_scale=1)
    
    pollutants = ["O3", "PM10", "PM25", "NO2"]
    nrows = 3 if errors is not None else 2
    fig, axs = plt.subplots(nrows, 4, figsize=(14, 8 if errors is not None else 7))

    obs_images = []
    pred_images = []
    error_images = []

    v_mins = [np.nanmin(ground_truth[:, i]) for i in range(len(pollutants))]
    v_maxs = [np.nanmax(ground_truth[:, i]) for i in range(len(pollutants))]

    for i, pollutant in enumerate(pollutants):
        # Ground Truth
        im_obs = axs[0, i].imshow(ground_truth[0, i], animated=True, cmap='viridis', vmin=v_mins[i], vmax=v_maxs[i])
        axs[0, i].set_title(f"{pollutant} Ground Truth")
        obs_images.append(im_obs)
        _add_aligned_colorbar(im_obs, axs[0, i])

        # Predictions
        im_pred = axs[1, i].imshow(predictions[0, i], animated=True, cmap='viridis', vmin=v_mins[i], vmax=v_maxs[i])
        axs[1, i].set_title(f"{pollutant} Model Prediction")
        pred_images.append(im_pred)
        _add_aligned_colorbar(im_pred, axs[1, i])

        # Errors
        if errors is not None:
            im_err = axs[2, i].imshow(errors[0, i], animated=True, cmap='coolwarm')
            axs[2, i].set_title(f"{pollutant} Prediction Errors")
            error_images.append(im_err)
            _add_aligned_colorbar(im_err, axs[2, i])

    def update(frame):
        for i in range(len(pollutants)):
            obs_images[i].set_array(ground_truth[frame, i])
            pred_images[i].set_array(predictions[frame, i])
            if errors is not None:
                error_images[i].set_array(errors[frame, i])
        return obs_images + pred_images + (error_images if errors is not None else [])

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True, interval=200)
    plt.tight_layout()
    plt.close(fig)

    # Save as GIF or return HTML
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        ani.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
        reset_theme()
        return None
    else:
        reset_theme()
        return HTML(ani.to_jshtml())


def plot_training(training_losses: list[float], validation_errors: list[float], save_dir: str):
    epochs = list(range(1, len(training_losses) + 1))
    val_epochs = list(range(5, len(training_losses) + 1, 5))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss on the left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.plot(epochs, training_losses, color='blue', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for validation error
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Relative Error', color='red')
    ax2.plot(val_epochs, validation_errors, color='red', marker='o', label='Validation Error')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Training Loss and Validation Error Over Epochs')
    fig.tight_layout()
    plt.grid(True)

    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

    reset_theme()


def plot_experiments(
    experiment_names: list[str],
    experiment_results: list[dict[str, np.ndarray]],
    pollutants: list[str] = ["o3", "pm10", "pm25", "no2"],
    save_dir: str = None
) -> None:

    pollutant_index = {"o3": 0, "pm10": 1, "pm25": 2, "no2": 3}
    num_models = len(experiment_names)
    num_pollutants = len(pollutants)

    if not experiment_results:
        print("Warning: experiment_results is empty. Cannot generate plots.")
        return

    # 1. For Ground Truth and Predictions plot
    pred_plot_ranges = {}
    ground_truth_base_data = experiment_results[0]['ground_truth'][0]

    for pollutant in pollutants:
        idx = pollutant_index.get(pollutant)
        if idx is None:
            print(f"Warning: Pollutant '{pollutant}' not in pollutant_index. Skipping range calculation for predictions.")
            pred_plot_ranges[pollutant] = (None, None)
            continue

        current_min_val = np.inf
        current_max_val = -np.inf
        data_found_for_pollutant = False

        # Check ground truth data for this pollutant
        if idx < ground_truth_base_data.shape[0]:
            gt_slice = ground_truth_base_data[idx]
            if gt_slice.size > 0 and not np.all(np.isnan(gt_slice)):
                current_min_val = np.nanmin([current_min_val, np.nanmin(gt_slice)])
                current_max_val = np.nanmax([current_max_val, np.nanmax(gt_slice)])
                data_found_for_pollutant = True
        else:
            print(f"Warning: Index {idx} for pollutant '{pollutant}' is out of bounds for ground truth data shape {ground_truth_base_data.shape} during range calculation.")


        # Check predictions data for this pollutant from all experiments
        for res in experiment_results:
            predictions_data_for_model = res['predictions'][0]
            if idx < predictions_data_for_model.shape[0]:
                pred_slice = predictions_data_for_model[idx]
                if pred_slice.size > 0 and not np.all(np.isnan(pred_slice)):
                    current_min_val = np.nanmin([current_min_val, np.nanmin(pred_slice)])
                    current_max_val = np.nanmax([current_max_val, np.nanmax(pred_slice)])
                    data_found_for_pollutant = True

        if data_found_for_pollutant and not (np.isinf(current_min_val) or np.isinf(current_max_val)):
            pred_plot_ranges[pollutant] = (current_min_val, current_max_val)
        else:
            pred_plot_ranges[pollutant] = (None, None)

    # 2. For Errors plot (symmetric range)
    error_plot_ranges = {}
    if num_models > 0:
        for pollutant in pollutants:
            idx = pollutant_index.get(pollutant)
            if idx is None:
                print(f"Warning: Pollutant '{pollutant}' not in pollutant_index. Skipping range calculation for errors.")
                error_plot_ranges[pollutant] = (None, None)
                continue

            max_abs_error = 0.0
            data_found_for_pollutant_error = False

            for res in experiment_results:
                errors_data_for_model = res['local_errors'][0]
                if idx < errors_data_for_model.shape[0]:
                    error_slice = errors_data_for_model[idx]
                    if error_slice.size > 0 and not np.all(np.isnan(error_slice)):
                        current_max_abs = np.nanmax(np.abs(error_slice))
                        if not np.isnan(current_max_abs):
                            max_abs_error = np.nanmax([max_abs_error, current_max_abs])
                            data_found_for_pollutant_error = True


            if data_found_for_pollutant_error:
                if max_abs_error == 0: # All errors are exactly zero
                     error_plot_ranges[pollutant] = (0.0, 0.0)
                else:
                     error_plot_ranges[pollutant] = (-max_abs_error, max_abs_error)
            else:
                error_plot_ranges[pollutant] = (None, None) # Fallback

    cols_pred_plot = 1 + num_models
    fig_pred, axs_pred = plt.subplots(
        nrows=num_pollutants,
        ncols=cols_pred_plot,
        figsize=(4 * cols_pred_plot, 3.5 * num_pollutants),
        squeeze=False
    )
    fig_pred.suptitle("Ground Truth vs. Model Predictions", fontsize=16, y=1.0)

    ground_truth_data_for_plot = experiment_results[0]['ground_truth'][0]

    for row_idx, pollutant in enumerate(pollutants):
        idx = pollutant_index.get(pollutant)
        vmin_pred, vmax_pred = pred_plot_ranges.get(pollutant, (None, None))

        if idx is None:
            print(f"Warning: Pollutant '{pollutant}' not found in pollutant_index. Skipping row in predictions plot.")
            for col_jdx in range(cols_pred_plot):
                if axs_pred[row_idx, col_jdx] is not None:
                     axs_pred[row_idx, col_jdx].set_title(f"Unknown Pollutant: {pollutant}", fontsize=10)
                     axs_pred[row_idx, col_jdx].axis('off')
            continue

        ax = axs_pred[row_idx, 0]
        if idx >= ground_truth_data_for_plot.shape[0]:
            print(f"Warning: Index {idx} for pollutant '{pollutant}' out of bounds for ground truth data. Skipping GT plot.")
            ax.set_title(f"{pollutant.upper()} GT (Index Error)", fontsize=12)
            ax.axis('off')
        else:
            try:
                im = ax.imshow(ground_truth_data_for_plot[idx], cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
                ax.axis('off')
                ax.set_title(f"{pollutant.upper()} Ground Truth", fontsize=12)
                fig_pred.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except IndexError:
                print(f"Error: Could not plot ground truth for '{pollutant}' at index {idx}. Data shape: {ground_truth_data_for_plot.shape}")
                ax.set_title(f"{pollutant.upper()} GT (Plot Error)", fontsize=12)
                ax.axis('off')
            except ValueError as e:
                print(f"Error plotting ground truth for '{pollutant}': {e}")
                ax.set_title(f"{pollutant.upper()} GT (Value Error)", fontsize=12)
                ax.axis('off')


        for model_idx, (name, res) in enumerate(zip(experiment_names, experiment_results)):
            col_pred = 1 + model_idx
            predictions_data = res['predictions'][0]

            ax_pred = axs_pred[row_idx, col_pred]
            if idx >= predictions_data.shape[0]:
                ax_pred.set_title(f"{name}\n{pollutant.upper()} Pred (Index Error)", fontsize=11)
                ax_pred.axis('off')
                continue

            try:
                im_pred = ax_pred.imshow(predictions_data[idx], cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
                ax_pred.axis('off')
                ax_pred.set_title(f"{name}\n{pollutant.upper()} Prediction", fontsize=11)
                fig_pred.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            except IndexError:
                print(f"Error: Could not plot prediction for {name}, '{pollutant}' at index {idx}. Data shape: {predictions_data.shape}")
                ax_pred.set_title(f"{name}\n{pollutant.upper()} Pred (Plot Error)", fontsize=11)
                ax_pred.axis('off')
            except ValueError as e:
                print(f"Error plotting prediction for {name}, '{pollutant}': {e}")
                ax_pred.set_title(f"{name}\n{pollutant.upper()} Pred (Value Error)", fontsize=11)
                ax_pred.axis('off')

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))

    cols_err_plot = num_models
    fig_err = None
    axs_err = None
    if cols_err_plot > 0:
        fig_err, axs_err = plt.subplots(
            nrows=num_pollutants,
            ncols=cols_err_plot,
            figsize=(4 * cols_err_plot, 3.5 * num_pollutants),
            squeeze=False
        )
        fig_err.suptitle("Model Prediction Errors", fontsize=16, y=1.0)

        for model_idx, (name, res) in enumerate(zip(experiment_names, experiment_results)):
            col_err = model_idx
            errors_data = res['local_errors'][0]

            for row_idx, pollutant in enumerate(pollutants):
                idx = pollutant_index.get(pollutant)
                vmin_err, vmax_err = error_plot_ranges.get(pollutant, (None, None))
                ax_err = axs_err[row_idx, col_err]

                if idx is None:
                    if ax_err is not None:
                        ax_err.set_title(f"{name}\nUnknown Pollutant Error", fontsize=11)
                        ax_err.axis('off')
                    continue

                if idx >= errors_data.shape[0]:
                    ax_err.set_title(f"{name}\n{pollutant.upper()} Error (Index Error)", fontsize=11)
                    ax_err.axis('off')
                    continue
                
                try:
                    error_data_slice = errors_data[idx]
                    im_err = ax_err.imshow(error_data_slice, cmap='coolwarm', vmin=vmin_err, vmax=vmax_err)
                    ax_err.axis('off')
                    ax_err.set_title(f"{name}\n{pollutant.upper()} Error", fontsize=11)
                    fig_err.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
                except (IndexError, ValueError) as e:
                    print(f"Error: Could not plot error for {name}, '{pollutant}' at index {idx}. Data shape: {errors_data.shape}. Error: {e}")
                    ax_err.set_title(f"{name}\n{pollutant.upper()} Error (Plot Error)", fontsize=11)
                    ax_err.axis('off')

        plt.tight_layout(rect=(0, 0.03, 1, 0.97))

    if save_dir:
        output_folder = os.path.dirname(save_dir)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        base_name, ext = os.path.splitext(save_dir)
        if not ext:
             if os.path.isdir(save_dir) and not base_name.endswith(os.path.sep):
                 base_name = os.path.join(save_dir, "plot_experiment")
             elif not os.path.isdir(save_dir) and output_folder == "":
                 base_name = save_dir
             ext = '.png'


        save_path_pred = f"{base_name}_predictions{ext}"
        save_path_err = f"{base_name}_errors{ext}"

        fig_pred.savefig(save_path_pred, bbox_inches='tight', dpi=150)
        plt.close(fig_pred)
        print(f"Saved predictions plot to {save_path_pred}")

        if fig_err is not None:
            fig_err.savefig(save_path_err, bbox_inches='tight', dpi=150)
            plt.close(fig_err)
            print(f"Saved errors plot to {save_path_err}")
        else:
            if num_models > 0 :
                 print("Skipping error plot saving as it was not generated (or num_models was 0).")
    else:
        plt.show()

    reset_theme()


def plot_error_distributions(
    experiment_names: list[str], 
    experiment_results: list[dict[str, np.ndarray]],
    save_dir: str = None
) -> None:
    sns.set_theme(style="white", context="talk", font_scale=1.2)

    pollutant_titles = ["O3", "PM10", "PM25", "NO2"]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        for exp_idx, (exp_name, exp_res) in enumerate(zip(experiment_names, experiment_results)):
            per_pollutant_errors = (
                np.sqrt(np.sum(exp_res['local_errors']**2, axis=(2, 3)))
                / (np.sqrt(np.sum(exp_res['ground_truth']**2, axis=(2, 3))) + 1e-8)
            )

            for pol_idx in range(4):
                data = per_pollutant_errors[:, pol_idx]

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.hist(data, bins=30, alpha=0.7)

                mu = data.mean()
                ax.axvline(mu, linestyle='--', linewidth=1.5, color='red',
                           label=f"mean = {mu:.3f}")
                ylim = ax.get_ylim()
                ax.text(mu + 0.02, ylim[1]*0.9, f"{mu:.3f}",
                        rotation=0, va='top', ha='left', fontsize=10)

                # ax.set_title(f"{pollutant_titles[pol_idx]} – {exp_name}", fontsize=12)
                ax.legend(fontsize=9, loc='upper right')

                filename = f"{exp_name.replace(' ', '_')}_{pollutant_titles[pol_idx].lower()}_error_hist.png"
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
                plt.close(fig)

    else:
        fig, axs = plt.subplots(
            nrows=4, ncols=len(experiment_names),
            figsize=(10 * len(experiment_names), 20),
            sharex=True
        )

        for exp_idx, (exp_name, exp_res) in enumerate(zip(experiment_names, experiment_results)):
            per_pollutant_errors = (
                np.sqrt(np.sum(exp_res['local_errors']**2, axis=(2, 3)))
                / (np.sqrt(np.sum(exp_res['ground_truth']**2, axis=(2, 3))) + 1e-8)
            )

            for pol_idx in range(4):
                ax = axs[pol_idx, exp_idx]
                data = per_pollutant_errors[:, pol_idx]

                ax.hist(data, bins=30, alpha=0.7)

                mu = data.mean()
                ax.axvline(mu, linestyle='--', linewidth=1.5, color='red',
                           label=f"mean = {mu:.3f}")
                ylim = ax.get_ylim()
                ax.text(mu + 0.02, ylim[1]*0.9, f"{mu:.3f}",
                        rotation=0, va='top', ha='left', fontsize=8)

                ax.set_title(pollutant_titles[pol_idx], fontsize=10)
                ax.legend(fontsize=8, loc='upper right')

            axs[0, exp_idx].annotate(
                f"{exp_name}\nε = {np.mean(exp_res['errors']):.4f}",
                xy=(0.5, 1.0), xycoords='axes fraction',
                xytext=(0.5, 1.15), textcoords='axes fraction',
                fontsize=10, ha='center', va='bottom',
                annotation_clip=False
            )

        for ax in axs.flat:
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

        fig.tight_layout(rect=(0, 0, 1, 0.99))
        plt.show()

    reset_theme()


def plot_one_model_error_distribution(
    experiment_result: dict[str, np.ndarray],
    save_dir: str = None,
    color: str = sns.color_palette("Blues_d", 5)[2]
) -> None:
    pollutant_titles = ["O3", "PM10", "PM25", "NO2"]

    os.makedirs(save_dir, exist_ok=True)

    per_pollutant_errors = (
        np.sqrt(np.sum(experiment_result['local_errors']**2, axis=(2, 3)))
        / (np.sqrt(np.sum(experiment_result['ground_truth']**2, axis=(2, 3))) + 1e-8)
    )

    for pol_idx in range(4):
        data = per_pollutant_errors[:, pol_idx]

        fig, ax = plt.subplots(figsize=(5, 4))
        # ax.hist(data, bins=30, alpha=0.7)
        sns.histplot(data, bins=30, ax=ax, color=color, linewidth=0)

        ax.set_xlabel("L2 RE of Individual Timeframes", fontsize=15, labelpad=10)
        ax.set_ylabel("Frequency", fontsize=15, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14)

        mu = np.mean(data)
        std = np.std(data)

        ax.axvline(mu, linestyle='--', linewidth=1.5, color='red')
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.text(mu + xlim[1] * 0.12, ylim[1]*0.9, f"{mu:.3f}\n±{std:.3f}",
                rotation=0, va='top', ha='center', fontsize=15)

        filename = f"{pollutant_titles[pol_idx].lower()}_error_hist.png"
        fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

    reset_theme()


def plot_timewise_error_progression(
    experiment_names: list[str],
    experiment_results: list[dict[str, np.ndarray]],
    running_mean: int = 2,
    save_dir: str = None,
) -> None:
    num_experiments = len(experiment_names)

    pollutant_titles = ["O3", "PM10", "PM25", "NO2"]
    num_pollutants = len(pollutant_titles)

    fig, axs = plt.subplots(
        nrows=num_pollutants, ncols=num_experiments,
        figsize=(3.5 * num_experiments, 2.5 * num_pollutants),
        sharex=True, sharey='row'
    )

    # Ensure axs is always a 2D array for consistent indexing, even with 1 experiment/pollutant
    if num_pollutants == 1 and num_experiments == 1:
        axs = np.array([[axs]])
    elif num_pollutants == 1:
        axs = axs.reshape(1, num_experiments)
    elif num_experiments == 1:
        axs = axs.reshape(num_pollutants, 1)

    for exp_idx, (exp_name, exp_data) in enumerate(zip(experiment_names, experiment_results)):

        axs[0, exp_idx].annotate(
            f"{exp_name}\nAvg ε = {np.mean(exp_data['errors']):.4f}",
            xy=(0.5, 1.0), xycoords='axes fraction',
            xytext=(0.5, 1.15), textcoords='axes fraction', # Positioned above the top subplot
            fontsize=10, ha='center', va='bottom',
            annotation_clip=False # Allows text to go outside the axes area
        )

        for pol_idx in range(num_pollutants):

            pollutant_errors = np.array(compute_relative_error(
                torch.from_numpy(exp_data['ground_truth'][:, pol_idx]),
                torch.from_numpy(exp_data['predictions'][:, pol_idx])
            ))

            ax = axs[pol_idx, exp_idx]
            time_indices = np.arange(len(pollutant_errors))

            ax.fill_between(time_indices, pollutant_errors, alpha=0.7)
            ax.plot(uniform_filter1d(pollutant_errors, size=running_mean * 24, mode='nearest'), alpha=0.7, label=f'{running_mean} day running mean', color='orange')

            ax.set_title(pollutant_titles[pol_idx], fontsize=10)

            ax.set_xlabel("Day of December", fontsize=9)

            days = np.arange(len(pollutant_errors), step=2 * 24)
            
            ax.set_xticks(days, labels=days // 24)
            ax.tick_params(which='both', labelbottom=True, labelleft=True)

            ax.legend(loc='upper right')
    
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))

    if save_dir:
        try:
            plt.savefig(save_dir, bbox_inches='tight')
            print(f"Plot saved to {save_dir}")
        except Exception as e:
            print(f"Error saving plot to {save_dir}: {e}")
    else:
        plt.show()

    reset_theme()


def plot_one_model_timewise_error(
    experiment_results: dict[str, np.ndarray],
    running_mean: int = 2,
    save_dir: str = None,
    color: str = sns.color_palette("Blues_d", 5)[2]
) -> None:
    pollutant_titles = ["O3", "PM10", "PM25", "NO2"]
    num_pollutants = len(pollutant_titles)

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    for pol_idx in range(num_pollutants):
        pollutant_errors = np.array(compute_relative_error(
            torch.from_numpy(experiment_results['ground_truth'][:, pol_idx]),
            torch.from_numpy(experiment_results['predictions'][:, pol_idx])
        ))

        time_indices = np.arange(len(pollutant_errors))
        smoothed = uniform_filter1d(pollutant_errors, size=running_mean * 24, mode='nearest')

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.fill_between(time_indices, pollutant_errors, alpha=0.7, color=color)
        ax.plot(smoothed, alpha=0.7, label=f'{running_mean}-day running mean', color='orange')

        ax.set_xlabel("Days of December", fontsize=13)
        ax.set_ylabel("L2 RE of Timesteps", fontsize=13)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.legend(loc='upper right', fontsize=13)

        days = np.arange(len(pollutant_errors), step=2 * 24)
        ax.set_xticks(days, labels=days // 24)
        # ax.tick_params(labelbottom=True, labelleft=True)

        filename = f"{pollutant_titles[pol_idx].lower()}_temporal_error.png"
        fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

    reset_theme()

    
def plot_performance_varying_value(
    experiment_names: list[list[str]],
    experiment_results: list[list[dict[str, np.ndarray]]],
    varying_values: list[int],
    x_label: str = "Number of Sensors",
    save_dir: str = None,
) -> None:
    plotted_values = {}

    for experiment_batch_names, experiment_batch_results, value in zip(
        experiment_names, experiment_results, varying_values
    ):
        for experiment_name, experiment_result in zip(experiment_batch_names, experiment_batch_results):
            avg_error = np.mean(experiment_result["errors"])
            current_values = plotted_values.get(experiment_name, [])
            current_values.append((value, avg_error))
            plotted_values[experiment_name] = current_values

    # Plotting
    plt.figure(figsize=(8, 6))
    
    for model_name, values in plotted_values.items():
        # Sort by sensor number just in case
        values.sort(key=lambda x: x[0])
        sensor_nums, avg_errors = zip(*values)
        plt.plot(sensor_nums, avg_errors, marker='o', label=model_name)

    plt.xlabel(x_label)
    plt.ylabel("Average Relative Error")
    plt.legend(title="Model")
    plt.grid(True)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(f"{save_dir}/performance_vs_sensor_number.png")
        print(f"Saved plot to {save_dir}/performance_vs_sensor_number.png")
    else:
        plt.show()

    reset_theme()

def plot_one_model_predictions(
    results: dict[str, np.ndarray],
    save_dir: str | None = None,
    n_examples: int = 5, 
    pollutant: str = "o3",
) -> None:
    sns.set_theme(style="white", context="talk", font_scale=1.4)

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    pollutants_idx = {
        "o3": 0,
        "pm10": 1,
        "pm25": 2,
        "no2": 3
    }

    pollutant_idx = pollutants_idx[pollutant]
    n_predictions = results['predictions'].shape[0]
    indices = np.arange(start=0, stop=n_predictions, step=max(1, n_predictions // n_examples))[:n_examples]

    v_max = np.max(results['ground_truth'][indices, pollutant_idx])
    v_max_errors = np.max(np.abs(results['local_errors'][indices, pollutant_idx]))

    for i, idx in enumerate(indices):
        gt_img = results['ground_truth'][idx, pollutant_idx]
        pred_img = results['predictions'][idx, pollutant_idx]
        err_img = np.abs(results['local_errors'][idx, pollutant_idx])

        if save_dir is not None:
            # Save each subfigure individually
            for name, img, cmap, vmax in [
                ("ground_truth", gt_img, "viridis", v_max),
                ("prediction", pred_img, "viridis", v_max),
                ("error", err_img, "jet", v_max_errors)
            ]:
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(img, vmin=0, vmax=vmax, cmap=cmap)
                ax.axis('off')
                _add_aligned_colorbar(im, ax, pad=0.05)

                filename = f"{pollutant}_{name}_example_{i + 1}.png"
                fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
                plt.close(fig)
        else:
            # Plot in a shared figure
            if i == 0:
                fig, ax = plt.subplots(3, n_examples, figsize=(n_examples * 5, 10))

            im0 = ax[0][i].imshow(gt_img, vmin=0, vmax=v_max)
            ax[0][i].set_title("Ground Truth", fontsize=16)
            ax[0][i].axis('off')
            _add_aligned_colorbar(im0, ax[0][i], pad=0.05)

            im1 = ax[1][i].imshow(pred_img, vmin=0, vmax=v_max)
            ax[1][i].set_title("Prediction", fontsize=16)
            ax[1][i].axis('off')
            _add_aligned_colorbar(im1, ax[1][i], pad=0.05)

            im2 = ax[2][i].imshow(err_img, vmin=0, vmax=v_max_errors, cmap='jet')
            ax[2][i].set_title("Absolute Errors", fontsize=16)
            ax[2][i].axis('off')
            _add_aligned_colorbar(im2, ax[2][i], pad=0.05)

    if save_dir is None:
        fig.suptitle(pollutant.upper(), fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    reset_theme()

def plot_model_prediction_comparison(
    all_results: dict[str, dict[str, np.ndarray]],
    save_dir: str | None = None,
    pollutant: str = "o3",
    show_errors: bool = True,
    sample_idx: int = -100,
    individual_model = False,
    v_max = None,
    v_max_errors = None
) -> None:
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    pollutants_idx = {
        "o3": 0,
        "pm10": 1,
        "pm25": 2,
        "no2": 3
    }
    if individual_model:
        pollutant_idx = 0
    else:
        pollutant_idx = pollutants_idx[pollutant]
    n_models = len(all_results)
    if v_max is None:
        v_max = np.max([max(np.max(res['simulated_ground_truths'][sample_idx, pollutant_idx]), np.max(res['simulated_predictions'][sample_idx, pollutant_idx])) for res in all_results.values()])

    if (show_errors) and (v_max_errors is None):
        try:
            v_max_errors = max([np.max(np.abs(res['real_local_errors'][sample_idx, pollutant_idx])) for res in all_results.values()])
        except:
            v_max_errors = max([
                np.max(np.abs(compute_local_error(torch.from_numpy(res['simulated_ground_truths'][sample_idx, pollutant_idx]), torch.from_numpy(res['simulated_predictions'][sample_idx, pollutant_idx])).numpy())) 
                for res in all_results.values()
                ])

    for i, (model_name, results) in enumerate(all_results.items()):
        gt_img = results['simulated_ground_truths'][sample_idx, pollutant_idx]
        pred_img = results['simulated_predictions'][sample_idx, pollutant_idx]

        if show_errors:
            try:
                err_img = np.abs(results['real_local_errors'][sample_idx, pollutant_idx])
            except:
                err_img = np.abs(
                    compute_local_error(
                        torch.from_numpy(gt_img),
                        torch.from_numpy(pred_img)
                    ).numpy()
                )

        if save_dir is not None:
            # Save each subfigure individually
            plot_info = [
                ("ground_truth", gt_img, "viridis", 0, v_max),
                ("prediction", pred_img, "viridis", 0, v_max),
            ]
            if show_errors:
                plot_info.append(("error", err_img, "jet", 0, v_max_errors))
            for name, img, cmap, vmin, vmax in plot_info:
                fig, ax = plt.subplots(figsize=(5, 5))
                # im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
                im = print_with_borders(img, ax, vmax, cmap)
                ax.axis('off')
                _add_aligned_colorbar(im, ax, pad=0.05)
                filename = f"{pollutant}_{model_name}_{name}.png"
                fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
                plt.close(fig)
        else:
            # Plot in a shared figure
            if i == 0:
                n_rows = 3 if show_errors else 2
                fig, ax = plt.subplots(n_rows, n_models, figsize=(n_models * 5, 10))
            im0 = ax[0][i].imshow(gt_img, vmin=0, vmax=v_max)
            ax[0][i].set_title("Ground Truth", fontsize=16)
            ax[0][i].axis('off')
            _add_aligned_colorbar(im0, ax[0][i], pad=0.05)
            im1 = ax[1][i].imshow(pred_img, vmin=0, vmax=v_max)
            ax[1][i].set_title(f"Prediction ({model_name})", fontsize=16)
            ax[1][i].axis('off')
            _add_aligned_colorbar(im1, ax[1][i], pad=0.05)
            if show_errors:
                im2 = ax[2][i].imshow(err_img, vmin=0, vmax=v_max_errors, cmap='jet')
                ax[2][i].set_title("Absolute Errors", fontsize=16)
                ax[2][i].axis('off')
                _add_aligned_colorbar(im2, ax[2][i], pad=0.05)

    if save_dir is None:
        fig.suptitle(pollutant.upper(), fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    reset_theme()


def plot_distribution_comparison(save_dir: str | None = None) -> None:
    reset_theme()

    real_data, _ = load_real_obs_data('vitae', sensor_type='real-random', scale=False, pollutant_type=pollutant_type)
    real_obs = torch.stack([obs for obs, _, _ in real_data], dim=0)

    training_dataset, *_ = load_vitae_data(scaling_type='none', sensor_type='real-random')
    train_obs = torch.stack([obs for obs, _, _ in training_dataset], dim=0)

    real_obs_flat = torch.clamp(real_obs.permute(1, 0, 2, 3).reshape(4, -1), 0).numpy()
    real_obs_flat_mask = real_obs_flat != 0
    train_obs_flat = torch.clamp(train_obs.permute(1, 0, 2, 3).reshape(4, -1), 0).numpy()
    train_obs_flat_mask = train_obs_flat != 0

    for i, pollutant in enumerate(["O3", "PM10", "PM25", "NO2"]):
        real_obs_values = real_obs_flat[i][real_obs_flat_mask[i]]
        train_obs_values = train_obs_flat[i][train_obs_flat_mask[i]]

        bins = np.histogram_bin_edges(
            np.concatenate([real_obs_values, train_obs_values]),
            bins=50
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.hist(real_obs_values, bins=bins, alpha=0.5, label="Real-World", density=True)
        ax.hist(train_obs_values, bins=bins, alpha=0.5, label="Synthethic", density=True)

        # ax.set_xlabel(f"{pollutant} Concentration Levels", fontsize=16, labelpad=10)
        # ax.set_ylabel("Frequency", fontsize=16, labelpad=10)

        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.legend(fontsize=24)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{pollutant.lower()}_distribution_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    reset_theme()


def plot_noise_effects(
    noise_type: str, 
    noise_mean: torch.Tensor, 
    noise_std: torch.Tensor,
    save: bool = False, 
    save_dir: str | None = None,
    **kwargs: dict,
) -> torch.Tensor:
    sns.reset_defaults()
    
    evaluation_dataset, _ = load_real_obs_data('vitae', sensor_type='real-random', scale=False, pollutant_type=pollutant_type)
    evaluation_obs = torch.stack([obs for obs, _, _ in evaluation_dataset], dim=0)
    
    training_dataset, _, _, stats = load_vitae_data(scaling_type='none', sensor_type='real-random')

    obs_mask = torch.stack([obs for obs, _, _ in training_dataset], dim=0) != 0
    clean_gts = torch.stack([gt for _, gt, _ in training_dataset], dim=0)

    noise = get_custom_noise(
        target_shape= obs_mask.shape,
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_type=noise_type,
        **kwargs
    )

    noised_gts = clean_gts + noise
    noised_obs = noised_gts * obs_mask

    noised_obs_flat = torch.clamp(noised_obs.permute(1, 0, 2, 3).reshape(4, -1), 0).numpy()
    noise_obs_mask = noised_obs_flat != 0
    evaluation_obs_flat = torch.clamp(evaluation_obs.permute(1, 0, 2, 3).reshape(4, -1), 0).numpy()
    evaluation_obs_mask = evaluation_obs_flat != 0

    if save_dir is None:
        _, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i, pollutant in enumerate(["O3", "PM10", "PM25", "NO2"]):
        if save_dir is not None:
            _, ax = plt.subplots(figsize=(6, 5))
        else:
            ax = axs[i]

        noised_values = noised_obs_flat[i][noise_obs_mask[i]]
        clean_values = evaluation_obs_flat[i][evaluation_obs_mask[i]]

        bins = np.histogram_bin_edges(
            np.concatenate([noised_values, clean_values]),
            bins=50
        )

        ax.hist(clean_values, bins=bins, alpha=0.5, label="Real-World", density=True)
        ax.hist(noised_values, bins=bins, alpha=0.5, label="Synthethic", density=True)

        if save_dir is None: ax.set_title(f"{pollutant} Distribution")
        # ax.set_xlabel(f"{pollutant} Concentration Levels", fontsize=16, labelpad=10)
        # ax.set_ylabel("Frequency", fontsize=16, labelpad=10)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=24)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{pollutant.lower()}_distribution_{noise_type}_noise.png", dpi=300, bbox_inches='tight')
            plt.close(ax.figure)

    if save_dir is None:
        plt.tight_layout()
        plt.suptitle(f"Effect of {noise_type} Noise (mean={noise_mean}, std={noise_std})", fontsize=16, y=1.05)
        plt.show()

    if save:
        data_min = torch.from_numpy(stats["data_min"].reshape(-1))
        data_max = torch.from_numpy(stats["data_max"].reshape(-1))

        torch.save({
            "noise_mean": (noise_mean - data_min) / (data_max - data_min),
            "noise_std": noise_std / (data_max - data_min),
        }, f'results/misc/noise_params_{noise_type}.pth'
        )

    reset_theme()

    return noised_gts, clean_gts


def read_noise_results_df() -> pd.DataFrame:

    # CLSTM

    clstm_results = np.load('results/predictions/clstm/sparse_real_random_predictions.npz')
    clstm_results_0 = np.load('results/predictions/clstm/sparse_real_random_0_predictions.npz')
    clstm_results_100 = np.load('results/predictions/clstm/sparse_real_random_100_predictions.npz')

    clstm_results_gaussian = np.load('results/predictions/clstm/sparse_real_random_gaussian_full_predictions.npz')
    clstm_results_gaussian_0 = np.load('results/predictions/clstm/sparse_real_random_gaussian_full_0_predictions.npz')
    clstm_results_gaussian_100 = np.load('results/predictions/clstm/sparse_real_random_gaussian_full_100_predictions.npz')

    clstm_results_time_gaussian = np.load('results/predictions/clstm/sparse_real_random_time_gaussian_full_predictions.npz')
    clstm_results_time_gaussian_0 = np.load('results/predictions/clstm/sparse_real_random_time_gaussian_full_0_predictions.npz')
    clstm_results_time_gaussian_100 = np.load('results/predictions/clstm/sparse_real_random_time_gaussian_full_100_predictions.npz')

    clstm_results_perlin = np.load('results/predictions/clstm/sparse_real_random_perlin_full_predictions.npz')
    clstm_results_perlin_0 = np.load('results/predictions/clstm/sparse_real_random_perlin_full_0_predictions.npz')
    clstm_results_perlin_100 = np.load('results/predictions/clstm/sparse_real_random_perlin_full_100_predictions.npz')

    clstm_results_correlated = np.load('results/predictions/clstm/sparse_real_random_correlated_full_predictions.npz')
    clstm_results_correlated_0 = np.load('results/predictions/clstm/sparse_real_random_correlated_full_0_predictions.npz')
    clstm_results_correlated_100 = np.load('results/predictions/clstm/sparse_real_random_correlated_full_100_predictions.npz')

    # ViTAE

    vitae_results = np.load("results/predictions/vitae/sparse_real_random_predictions.npz")
    vitae_results_0 = np.load("results/predictions/vitae/sparse_real_random_0_predictions.npz")
    vitae_results_100 = np.load("results/predictions/vitae/sparse_real_random_100_predictions.npz")

    vitae_results_gaussian = np.load('results/predictions/vitae/sparse_real_random_gaussian_full_predictions.npz')
    vitae_results_gaussian_0 = np.load('results/predictions/vitae/sparse_real_random_gaussian_full_0_predictions.npz')
    vitae_results_gaussian_100 = np.load('results/predictions/vitae/sparse_real_random_gaussian_full_100_predictions.npz')

    vitae_results_time_gaussian = np.load('results/predictions/vitae/sparse_real_random_time_gaussian_full_predictions.npz')
    vitae_results_time_gaussian_0 = np.load('results/predictions/vitae/sparse_real_random_time_gaussian_full_0_predictions.npz')
    vitae_results_time_gaussian_100 = np.load('results/predictions/vitae/sparse_real_random_time_gaussian_full_100_predictions.npz')

    vitae_results_perlin = np.load('results/predictions/vitae/sparse_real_random_perlin_full_predictions.npz')
    vitae_results_perlin_0 = np.load('results/predictions/vitae/sparse_real_random_perlin_full_0_predictions.npz')
    vitae_results_perlin_100 = np.load('results/predictions/vitae/sparse_real_random_perlin_full_100_predictions.npz')

    vitae_results_correlated = np.load('results/predictions/vitae/sparse_real_random_correlated_full_predictions.npz')
    vitae_results_correlated_0 = np.load('results/predictions/vitae/sparse_real_random_correlated_full_0_predictions.npz')
    vitae_results_correlated_100 = np.load('results/predictions/vitae/sparse_real_random_correlated_full_100_predictions.npz')

    # VUNet

    vunet_results = np.load('results/predictions/vunet/sparse_real_random_predictions.npz')
    vunet_results_0 = np.load('results/predictions/vunet/sparse_real_random_0_predictions.npz')
    vunet_results_100 = np.load('results/predictions/vunet/sparse_real_random_100_predictions.npz')


    vunet_results_gaussian = np.load('results/predictions/vunet/sparse_real_random_gaussian_full_predictions.npz')
    vunet_results_gaussian_0 = np.load('results/predictions/vunet/sparse_real_random_gaussian_full_0_predictions.npz')
    vunet_results_gaussian_100 = np.load('results/predictions/vunet/sparse_real_random_gaussian_full_100_predictions.npz')

    vunet_results_time_gaussian = np.load('results/predictions/vunet/sparse_real_random_time_gaussian_full_predictions.npz')
    vunet_results_time_gaussian_0 = np.load('results/predictions/vunet/sparse_real_random_time_gaussian_full_0_predictions.npz')
    vunet_results_time_gaussian_100 = np.load('results/predictions/vunet/sparse_real_random_time_gaussian_full_100_predictions.npz')

    vunet_results_perlin = np.load('results/predictions/vunet/sparse_real_random_perlin_full_predictions.npz')
    vunet_results_perlin_0 = np.load('results/predictions/vunet/sparse_real_random_perlin_full_0_predictions.npz')
    vunet_results_perlin_100 = np.load('results/predictions/vunet/sparse_real_random_perlin_full_100_predictions.npz')

    vunet_results_correlated = np.load('results/predictions/vunet/sparse_real_random_correlated_full_predictions.npz')
    vunet_results_correlated_0 = np.load('results/predictions/vunet/sparse_real_random_correlated_full_0_predictions.npz')
    vunet_results_correlated_100 = np.load('results/predictions/vunet/sparse_real_random_correlated_full_100_predictions.npz')

    # Kriging

    kriging_results_real = np.load('results/predictions/kriging/real_dataset_predictions.npz')

    models_data = {
        "Experiment Name": [
            "CLSTM",
            "CLSTM",
            "CLSTM",
            "CLSTM - Gaussian",
            "CLSTM - Gaussian",
            "CLSTM - Gaussian",
            "CLSTM - Time-aware Gaussian",
            "CLSTM - Time-aware Gaussian",
            "CLSTM - Time-aware Gaussian",
            "CLSTM - Perlin",
            "CLSTM - Perlin",
            "CLSTM - Perlin",
            "CLSTM - Correlated",
            "CLSTM - Correlated",
            "CLSTM - Correlated",

            "ViTAE",
            "ViTAE",
            "ViTAE",
            "ViTAE - Gaussian",
            "ViTAE - Gaussian",
            "ViTAE - Gaussian",
            "ViTAE - Time-aware Gaussian",
            "ViTAE - Time-aware Gaussian",
            "ViTAE - Time-aware Gaussian",
            "ViTAE - Perlin",
            "ViTAE - Perlin",
            "ViTAE - Perlin",
            "ViTAE - Correlated",
            "ViTAE - Correlated",
            "ViTAE - Correlated",

            "VUNet",
            "VUNet",
            "VUNet",
            "VUNet - Gaussian",
            "VUNet - Gaussian",
            "VUNet - Gaussian",
            "VUNet - Time-aware Gaussian",
            "VUNet - Time-aware Gaussian",
            "VUNet - Time-aware Gaussian",
            "VUNet - Perlin",
            "VUNet - Perlin",
            "VUNet - Perlin",
            "VUNet - Correlated",
            "VUNet - Correlated",
            "VUNet - Correlated",
            # "VUNet - Fine-tuned",

            "Kriging"
        ],
        "Error": [
            np.mean(clstm_results['errors']),
            np.mean(clstm_results_0['errors']),
            np.mean(clstm_results_100['errors']),
            np.mean(clstm_results_gaussian['errors']),
            np.mean(clstm_results_gaussian_0['errors']),
            np.mean(clstm_results_gaussian_100['errors']),
            np.mean(clstm_results_time_gaussian['errors']),
            np.mean(clstm_results_time_gaussian_0['errors']),
            np.mean(clstm_results_time_gaussian_100['errors']),
            np.mean(clstm_results_perlin['errors']),
            np.mean(clstm_results_perlin_0['errors']),
            np.mean(clstm_results_perlin_100['errors']),
            np.mean(clstm_results_correlated['errors']),
            np.mean(clstm_results_correlated_0['errors']),
            np.mean(clstm_results_correlated_100['errors']),

            np.mean(vitae_results['errors']),
            np.mean(vitae_results_0['errors']),
            np.mean(vitae_results_100['errors']),
            np.mean(vitae_results_gaussian['errors']),
            np.mean(vitae_results_gaussian_0['errors']),
            np.mean(vitae_results_gaussian_100['errors']),
            np.mean(vitae_results_time_gaussian['errors']),
            np.mean(vitae_results_time_gaussian_0['errors']),
            np.mean(vitae_results_time_gaussian_100['errors']),
            np.mean(vitae_results_perlin['errors']),
            np.mean(vitae_results_perlin_0['errors']),
            np.mean(vitae_results_perlin_100['errors']),
            np.mean(vitae_results_correlated['errors']),
            np.mean(vitae_results_correlated_0['errors']),
            np.mean(vitae_results_correlated_100['errors']),

            np.mean(vunet_results['errors']),
            np.mean(vunet_results_0['errors']),
            np.mean(vunet_results_100['errors']),
            # np.mean(vunet_results_1['errors']),
            np.mean(vunet_results_gaussian['errors']),
            np.mean(vunet_results_gaussian_0['errors']),
            np.mean(vunet_results_gaussian_100['errors']),
            np.mean(vunet_results_time_gaussian['errors']),
            np.mean(vunet_results_time_gaussian_0['errors']),
            np.mean(vunet_results_time_gaussian_100['errors']),
            np.mean(vunet_results_perlin['errors']),
            np.mean(vunet_results_perlin_0['errors']),
            np.mean(vunet_results_perlin_100['errors']),
            np.mean(vunet_results_correlated['errors']),
            np.mean(vunet_results_correlated_0['errors']),
            np.mean(vunet_results_correlated_100['errors']),
            # np.mean(vunet_results_fine_tuned['errors']),

            np.mean(kriging_results_real['errors'])
        ],
        "Seed": [
            42, 0, 100,
            42, 0, 100,
            42, 0, 100,
            42, 0, 100,
            42, 0, 100,

            42, 0, 100,
            42, 0, 100,
            42, 0, 100,
            42, 0, 100,
            42, 0, 100,

            42, 0, 100,
            42, 0, 100,
            42, 0, 100,
            42, 0, 100,
            42, 0, 100,

            42
        ]
    }
    df = pd.DataFrame(models_data)

    def extract_category(model_name):
        if "CLSTM" in model_name:
            return "CLSTM"
        elif "VUNet" in model_name:
            return "VUNet"
        elif "ViTAE" in model_name:
            return "ViTAE"
        elif "Kriging" in model_name:
            return "Kriging"
        else:
            return "Other"
        
    df['Model'] = df['Experiment Name'].apply(extract_category)

    return df


def plot_real_data_results(save: bool = True):
    df = read_noise_results_df()

    # Define color palettes
    palette_names = {
        "CLSTM": "Greens",
        "VUNet": "Blues",
        "ViTAE": "Oranges",
        "Kriging": "Greys",
        "VCNN": "Purples",
    }

    def generate_color_dict(df_subset):
        color_dict = {}
        for category in df_subset['Category'].unique():
            cat_df = df_subset[df_subset['Category'] == category]
            n = len(cat_df)
            palette = sns.color_palette(palette_names.get(category, "viridis"), n)
            sorted_indices = cat_df.sort_values("MeanRelativeError").index
            for idx, color in zip(sorted_indices, palette):
                color_dict[df_subset.loc[idx, "Model"]] = color
        return color_dict

    def plot_df(df_subset, color_dict, save_path):
        stats = df_subset.groupby("Model")["MeanRelativeError"].agg(["mean", "std"]).reset_index()
        ordering = stats.sort_values("mean")["Model"].tolist()

        plt.figure(figsize=(12, 6))
        bars = sns.barplot(
            data=df_subset,
            x="Model",
            y="MeanRelativeError",
            hue="Model",
            palette=color_dict,
            legend=False,
            errorbar='sd',
            order=ordering
        )

        # Annotate bars
        for i, p in enumerate(bars.patches):
            model = ordering[i]  # model name
            mean_val = stats.loc[stats["Model"] == model, "mean"].values[0]
            std_val = stats.loc[stats["Model"] == model, "std"].values[0]

            bars.annotate(f'{mean_val:.3f}\n±{std_val:.3f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12, color='black',
                        xytext=(0, 5), textcoords='offset points')

        # Axis formatting
        bars.set_xlabel("Noising Method", fontsize=15, labelpad=10)
        bars.set_ylabel("Mean Relative L2 Error", fontsize=15, labelpad=10)
        bars.tick_params(axis='x', labelsize=13)
        bars.tick_params(axis='y', labelsize=13)
        plt.xticks(rotation=30, ha='right')
        bars.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        bars.grid(axis='y', linestyle='--', alpha=0.6)
        bars.set_ylim(0, 1.0)

        # Legend formatting
        legend_patches = [
            mpatches.Patch(color=sns.color_palette("Greens", 5)[2], label='CLSTM'),
            mpatches.Patch(color=sns.color_palette("Blues", 5)[2], label='VUNet'),
            mpatches.Patch(color=sns.color_palette("Oranges", 5)[2], label='ViTAE'),
            mpatches.Patch(color=sns.color_palette("Greys", 5)[2], label='Kriging'),
        ]
        
        plt.legend(
            handles=legend_patches,
            title="Model Category",
            title_fontsize=13,
            fontsize=12,
            loc="upper right",
            frameon=False
        )

        plt.tight_layout(pad=1.5)
        if save: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    color_dict_all = generate_color_dict(df)
    plot_df(df, color_dict_all, "report_images/experiments/real/errors_bar_all.png")

    df_clstm = df[df['Category'] == 'CLSTM']
    color_dict_clstm = generate_color_dict(df_clstm)
    plot_df(df_clstm, color_dict_clstm, "report_images/experiments/real/errors_bar_clstm.png")

    df_vunet = df[df['Category'] == 'VUNet']
    color_dict_vunet = generate_color_dict(df_vunet)
    plot_df(df_vunet, color_dict_vunet, "report_images/experiments/real/errors_bar_vunet.png")

    df_vitae = df[df['Category'] == 'ViTAE']
    color_dict_vitae = generate_color_dict(df_vitae)
    plot_df(df_vitae, color_dict_vitae, "report_images/experiments/real/errors_bar_vitae.png")

    reset_theme()


def overlay_images(
        img_paths: list[tuple[str, tuple[int, int]]], 
        save_dir: str,
        output_filename: str
        ) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Load images and track maximum extents
    images = []
    max_width = 0
    max_height = 0

    for path, (x_offset, y_offset) in img_paths:
        img = Image.open(path).convert("RGBA")
        images.append((img, (x_offset, y_offset)))
        w, h = img.size
        max_width = max(max_width, x_offset + w)
        max_height = max(max_height, y_offset + h)

    # Create a transparent background large enough to hold all images
    composite = Image.new("RGBA", (max_width, max_height), (0, 0, 0, 0))

    # Paste each image
    for img, offset in images:
        composite.paste(img, offset, img)

    save_path = os.path.join(save_dir, output_filename)

    # Save the final overlapped image
    composite.save(save_path)

    reset_theme()


def split_and_save_chunks(
    input_array: np.ndarray,
    output_dir: str = "chunks"
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    C, H, W = input_array.shape
    h_splits = np.array_split(np.arange(H), 3)
    w_splits = np.array_split(np.arange(W), 3)

    pollutants = ["O3", "PM10", "PM25", "NO2"]

    for idx, pollutant in enumerate(pollutants):
        # Save full image with grid overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(input_array[idx], cmap='viridis')
        
        # Draw horizontal lines
        for h_idx in h_splits[:-1]:
            y = h_idx[-1] + 0.5
            plt.axhline(y=y, color='red', linewidth=2)

        # Draw vertical lines
        for w_idx in w_splits[:-1]:
            x = w_idx[-1] + 0.5
            plt.axvline(x=x, color='red', linewidth=2)

        plt.axis('off')
        full_img_filename = os.path.join(output_dir, f"{pollutant}_full_with_grid.png")
        plt.savefig(full_img_filename, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.imshow(input_array[idx], cmap='viridis')
        plt.axis('off')
        full_img_filename = os.path.join(output_dir, f"{pollutant}_full.png")
        plt.savefig(full_img_filename, bbox_inches='tight')
        plt.close()

        # Save individual chunks
        for i, h_idx in enumerate(h_splits):
            for j, w_idx in enumerate(w_splits):
                chunk = input_array[idx, h_idx[0]:h_idx[-1]+1, w_idx[0]:w_idx[-1]+1]
                chunk_filename = os.path.join(output_dir, f"{pollutant}_{i}_{j}.png")

                plt.figure(figsize=(10, 10))
                plt.imshow(chunk, cmap='viridis')
                plt.axis('off')
                plt.savefig(chunk_filename, bbox_inches='tight')
                plt.close()

    for i, h_idx in enumerate(h_splits):
        for j, w_idx in enumerate(w_splits):
            images = []

            for idx, pollutant in enumerate(pollutants):
                chunk_filename = os.path.join(output_dir, f"{pollutant}_{i}_{j}.png")
                images.append((chunk_filename, ((len(pollutants) - idx) * 50, (len(pollutants) - idx) * 50)))

                # os.remove(chunk_filename)  # Remove individual chunk files after overlaying

            overlay_images(reversed(images), output_dir, f"chunk_{i}_{j}_overlay.png")

            for img, _ in images:
                os.remove(img)

    images = []
    clean_images = []
    for idx, pollutant in enumerate(pollutants):
        full_img_filename = os.path.join(output_dir, f"{pollutant}_full_with_grid.png")
        clean_img_filename = os.path.join(output_dir, f"{pollutant}_full.png")
        images.append((full_img_filename, ((len(pollutants) - idx) * 50, (len(pollutants) - idx) * 50)))
        clean_images.append((clean_img_filename, ((len(pollutants) - idx) * 50, (len(pollutants) - idx) * 50)))

        # os.remove(full_img_filename)  # Remove full image files after overlaying

    overlay_images(reversed(images), output_dir, "full_overlay.png")
    overlay_images(reversed(clean_images), output_dir, "full_clean_overlay.png")

    for img, _ in images:
        os.remove(img)

    for img, _ in clean_images:
        os.remove(img)

    reset_theme()


def plot_real_sensor_partitioning() -> None:
    real_data = read_real_observation_files()
    all_positions = np.logical_or.reduce(real_data, axis=0)

    center_i_min, center_i_max = 30, 45
    center_j_min, center_j_max = 45, 65

    C, _, _ = all_positions.shape

    inner_mask = np.zeros_like(all_positions)
    outer_mask = np.zeros_like(all_positions)

    for c in range(C):
        # Get all valid positions
        valid_positions = np.nonzero(all_positions[c])

        # Split into center and outer positions
        i_vals, j_vals = valid_positions[0], valid_positions[1]
        is_center_i = (i_vals >= center_i_min) & (i_vals < center_i_max)
        is_center_j = (j_vals >= center_j_min) & (j_vals < center_j_max)

        inner_mask[c, i_vals[is_center_i & is_center_j], j_vals[is_center_i & is_center_j]] = 1
        outer_mask[c, i_vals[~(is_center_i & is_center_j)], j_vals[~(is_center_i & is_center_j)]] = 1

    # Transparent for class 0, vivid colors for class 1
    cmap_inner = ListedColormap(["#D6D6D6", "#0072B2"])  # blue
    cmap_outer = ListedColormap([(0, 0, 0, 0), "#B90F0F"])  # orange

    pollutants = ["O3", "PM10", "PM25", "NO2"]

    for idx, pollutant in enumerate(pollutants):
        _, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(inner_mask[idx], cmap=cmap_inner, interpolation='nearest')
        ax.imshow(outer_mask[idx], cmap=cmap_outer, interpolation='nearest', alpha=0.8)
        ax.axis('off')

        # Draw rectangle (note: xy=(x, y) = (col, row))
        rect_width = center_j_max - center_j_min
        rect_height = center_i_max - center_i_min
        rect = Rectangle(
            (center_j_min, center_i_min),  # bottom-left corner
            rect_width,
            rect_height,
            linewidth=1,
            edgecolor="#000000",
            facecolor='none'
        )
        ax.add_patch(rect)

        # Create legend handles
        legend_handles = [
            Patch(facecolor="#0072B2", label="Inner City"),
            Patch(facecolor="#B90F0F", label="Outer City"),
            Patch(edgecolor="#000000", facecolor='none', label="City Boundary")
        ]

        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=14)
        plt.savefig(f"report_images/methodology/evaluation/eval_location_{pollutant}.png", bbox_inches='tight', dpi=300)
        plt.close()

    reset_theme()

def plot_avg_local_relative_error(
    predictions: np.array,
    ground_truth: np.array,
    save_dir: str = None,
    normalize: bool = True,
) -> None:
    pollutants = ["O3", "PM10", "PM25", "NO2"]
    for idx, pollutant in enumerate(pollutants):
        relative_local_errors = compute_local_relative_error(
            torch.from_numpy(ground_truth[:, idx]), 
            torch.from_numpy(predictions[:, idx])
        ).numpy()

        local_std = np.std(ground_truth[:, idx], axis=0)

        relative_local_errors = np.where(relative_local_errors > 5.0, 0, relative_local_errors)

        avg_local_errors = np.mean(relative_local_errors, axis=0)

        if normalize:
            avg_local_errors = avg_local_errors / local_std
            vmax = 0.3
        else:
            vmax = 0.5

        _, ax = plt.subplots(figsize=(4, 5))
        im = ax.imshow(avg_local_errors, cmap='jet', vmin=0, vmax=vmax)
        ax.axis('off')
        _add_aligned_colorbar(im, ax)

        if save_dir:
            
            if normalize:
                file_name = f'avg_local_error_{pollutant}_normalized.png'
            else:
                file_name = f'avg_local_error_{pollutant}.png'

            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/{file_name}', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    reset_theme()


def read_tuning_results_df() -> pd.DataFrame:

    # CLSTM

    clstm_noise_0 = np.load('results/predictions/clstm/sparse_real_random_0_predictions.npz')
    clstm_noise_42 = np.load('results/predictions/clstm/sparse_real_random_predictions.npz')
    clstm_noise_100 = np.load('results/predictions/clstm/sparse_real_random_100_predictions.npz')

    clstm_noiseless_no_pre_0 = np.load('results/predictions/clstm/sparse_fine_tuned_no_pre_0_predictions.npz')
    clstm_noiseless_no_pre_42 = np.load('results/predictions/clstm/sparse_fine_tuned_no_pre_42_predictions.npz')
    clstm_noiseless_no_pre_100 = np.load('results/predictions/clstm/sparse_fine_tuned_no_pre_100_predictions.npz')

    # ViTAE

    vitae_noise_0 = np.load("results/predictions/vitae/sparse_real_random_0_predictions.npz")
    vitae_noise_42 = np.load("results/predictions/vitae/sparse_real_random_predictions.npz")
    vitae_noise_100 = np.load("results/predictions/vitae/sparse_real_random_100_predictions.npz")

    vitae_noiseless_tuned_0 = np.load('results/predictions/vitae/sparse_fine_tuned_noiseless_0_predictions.npz')
    vitae_noiseless_tuned_42 = np.load('results/predictions/vitae/sparse_fine_tuned_noiseless_42_predictions.npz')
    vitae_noiseless_tuned_100 = np.load('results/predictions/vitae/sparse_fine_tuned_noiseless_100_predictions.npz')

    vitae_noise_tuned_0 = np.load('results/predictions/vitae/sparse_fine_tuned_long_0_predictions.npz')
    vitae_noise_tuned_42 = np.load('results/predictions/vitae/sparse_fine_tuned_long_42_predictions.npz')
    vitae_noise_tuned_100 = np.load('results/predictions/vitae/sparse_fine_tuned_long_100_predictions.npz')

    vitae_noiseless_no_pre_0 = np.load('results/predictions/vitae/sparse_fine_tuned_no_pre_0_predictions.npz')
    vitae_noiseless_no_pre_42 = np.load('results/predictions/vitae/sparse_fine_tuned_no_pre_42_predictions.npz')
    vitae_noiseless_no_pre_100 = np.load('results/predictions/vitae/sparse_fine_tuned_no_pre_100_predictions.npz')

    # VUNet

    vunet_noise_0 = np.load('results/predictions/vunet/sparse_real_random_0_predictions.npz')
    vunet_noise_42 = np.load('results/predictions/vunet/sparse_real_random_predictions.npz')
    vunet_noise_100 = np.load('results/predictions/vunet/sparse_real_random_100_predictions.npz')

    vunet_noiseless_tuned_0 = np.load('results/predictions/vunet/sparse_fine_tuned_noiseless_0_predictions.npz')
    vunet_noiseless_tuned_42 = np.load('results/predictions/vunet/sparse_fine_tuned_noiseless_42_predictions.npz')
    vunet_noiseless_tuned_100 = np.load('results/predictions/vunet/sparse_fine_tuned_noiseless_100_predictions.npz')

    vunet_noise_tuned_0 = np.load('results/predictions/vunet/sparse_fine_tuned_long_0_predictions.npz')
    vunet_noise_tuned_42 = np.load('results/predictions/vunet/sparse_fine_tuned_long_predictions.npz')
    vunet_noise_tuned_100 = np.load('results/predictions/vunet/sparse_fine_tuned_long_100_predictions.npz')

    vunet_noiseless_no_pre_0 = np.load('results/predictions/vunet/sparse_fine_tuned_no_pre_0_predictions.npz')
    vunet_noiseless_no_pre_42 = np.load('results/predictions/vunet/sparse_fine_tuned_no_pre_42_predictions.npz')
    vunet_noiseless_no_pre_100 = np.load('results/predictions/vunet/sparse_fine_tuned_no_pre_100_predictions.npz')

    # Kriging

    kriging_results_real = np.load('results/predictions/kriging/real_dataset_predictions.npz')

    models_data = {
        "Experiment Name": [
            "CLSTM",
            "CLSTM",
            "CLSTM",
            "CLSTM - Joint Training",
            "CLSTM - Joint Training",
            "CLSTM - Joint Training",

            "ViTAE",
            "ViTAE",
            "ViTAE",
            "ViTAE - Fine-tuned",
            "ViTAE - Fine-tuned",
            "ViTAE - Fine-tuned",
            "ViTAE - Joint Training",
            "ViTAE - Joint Training",
            "ViTAE - Joint Training",

            "VUNet",
            "VUNet",
            "VUNet",
            "VUNet - Fine-tuned",
            "VUNet - Fine-tuned",
            "VUNet - Fine-tuned",
            "VUNet - Joint Training",
            "VUNet - Joint Training",
            "VUNet - Joint Training",

            "Kriging"
        ],
        "Error": [
            np.mean(clstm_noise_0['errors']),
            np.mean(clstm_noise_42['errors']),
            np.mean(clstm_noise_100['errors']),
            np.mean(clstm_noiseless_no_pre_0['errors']),
            np.mean(clstm_noiseless_no_pre_42['errors']),
            np.mean(clstm_noiseless_no_pre_100['errors']),

            np.mean(vitae_noise_0['errors']),
            np.mean(vitae_noise_42['errors']),
            np.mean(vitae_noise_100['errors']),
            np.mean(vitae_noiseless_tuned_0['errors']),
            np.mean(vitae_noiseless_tuned_42['errors']),
            np.mean(vitae_noiseless_tuned_100['errors']),
            np.mean(vitae_noiseless_no_pre_0['errors']),
            np.mean(vitae_noiseless_no_pre_42['errors']),
            np.mean(vitae_noiseless_no_pre_100['errors']),

            np.mean(vunet_noise_0['errors']),
            np.mean(vunet_noise_42['errors']),
            np.mean(vunet_noise_100['errors']),
            np.mean(vunet_noiseless_tuned_0['errors']),
            np.mean(vunet_noiseless_tuned_42['errors']),
            np.mean(vunet_noiseless_tuned_100['errors']),
            np.mean(vunet_noiseless_no_pre_0['errors']),
            np.mean(vunet_noiseless_no_pre_42['errors']),
            np.mean(vunet_noiseless_no_pre_100['errors']),

            np.mean(kriging_results_real['errors'])
        ],
        "Seed": [
            0, 42, 100,
            0, 42, 100,

            0, 42, 100,
            0, 42, 100,
            0, 42, 100,

            0, 42, 100,
            0, 42, 100,
            0, 42, 100,

            42
        ]
    }
    df = pd.DataFrame(models_data)

    def extract_category(model_name):
        if "CLSTM" in model_name:
            return "CLSTM"
        elif "VUNet" in model_name:
            return "VUNet"
        elif "ViTAE" in model_name:
            return "ViTAE"
        elif "Kriging" in model_name:
            return "Kriging"
        else:
            return "Other"
        
    def extract_training_method(model_name):
        if "Fine-tuned" in model_name:
            return "Fine-tuned"
        elif "Joint Training" in model_name:
            return "Joint Training"
        else:
            return "Standard"

    df['Architecture'] = df['Experiment Name'].apply(extract_category)
    df['TrainingMethod'] = df['Experiment Name'].apply(extract_training_method)

    return df


def plot_fine_tuned_comparison(save: bool = True):
    results = read_tuning_results_df()

    # Colormaps per architecture (same as before)
    palette_names = {
        "CLSTM": "Greens",
        "VUNet": "Blues",
        "ViTAE": "Oranges",
        "Kriging": "Greys",
        "VCNN": "Purples",
    }

    categories = sorted(results["Architecture"].unique())
    x_centers = np.arange(len(categories))

    plt.figure(figsize=(14, 6))

    legend_handles = []
    legend_labels = []

    for cat_idx, category in enumerate(categories):
        cat_df = results[results["Architecture"] == category]
        cat_methods = sorted(cat_df["TrainingMethod"].unique())
        k = len(cat_methods)
        if k == 0:
            continue

        total_width = 0.8
        bar_width = total_width / k
        offsets = (np.arange(k) - (k - 1) / 2.0) * bar_width

        cmap = plt.cm.get_cmap(palette_names.get(category, "Greys"))
        if k == 1:
            shades = [0.65]
        elif k == 2:
            shades = [0.45, 0.75]
        else:  # k >= 3
            shades = np.linspace(0.35, 0.85, k)

        for i, (method, shade) in enumerate(zip(cat_methods, shades)):
            sub = cat_df[cat_df["TrainingMethod"] == method]["Error"]
            if len(sub) > 0:
                mean = float(np.nanmean(sub))
                std = float(np.nanstd(sub))
            else:
                mean, std = 0.0, 0.0

            x = x_centers[cat_idx] + offsets[i]
            bars = plt.bar(
                x, mean, bar_width,
                color=cmap(shade),
                yerr=(std if (std and not np.isnan(std)) else None),
                capsize=3, alpha=0.8, edgecolor='black', linewidth=0.5
            )

            if mean != 0.0:
                txt = f"{mean:.3f}" + (f"\n±{std:.3f}" if (std and not np.isnan(std)) else "")
                plt.text(
                    bars.patches[0].get_x() + bars.patches[0].get_width() / 2.,
                    bars.patches[0].get_height() + 0.02,
                    txt, ha='center', va='bottom', fontsize=14, color='black'
                )

            legend_handles.append(bars)
            legend_labels.append(f"{category} ({method})")

    plt.xticks(x_centers, categories)
    plt.xlabel("Training Methods grouped by Architecture", fontsize=17, labelpad=10)
    plt.ylabel("Mean L2 MRE (mean ± std over seeds)", fontsize=17, labelpad=10)

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.grid(axis="y", linestyle=":", alpha=0.6)

    plt.ylim(0, 0.5)

    legend_info = list(zip(legend_labels, legend_handles))
    legend_info.sort(key=lambda x: x[0])
    legend_labels = [label for label, _ in legend_info]
    legend_handles = [handle for _, handle in legend_info]

    proxy_handles = []
    for h in legend_handles:
        if hasattr(h, "patches") and len(h.patches) > 0:
            p0 = h.patches[0]
            fc = p0.get_facecolor()
            ec = p0.get_edgecolor()
            proxy_handles.append(Patch(facecolor=fc, edgecolor=ec))
        else:
            proxy_handles.append(h)
    legend_handles = proxy_handles

    # Split into uneven columns: 4 columns with 2, 1, 3, 3 rows respectively
    layout = [2, 1, 3, 3]
    chunks = []
    start = 0
    for n in layout:
        end = min(start + n, len(legend_handles))
        chunks.append((legend_handles[start:end], legend_labels[start:end]))
        start = end

    plt.subplots_adjust(top=0.82)
    fig = plt.gcf()
    ax = plt.gca()

    x_anchors = np.linspace(0.20, 0.85, num=4)
    y_anchor = 0.88

    for (h_chunk, l_chunk), xa in zip(chunks, x_anchors):
        leg = ax.legend(
            h_chunk, l_chunk,
            loc="center",
            bbox_to_anchor=(xa, y_anchor),
            frameon=False,
            fontsize=15,
            handlelength=1.4,
            borderpad=0.2,
            labelspacing=0.3,
            ncol=1,
            borderaxespad=0.0,
            bbox_transform=fig.transFigure
        )
        ax.add_artist(leg)


    plt.tight_layout()
    if save:
        plt.savefig(
            "report_images/experiments/fine_tuned/errors_bar_all.png",
            dpi=300, bbox_inches='tight'
        )
    plt.show()
    plt.close()

    reset_theme()

def plot_noise_experiment_results():
    df = read_noise_results_df()

    save_dir = "report_images/experiments/real"
    os.makedirs(save_dir, exist_ok=True)

    plot_experiment_barcharts(
        df, 
        figsize=(10, 6), capsize=4, annot_size=10,
        file_name=f"{save_dir}/errors_bar_all.png"
        )
    
    plot_experiment_barcharts(
        df[df['Model'].isin(["CLSTM", "Kriging"])], 
        figsize=(8, 6), capsize=4, annot_size=13,
        file_name=f"{save_dir}/errors_bar_clstm.png"
        )

    plot_experiment_barcharts(
        df[df['Model'].isin(["VUNet", "Kriging"])], 
        figsize=(8, 6), capsize=4, annot_size=13,
        file_name=f"{save_dir}/errors_bar_vunet.png"
        )
    
    plot_experiment_barcharts(
        df[df['Model'].isin(["ViTAE", "Kriging"])], 
        figsize=(8, 6), capsize=4, annot_size=13,
        file_name=f"{save_dir}/errors_bar_vitae.png"
        )

from matplotlib.patches import Patch

def plot_experiment_barcharts(
    df: pd.DataFrame, 
    figsize=(10, 6), capsize=4, 
    annot_size: int = 12, ylim: tuple[float, float] = (0, 0.5),
    y_axis_name: str = "Mean L2 MRE (mean ± std over seeds)",
    file_name: str = None
) -> None:

    palette_names = {
        "CLSTM": "#5F8A5F",
        "VUNet": "#6495ED",
        "ViTAE": "#CD853F",
        "Kriging": "#808080",
        "Diffusion": "#7A42EB",
    }

    agg = (
        df.groupby(["Experiment Name", "Model"], as_index=False)
          .agg(
              mean_error=("Error", "mean"),
              std_error=("Error", "std"),
              n=("Seed", "nunique"),
          )
    )

    exp_avg_error = (
        agg.groupby("Experiment Name")["mean_error"]
           .mean()
           .sort_values()
    )
    experiments = exp_avg_error.index.tolist()
    models_in_data = agg["Model"].unique().tolist()

    ordered_models = (
        [m for m in palette_names if m in models_in_data] +
        [m for m in models_in_data if m not in palette_names]
    )

    model_colors = {}
    fallback_colors = ["#BC8F8F", "#20B2AA", "#F0E68C", "#CD5C5C", "#90EE90"]
    fallback_idx = 0

    for m in ordered_models:
        if m in palette_names:
            model_colors[m] = palette_names[m]
        else:
            model_colors[m] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1

    x = np.arange(len(experiments))

    # Identify "No noise" experiments
    no_noise_mask = np.array([
        (" - " not in exp_name) for exp_name in experiments
    ])

    fig, ax = plt.subplots(figsize=figsize)

    for m in ordered_models:
        m_df = (
            agg[agg["Model"] == m]
            .set_index("Experiment Name")
            .reindex(experiments)
        )

        means = m_df["mean_error"].values
        stds = m_df["std_error"].values

        mask = ~np.isnan(means)

        bar_info = list(zip(x[mask], means[mask], stds[mask]))
        bar_info.sort(key=lambda tup: tup[1])

        model_bar_coord = [info[0] for info in bar_info]
        model_bar_means = [info[1] for info in bar_info]
        model_bar_stds = [info[2] for info in bar_info]

        bars = ax.bar(
            model_bar_coord,
            model_bar_means,
            yerr=model_bar_stds,
            width=0.9,
            color=model_colors[m],
            edgecolor="black",
            linewidth=0.4,
            capsize=capsize,
        )

        for bar, mean, std, x_idx in zip(
            bars.patches,
            model_bar_means,
            model_bar_stds,
            model_bar_coord,
        ):
            if no_noise_mask[x_idx]:
                bar.set_hatch("//")
                bar.set_linewidth(0.6)

            if not np.isnan(std):
                text = f"{mean:.2f}\n±{std:.2f}"
            else:
                text = f"{mean:.2f}"

            ax.annotate(
                text,
                (bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 0.01),
                ha="center",
                va="bottom",
                fontsize=annot_size,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("Noising Method", fontsize=14, labelpad=10)
    ax.set_ylabel(y_axis_name, fontsize=14, labelpad=10)

    noising_methods = [
        exp_name.split(" - ")[-1] if " - " in exp_name else "No noise"
        for exp_name in experiments
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(noising_methods, rotation=30, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.set_ylim(ylim)
    ax.margins(x=0.02)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # --- CLEAN LEGEND (no hatch, always solid) ---
    legend_handles = [
        Patch(
            facecolor=model_colors[m],
            edgecolor="black",
            linewidth=0.6,
            label=m,
        )
        for m in ordered_models
    ]

    # --- Legend 1: Model architecture ---
    legend_models = ax.legend(
        handles=legend_handles,
        title="Model Architecture",
        frameon=False,
        fontsize=12,
        title_fontsize=13,
        ncol=len(ordered_models),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )

    ax.add_artist(legend_models)  # IMPORTANT: keep first legend

    # --- Legend 2: Noise vs No-noise ---
    noise_legend_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            label="With noise augmentation",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="//",
            linewidth=0.6,
            label="No noise",
        ),
    ]

    ax.legend(
        handles=noise_legend_handles,
        title="Noise Setting",
        frameon=False,
        fontsize=11,
        title_fontsize=12,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
    )

    fig.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()
    reset_theme()


def plot_real_timewise_error(
    experiment_results: dict[str, np.ndarray],
    running_mean: int = 2,
    save_dir: str = None,
    color: str = sns.color_palette("Blues_d", 5)[2]
) -> None:
    pollutant_titles = ["O3", "PM10", "PM25", "NO2"]
    num_pollutants = len(pollutant_titles)

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    dataset, _ = load_real_obs_data(model_type="clstm", sensor_type="real-random", timesteps=8, val_set=False, pollutant_type=pollutant_type)
    mask = torch.stack([target_mask for _, _, target_mask in dataset], dim=0)

    for pol_idx in range(num_pollutants):
        gts = experiment_results['ground_truth'][:, pol_idx]
        preds = experiment_results['predictions'][:, pol_idx]

        pollutant_errors = np.array(compute_relative_error(
            torch.from_numpy(gts) * mask[:, pol_idx],
            torch.from_numpy(preds) * mask[:, pol_idx]
        ))

        time_indices = np.arange(len(pollutant_errors))
        smoothed = uniform_filter1d(pollutant_errors, size=running_mean * 24, mode='nearest')

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.fill_between(time_indices, pollutant_errors, alpha=0.7, color=color)
        ax.plot(smoothed, alpha=0.7, label=f'{running_mean}-day running mean', color='orange')

        ax.set_xlabel("Days of December", fontsize=13)
        ax.set_ylabel("Mean Relative L2 Error", fontsize=13)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.legend(loc='upper right', fontsize=13)

        ax.set_ylim(0, 1.0)

        days = np.arange(len(pollutant_errors), step=2 * 24)
        ax.set_xticks(days, labels=days // 24)
        # ax.tick_params(labelbottom=True, labelleft=True)

        filename = f"{pollutant_titles[pol_idx].lower()}_temporal_error.png"
        fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

    reset_theme()

def plot_real_error_distribution(
    experiment_results: dict[str, np.ndarray],
    save_dir: str = None,
    color: str = sns.color_palette("Blues_d", 5)[2]
) -> None:
    pollutant_titles = ["O3", "PM10", "PM25", "NO2"]
    num_pollutants = len(pollutant_titles)

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    dataset, _ = load_real_obs_data(model_type="clstm", sensor_type="real-random", timesteps=8, val_set=False, pollutant_type=pollutant_type)
    mask = torch.stack([target_mask for _, _, target_mask in dataset], dim=0)

    pol_x_lims = [5.0, 2.5, 3.0, 2.0]
    pol_y_lims = [300, 150, 250, 150]

    for pol_idx in range(num_pollutants):
        gts = experiment_results['ground_truth'][:, pol_idx]
        preds = experiment_results['predictions'][:, pol_idx]

        pollutant_errors = np.array(compute_relative_error(
            torch.from_numpy(gts) * mask[:, pol_idx],
            torch.from_numpy(preds) * mask[:, pol_idx]
        ))

        pollutant_errors = np.where(pollutant_errors > 5.0, 5.0, pollutant_errors)

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.histplot(pollutant_errors, bins=30, ax=ax, color=color, linewidth=0)

        ax.set_xlim(0, pol_x_lims[pol_idx])
        ax.set_ylim(0, pol_y_lims[pol_idx])

        ax.set_xlabel("L2 RE of Individual Timeframes", fontsize=15, labelpad=10)
        ax.set_ylabel("Frequency", fontsize=15, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14)

        mu = np.mean(pollutant_errors)
        std = np.std(pollutant_errors)

        ax.axvline(mu, linestyle='--', linewidth=1.5, color='red')
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.text(mu + xlim[1] * 0.12, ylim[1]*0.9, f"{mu:.3f}\n±{std:.3f}",
                rotation=0, va='top', ha='center', fontsize=15)

        filename = f"{pollutant_titles[pol_idx].lower()}_error_hist.png"
        fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

    reset_theme()


def plot_joint_results_comparison():
    def plot_joint_results_comparison_helper(
        df: pd.DataFrame,
        figsize=(12, 6),
        capsize: int = 4,
        annot_size: int = 11,
        file_name: str | None = None,
    ) -> None:
        palette_names = {
                "CLSTM": "#5F8A5F",
                "VUNet": "#6495ED",
                "ViTAE": "#CD853F",
                "Kriging": "#808080",
                "VCNN": "#9370DB",
            }

        agg = (
            df.groupby(["Experiment Name", "Architecture"], as_index=False)
            .agg(mean_error=("Error", "mean"),
                std_error=("Error", "std"),
                n=("Seed", "nunique"))
        )

        exp_avg_error = agg.groupby("Experiment Name")["mean_error"].mean().sort_values()
        experiments = exp_avg_error.index.tolist()
        models_in_data = agg["Architecture"].unique().tolist()

        ordered_models = [m for m in palette_names if m in models_in_data] + \
                        [m for m in models_in_data if m not in palette_names]

        model_colors = {}
        fallback_colors = ["#BC8F8F", "#20B2AA", "#F0E68C", "#CD5C5C", "#90EE90"]
        fallback_idx = 0

        for m in ordered_models:
            if m in palette_names:
                model_colors[m] = palette_names[m]
            else:
                model_colors[m] = fallback_colors[fallback_idx % len(fallback_colors)]
                fallback_idx += 1

        x = np.arange(len(experiments))
        M = len(ordered_models)

        fig, ax = plt.subplots(figsize=figsize)

        for i, m in enumerate(ordered_models):
            m_df = agg[agg["Architecture"] == m].set_index("Experiment Name").reindex(experiments)

            means = m_df["mean_error"].values
            stds = m_df["std_error"].values

            mask = ~np.isnan(means)

            bar_info = list(zip(x[mask], means[mask], stds[mask]))
            bar_info.sort(key=lambda tup: tup[1])

            model_bar_coord = [info[0] for info in bar_info]
            model_bar_means = [info[1] for info in bar_info]
            model_bar_stds = [info[2] for info in bar_info]

            bars = ax.bar(
                model_bar_coord,
                model_bar_means,
                yerr=model_bar_stds,
                width=0.9,
                label=m,
                color=model_colors[m],
                edgecolor="black",
                linewidth=0.4,
                capsize=capsize,
            )

            for bar, mean, std in zip(bars.patches, model_bar_means, model_bar_stds):
                
                if not np.isnan(std): 
                    text = f"{mean:.3f}\n±{std:.3f}"
                else: 
                    text = f"{mean:.3f}"
                
                plt.annotate(
                    text,
                    (bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01),
                    ha="center", va="bottom", fontsize=annot_size, color="black",
                    xytext=(0, 5), textcoords="offset points"
                )

        ax.set_xlabel("Training Method", fontsize=14, labelpad=10)
        ax.set_ylabel("Mean L2 MRE (mean ± std over seeds)", fontsize=14, labelpad=10)

        noising_methods = [exp_name.split(" - ")[-1] if " - " in exp_name else "Standard" for exp_name in experiments]
        ax.set_xticks(x)
        ax.set_xticklabels(noising_methods, rotation=30, ha="right", fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.set_ylim(0, 0.5)

        ax.legend(title="Model", frameon=False, fontsize=12, title_fontsize=13)

        ax.margins(x=0.02)

        ax.grid(axis="y", linestyle=":", alpha=0.6)

        fig.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        reset_theme()

    df = read_tuning_results_df()

    os.makedirs("report_images/experiments/fine_tuned/error_bars", exist_ok=True)

    plot_joint_results_comparison_helper(
        df,
        figsize=(10, 6), capsize=4, annot_size=11, 
        file_name="test.png"
    )

    plot_joint_results_comparison_helper(
        df[df['Architecture'].isin(["CLSTM", "Kriging"])],
        figsize=(5, 5), capsize=4, annot_size=13, 
        file_name="report_images/experiments/fine_tuned/error_bars/clstm.png"
    )

    plot_joint_results_comparison_helper(
        df[df['Architecture'].isin(["VUNet", "Kriging"])],
        figsize=(5, 5), capsize=4, annot_size=13, 
        file_name="report_images/experiments/fine_tuned/error_bars/vunet.png"
    )

    plot_joint_results_comparison_helper(
        df[df['Architecture'].isin(["ViTAE", "Kriging"])],
        figsize=(5, 5), capsize=4, annot_size=13, 
        file_name="report_images/experiments/fine_tuned/error_bars/vitae.png"
    )


def plot_grouped_paper_real_results(results: pd.DataFrame, save: bool = True):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple

    palette_names = {
        "CLSTM": "Greens",
        "VUNet": "Blues",
        "ViTAE": "Oranges",
        "Kriging": "Greys",
        "Diffusion": "Purples",
    }

    # NEW: shrink entire group width symmetrically
    group_width_scale = {
        "Kriging": 0.55,   # thinner baseline group
    }

    categories = sorted(results["Model"].unique())
    x_centers = np.arange(len(categories))

    plt.figure(figsize=(16, 7))

    ribbon_handles = []
    ribbon_labels = []

    xtick_positions = []
    xtick_labels = []

    for cat_idx, category in enumerate(categories):
        cat_df = results[results["Model"] == category]
        cat_methods = sorted(cat_df["Noising Method"].unique())
        k = len(cat_methods)
        if k == 0:
            continue

        # --- GROUP GEOMETRY (FIXED) ---
        base_total_width = 0.8
        group_scale = group_width_scale.get(category, 1.0)
        total_width = base_total_width * group_scale

        bar_width = total_width / k
        offsets = (np.arange(k) - (k - 1) / 2.0) * bar_width

        cmap = plt.cm.get_cmap(palette_names.get(category, "Greys"))
        shades = (
            [0.65] if k == 1 else
            [0.45, 0.75] if k == 2 else
            np.linspace(0.35, 0.85, k)
        )

        for i, (method, shade) in enumerate(zip(cat_methods, shades)):
            sub = cat_df[cat_df["Noising Method"] == method]["Error"]
            mean = float(np.nanmean(sub)) if len(sub) else 0.0
            std = float(np.nanstd(sub)) if len(sub) else 0.0

            x = x_centers[cat_idx] + offsets[i]

            bars = plt.bar(
                x,
                mean,
                bar_width,
                color=cmap(shade),
                yerr=(std if (std and not np.isnan(std)) else None),
                capsize=3,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

            if mean != 0.0:
                plt.text(
                    x,
                    mean + 0.02,
                    f"{mean:.3f}" + (f"\n±{std:.3f}" if std else ""),
                    ha="center",
                    va="bottom",
                    fontsize=15,
                    rotation=45,
                    
                )

            xtick_positions.append(x)
            xtick_labels.append(method)

        # --- Ribbon legend entry ---
        patches = tuple(
            Patch(
                facecolor=cmap(shade),
                edgecolor="black",
                linewidth=0.5
            )
            for shade in shades
        )

        ribbon_handles.append(patches)
        ribbon_labels.append(category)

    plt.xticks(
        xtick_positions,
        xtick_labels,
        rotation=45,
        ha="right",
        fontsize=15
    )

    plt.xlabel("Noising Methods (grouped by Architecture)", fontsize=17, labelpad=25)
    plt.ylabel("Global Relative Error", fontsize=17)

    plt.tick_params(axis="y", labelsize=15)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.ylim(0, 0.6)

    ax = plt.gca()
    model_legend = ax.legend(
        ribbon_handles,
        ribbon_labels,
        title="Model Architecture",
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=16,
        title_fontsize=19,
        handlelength=4.2,
        handletextpad=1.8,
        columnspacing=3.0,
        ncol=len(ribbon_labels),
    )

    ax.add_artist(model_legend)

    plt.tight_layout()
    if save:
        plt.savefig("paper_images/grouped_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    reset_theme()


# The gt_image and pred_image numpy arrays have 3 dimensions, representing (time, width, height). Another feature of the data is that the gt_image array has a lot of 0 values. That's missing data that the model is supposed to predict.  However, in this function, we are going to use the ground truth data that we have to evaluate the model.



# Thus, I would like this function to generate a plot  where the x axis is the gt value and the y axis is the predicted value. Ignore values pixels where gt is 0.



# def plot_gt_vs_pred(
#     all_results: dict[str, dict[str, np.ndarray]],
#     save_dir: str,
#     pollutant: str,
# ) -> None:
#     pollutants_idx = {
#         "o3": 0,
#         "pm10": 1,
#         "pm25": 2,
#         "no2": 3
#     }
#     pollutant_idx = pollutants_idx[pollutant]
#     n_models = len(all_results)

#     for i, (model_name, results) in enumerate(all_results.items()):
#         gt_img = results['simulated_ground_truths'][:, pollutant_idx]
#         pred_img = results['simulated_predictions'][:, pollutant_idx]

#         print(gt_img.shape, pred_img.shape)
def draw_ellipse(gt_valid, pred_valid, ax, model_name):
        
    # Add confidence ellipse
        cov = np.cov(gt_valid, pred_valid)
        mean_x = np.mean(gt_valid)
        mean_y = np.mean(pred_valid)
        if model_name == 'Kriging':
            edgec = 'green'
        else:
            edgec = 'blue'

        # Ellipse radii
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvalues (largest first)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Angle of ellipse
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

        width, height = 2 * 2 * np.sqrt(eigvals)


        ellipse = Ellipse((mean_x, mean_y),
                      width=width,
                      height=height,
                      angle=angle,
                      facecolor='none',
                      edgecolor=edgec,
                      alpha=0.8,
                      label = 'confidence ellipse '+ model_name)
        
        return ax.add_patch(ellipse)

def plot_gt_vs_pred(
    all_results: dict[str, dict[str, np.ndarray]],
    save_dir: str,
    pollutant: str,
    gt_column: str = 'real_ground_truths',
    pred_column: str = 'real_predictions',
) -> None:
    from scipy import stats
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    
    pollutants_idx = {
        "o3": 0,
        "pm10": 1,
        "pm25": 2,
        "no2": 3
    }
    title_dict = {
        "o3": "O3",
        "pm10": "PM10",
        "pm25": "PM2.5",
        "no2": "NO2"
    }
    pollutant_idx = pollutants_idx[pollutant]
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, results in all_results.items():

        gt_img = results[gt_column][:, pollutant_idx]
        pred_img = results[pred_column][:, pollutant_idx]
        pred_mask = results["real_target_masks"][:, [pollutant_idx]]
        
        # Get the shape to reconstruct spatial coordinates
        time_steps, height, width = gt_img.shape
        
        # Create coordinate arrays
        # Shape will be (time, height, width)
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Broadcast to match time dimension
        x_coords_full = np.broadcast_to(x_coords[np.newaxis, :, :], (time_steps, height, width))
        y_coords_full = np.broadcast_to(y_coords[np.newaxis, :, :], (time_steps, height, width))
        
        # Flatten the arrays
        pred_mask_flat = pred_mask.flatten()
        gt_flat = gt_img.flatten()
        pred_flat = pred_img.flatten()
        x_flat = x_coords_full.flatten()
        y_flat = y_coords_full.flatten()

        if pred_mask_flat.shape[0] > gt_flat.shape[0]:
            pred_mask_flat = pred_mask_flat[:gt_flat.shape[0]]

        gt_flat = gt_flat[:pred_mask_flat.shape[0]]
        pred_flat = pred_flat[:pred_mask_flat.shape[0]]
        x_flat = x_flat[:pred_mask_flat.shape[0]]
        y_flat = y_flat[:pred_mask_flat.shape[0]]
        
        # Filter out pixels where ground truth is close to 0 (missing data)
        mask = gt_flat > 1e-6
        mask = pred_mask_flat > 1e-6
        gt_valid = gt_flat[mask]
        pred_valid = pred_flat[mask]
        x_valid = x_flat[mask]
        y_valid = y_flat[mask]
        
        # Classify as inner city or outer city
        # Inner city: x between 45 and 65, y between 30 and 45
        inner_city_mask = (x_valid >= 45) & (x_valid <= 65) & (y_valid >= 30) & (y_valid <= 45)
        
        # Create figure
        _, ax = plt.subplots(figsize=(5, 5))
        
        # Plot outer city points
        outer_mask = ~inner_city_mask
        if np.any(outer_mask):
            ax.scatter(gt_valid[outer_mask], pred_valid[outer_mask], 
                    alpha=0.5, s=18, color='coral', edgecolor='none',
                    label='Outer city')
        
        # Plot inner city points
        if np.any(inner_city_mask):
            ax.scatter(gt_valid[inner_city_mask], pred_valid[inner_city_mask], 
                    alpha=0.5, s=18, color='steelblue', edgecolor='none',
                    label='Inner city')
        
        # Add diagonal line (perfect prediction)
        min_val = min(gt_valid.min(), pred_valid.min())
        max_val = max(gt_valid.max(), pred_valid.max()) * 1.05
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect prediction', zorder=5, alpha=0.7)
        
        # Labels and title
        ax.set_xlabel('Reference '+ title_dict[pollutant], fontsize=20)
        ax.set_ylabel('Predicted', fontsize=20)
        #ax.legend(fontsize=16, loc='upper left')

        if pollutant == "pm10" or pollutant == "pm25" or pollutant == "o3":
            ax.set_xlim(-5, 80)
            ax.set_ylim(-5, 80)
        elif pollutant == "no2":
            ax.set_xlim(-5, 90)
            ax.set_ylim(-5, 90)
        
        # Add r-value text below the legend
        # Calculate Pearson correlation coefficient
        r_value, p_value = stats.pearsonr(gt_valid, pred_valid)

        #ax.text(0.04, 0.75, f'r = {r_value:.3f}', 
        #        transform=ax.transAxes, fontsize=14,
        #        verticalalignment='top', horizontalalignment='left',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Make axes equal for better comparison
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()

        #plt.title(pred_column)
        
        # Save the figure
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        
        save_path = os.path.join(save_dir, f'{safe_model_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()

    reset_theme()


def plot_gt_vs_pred_simulation(
    all_results: dict[str, dict[str, np.ndarray]],
    save_dir: str,
    pollutant: str,
    gt_column: str = 'real_ground_truths',
    pred_column: str = 'real_predictions',
) -> None:
    from scipy import stats
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    
    pollutants_idx = {
        "o3": 0,
        "pm10": 1,
        "pm25": 2,
        "no2": 3
    }

    title_dict = {
        "o3": "O3",
        "pm10": "PM10",
        "pm25": "PM2.5",
        "no2": "NO2"
    }
    pollutant_idx = pollutants_idx[pollutant]
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    gt_valid_krig = None
    pred_valid_krig = None
    
    for model_name, results in all_results.items():

        gt_img = results[gt_column][:, pollutant_idx]
        pred_img = results[pred_column][:, pollutant_idx]
        pred_mask = results["real_target_masks"][:, [pollutant_idx]]
        
        # Get the shape to reconstruct spatial coordinates
        time_steps, height, width = gt_img.shape
        
        # Create coordinate arrays
        # Shape will be (time, height, width)
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Broadcast to match time dimension
        x_coords_full = np.broadcast_to(x_coords[np.newaxis, :, :], (time_steps, height, width))
        y_coords_full = np.broadcast_to(y_coords[np.newaxis, :, :], (time_steps, height, width))
        
        # Flatten the arrays
        pred_mask_flat = pred_mask.flatten()
        gt_flat = gt_img.flatten()
        pred_flat = pred_img.flatten()
        x_flat = x_coords_full.flatten()
        y_flat = y_coords_full.flatten()

        if pred_mask_flat.shape[0] > gt_flat.shape[0]:
            pred_mask_flat = pred_mask_flat[:gt_flat.shape[0]]

        gt_flat = gt_flat[:pred_mask_flat.shape[0]]
        pred_flat = pred_flat[:pred_mask_flat.shape[0]]
        x_flat = x_flat[:pred_mask_flat.shape[0]]
        y_flat = y_flat[:pred_mask_flat.shape[0]]
        
        # Filter out pixels where ground truth is close to 0 (missing data)
        mask = gt_flat > 1e-6
        #mask_plot = pred_mask_flat > 1e-6
        gt_valid = gt_flat[mask]
        pred_valid = pred_flat[mask]
        
        
        # Create figure
        _, ax = plt.subplots(figsize=(5, 5))

        # Select random points to plot
        k_size = int(0.0001*len(gt_valid))
        sort_idx = np.argsort(gt_valid)
        groups = np.array_split(sort_idx, k_size)

        random_idx = np.array([np.random.choice(g) for g in groups], dtype=int)
        random_idx = np.sort(random_idx)
        
        ax.scatter(gt_flat[random_idx], pred_flat[random_idx], 
                alpha=0.5, s=18, color='coral', edgecolor='none')
        
        if model_name == "Kriging":
            gt_valid_krig = gt_valid
            pred_valid_krig = pred_valid
        
        # Add confidence ellipse
        draw_ellipse(gt_valid, pred_valid, ax, model_name)
        if model_name != 'Kriging':
            draw_ellipse(gt_valid_krig, pred_valid_krig, ax, "Kriging")
        
        # Labels and title
        ax.set_xlabel('Reference '+ title_dict[pollutant], fontsize=20)
        ax.set_ylabel('Predicted', fontsize=20)
        if pollutant == "no2":
            ax.set_xlim(-20, 60)
            ax.set_ylim(-20, 60)
        if pollutant == "pm10":
            ax.set_xlim(-5, 35)
            ax.set_ylim(-5, 35)
        if pollutant == "pm25":
            ax.set_xlim(-5, 30)
            ax.set_ylim(-5, 30)
        
        # Make axes equal for better comparison
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()

        # Save the figure
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        
        save_path = os.path.join(save_dir, f'{safe_model_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()

    reset_theme()