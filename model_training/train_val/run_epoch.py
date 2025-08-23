from time import perf_counter

import mlflow
import torch
import numpy as np
from eitools.utils import progress_bar, seconds_to_string
from matplotlib import pyplot as plt
from torch import nn

from train_val.epoch_results import EpochResults
from train_val.utils import TrainingModule, EpochInfo
from utils.preprocess_hsi import preprocess_hsi
from eitools.hs.hs_to_rgb import hs_to_rgb_from_array
from utils.random_channel_drop import RandomChannelDrop
from eitools.utils.progress_bar import ProgressBar

import logging
import csv
import os
import pandas as pd
from datetime import datetime
logger = logging.getLogger()



def log_metrics(metrics: dict[str, float], current_step: int, csv_path: str = None) -> None:
    """ Log the metrics in mlflow and CSV. """

    for key, value in metrics.items():
        mlflow.log_metric(key=f"{key}", value=value, step=current_step)
    
    # Log to CSV if path provided
    if csv_path:
        log_metrics_to_csv(metrics, current_step, csv_path)


def log_metrics_to_csv(metrics: dict[str, float], current_step: int, csv_path: str) -> None:
    """ Log metrics to CSV file. """
    
    # Prepare the row data
    row_data = {'step': current_step, 'timestamp': datetime.now().isoformat()}
    row_data.update(metrics)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_path)
    
    # Create directory if it doesn't exist
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:  # Only create if there's a directory path
        os.makedirs(csv_dir, exist_ok=True)
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)


def run_one_epoch(epoch_info: EpochInfo, train_module: TrainingModule, loader, loss_fn, metric_fn, show_predictions, csv_path: str = None) -> tuple[float, float]:
    """Train the model for one epoch and log task-specific losses."""
    t_start_epoch = perf_counter()
    mode = 'train' if train_module.model.training else 'val'

    # Initialize accumulators
    epoch_loss = 0.0
    epoch_metric = 0.0
    total_samples = 0

    nb_batch = len(loader)

    # Determine the device from the model's parameters.
    device = next(train_module.model.parameters()).device

    for i, batch in enumerate(loader, 1):

        hs_cube, label, rgb = batch
        hs_cube = hs_cube.to(device, non_blocking=True)

        num_samples = hs_cube.size(0)
        hs_cube = preprocess_hsi(hs_cube)
        # hs_cube = hs_cube[:, ::2]

        train_module.zero_grad()  # Reset gradients

        loss_batch, pred, mask = train_module.model(hs_cube)
        loss = torch.mean(loss_batch)

        # Access the decoder depending on whether the model is wrapped or not.
        if isinstance(train_module.model, nn.DataParallel):
            decoder = train_module.model.module.decoder
        else:
            decoder = train_module.model.decoder

        reconstructed_output = decoder.unpatchify(pred)
        reconstructed_output = reconstructed_output.squeeze(1)
        metric = metric_fn(reconstructed_output, hs_cube).item()

        # Backpropagation and optimizer step
        train_module.backward(loss)  # Ensure all losses contribute to gradients

        # Accumulate total loss and segmentation metric
        epoch_loss += loss.item() * num_samples
        epoch_metric += metric * num_samples
        total_samples += num_samples

        # Plot the prediction
        if show_predictions:
            hs_cube = hs_cube.detach().cpu().numpy()
            reconstructed_output = reconstructed_output.detach().cpu().numpy()
            rgb = rgb.detach().cpu().numpy()

            for plot_name, show_prediction in show_predictions.items():
                fig = show_prediction(hs_cube, reconstructed_output, rgb, label)
                mlflow.log_figure(fig, artifact_file=f'{epoch_info.epoch:0>4}_{i:0>3}_{plot_name}.png')
                plt.close(fig)

        # Get CUDA memory stats if on a CUDA device
        if device.type == 'cuda':
            cuda_memory = f"{torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB"
            cuda_cached = f"{torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
        else:
            cuda_memory = "N/A"
            cuda_cached = "N/A"

        # Update progress bar
        progress_bar(i - 1, nb_batch, msg=f'[{i}/{nb_batch}] Loss: {loss.item():.3e}, Acc: {metric:.2e}, cuda_memory: {cuda_memory}, cuda_cached={cuda_cached}', colored=False)

    # Clear the progress bar
    print('\r', end='')

    # Calculate the epoch loss and metric
    epoch_loss = epoch_loss / total_samples
    epoch_metric = epoch_metric / total_samples

    # log metric
    log_values = {
        f'{loss_fn.__class__.__name__} {mode}': round(epoch_loss, 5),
        f'{metric_fn.__class__.__name__} {mode}': round(epoch_metric, 5),
    }
    if mode == 'train':
        log_values['Learning Rate'] = train_module.optimizer.param_groups[0]['lr']
    log_metrics(log_values, epoch_info.epoch, csv_path)

    elapsed_time = seconds_to_string(perf_counter() - t_start_epoch)

    if mode == 'train':
        prefix = f'[{epoch_info.epoch}/{epoch_info.nb_epochs}] TRAIN -'
    else:
        spaces = ' ' * len(f'[{epoch_info.epoch}/{epoch_info.nb_epochs}]')
        prefix = f'{spaces} VALID -'

    logger.info(f'{prefix} Loss: {epoch_loss:.3e}, Acc: {epoch_metric:.3f} ({elapsed_time})')

    return epoch_loss, epoch_metric


