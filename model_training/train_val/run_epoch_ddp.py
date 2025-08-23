import torch
import mlflow
import gc
from time import perf_counter
from eitools.utils import progress_bar, seconds_to_string
from matplotlib import pyplot as plt

import logging
import csv
import os
import pandas as pd
from datetime import datetime
from utils.preprocess_hsi import preprocess_hsi

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

def run_one_epoch(epoch_info, train_module, loader, loss_fn, metric_fn, show_predictions, device, csv_path: str = None):
    """Train or validate for one epoch using the specified device."""
    t_start_epoch = perf_counter()
    mode = 'train' if train_module.model.training else 'val'

    epoch_loss = 0.0
    epoch_metric = 0.0
    total_samples = 0
    nb_batch = len(loader)

    for i, batch in enumerate(loader, 1):
        hs_cube, label, rgb = batch

        # Move hs_cube to the proper device (instead of hardcoding "cuda")
        hs_cube = hs_cube.to(device, non_blocking=True)
        num_samples = hs_cube.size(0)
        hs_cube = preprocess_hsi(hs_cube)

        train_module.zero_grad()

        # Forward pass: assume model returns (loss_batch, pred, mask)
        loss_batch, pred, mask = train_module.model(hs_cube)
        loss = torch.mean(loss_batch)

        # Reconstruct output (ensure the model's decoder is on the same device)
        reconstructed_output = train_module.model.module.decoder.unpatchify(pred)
        reconstructed_output = reconstructed_output.squeeze(1)

        # Compute metric (both tensors are now on the same device)
        metric = metric_fn(reconstructed_output, hs_cube).item()

        # Backpropagation & Optimization
        train_module.backward(loss)

        # Accumulate total loss & metric
        epoch_loss += loss.item() * hs_cube.size(0)
        epoch_metric += metric * hs_cube.size(0)
        total_samples += hs_cube.size(0)

        # Logging Figures to MLflow (move tensors to CPU for plotting)
        if show_predictions:
            hs_cube_cpu = hs_cube.detach().cpu().numpy()
            reconstructed_output_cpu = reconstructed_output.detach().cpu().numpy()
            rgb_cpu = rgb.detach().cpu().numpy()

            for plot_name, show_prediction in show_predictions.items():
                fig = show_prediction(hs_cube_cpu, reconstructed_output_cpu, rgb_cpu, label)
                mlflow.log_figure(fig, artifact_file=f'{epoch_info.epoch:04}_{i:03}_{plot_name}.png')
                plt.close(fig)

        # Update Progress Bar
        cuda_memory = f"{torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB"
        cuda_cached = f"{torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
        progress_bar(
            i - 1, nb_batch,
            msg=f'[{i}/{nb_batch}] Loss: {loss.item():.3e}, Acc: {metric:.2e}, cuda_mem: {cuda_memory}, cached={cuda_cached}',
            colored=False
        )

    print('\r', end='')

    # Final Calculation
    epoch_loss = epoch_loss / total_samples
    epoch_metric = epoch_metric / total_samples

    # Log to MLflow
    log_values = {
        f'{loss_fn.__class__.__name__} {mode}': round(epoch_loss, 5),
        f'{metric_fn.__class__.__name__} {mode}': round(epoch_metric, 5),
    }
    if mode == 'train':
        log_values['Learning Rate'] = train_module.optimizer.param_groups[0]['lr']
    log_metrics(log_values, epoch_info.epoch, csv_path)

    elapsed_time = seconds_to_string(perf_counter() - t_start_epoch)

    # Logging final results
    if mode == 'train':
        prefix = f'[{epoch_info.epoch}/{epoch_info.nb_epochs}] TRAIN -'
    else:
        spaces = ' ' * len(f'[{epoch_info.epoch}/{epoch_info.nb_epochs}]')
        prefix = f'{spaces} VALID -'
    logger.info(f'{prefix} Loss: {epoch_loss:.3e}, Acc: {epoch_metric:.3f} ({elapsed_time})')

    return epoch_loss, epoch_metric
