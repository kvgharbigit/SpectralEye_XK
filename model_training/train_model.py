import gc
import logging
from time import perf_counter
import hydra
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from eitools.utils import seconds_to_string
from eitorch.training.config.utils import get_all_hydra_parameters, get_instance_repr
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from path import Path
from skimage.io import imsave
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision.transforms import Compose
import os
from datetime import datetime

from model_training.plots.plot_bottleneck import display_rgb, display_latent
from model_training.train_val.epoch_results import EpochResults

from model_training.train_val.run_epoch import run_one_epoch
from eitorch.model_io import save_model_state

from model_training.train_val.utils import create_training_module, get_datasets, get_dataloaders, EpochInfo
from model_training.utils.save_model import save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class PrefetchLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.iter = iter(self.dataloader)
        self.stream = torch.cuda.Stream()

    def preload(self):
        try:
            self.batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            self.batch = next(self.iter)
        with torch.cuda.stream(self.stream):
            self.batch = {k: v.to(self.device, non_blocking=True) for k, v in self.batch.items()}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def __iter__(self):
        self.preload()
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.dataloader)


def create_model_info_file(output_dir: str, cfg: DictConfig, model, train_module, dataset_info: dict = None) -> None:
    """Create a comprehensive model_info.txt file with all training details"""
    import os
    from datetime import datetime
    import torch
    
    info_path = os.path.join(output_dir, 'model_info.txt')
    
    with open(info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL TRAINING INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        # Timestamp
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training mode: {'DDP' if cfg.general.use_ddp else 'Single GPU'}\n\n")
        
        # Model Architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model name: {cfg.model.name}\n")
        f.write(f"Model class: {model.__class__.__name__}\n")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}\n\n")
        
        # Model specific config
        if hasattr(cfg.model, 'img_size'):
            f.write("Model Configuration:\n")
            for key, value in cfg.model.items():
                if key != '_target_' and key != 'name':
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Training Configuration
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epochs: {cfg.hparams.nb_epochs}\n")
        f.write(f"Batch size: {cfg.hparams.batch_size}\n")
        f.write(f"Learning rate: {cfg.hparams.lr}\n")
        f.write(f"Validation interval: {cfg.hparams.valid_interval}\n")
        if hasattr(cfg.hparams, 'accumulate_grad_batches'):
            f.write(f"Gradient accumulation: {cfg.hparams.accumulate_grad_batches}\n")
        f.write("\n")
        
        # Optimizer
        f.write("OPTIMIZER\n")
        f.write("-" * 40 + "\n")
        f.write(f"Type: {train_module.optimizer.__class__.__name__}\n")
        f.write(f"Config: {cfg.optimizer}\n\n")
        
        # Scheduler
        f.write("SCHEDULER\n")
        f.write("-" * 40 + "\n")
        if hasattr(train_module, 'scheduler') and train_module.scheduler:
            f.write(f"Type: {train_module.scheduler.__class__.__name__}\n")
            f.write(f"Config: {cfg.scheduler}\n")
        else:
            f.write("No scheduler configured\n")
        f.write("\n")
        
        # Loss Function
        f.write("LOSS FUNCTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Type: {getattr(cfg.loss, '_target_', 'unknown')}\n")
        for key, value in cfg.loss.items():
            if key != '_target_':
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Metric Function
        f.write("METRIC FUNCTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Type: {getattr(cfg.metric, '_target_', 'unknown')}\n\n")
        
        # Dataset Information
        f.write("DATASET\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset config: {cfg.dataset.name if hasattr(cfg.dataset, 'name') else 'unknown'}\n")
        if hasattr(cfg.dataset, 'csv_path'):
            f.write(f"CSV path: {cfg.dataset.csv_path}\n")
        if hasattr(cfg.dataset, 'trial_mode'):
            f.write(f"Trial mode: {cfg.dataset.trial_mode}\n")
            if cfg.dataset.trial_mode and hasattr(cfg.dataset, 'trial_size'):
                f.write(f"Trial size: {cfg.dataset.trial_size}\n")
        if dataset_info:
            f.write(f"Train samples: {dataset_info.get('train_size', 'N/A')}\n")
            f.write(f"Val samples: {dataset_info.get('val_size', 'N/A')}\n")
        f.write("\n")
        
        # Data Augmentation
        f.write("DATA AUGMENTATION\n")
        f.write("-" * 40 + "\n")
        if hasattr(cfg, 'augmentation'):
            try:
                # Handle different augmentation config structures
                if hasattr(cfg.augmentation, 'transforms'):
                    for aug in cfg.augmentation.transforms:
                        f.write(f"  - {getattr(aug, 'name', str(aug))}: {aug}\n")
                else:
                    # If transforms is not available, just show the config
                    f.write(f"Augmentation config: {cfg.augmentation}\n")
            except Exception as e:
                f.write(f"Augmentation config (could not parse): {cfg.augmentation}\n")
        else:
            f.write("No augmentation configured\n")
        f.write("\n")
        
        # Hardware Configuration
        f.write("HARDWARE CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Use CUDA: {cfg.general.use_cuda}\n")
        if cfg.general.use_ddp:
            f.write(f"DDP enabled: True\n")
            f.write(f"World size: {torch.cuda.device_count()}\n")
        elif cfg.general.parallel.use_parallel:
            f.write(f"DataParallel enabled: True\n")
            f.write(f"Device IDs: {cfg.general.parallel.device_ids}\n")
        else:
            f.write(f"Device ID: {cfg.general.device_id}\n")
        f.write(f"Num workers: {cfg.dataloader.num_workers}\n")
        f.write(f"Pin memory: {cfg.dataloader.pin_memory}\n")
        f.write(f"Use AMP: {cfg.general.use_amp}\n\n")
        
        # Output Configuration
        f.write("OUTPUT CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Save model: {cfg.general.save_model}\n")
        f.write(f"Save results: {cfg.general.save_results}\n\n")
        
        # Model String Representation
        f.write("FULL MODEL ARCHITECTURE\n")
        f.write("-" * 40 + "\n")
        f.write(str(model))
        
    logger.info(f"Created model info file: {info_path}")


def generate_training_plots(csv_path: str, output_dir: str) -> None:
    """Generate training plots from CSV metrics data"""
    import time
    
    try:
        # Add a small delay to ensure CSV write is complete
        time.sleep(0.1)
        
        # Read the CSV file with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = pd.read_csv(csv_path)
                break
            except (pd.errors.EmptyDataError, FileNotFoundError, PermissionError) as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Attempt {attempt + 1} failed to read CSV: {e}, retrying...")
                    time.sleep(0.2)
                else:
                    logger.warning(f"Failed to read CSV after {max_retries} attempts: {e}")
                    return
        
        if len(df) == 0:
            logger.debug("CSV file is empty, skipping plot generation")
            return
            
        logger.debug(f"Generating plots from {csv_path} with {len(df)} epochs")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training and Validation Loss
        # Debug: print column names to understand structure
        logger.debug(f"CSV columns: {list(df.columns)}")
        
        # Find loss columns (could be CustomLoss, MSELoss, etc.)
        train_loss_cols = [col for col in df.columns if 'Loss train' in col or (col.endswith(' train') and 'Loss' in col)]
        val_loss_cols = [col for col in df.columns if 'Loss val' in col or (col.endswith(' val') and 'Loss' in col)]
        
        logger.debug(f"Train loss columns found: {train_loss_cols}")
        logger.debug(f"Val loss columns found: {val_loss_cols}")
        
        if train_loss_cols:
            ax1.plot(df['step'], df[train_loss_cols[0]], label='Train Loss', linewidth=2)
        if val_loss_cols:
            val_df = df[df[val_loss_cols[0]].notna()]
            if len(val_df) > 0:
                # Only show markers if validation data is sparse (not every epoch)
                total_epochs = len(df)
                val_epochs = len(val_df)
                is_sparse = val_epochs < (total_epochs * 0.8)  # Less than 80% of epochs have val data
                
                if is_sparse:
                    ax1.plot(val_df['step'], val_df[val_loss_cols[0]], label='Val Loss', 
                            marker='o', markersize=6, linewidth=2)
                else:
                    ax1.plot(val_df['step'], val_df[val_loss_cols[0]], label='Val Loss', 
                            linewidth=2)  # No markers for dense validation data
            else:
                logger.debug("No validation data found in CSV")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reconstruction MSE or Metric
        # Find metric columns (could be ReconstructionMSE, L1Loss, etc.)
        train_metric_cols = [col for col in df.columns if col.endswith(' train') and not any(x in col for x in ['Loss train', 'Learning Rate'])]
        val_metric_cols = [col for col in df.columns if col.endswith(' val') and 'Loss val' not in col]
        
        logger.debug(f"Train metric columns found: {train_metric_cols}")
        logger.debug(f"Val metric columns found: {val_metric_cols}")
        
        if train_metric_cols:
            ax2.plot(df['step'], df[train_metric_cols[0]], label=f'Train {train_metric_cols[0].replace(" train", "")}', linewidth=2)
        if val_metric_cols:
            val_df = df[df[val_metric_cols[0]].notna()]
            if len(val_df) > 0:
                # Only show markers if validation data is sparse (not every epoch)
                total_epochs = len(df)
                val_epochs = len(val_df)
                is_sparse = val_epochs < (total_epochs * 0.8)  # Less than 80% of epochs have val data
                
                if is_sparse:
                    ax2.plot(val_df['step'], val_df[val_metric_cols[0]], label=f'Val {val_metric_cols[0].replace(" val", "")}', 
                            marker='o', markersize=6, linewidth=2)
                else:
                    ax2.plot(val_df['step'], val_df[val_metric_cols[0]], label=f'Val {val_metric_cols[0].replace(" val", "")}', 
                            linewidth=2)  # No markers for dense validation data
            else:
                logger.debug("No validation metric data found in CSV")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.set_title('Reconstruction MSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate
        if 'Learning Rate' in df.columns:
            ax3.plot(df['step'], df['Learning Rate'], color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: Loss components if available
        loss_components = [col for col in df.columns if 'train' in col and col != 'CustomLoss train' 
                          and col != 'ReconstructionMSE train']
        if loss_components:
            for comp in loss_components:
                ax4.plot(df['step'], df[comp], label=comp.replace(' train', ''), linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Value')
            ax4.set_title('Loss Components')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # If no loss components, show train vs val gap
            if train_loss_cols and val_loss_cols:
                train_col = train_loss_cols[0]
                val_col = val_loss_cols[0]
                # Get validation epochs only
                val_df = df[df[val_col].notna()].copy()
                if len(val_df) > 0:
                    # Calculate gap for validation epochs only
                    gap = val_df[val_col] - val_df[train_col]
                    ax4.plot(val_df['step'], gap, color='red', linewidth=2)
                    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax4.set_xlabel('Epoch')
                    ax4.set_ylabel('Val Loss - Train Loss')
                    ax4.set_title('Generalization Gap')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No validation data available', 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=ax4.transAxes)
                    ax4.set_title('Generalization Gap')
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for gap calculation', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=ax4.transAxes)
                ax4.set_title('Generalization Gap')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training plots to {plot_path}")
        
        # Generate summary statistics
        summary_path = os.path.join(output_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Training Summary\n")
            f.write("================\n\n")
            f.write(f"Total Epochs: {len(df)}\n")
            
            if 'CustomLoss train' in df.columns:
                f.write(f"Final Train Loss: {df['CustomLoss train'].iloc[-1]:.6f}\n")
                f.write(f"Best Train Loss: {df['CustomLoss train'].min():.6f} (epoch {df['CustomLoss train'].idxmin() + 1})\n")
            
            if 'CustomLoss val' in df.columns:
                val_df = df[df['CustomLoss val'].notna()]
                if len(val_df) > 0:
                    f.write(f"\nFinal Val Loss: {val_df['CustomLoss val'].iloc[-1]:.6f}\n")
                    f.write(f"Best Val Loss: {val_df['CustomLoss val'].min():.6f} (epoch {val_df['CustomLoss val'].idxmin() + 1})\n")
            
            if 'ReconstructionMSE train' in df.columns:
                f.write(f"\nFinal Train MSE: {df['ReconstructionMSE train'].iloc[-1]:.6f}\n")
            
            if 'ReconstructionMSE val' in df.columns:
                val_df = df[df['ReconstructionMSE val'].notna()]
                if len(val_df) > 0:
                    f.write(f"Final Val MSE: {val_df['ReconstructionMSE val'].iloc[-1]:.6f}\n")
        
        logger.info(f"Saved training summary to {summary_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")


def copy_small_files_to_network(local_dir: str, run_name: str, rank: int = 0) -> None:
    """Copy small files to network drive (only rank 0 should call this)"""
    if rank != 0:
        return
        
    import shutil
    try:
        network_base = r"Z:\Projects\Ophthalmic neuroscience\Projects\Kayvan\SpectralEye_XK_Outputs"
        network_dir = os.path.join(network_base, run_name)
        
        # Create network directory
        os.makedirs(network_dir, exist_ok=True)
        
        # Files to copy (small files only)
        small_files = [
            "metrics.csv",
            "model_info.txt", 
            "training_summary.txt",
            "training_curves.png"
        ]
        
        for filename in small_files:
            local_file = os.path.join(local_dir, filename)
            if os.path.exists(local_file):
                network_file = os.path.join(network_dir, filename)
                shutil.copy2(local_file, network_file)
                logger.debug(f"Copied {filename} to network drive")
        
        # Copy experiment info if it exists
        exp_info_pattern = os.path.join(local_dir, "*experiment_info.json")
        import glob
        for exp_file in glob.glob(exp_info_pattern):
            filename = os.path.basename(exp_file)
            network_file = os.path.join(network_dir, filename)
            shutil.copy2(exp_file, network_file)
            logger.debug(f"Copied {filename} to network drive")
            
        logger.info(f"Small files copied to network drive: {network_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to copy files to network drive: {e}")


def run_training(cfg: DictConfig):
    # Get Hydra runtime configuration (for saving outputs, etc.)
    hydra_config = HydraConfig.get()

    # --- Determine the primary device ---
    # If CUDA is enabled and available:
    if cfg.general.use_cuda and torch.cuda.is_available():
        if cfg.general.parallel.use_parallel:
            # Use the first device from the parallel device list.
            primary_device = torch.device(f"cuda:{cfg.general.parallel.device_ids[0]}")
            logger.info(f"Parallel mode enabled. Primary device: {primary_device}")
        else:
            # Use the device specified in the config.
            primary_device = torch.device(f"cuda:{cfg.general.device_id}")
            logger.info(f"Non-parallel mode. Using device: {primary_device}")
    else:
        primary_device = torch.device("cpu")
        logger.info("CUDA not available or disabled. Using CPU.")

    # Set seeds for reproducibility.
    torch.manual_seed(cfg.general.seed)
    if primary_device.type == "cuda":
        torch.cuda.manual_seed(cfg.general.seed)

    # --- Prepare Data ---
    logger.info("Preparing data...")
    dataset = get_datasets(cfg)
    dl_train = get_dataloaders(cfg, dataset[0], mode="train")
    dl_val = get_dataloaders(cfg, dataset[1], mode="val")

    # --- Instantiate loss and metric functions ---
    logger.info("Instantiating loss and metric functions...")
    loss_fn = instantiate(cfg.loss).to(primary_device)
    metric_fn = instantiate(cfg.metric).to(primary_device)

    # --- Create training module (which contains your model, optimizer, etc.) ---
    train_module = create_training_module(cfg)

    # --- Set up the model and device ---
    logger.info("Setting up model and moving to device...")
    if cfg.general.use_cuda and cfg.general.parallel.use_parallel:
        # In parallel mode, move the model to the primary device then wrap in DataParallel.
        train_module.model.to(primary_device)
        logger.info(f"Wrapping model in DataParallel on devices: {cfg.general.parallel.device_ids}")
        train_module.model = nn.DataParallel(train_module.model, device_ids=cfg.general.parallel.device_ids)
    else:
        # Non-parallel mode: simply move the model to the chosen device.
        logger.info(f"Using non-parallel mode. Moving model to {primary_device}")
        train_module.model.to(primary_device)

    # Ensure loss and metric functions are on the same device.
    loss_fn.to(primary_device)
    if metric_fn is not None:
        metric_fn.to(primary_device)

    # --- Log hyperparameters ---
    logger.info("Starting training loop with hyperparameters:")
    for k, v in cfg.hparams.items():
        logger.info(f"\t{k}: {v}")

    logger.info(f'Creating plot/save prediction:')
    if cfg.show_prediction:
        show_predictions = {}
        for k, v in cfg.show_prediction.plots.items():
            logger.info(f'\t {k}: {get_instance_repr(v)}')
            show_predictions[k] = instantiate(v)
    else:
        logger.info(f'No plot/save prediction defined')
        show_predictions = None

    # --- Set up MLflow experiment ---
    # Set tracking URI to model_training/mlruns
    mlflow.set_tracking_uri("file:model_training/mlruns")
    my_experiment = mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.run_name
    logger.info(f"Starting experiment: {run_name}")
    
    # --- Create experiment mapping for easy identification ---
    def update_experiment_mapping():
        import json
        import os
        from datetime import datetime
        
        mapping_file = "model_training/mlruns/experiment_mapping.json"
        os.makedirs("model_training/mlruns", exist_ok=True)
        
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
        else:
            mapping = {}
        
        exp_id = str(my_experiment.experiment_id)
        mapping[exp_id] = {
            "name": cfg.mlflow.experiment_name,
            "model": cfg.model.name,
            "dataset": getattr(cfg.dataset, 'name', 'unknown'),
            "run_name": run_name,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        # Also create a mapping file in the experiment directory
        exp_dir = f"model_training/mlruns/{exp_id}"
        if os.path.exists(exp_dir):
            exp_mapping_file = os.path.join(exp_dir, "experiment_info.json")
            exp_info = {
                "experiment_id": exp_id,
                "experiment_name": cfg.mlflow.experiment_name,
                "model": cfg.model.name,
                "dataset": getattr(cfg.dataset, 'name', 'unknown'),
                "run_name": run_name,
                "created": datetime.now().isoformat(),
                "config": {
                    "epochs": cfg.hparams.nb_epochs,
                    "batch_size": cfg.hparams.batch_size,
                    "learning_rate": cfg.hparams.lr,
                    "optimizer": cfg.optimizer.name if hasattr(cfg.optimizer, 'name') else 'unknown'
                }
            }
            with open(exp_mapping_file, 'w') as f:
                json.dump(exp_info, f, indent=2)
            logger.info(f"Created experiment info file: {exp_mapping_file}")
        
        logger.info(f"Experiment mapping updated: ID {exp_id} -> {cfg.model.name}")
    
    update_experiment_mapping()

    with mlflow.start_run(experiment_id=my_experiment.experiment_id, run_name=run_name):
        mlflow.log_params(get_all_hydra_parameters(cfg))
        
        # --- Setup CSV logging ---
        hydra_config = HydraConfig.get()
        csv_path = f"{hydra_config.runtime.output_dir}/metrics.csv"
        logger.info(f"Metrics will be saved to: {csv_path}")
        
        # --- Create comprehensive model info file ---
        dataset_info = {"train_size": len(dl_train) * cfg.hparams.batch_size, "val_size": len(dl_val) * cfg.hparams.batch_size}
        create_model_info_file(hydra_config.runtime.output_dir, cfg, train_module.model, train_module, dataset_info)
        
        # --- Training loop ---
        for epoch in range(1, cfg.hparams.nb_epochs + 1):
            epoch_info = EpochInfo(epoch, cfg.hparams.nb_epochs, cfg.hparams.batch_size * epoch)
            train_module.model.train()
            loss, acc = run_one_epoch(epoch_info, train_module, dl_train, loss_fn, metric_fn, None, csv_path)
            # logger.info(f"Epoch {epoch} TRAIN: Loss = {loss:.4f}, Metric = {acc:.4f}")

            # Validation loss calculation EVERY epoch (lightweight)
            train_module.model.eval()
            with torch.no_grad():
                # Always calculate validation loss but only generate images at intervals
                show_val_predictions = show_predictions if epoch % cfg.hparams.valid_interval == 0 else None
                loss_val, acc_val = run_one_epoch(epoch_info, train_module, dl_val, loss_fn, metric_fn, show_val_predictions, csv_path)
                # logger.info(f"Epoch {epoch} VAL: Loss = {loss_val:.4f}, Metric = {acc_val:.4f}")

            if train_module.scheduler:
                if isinstance(train_module.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    train_module.scheduler.step(loss)
                else:
                    train_module.scheduler.step()

            if epoch % cfg.hparams.valid_interval == 0 and cfg.general.save_model:
                model_path = cfg.general.model_path or f"{hydra_config.runtime.output_dir}/model_{epoch}.pth"
                save_model(train_module.model, model_path, cfg.general.parallel.use_parallel)
                logger.info(f"Saved model checkpoint: {model_path}")

            # Generate plots after every CSV update (every epoch)
            if os.path.exists(csv_path):
                generate_training_plots(csv_path, hydra_config.runtime.output_dir)
                # Copy small files to network drive after every CSV update
                run_name = os.path.basename(hydra_config.runtime.output_dir)
                copy_small_files_to_network(hydra_config.runtime.output_dir, run_name, rank=0)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Generate final plots and copy to network drive at end of training
        if os.path.exists(csv_path):
            logger.info("Generating final training plots...")
            generate_training_plots(csv_path, hydra_config.runtime.output_dir)
            # Final copy to network drive
            run_name = os.path.basename(hydra_config.runtime.output_dir)
            copy_small_files_to_network(hydra_config.runtime.output_dir, run_name, rank=0)




@hydra.main(version_base='1.3', config_path="conf", config_name="config")
def main(cfg: DictConfig):
    try:
        start_training_time = perf_counter()
        run_training(cfg)
        logger.info(f'Experiment finished in {seconds_to_string(perf_counter() - start_training_time)}')

    except Exception as err:
        logger.error(err)


if __name__ == '__main__':
    main()
