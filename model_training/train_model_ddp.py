#!/usr/bin/env python
# IMPORTANT: This file should be run as your training entry point.

import os
# Set the environment variable for libuv BEFORE any torch imports.
os.environ["TORCH_DISTRIBUTED_USE_LIBUV"] = "0"
# (Optional) Register a custom resolver for time-based interpolations.
from omegaconf import OmegaConf
from datetime import datetime
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))

import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Set the sharing strategy to use the file system
mp.set_sharing_strategy('file_system')
import torch.nn as nn
import mlflow
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import your custom modules.
# Adjust the import paths based on your project structure.
from model_training.train_val.run_epoch_ddp import run_one_epoch
from model_training.train_val.utils import create_training_module, get_datasets, EpochInfo
from model_training.train_val.utils_ddp import get_dataloaders
from model_training.utils.save_model import save_model
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def create_model_info_file(output_dir: str, cfg: DictConfig, model, train_module, dataset_info: dict = None) -> None:
    """Create a comprehensive model_info.txt file with all training details"""
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
        try:
            if hasattr(cfg.model, 'img_size'):
                f.write("Model Configuration:\n")
                for key, value in cfg.model.items():
                    if key != '_target_' and key != 'name':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        except Exception as e:
            f.write(f"Model config (could not parse): {cfg.model}\n\n")
        
        # Training Configuration
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        try:
            f.write(f"Epochs: {cfg.hparams.nb_epochs}\n")
            f.write(f"Batch size: {cfg.hparams.batch_size}\n")
            f.write(f"Learning rate: {cfg.hparams.lr}\n")
            f.write(f"Validation interval: {cfg.hparams.valid_interval}\n")
            if hasattr(cfg.hparams, 'accumulate_grad_batches'):
                f.write(f"Gradient accumulation: {cfg.hparams.accumulate_grad_batches}\n")
        except Exception as e:
            f.write(f"Training config (could not parse): {cfg.hparams}\n")
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


def run_training(rank: int, world_size: int, cfg: DictConfig) -> None:
    """
    Training function to be run in each process.
    This function sets the proper device, initializes DDP (if enabled),
    and calls run_one_epoch (passing the device so that tensors are moved correctly).
    """
    # Each spawned process uses its corresponding GPU (or CPU if not available)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.general.seed)
    logger.info(f"[Rank {rank}] Running on device {device}")
    
    # Set MLflow tracking URI in each process to ensure proper connection
    # This is critical for DDP as spawned processes don't inherit the MLflow context
    mlflow.set_tracking_uri("file:model_training/mlruns")
    logger.info(f"[Rank {rank}] MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    # Initialize DDP if enabled
    if cfg.general.use_ddp:
        # Force disable libuv for Windows compatibility
        os.environ["USE_LIBUV"] = "0"
        os.environ["GLOO_SOCKET_IFNAME"] = ""
        
        # Use TCP store directly with correct parameters
        from torch.distributed import TCPStore
        import datetime
        timeout = datetime.timedelta(seconds=300)
        
        if rank == 0:
            store = TCPStore("127.0.0.1", 29500, world_size=world_size, is_master=True, 
                           timeout=timeout, wait_for_workers=True, use_libuv=False)
        else:
            store = TCPStore("127.0.0.1", 29500, world_size=world_size, is_master=False, 
                           timeout=timeout, wait_for_workers=True, use_libuv=False)
        
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=rank,
            world_size=world_size
        )

    # Load dataset and dataloaders
    dataset = get_datasets(cfg)
    dl_train = get_dataloaders(cfg, dataset[0], mode="train", use_ddp=cfg.general.use_ddp)
    dl_val   = get_dataloaders(cfg, dataset[1], mode="val", use_ddp=cfg.general.use_ddp)

    # Create the training module (this should include the model, optimizer, scheduler, etc.)
    train_module = create_training_module(cfg)
    train_module.model.to(device)

    # Wrap the model in DistributedDataParallel if using DDP
    if cfg.general.use_ddp:
        train_module.model = nn.parallel.DistributedDataParallel(train_module.model, device_ids=[rank])

    # Import instantiate at the beginning
    from hydra.utils import instantiate
    
    # Optionally instantiate prediction functions from your configuration.
    if cfg.show_prediction:
        show_predictions = {}
        for key, pred_cfg in cfg.show_prediction.plots.items():
            logger.info(f"[Rank {rank}] Creating prediction: {key}")
            show_predictions[key] = instantiate(pred_cfg)
    else:
        show_predictions = None

    # Set up mlflow experiment and run name (custom resolvers like ${now:...} work now)
    # Only rank 0 creates/sets the experiment to avoid race conditions
    if rank == 0:
        my_experiment = mlflow.set_experiment(cfg.mlflow.experiment_name)
        run_name = cfg.mlflow.run_name
        logger.info(f"[Rank {rank}] Starting experiment: {run_name}")
    else:
        my_experiment = None
        run_name = None
    
    # --- Create experiment mapping for easy identification ---
    if rank == 0:  # Only main process updates mapping
        def update_experiment_mapping():
            import json
            import os
            from datetime import datetime
            
            # Update global mapping file
            mapping_file = "model_training/mlruns/experiment_mapping.json"
            os.makedirs("model_training/mlruns", exist_ok=True)
            
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
            else:
                mapping = {}
            
            exp_id = str(my_experiment.experiment_id)
            now = datetime.now()
            
            mapping[exp_id] = {
                "name": cfg.mlflow.experiment_name,
                "model": cfg.model.name,
                "dataset": getattr(cfg.dataset, 'name', 'unknown'),
                "run_name": run_name,
                "last_updated": now.isoformat()
            }
            
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            
            # Create detailed experiment info in the experiment's directory
            exp_dir = f"model_training/mlruns/{exp_id}"
            if os.path.exists(exp_dir):
                exp_info_file = os.path.join(exp_dir, "experiment_info.json")
                
                # Convert MLflow timestamps to human-readable format
                creation_time_ms = my_experiment.creation_time
                creation_time = datetime.fromtimestamp(creation_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
                last_update_ms = my_experiment.last_update_time
                last_update_time = datetime.fromtimestamp(last_update_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
                
                exp_info = {
                    "experiment_id": exp_id,
                    "name": cfg.mlflow.experiment_name,
                    "model": cfg.model.name,
                    "dataset": getattr(cfg.dataset, 'name', 'unknown'),
                    "artifact_location": my_experiment.artifact_location,
                    "lifecycle_stage": my_experiment.lifecycle_stage,
                    "creation_time": creation_time,
                    "creation_time_ms": creation_time_ms,
                    "last_update_time": last_update_time,
                    "last_update_time_ms": last_update_ms,
                    "current_run_name": run_name,
                    "current_run_start": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": {
                        "batch_size": cfg.hparams.batch_size,
                        "learning_rate": cfg.hparams.lr,
                        "epochs": cfg.hparams.nb_epochs,
                        "valid_interval": cfg.hparams.valid_interval,
                        "optimizer": getattr(cfg.optimizer, '_target_', 'unknown').split('.')[-1],
                        "loss": getattr(cfg.loss, '_target_', 'unknown').split('.')[-1],
                        "use_ddp": cfg.general.use_ddp,
                        "device_ids": list(getattr(cfg.general.parallel, 'device_ids', [cfg.general.device_id]))
                    }
                }
                
                with open(exp_info_file, 'w') as f:
                    json.dump(exp_info, f, indent=2)
                
                logger.info(f"[Rank {rank}] Created experiment info: {exp_info_file}")
            
            logger.info(f"[Rank {rank}] Experiment mapping updated: ID {exp_id} -> {cfg.model.name}")
        
        update_experiment_mapping()

    # Start an mlflow run (only on rank 0 to avoid duplicate runs)
    if rank == 0:
        mlflow_run = mlflow.start_run(experiment_id=my_experiment.experiment_id, run_name=run_name)
        logger.info(f"[Rank {rank}] Started MLflow run: {mlflow_run.info.run_id}")
        logger.info(f"[Rank {rank}] MLflow experiment ID: {my_experiment.experiment_id}")
        logger.info(f"[Rank {rank}] MLflow tracking URI: {mlflow.get_tracking_uri()}")
        # (Optionally log the entire config or specific parameters)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        logger.info(f"[Rank {rank}] Logged parameters to MLflow")
    else:
        mlflow_run = None
        
    # --- Setup CSV logging (only for main process) ---
    csv_path = None
    output_dir = None
    if rank == 0:  # Only main process creates CSV
        # Create a simple output directory since HydraConfig isn't available in spawned processes
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"model_training/working_env/ddp_runs/ddp_run_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{output_dir}/metrics.csv"
        logger.info(f"[Rank {rank}] Metrics will be saved to: {csv_path}")
        
        # Create comprehensive model info file
        dataset_info = {"train_size": len(dl_train) * cfg.hparams.batch_size, "val_size": len(dl_val) * cfg.hparams.batch_size}
        create_model_info_file(output_dir, cfg, train_module.model, train_module, dataset_info)

    nb_epochs = cfg.hparams.nb_epochs
    
    # Instantiate loss and metric functions from config
    loss_fn = instantiate(cfg.loss).to(device)
    metric_fn = instantiate(cfg.metric).to(device)

    # Main training loop
    for epoch in range(1, nb_epochs + 1):
        epoch_info = EpochInfo(epoch, nb_epochs, cfg.hparams.batch_size * epoch)

        # Training epoch
        train_module.model.train()
        loss, acc = run_one_epoch(epoch_info, train_module, dl_train, loss_fn, metric_fn, show_predictions, device, csv_path, rank)
        if rank == 0:
            logger.info(f"Epoch {epoch} TRAIN: Loss={loss:.4f}, Metric={acc:.4f}")

        # Validation loss calculation EVERY epoch (lightweight)
        train_module.model.eval()
        with torch.no_grad():
            # Always calculate validation loss but only generate images at intervals
            show_val_predictions = show_predictions if epoch % cfg.hparams.valid_interval == 0 else None
            val_loss, val_acc = run_one_epoch(epoch_info, train_module, dl_val, loss_fn, metric_fn, show_val_predictions, device, csv_path, rank)
        if rank == 0:
            logger.info(f"Epoch {epoch} VAL: Loss={val_loss:.4f}, Metric={val_acc:.4f}")

        # (Optional) Step the scheduler
        if hasattr(train_module, "scheduler") and train_module.scheduler:
            train_module.scheduler.step(loss)

        # (Optional) Save the model checkpoint at intervals (only rank 0)
        if epoch % cfg.hparams.valid_interval == 0 and cfg.general.save_model and rank == 0:
            # Use the same output directory as CSV
            if output_dir:
                model_path = f"{output_dir}/model_{epoch}.pth"
                save_model(train_module.model, model_path, cfg.general.use_ddp or cfg.general.parallel.use_parallel)
                logger.info(f"Saved model checkpoint: {model_path}")
        
        # Generate plots and copy to server drive after every CSV update (every epoch)
        if rank == 0 and csv_path and output_dir and os.path.exists(csv_path):
            generate_training_plots(csv_path, output_dir)
            # Copy small files to network drive after every CSV update
            run_name = os.path.basename(output_dir)
            copy_small_files_to_network(output_dir, run_name, rank)
        
        # Clean up memory before synchronization
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Synchronize all processes at the end of each epoch
        if cfg.general.use_ddp:
            dist.barrier()

    # Generate final plots before ending (only rank 0)
    if rank == 0 and csv_path and output_dir and os.path.exists(csv_path):
        logger.info("Generating final training plots...")
        generate_training_plots(csv_path, output_dir)
        # Final copy to network drive
        run_name = os.path.basename(output_dir)
        copy_small_files_to_network(output_dir, run_name, rank)
    
    # End MLflow run if we started one
    if rank == 0 and mlflow_run:
        mlflow.end_run()
    
    # Clean up distributed processes if using DDP
    if cfg.general.use_ddp:
        torch.distributed.destroy_process_group()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra main function that spawns the training processes.
    If DDP is enabled, this spawns one process per GPU.
    Otherwise, it runs in a single process.
    """
    if cfg.general.use_ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(run_training, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        run_training(0, 1, cfg)


if __name__ == "__main__":
    main()
