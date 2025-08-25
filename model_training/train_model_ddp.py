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
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Generating plots from {csv_path} with {len(df)} epochs")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training and Validation Loss
        if 'CustomLoss train' in df.columns:
            ax1.plot(df['step'], df['CustomLoss train'], label='Train Loss', linewidth=2)
        if 'CustomLoss val' in df.columns:
            val_df = df[df['CustomLoss val'].notna()]
            ax1.plot(val_df['step'], val_df['CustomLoss val'], label='Val Loss', 
                    marker='o', markersize=6, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reconstruction MSE
        if 'ReconstructionMSE train' in df.columns:
            ax2.plot(df['step'], df['ReconstructionMSE train'], label='Train MSE', linewidth=2)
        if 'ReconstructionMSE val' in df.columns:
            val_df = df[df['ReconstructionMSE val'].notna()]
            ax2.plot(val_df['step'], val_df['ReconstructionMSE val'], label='Val MSE', 
                    marker='o', markersize=6, linewidth=2)
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
            if 'CustomLoss train' in df.columns and 'CustomLoss val' in df.columns:
                val_df = df[df['CustomLoss val'].notna()]
                gap = val_df['CustomLoss val'] - val_df['CustomLoss train']
                ax4.plot(val_df['step'], gap, color='red', linewidth=2)
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Val Loss - Train Loss')
                ax4.set_title('Generalization Gap')
                ax4.grid(True, alpha=0.3)
        
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
                        "device_ids": getattr(cfg.general.parallel, 'device_ids', [cfg.general.device_id])
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

    nb_epochs = cfg.hparams.nb_epochs
    loss_fn = torch.nn.MSELoss().to(device)
    metric_fn = torch.nn.L1Loss().to(device)

    # Main training loop
    for epoch in range(1, nb_epochs + 1):
        epoch_info = EpochInfo(epoch, nb_epochs, cfg.hparams.batch_size * epoch)

        # Training epoch
        train_module.model.train()
        loss, acc = run_one_epoch(epoch_info, train_module, dl_train, loss_fn, metric_fn, show_predictions, device, csv_path, rank)
        if rank == 0:
            logger.info(f"Epoch {epoch} TRAIN: Loss={loss:.4f}, Metric={acc:.4f}")

        # Validation epoch (if needed)
        if epoch % cfg.hparams.valid_interval == 0:
            train_module.model.eval()
            with torch.no_grad():
                val_loss, val_acc = run_one_epoch(epoch_info, train_module, dl_val, loss_fn, metric_fn, show_predictions, device, csv_path, rank)
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
                
                # Generate plots after each validation
                if csv_path and os.path.exists(csv_path):
                    generate_training_plots(csv_path, output_dir)
        
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
