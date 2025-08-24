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
    my_experiment = mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.run_name
    logger.info(f"[Rank {rank}] Starting experiment: {run_name}")
    
    # --- Create experiment mapping for easy identification ---
    if rank == 0:  # Only main process updates mapping
        def update_experiment_mapping():
            import json
            import os
            from datetime import datetime
            
            mapping_file = "mlruns/experiment_mapping.json"
            os.makedirs("mlruns", exist_ok=True)
            
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
            
            logger.info(f"[Rank {rank}] Experiment mapping updated: ID {exp_id} -> {cfg.model.name}")
        
        update_experiment_mapping()

    # Start an mlflow run (only on rank 0 to avoid duplicate runs)
    if rank == 0:
        mlflow_run = mlflow.start_run(experiment_id=my_experiment.experiment_id, run_name=run_name)
        # (Optionally log the entire config or specific parameters)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    else:
        mlflow_run = None
        
    # --- Setup CSV logging (only for main process) ---
    csv_path = None
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
            # Use the same output directory as CSV (reuse timestamp from earlier)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"model_training/working_env/ddp_runs/ddp_run_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            model_path = f"{output_dir}/model_{epoch}.pth"
            save_model(train_module.model, model_path, cfg.general.use_ddp or cfg.general.parallel.use_parallel)
            logger.info(f"Saved model checkpoint: {model_path}")
        
        # Synchronize all processes at the end of each epoch
        if cfg.general.use_ddp:
            dist.barrier()

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
