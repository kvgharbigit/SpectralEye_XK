import gc
import logging
from time import perf_counter
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from eitools.utils import seconds_to_string
from eitorch.training_loop.hydra_utils import get_all_hydra_parameters, get_instance_repr
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from path import Path
from skimage.io import imsave
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision.transforms import Compose

from src.model_training.plots.plot_bottleneck import display_rgb, display_latent
from src.model_training.train_val.epoch_results import EpochResults

from src.model_training.train_val.run_epoch import run_one_epoch
from eitorch.model_io import save_model_state

from src.model_training.train_val.utils import create_training_module, get_datasets, get_dataloaders, EpochInfo
from src.model_training.utils.save_model import save_model

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


def run_training(cfg: DictConfig):
    hydra_config = HydraConfig.get()
    device = torch.device('cuda') if cfg.general.use_cuda and torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)

    logger.info('Preparing data...')
    dataset = get_datasets(cfg)

    dl_train = get_dataloaders(cfg, dataset[0], mode='train')
    # dl_train = PrefetchLoader(dl_train, device)
    dl_val = get_dataloaders(cfg, dataset[1], mode='val')
    # dl_val = PrefetchLoader(dl_val, device)
    # val_id = dataset[2]

    logger.info('Instantiating loss and metric functions...')
    loss_fn = instantiate(cfg.loss).to(device)
    metric_fn = instantiate(cfg.metric).to(device)

    train_module = create_training_module(cfg)

    logger.info(f'Creating plot/save prediction:')
    if cfg.show_prediction:
        show_predictions = {}
        for k, v in cfg.show_prediction.plots.items():
            logger.info(f'\t {k}: {get_instance_repr(v)}')
            show_predictions[k] = instantiate(v)
    else:
        logger.info(f'No plot/save prediction defined')
        show_predictions = None

    # Move model to device and set parallel mode
    logger.info(f'Creating device and setting parallel mode')
    device = torch.device('cuda') if cfg.general.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Using device: {device}')
    if cfg.general.use_cuda and cfg.general.parallel.use_parallel:
        logger.info(f'Using parallel mode: {cfg.general.parallel.device_ids}')
        train_module.model = nn.DataParallel(train_module.model, device_ids=cfg.general.parallel.device_ids)

    logger.info(f'Moving model to {device}')
    train_module.model.to(device)
    loss_fn.to(device)
    # train_module.optimizer.to(device)
    if metric_fn is not None:
        metric_fn.to(device)

    logger.info(f'Starting training loop with hyperparameters:')
    max_len = max([len(k) for k in cfg.hparams])
    for k, v in cfg.hparams.items():
        logger.info(f'\t {k:<{max_len}}: {v}')


    my_experiment = mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = cfg.mlflow.run_name
    logger.info(f'Starting experiment: {run_name}')

    with mlflow.start_run(experiment_id=my_experiment.experiment_id, run_name=run_name):
        # Log configuration and parameters
        mlflow.log_params(get_all_hydra_parameters(cfg))

        logger.info('Starting training...')
        for epoch in range(1, cfg.hparams.nb_epochs + 1):
            epoch_info = EpochInfo(epoch, cfg.hparams.nb_epochs, cfg.hparams.batch_size * epoch)
            train_module.model.train()
            loss, acc = run_one_epoch(epoch_info, train_module, dl_train, loss_fn, metric_fn, None)

            if epoch % cfg.hparams.valid_interval == 0:
                train_module.model.eval()
                with torch.no_grad():
                    loss_val, acc_val = run_one_epoch(epoch_info, train_module, dl_val, loss_fn, metric_fn, show_predictions)

            if train_module.scheduler:
                if isinstance(train_module.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    train_module.scheduler.step(loss)
                else:
                    train_module.scheduler.step()

            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()

            if epoch % cfg.hparams.valid_interval == 0:
                if cfg.general.save_model:
                    model_path = cfg.general.model_path or rf'{hydra_config.runtime.output_dir}\model_{epoch}.pth'
                    save_model(train_module.model, model_path, cfg.general.parallel.use_parallel)

            # Cleanup memory after epoch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def filter_epoch_results(results: EpochResults, mode: str):
    # Return a copy of epoch result only where the labels match the mode
    labels = results.labels
    latent_outputs = results.spectral_latent_outputs
    rgb_images = results.rgb_images
    mse_spectra = results.mse_spectra
    input_spectra = results.input_spectra
    output_spectra = results.output_spectra
    idx = np.where(labels == mode)
    return EpochResults(
        loss=results.loss,
        metric=results.metric,
        spectral_latent_outputs=latent_outputs[idx],
        rgb_images=rgb_images[idx],
        mse_spectra=mse_spectra[idx],
        input_spectra=input_spectra[idx],
        output_spectra=output_spectra[idx],
        labels=labels[idx]
    )


def log_epoch_results(results: EpochResults, epoch: int, mode: str):
    """Log and print results for an epoch, including task-specific losses."""
    print(f"\n--- {mode.capitalize()} Epoch {epoch} ---")
    print(f"{mode.capitalize()} Loss: {results.loss:.4f}")
    mlflow.log_metric(f"{mode}_loss", results.loss, step=epoch)

    print(f"Segmentation Metric: {results.metric:.4f}")
    mlflow.log_metric(f"{mode}_segmentation_metric", results.metric, step=epoch)


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
