""" """

import logging
import time
from collections import namedtuple

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from path import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from eitorch.model_io import load_model_with_loader
from eitorch.training.config.utils import get_instance_repr
from eitools.utils.terminal_print import seconds_to_string

logger = logging.getLogger()

EpochInfo = namedtuple(typename='EpochInfo', field_names=['epoch', 'nb_epochs', 'nb_steps'])


class TrainingModule:

    def __init__(self, model: nn.Module, optimizer: nn.Module, scheduler: nn.Module, use_amp: bool = False):
        self.model: nn.Module = model
        self.optimizer: nn.Module = optimizer
        self.scheduler: nn.Module = scheduler
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def zero_grad(self):
        if not self.model.training:
            return

        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: torch.Tensor):
        if not self.model.training:
            return

        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


def print_overridden_parameters() -> str:
    """ Print the override parameters and return the name of the task.

    :return: The name of the task composed of all the param names that are overridden.
    """

    hydra_config = HydraConfig.get()
    override = hydra_config.overrides.task
    name = '__'.join([e.split('=')[0].split('.')[-1] for e in override])
    if override:
        logger.info(f'Overrides - {name}:')
        for elem in override:
            logger.info(f'\t{elem}')

    return name



def get_model(cfg) -> nn.Module:
    """ Create the model to train. """

    model = instantiate(cfg.model.model)
    logger.info(f'\t {get_instance_repr(cfg.model.model)}')
    if cfg.general.use_pretrained:
        model_file = Path(cfg.general.pretrained_path)
        if model_file.exists():
            logger.info(f'Loading pretrained model from {model_file}')
            model.load_state_dict(torch.load(model_file, weights_only=True))
        else:
            logger.warning(f'Pretrained model not found at {model_file}')

    return model


def create_training_module(cfg) -> TrainingModule:
    """ Create the model, optimizer and scheduler. """

    logger.info(f'Creating model:')
    model = get_model(cfg)

    logger.info(f'Creating optimizer:')
    logger.info(f'\t {get_instance_repr(cfg.optimizer)}')
    optim_partial = instantiate(cfg.optimizer)
    optimizer = optim_partial(model.parameters())

    logger.info(f'Creating scheduler:')
    if cfg.scheduler:
        logger.info(f'\t {get_instance_repr(cfg.scheduler)}')
        scheduler = instantiate(cfg.scheduler, optimizer)
    else:
        logger.info(f'\t No scheduler defined')
        scheduler = None

    use_amp = getattr(cfg.general, 'use_amp', False)
    return TrainingModule(model, optimizer, scheduler, use_amp)



def get_datasets(cfg: DictConfig):
    transform = get_transform(cfg.augmentation) if cfg.augmentation.use_transform else None
    dataset_fn = instantiate(cfg.dataset)
    datasets = dataset_fn(transform=transform)
    return datasets


def get_dataloaders(cfg: DictConfig, dataset, mode='train'):
    batch_size = cfg.hparams.batch_size
    shuffle = cfg.dataloader.shuffle
    num_workers = cfg.dataloader.num_workers
    pin_memory = cfg.dataloader.pin_memory
    prefetch_factor = cfg.dataloader.prefetch_factor
    persistent_workers = cfg.dataloader.persistent_workers

    if mode == 'train':
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 prefetch_factor=prefetch_factor,
                                 persistent_workers=persistent_workers)

    elif mode == 'val':
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 prefetch_factor=prefetch_factor,
                                 persistent_workers=persistent_workers)
    else:
        raise ValueError(f"Invalid mode: {mode}: must be 'train' or 'val'.")

    return data_loader



def get_transform(cfg) -> Compose:
    """ Create the transformation to apply to the data. """

    logger.info(f'Creating transformations:')
    transform = Compose([instantiate(tr) for elem in cfg.transform for tr in elem.values()])

    # Log transform
    max_len = max([len(k) for tr in cfg.transform for k in tr.keys()])
    for tr in cfg.transform:
        for k, v in tr.items():
            params = ', '.join([f'{name}: {value}' for name, value in v.items() if not name.startswith('_')])
            logger.info(f'\t {k:<{max_len}} | {params}')

    return transform

