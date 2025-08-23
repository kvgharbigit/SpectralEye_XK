import logging
from collections import namedtuple

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import Compose

logger = logging.getLogger()

EpochInfo = namedtuple(typename='EpochInfo', field_names=['epoch', 'nb_epochs', 'nb_steps'])


class TrainingModule:
    def __init__(self, model, optimizer, scheduler, use_ddp=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_ddp = use_ddp

    def zero_grad(self):
        if not self.model.training:
            return

        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss):
        if not self.model.training:
            return
        loss.backward()
        self.optimizer.step()
        if self.use_ddp:
            torch.cuda.synchronize()


def get_model(cfg, use_ddp=False):
    model = instantiate(cfg.model.model)
    if cfg.general.use_pretrained:
        model.load_state_dict(torch.load(cfg.general.pretrained_path, map_location="cpu"))

    if cfg.general.use_cuda:
        model = model.to("cuda")

    if cfg.general.parallel.use_parallel and not use_ddp:
        model = nn.DataParallel(model, device_ids=cfg.general.parallel.device_ids)

    return model


def create_training_module(cfg, use_ddp=False):
    model = get_model(cfg, use_ddp)
    optimizer = instantiate(cfg.optimizer)(model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer) if cfg.scheduler else None
    return TrainingModule(model, optimizer, scheduler, use_ddp)


def get_datasets(cfg: DictConfig):
    transform = get_transform(cfg.augmentation) if cfg.augmentation.use_transform else None
    dataset_fn = instantiate(cfg.dataset)
    return dataset_fn(transform=transform)


def get_dataloaders(cfg, dataset, mode='train', use_ddp=False):
    batch_size = cfg.hparams.batch_size
    num_workers = cfg.dataloader.num_workers
    pin_memory = cfg.dataloader.pin_memory
    prefetch_factor = cfg.dataloader.prefetch_factor
    persistent_workers = cfg.dataloader.persistent_workers

    if mode == 'train':
        shuffle = not use_ddp  # In DDP, shuffling should be handled by DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
        drop_last = True  # Prevents uneven batches in DDP
    elif mode == 'val':
        shuffle = False
        sampler = DistributedSampler(dataset, shuffle=False) if use_ddp else None
        drop_last = False
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        sampler=sampler
    )


def get_transform(cfg):
    return Compose([instantiate(tr) for elem in cfg.transform for tr in elem.values()])
