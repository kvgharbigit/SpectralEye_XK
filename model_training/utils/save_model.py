from collections import OrderedDict

import torch
import torch.nn as nn


def save_model(model: nn.Module, file_path: str, use_parallel: bool) -> None:
    """ Save the model to the given path, if parallel is used removed the "module." from each key. """

    state_dict = model.state_dict()
    # Handle both DataParallel and DDP which add "module." prefix
    if use_parallel or any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = OrderedDict({k.replace('module.', ''): v for k, v in state_dict.items()})

    torch.save(state_dict, file_path)
