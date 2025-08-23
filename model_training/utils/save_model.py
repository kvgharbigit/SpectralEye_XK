from collections import OrderedDict

import torch
import torch.nn as nn


def save_model(model: nn.Module, file_path: str, use_parallel: bool) -> None:
    """ Save the model to the given path, if parallel is used removed the "module." from each key. """

    state_dict = model.state_dict()
    if use_parallel:
        state_dict = OrderedDict({k[7:]: v for k, v in state_dict.items()})

    torch.save(state_dict, file_path)
