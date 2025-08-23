""" """

import logging
from time import perf_counter
from typing import Callable

import mlflow
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from eitorch.utils.terminal_print import seconds_to_string, progress_bar
from eitorch.training_loop.segmentation.utils import EpochInfo, TrainingModule
from eitorch.plot import ShowPredictionFct
from eitorch.metrics import MetricClass

logger = logging.getLogger()

CleanPredictionFct = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def move_data_to_device(inputs: torch.Tensor,
                        targets: torch.Tensor,
                        model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """ Move the data to the device of the model.

    :param inputs: The input data.
    :param targets: The target data.
    :param model: The model.
    :return: The input and target data moved to the device of the model.
    """

    if next(model.parameters()).is_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()

    return inputs, targets


def forward_pass(model: nn.Module,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 pp_fct: CleanPredictionFct | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """ Forward pass of the model.

    :param model: The model.
    :param inputs: The input data.
    :param targets: The target data.
    :param pp_fct: A function to post-process the predictions.
    :return: The predictions and the targets.
    """

    predictions = model(inputs)

    if pp_fct is not None:
        if isinstance(predictions, tuple):
            for pred_index in range(len(predictions)):
                predictions[pred_index], targets = pp_fct(predictions[pred_index], targets)
        else:
            predictions, targets = pp_fct(predictions, targets)

    return predictions, targets


def calculate_loss(predictions: torch.Tensor | tuple[torch.Tensor],
                   targets: torch.Tensor,
                   criterion: nn.Module) -> torch.Tensor:
    """ Calculate the loss.

    :param predictions: The predictions.
    :param targets: The target data.
    :param criterion: The loss function.
    :return: The loss.
    """

    if isinstance(predictions, tuple):
        loss = sum(criterion(pred, targets) for pred in predictions)
    else:
        loss = criterion(predictions, targets)

    return loss


def calculate_accuracy(predictions: torch.Tensor | tuple[torch.Tensor],
                       targets: torch.Tensor,
                       metric: MetricClass | None) -> float:
    """ Calculate the accuracy.

    :param predictions: The predictions.
    :param targets: The target data.
    :param metric: The accuracy function.
    :return: The accuracy.
    """

    if metric is None:
        return 0.0

    if isinstance(predictions, tuple):
        predictions = predictions[-1]

    return metric(predictions, targets)


def log_metrics(metrics: dict[str, float], current_step: int) -> None:
    """ Log the metrics in mlflow. """

    for key, value in metrics.items():
        mlflow.log_metric(key=f"{key}", value=value, step=current_step)


def run_one_epoch(epoch_info: EpochInfo,
                  train_module: TrainingModule,
                  loader: DataLoader,
                  criterion: nn.Module,
                  metric: MetricClass | None = None,
                  pp_fct: CleanPredictionFct | None = None,
                  show_predictions: dict[str, ShowPredictionFct] | None = None
                  ) -> tuple[float, float]:
    """ """

    t_start_epoch = perf_counter()
    mode = 'Train' if train_module.model.training else 'Val'

    total_loss, total_acc = 0, 0
    nb_batch = len(loader)

    for i, (inputs, targets, file_names) in enumerate(loader, 1):
        # Move data to device
        inputs, targets = move_data_to_device(inputs, targets, train_module.model)

        # Zero the parameter gradients
        train_module.zero_grad()

        # Forward pass
        predictions, targets = forward_pass(train_module.model, inputs, targets, pp_fct)

        # Calculate loss
        loss = calculate_loss(predictions, targets, criterion)
        loss_value = loss.detach().item()

        # Backward pass
        train_module.backward(loss)

        # Calculate acc
        acc = calculate_accuracy(predictions, targets, metric)

        # Update total loss and acc
        total_loss += loss_value
        total_acc += acc

        # Plot the prediction
        if show_predictions:
            if isinstance(predictions, tuple):
                predictions = predictions[-1]

            inputs = inputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()

            for plot_name, show_prediction in show_predictions.items():
                fig = show_prediction(inputs, targets, predictions, file_names)
                mlflow.log_figure(fig, artifact_file=f'{epoch_info.epoch:0>4}_{i:0>3}_{plot_name}.png')
                plt.close(fig)

        # Update progress bar
        progress_bar(i - 1, nb_batch, msg=f'[{i}/{nb_batch}] Loss: {loss_value:.3e}, Acc: {acc:.2f}', colored=False)

    # Clear the progress bar
    print('\r', end='')

    loss, acc = total_loss / len(loader), total_acc / len(loader)

    # log metric
    log_values = {
        f'{criterion.__class__.__name__} {mode}': round(loss, 3),
        f'{metric.__class__.__name__} {mode}': round(acc, 3),
    }
    if mode == 'Train':
        log_values['Learning Rate'] = train_module.optimizer.param_groups[0]['lr']
    log_metrics(log_values, epoch_info.epoch)

    elapsed_time = seconds_to_string(perf_counter() - t_start_epoch)

    if mode == 'Train':
        prefix = f'[{epoch_info.epoch}/{epoch_info.nb_epochs}] TRAIN -'
    else:
        spaces = ' ' * len(f'[{epoch_info.epoch}/{epoch_info.nb_epochs}]')
        prefix = f'{spaces} VALID -'

    logger.info(f'{prefix} Loss: {loss:.3e}, Acc: {acc:.3f} ({elapsed_time})')

    return loss, acc

