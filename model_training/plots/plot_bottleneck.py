
""" Defines the function to plot segmentation data. """
from operator import index

import pandas as pd
import torch
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from torch import nn

from model_training.train_val.epoch_results import EpochResults
import numpy as np
from matplotlib import pyplot as plt

from .plot_utils import (GridLayout, convert_if_one_hot, get_random_frame_indexes, get_colored_image, get_colored_label,
                         create_oriented_fig, cast_to_uint8)

__all__ = ['plot_seg_labeled_images', 'plot_seg_predictions', 'plot_seg_probabilities', 'plot_seg_all']


def plot_seg_labeled_images(images: np.ndarray, targets: np.ndarray, filenames: list[str], *, nb_classes: int = -1,
                            nb_to_plot: int = -1, layout: GridLayout = 'horizontal', suptitle: str = '',
                            size: int = 2) -> plt.Figure:
    """ Plot the image with the prediction overlayed. """

    # Ensure the inputs are consistent
    assert len(images) == len(targets) == len(
        filenames), "The number of images, targets, and filenames must be the same."

    # Get indexes to plot
    frame_indexes = get_random_frame_indexes(len(images), nb_to_plot)

    # Create the arrays and titles
    arrays, titles = [], []
    for idx in frame_indexes:
        image = get_colored_image(images[idx])
        target = convert_if_one_hot(targets[idx])
        label = get_colored_label(target, nb_classes=nb_classes)

        # Combine the original image and the colored prediction
        overlay_image = np.clip(0.7 * image + 0.3 * label, a_min=0, a_max=255)
        arrays.append(overlay_image.astype(np.uint8))

        titles.append(filenames[idx])

    return create_oriented_fig(layout, 1, arrays, titles, suptitle, size)


def plot_seg_predictions(images: np.ndarray, targets: np.ndarray, predictions: np.ndarray, filenames: list[str], *,
                         nb_to_plot: int = -1, layout: GridLayout = 'horizontal',
                         suptitle: str = '', size: int = 2) -> plt.Figure:
    """ Create a figure displaying images, targets, predictions, and their differences. """

    # Ensure the inputs are consistent
    assert len(images) == len(targets) == len(predictions) == len(filenames), \
        "The number of images, targets, predictions, and image names must be the same."

    # Get indexes to plot
    frame_indexes = get_random_frame_indexes(len(images), nb_to_plot)

    # Create the arrays and titles
    nb_classes = predictions.shape[1]
    chunk_size = 4
    arrays, titles = [], []
    for idx in frame_indexes:
        image = get_colored_image(images[idx])
        target = convert_if_one_hot(targets[idx])
        prediction = np.argmax(predictions[idx], axis=0)
        diff = prediction != target
        arrays.extend([image, get_colored_label(target, nb_classes), get_colored_label(prediction, nb_classes), diff])
        titles.extend([filenames[idx], 'Target', 'Prediction', f'Diff ({np.sum(diff)})'])

    return create_oriented_fig(layout, chunk_size, arrays, titles, suptitle, size)


def plot_seg_probabilities(images: np.ndarray, targets: np.ndarray, predictions: np.ndarray, filenames: list[str], *,
                           nb_to_plot: int = -1, layout: GridLayout = 'horizontal',
                           suptitle: str = '', size: int = 2) -> plt.Figure:
    """ Create a figure displaying images, targets, and prediction probabilty map for each class. """

    # Ensure the inputs are consistent
    assert len(images) == len(targets) == len(predictions) == len(filenames), \
        "The number of images, targets, predictions, and image names must be the same."

    # Get indexes to plot
    frame_indexes = get_random_frame_indexes(len(images), nb_to_plot)

    # Create the arrays and titles
    nb_classes = predictions.shape[1]
    chunk_size = 2 + nb_classes
    arrays, titles = [], []
    for idx in frame_indexes:
        image = get_colored_image(images[idx])
        target = convert_if_one_hot(targets[idx])
        arrays.extend([image, get_colored_label(target, nb_classes)])
        titles.extend([filenames[idx], 'Target'])
        for cl in range(nb_classes):
            prediction = predictions[idx][cl]
            arrays.append(cast_to_uint8(prediction))
            titles.append(f'Prediction cl {cl}')

    return create_oriented_fig(layout, chunk_size, arrays, titles, suptitle, size)



def plot_seg_all(images: np.ndarray, targets: np.ndarray, predictions: np.ndarray, filenames: list[str], *,
                 nb_to_plot: int = -1, layout: GridLayout = 'horizontal',
                 suptitle: str = '', size: int = 2) -> plt.Figure:
    """ Create a figure displaying images, targets, and prediction probabilty map for each class. """

    # Ensure the inputs are consistent
    assert len(images) == len(targets) == len(predictions) == len(filenames), \
        "The number of images, targets, predictions, and image names must be the same."

    # Get indexes to plot
    frame_indexes = get_random_frame_indexes(len(images), nb_to_plot)

    # Create the arrays and titles
    nb_classes = predictions.shape[1]
    chunk_size = 3 + nb_classes
    arrays, titles = [], []
    for idx in frame_indexes:
        image = get_colored_image(images[idx])
        target = get_colored_label(convert_if_one_hot(targets[idx]), nb_classes)
        prediction = np.argmax(predictions[idx], axis=0)
        arrays.extend([image, target, get_colored_label(prediction, nb_classes)])
        titles.extend([filenames[idx], 'Target', 'Prediction'])
        for cl in range(nb_classes):
            prediction = predictions[idx][cl]
            arrays.append(cast_to_uint8(prediction))
            titles.append(f'Proba class {cl}')

    return create_oriented_fig(layout, chunk_size, arrays, titles, suptitle, size)


def display_rgb(hs_cubes, reconstructed_outputs, rgb_images, labels, nb_to_plot=5):
    frame_indexes = get_random_frame_indexes(len(hs_cubes), nb_to_plot)
    nb_to_plot = len(frame_indexes)
    wl = np.linspace(450, 900, 91)
    wl = wl[np.r_[0:58:2, 80]]

    fig = plt.figure(figsize=(12, nb_to_plot * 3))  # Adjust the figure size
    gs = GridSpec(nb_to_plot, 4, figure=fig, wspace=0.3, hspace=0.3)  # GridSpec for layout
    mse = nn.MSELoss(reduction='none')

    # Avoid creating new tensors - compute MSE directly on numpy arrays or use existing tensors
    if isinstance(reconstructed_outputs, np.ndarray) and isinstance(hs_cubes, np.ndarray):
        mse_spectra = np.mean((reconstructed_outputs - hs_cubes) ** 2, axis=(2, 3))
    else:
        # If they're already tensors, use them directly
        mse_spectra = mse(reconstructed_outputs, hs_cubes).mean(dim=(2, 3)).cpu().detach().numpy()

    input_spectra = hs_cubes.mean(axis=(2, 3))
    output_spectra = reconstructed_outputs.mean(axis=(2, 3))

    for cpt, f_index in enumerate(frame_indexes):
        reconstructed_output = reconstructed_outputs[f_index].transpose(1, 2, 0)
        rgb = rgb_images[f_index].transpose(1, 2, 0)
        mse = mse_spectra[f_index]
        input_spectrum = input_spectra[f_index]
        output_spectrum = output_spectra[f_index]

        # Keep channels
        reconstructed_rgb = reconstructed_output[:, :, [12, 8, 1]]

        # Column 1: Reconstructed RGB
        ax = fig.add_subplot(gs[cpt, 0])
        ax.imshow(np.clip(reconstructed_rgb, 0, 1))
        ax.axis('off')
        ax.set_title("Reconstructed RGB", fontsize=10)

        # Column 2: Original RGB
        ax = fig.add_subplot(gs[cpt, 1])
        ax.imshow(np.clip(rgb, 0, 1))
        ax.axis('off')
        ax.set_title("Original RGB", fontsize=10)

        # Column 3: MSE Plot
        ax = fig.add_subplot(gs[cpt, 2])
        ax.plot(wl, mse, 'b')
        ax.set_title("MSE per Wavelength", fontsize=10)
        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.set_ylabel("MSE", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Column 4: Input vs Output Spectrum
        ax = fig.add_subplot(gs[cpt, 3])
        ax.plot(wl, input_spectrum, 'k', label='Input')
        ax.plot(wl, output_spectrum, 'r', label='Output')
        # ax.set_ylim([-1.0, 1.0])
        ax.set_title("Spectral Comparison", fontsize=10)
        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    return fig


def display_latent(train_results: EpochResults, val_results: EpochResults, epoch: int, num_epochs: int):
    nb_image = 7
    nb_latent = train_results.spectral_latent_outputs[0].shape[0]
    fig = plt.figure(figsize=(nb_image * 2, nb_latent * 2))
    grid = gridspec.GridSpec(nb_latent, nb_image * 2, figure=fig)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    id_training = np.random.permutation(len(train_results.spectral_latent_outputs))
    if len(train_results.spectral_latent_outputs) < nb_image:
        nb_image = len(train_results.spectral_latent_outputs)
    id_val = np.random.permutation(len(val_results.spectral_latent_outputs))
    nb_image = min(nb_image, len(val_results.spectral_latent_outputs))
    id_val = id_val[:nb_image]
    id_training = id_training[:nb_image]

    for ii in range(nb_image):
        # train_name = train_results.names[id_training[ii]]
        train_latent = train_results.spectral_latent_outputs[id_training[ii]]
        val_latent = val_results.spectral_latent_outputs[id_val[ii]]

        for jj in range(nb_latent):
            ax = plt.subplot(grid[jj, ii])
            ax.imshow(train_latent[jj], cmap='gray')
            ax.axis('off')

            ax = plt.subplot(grid[jj, nb_image + ii])
            ax.imshow(val_latent[jj], cmap='gray')
            ax.axis('off')


    fig.suptitle(f'Epoch {epoch + 1}/{num_epochs}', fontsize=16)
    plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
    plt.subplots_adjust(top=0.9)  # Adjust the top to make room for the suptitle

    return fig