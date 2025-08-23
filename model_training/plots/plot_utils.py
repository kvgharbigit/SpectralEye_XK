
""" Defines the function to plot segmentation data. """

from typing import Literal, Any

import numpy as np
from matplotlib import pyplot as plt

__all__ = ['GridLayout', 'convert_if_one_hot', 'ensure_3_channel_image', 'cast_to_uint8', 'get_3_channels_colors',
           'get_colored_image', 'get_colored_label', 'get_random_frame_indexes',
           'create_fig_with_layout', 'reorder_for_layout', 'add_images_to_fig', 'create_oriented_fig']

GridLayout = Literal['horizontal', 'vertical', 'grid']


def convert_if_one_hot(array: np.ndarray) -> np.ndarray:
    """ Get the target array managing the case when it is a one_hot array. """

    if array.ndim == 3:
        array = np.argmax(array, axis=0)

    return array


def ensure_3_channel_image(image: np.ndarray) -> np.ndarray:
    """Ensure the image has exactly 3 channels.

    :param image: The input image as a numpy array.
    :return: The image with 3 channels.
    :raises ValueError: If the image has more than 3 channels.
    """

    # If the image has less than 3 dimensions, add a channel dimension
    if image.ndim < 3:
        image = image.reshape(image.shape[0], image.shape[1], 1)

    # If the image has a single channel, repeat it to make 3 channels
    if image.shape[2] == 1:
        image = np.repeat(image, repeats=3, axis=2)

    elif image.shape[2] == 2:
        image = np.concatenate([image, np.zeros_like(image[:, :, 0:1])], axis=2)

    # If the image has more than 3 channels, raise an error
    elif image.shape[2] != 3:
        raise ValueError(f"Image should have 1 or 3 channels: {image.shape[2]}")

    return image


def cast_to_uint8(image: np.ndarray) -> np.ndarray:
    """ Normalize and cast an image to uint8.

    :param image: The input image as a numpy array.
    :return: The image normalized and cast to uint8.
    """

    # Check there is only 0
    if np.all(image == 0):
        return np.zeros_like(image, dtype=np.uint8)

    # Normalize the image to the range [0, 1]
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Scale to the range [0, 255] and cast to uint8
    uint8_image = (normalized_image * 255).astype(np.uint8)

    return uint8_image


def get_3_channels_colors(nb_colors: int, cmap_name: str = 'hsv') -> np.ndarray:
    """ Get a list of colors for the given number of classes.

    :param nb_colors: The number of classes.
    :param cmap_name: The name of the colormap to use.
    :return: An array of colors.
    """

    if nb_colors == 0:
        return np.array([])

    cmap = plt.get_cmap(name=cmap_name, lut=nb_colors)
    colors = cmap(np.arange(nb_colors))[:, :3]

    return colors


def get_colored_image(image: np.ndarray) -> np.ndarray:
    """ """

    image = np.moveaxis(image, source=0, destination=-1)

    return cast_to_uint8(ensure_3_channel_image(image))


def get_colored_label(image: np.ndarray, nb_classes: int = -1) -> np.ndarray:
    """ """

    nb_colors = len(np.unique(image)) if nb_classes == -1 else nb_classes

    # Select a color for each of the N classes in the image
    colors = get_3_channels_colors(nb_colors=nb_colors)

    # Create the colored image skipping the background class
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors, 1):
        colored_image[image == i] = color * 255

    return colored_image


def get_random_frame_indexes(nb_frame: int, nb_to_select: int = -1) -> np.ndarray:
    """ """

    n = min(nb_frame, nb_to_select) if nb_to_select > 0 else nb_frame

    return np.random.choice(nb_frame, n, replace=False)


def create_fig_with_layout(layout: GridLayout, nb_frame: int, x: int, suptitle: str = '', size: int = 2) -> tuple[
    plt.Figure, np.ndarray]:
    """ """

    assert layout in ['horizontal', 'vertical', 'grid'], "The layout must be 'horizontal', 'vertical', or 'grid'."

    if layout == 'grid' and x != 1:
        layout = 'horizontal'

    if layout == 'horizontal':
        rows, cols = nb_frame, x
        figsize = size * x, size * nb_frame

    elif layout == 'vertical':
        rows, cols = x, nb_frame
        figsize = size * nb_frame, size * x

    else:
        rows = cols = np.ceil(np.sqrt(nb_frame)).astype(int)
        figsize = size * rows, size * cols

    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    if suptitle:
        fig.suptitle(suptitle)

    return fig, np.asarray(axs)


def reorder_for_layout(layout: GridLayout, elements: list[Any], chunk_size: int) -> list[Any]:
    """ """

    if layout == 'vertical':
        chunks = [elements[i::chunk_size] for i in range(chunk_size)]
        elements = [item for sublist in chunks for item in sublist]

    return elements


def add_images_to_fig(axes: np.ndarray, images: list[np.ndarray], names: list[str]) -> None:
    """ """

    for (ax, img, title) in zip(axes, images, names):
        ax.imshow(img)
        ax.set_title(title, fontsize=6)
        ax.axis('off')


def create_oriented_fig(layout: GridLayout, chunk_size: int, arrays: list[np.ndarray], titles: list[str],
                        suptitle: str = '', size: int = 2) -> plt.Figure:
    """ Create a figure with the given layout. """

    # Manage orientation
    fig, axs = create_fig_with_layout(layout, len(arrays) // chunk_size, x=chunk_size, suptitle=suptitle, size=size)
    arrays = reorder_for_layout(layout, arrays, chunk_size=chunk_size)
    titles = reorder_for_layout(layout, titles, chunk_size=chunk_size)

    # Add the images to the figure
    add_images_to_fig(axs.flatten(), arrays, titles)

    return fig
