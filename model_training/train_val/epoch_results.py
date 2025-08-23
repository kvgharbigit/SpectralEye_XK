from dataclasses import dataclass, field
import numpy as np


@dataclass
class EpochResults:
    loss: float
    metric: float
    spectral_latent_outputs: np.ndarray
    rgb_images: np.ndarray
    mse_spectra: np.ndarray
    input_spectra: np.ndarray
    output_spectra: np.ndarray
    labels: np.ndarray
    # unet_outputs: np.ndarray
    # targets: np.ndarray
