from eitools.image_reader import DataHsImage
from eitools.utils import ProgressBar
import numpy as np
from path import Path
import matplotlib.pyplot as plt


def get_normalising_spectra(hs_files, use_log=True):
    """ Get the normalising spectra for the hyperspectral dataset. """

    # Initialize the mean and variance
    all_spectra = []
    pb = ProgressBar(hs_files)
    # Accumulate the mean and variance of the spectra
    for hs_file in pb:
        hs = DataHsImage(hs_file).get_cube()
        bands, rows, cols = hs.shape
        hs = hs.reshape((bands, rows * cols))
        keep = hs[0, :] > 1e-4

        hs = hs[:, keep]
        if use_log:
            hs = np.log10(hs)
        all_spectra.append(hs)

    all_spectra = np.concatenate(all_spectra, axis=1)
    mean = np.mean(all_spectra, axis=1, dtype=np.float64)
    std = np.std(all_spectra, axis=1, dtype=np.float64)
    return mean, std


def main():
    # Path to the hyperspectral dataset
    folder = Path(r'F:\Foundational_model\data_256')
    hs_files = folder.glob('*/*/*.h5')

    # Get the normalising spectra
    mean, variance = get_normalising_spectra(hs_files, use_log=False)

    # Save the normalising spectra
    np.save('data/mean_spectrum.npy', mean)
    np.save('data/std_spectrum.npy', variance)


if __name__ == '__main__':
    main()

    # Read normalising spectra
    mean = np.load('data/mean_spectrum.npy')
    variance = np.load('data/std_spectrum.npy')

    old_mean = np.array([-2.0066323, -1.9721451, -1.9430325, -1.8872414, -1.8201746,
                             -1.7571307, -1.691031, -1.6496427, -1.6376318, -1.6083344,
                             -1.5398917, -1.4476203, -1.4196872, -1.3763593, -1.0936164,
                             -0.78906184, -0.6126504, -0.48976675, -0.39780718, -0.32077456,
                             -0.25213036, -0.18648516, -0.12758268, -0.0684865, -0.009604,
                             -0.03754743, -0.08673472, -0.13510391, -0.17267017])
    import matplotlib.pyplot as plt

    plt.plot(mean, 'k')
    plt.plot(old_mean, 'r')
    plt.show()
    plt.plot(variance)
    plt.show()


