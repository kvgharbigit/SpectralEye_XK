import torch
import numpy as np
from torchvision.transforms.functional import normalize


SP_MEAN_FILE = r'C:\Users\xhadoux\Data_projects\spectral_compression\src\data_preparation\data\mean_spectrum.npy'
SP_STD_FILE = r'C:\Users\xhadoux\Data_projects\spectral_compression\src\data_preparation\data\std_spectrum.npy'


def preprocess_hsi(hs):
    nb_bands = hs.size(1)
    mask = hs[:, 0] < 1e-3
    mask = mask.unsqueeze(1).expand(-1, nb_bands, -1, -1).to(hs.device)
    hs[mask] = 1
    hs = torch.log10(hs)

    hs = hs + 3
    hs = hs / 3
    return hs


def preprocess_hsi_std_all(hs, sp_mean_file=SP_MEAN_FILE, sp_std_file=SP_STD_FILE):
    sp_mean = np.load(sp_mean_file).tolist()
    sp_std = np.load(sp_std_file).tolist()
    hs = normalize(hs, mean=sp_mean, std=sp_std)
    return hs


def preprocess_hsi_old3(hs):
    nb_bands = hs.size(1)
    mask = hs[:, 0] < 1e-3
    mask = mask.unsqueeze(1).expand(-1, nb_bands, -1, -1).to(hs.device)
    hs[mask] = 1
    hs = torch.log10(hs)

    hs = hs + 3
    hs = hs / 3

    return hs


def preprocess_hsi_xx(hs):
    # For each image of the batch of size (B, C, R, C) remove mean and divide by std of the image channelwise
    # hs: (B, C, R, C)
    # sp_mean: (B, C, 1, 1)
    # sp_std: (B, C, 1, 1)
    # hs[hs < 0.001] = 1
    # hs = torch.log10(hs)
    mask = hs != 0
    sp_mean = torch.sum(hs * mask, dim=(2, 3), keepdim=True) / torch.sum(mask, dim=(2, 3), keepdim=True)

    sp_std = torch.sqrt(torch.sum(((hs - sp_mean) * mask) ** 2, dim=(2, 3), keepdim=True) / torch.sum(mask, dim=(2, 3), keepdim=True))
    hs = (hs - sp_mean) / sp_std
    return hs


def preprocess_hsi_old(hs, sp_mean_file=SP_MEAN_FILE, sp_std_file=SP_STD_FILE):
    sp_mean = np.load(sp_mean_file)
    sp_std = np.load(sp_std_file)
    sp_mean = torch.tensor(sp_mean, dtype=torch.float).to(hs.device)
    sp_std = torch.tensor(sp_std, dtype=torch.float).to(hs.device)
    nb_bands = hs.size(1)
    mask = hs[:, 0] < 1e-3
    mask = mask.unsqueeze(1).expand(-1, nb_bands, -1, -1).to(hs.device)
    hs[mask] = 1
    hs = torch.log10(hs)

    hs = hs - sp_mean.view(nb_bands, 1, 1)
    hs = hs / (6 * sp_std.view(nb_bands, 1, 1))

    return hs
