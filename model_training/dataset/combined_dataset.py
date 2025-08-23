import random
from typing import Dict, Optional
from typing import Tuple

import h5py
import pandas as pd
import numpy as np
import torch
from eitools.image_reader import DataHsImage
from path import Path
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from utils.split_participants import split_train_validation, get_unique_id_by_filename

import concurrent.futures
import h5py
import torch
import numpy as np
import gc
from skimage.io import imread
from torch.utils.data import Dataset

import concurrent.futures
import gc
import h5py
import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset


import concurrent.futures
import gc
import h5py
import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset

import concurrent.futures
import gc
import h5py
import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset



class SegmentationDataset(Dataset):
    """
    Optimized Dataset for hyperspectral images with segmentation.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Compose] = None):
        self.df = df
        self.transform = transform
        self.data_cache = {}  # To cache data for faster access

    def __len__(self):
        return len(self.df)

    def load_hdf5(self, file_path):
        """Loads hyperspectral data using h5py, ."""

        with h5py.File(file_path, 'r', swmr=True) as f:
            hs_data = f['/Cube/Images'][:]

        hs_tensor = torch.tensor(hs_data, dtype=torch.float32)
        # self.data_cache[file_path] = hs_tensor
        return hs_tensor

    def __getitem__(self, idx: int):
        # Load hyperspectral data
        hs_file = self.df.iloc[idx]['hs_file']
        # change drive letter from F to D
        # hs_file = hs_file.replace('F:', 'D:')

        hs = self.load_hdf5(hs_file)

        # Load RGB image
        rgb_file = self.df.iloc[idx]['rgb_file']
        # change drive letter from F to D
        # rgb_file = rgb_file.replace('F:', 'D:')

        rgb = imread(rgb_file).astype(np.float32) / 255.0
        rgb = torch.tensor(rgb).permute(2, 0, 1)

        # Apply transforms
        if self.transform and random.random() < 0.5:
            hs, rgb = self.transform((hs, rgb))

        label = self.df.iloc[idx]['label']

        return hs, label, rgb


class SegmentationDataset_slow(Dataset):
    """
    Dataset for hyperspectral images with segmentation.
    """

    def __init__(self,  df: pd.DataFrame, transform: Optional[Compose] = None):
        """
        Initialize the dataset with auxiliary tasks and transformations.

        Args:
            df (pd.DataFrame): DataFrame containing file paths and auxiliary task values.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        hs_file = self.df.iloc[idx]['hs_file']
        # Read cube with hdf5 reader
        with h5py.File(hs_file, 'r') as f:
            hs = torch.from_numpy(f['/Cube/Images'][:]).type(torch.float32)

        # hs = DataHsImage(hs_file).get_cube()
        # hs = torch.tensor(hs, dtype=torch.float32)

        rgb_file = self.df.iloc[idx]['rgb_file']
        rgb = imread(rgb_file)
        rgb = torch.tensor(rgb, dtype=torch.float) / 255.0
        rgb = rgb.permute(2, 0, 1)

        # Apply transformations if any
        if self.transform and random.random() < 0.5:
            hs, rgb = self.transform((hs, rgb))

        label = self.df.iloc[idx]['label']

        return hs, label, rgb


def get_dataset(csv_path: str, train_ratio: float, seed: int, transform: Optional[Compose] = None) -> tuple[Dataset, Dataset]:
    """
    Load the dataset from a CSV file, split it, and create training and validation datasets.

    Args:
        csv_path (str): Path to the CSV file containing the data.
        train_ratio (float): Ratio of training data.
        seed (int): Random seed for splitting.

    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """

    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    # Separate "Reflec" and other labels
    non_reflec_labels = df['label'].unique().tolist()
    non_reflec_labels.remove('All')

    # Initialize empty DataFrames for train and validation sets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()

    # Process each label separately
    for label in non_reflec_labels:
        # Filter participants with the current label
        df_label = df[df['label'] == label]

        # Randomly sample 10 participants for validation
        if label != "Hypercolour":
            val_sample = df_label.sample(n=10, random_state=seed)
        else:
            val_sample = df_label.sample(n=20, random_state=seed)
        # Add the selected validation samples
        df_val = pd.concat([df_val, val_sample])

        # Remove only the selected participants from the df_label to avoid index error
        df_label = df_label.drop(val_sample.index)

        # **Triplicate only if the label is NOT "Hypercolour"**
        if label != "Hypercolour":
            df_label = pd.concat([df_label] * 5, ignore_index=True)

        # Append the triplicated data to the training set
        df_train = pd.concat([df_train, df_label], ignore_index=True)

    # Add the "All" participants to the training set without modification
    df_train = pd.concat([df_train, df[df['label'] == 'All']], ignore_index=True)

    # Shuffle the datasets
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=seed).reset_index(drop=True)

    # create df train for rows with column label = All and df val for all the other ones
    # df_train = df[df['label'] == 'All']
    # df_val = df[df['label'] != 'All']

    # df_train = df_train[::3]
    # df_val = df_val[::2]


    # Split participants into training and validation sets
    # unique_ids = df['ID'].drop_duplicates().tolist()
    # train_ids, val_ids = split_train_validation(unique_ids, train_ratio=train_ratio, seed=seed)

    # keep every third participant for train and validation
    # train_ids = train_ids[::2]
    # val_ids = val_ids[::3]
    # Create DataFrames for training and validation
    # df_train = df[df['ID'].isin(train_ids)].copy()
    # df_val = df[df['ID'].isin(val_ids)].copy()

    # Create the datasets

    # Only keep every 10 participants for training
    # df_train = df_train[::10]

    train_dataset = SegmentationDataset(df=df_train, transform=transform)
    val_dataset = SegmentationDataset(df=df_val, transform=None)

    return train_dataset, val_dataset


