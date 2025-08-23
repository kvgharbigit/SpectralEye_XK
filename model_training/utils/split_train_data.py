from typing import List, Callable
import logging
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
from path import Path

logger = logging.getLogger()


get_id_func = Callable[[list[Path]], list[str]]


def split_train_validation_10fold(file_names: list[str], get_unique_id: callable):
    """ Split the data into 10 folds while ensuring unique IDs are consistently separated. """

    # Get shuffled random unique IDs from file_names
    unique_ids = get_unique_id(file_names)
    np.random.shuffle(unique_ids)

    # Initialize StratifiedKFold with 10 folds
    stratified_kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

    # Create lists to store train and validation files for each fold
    fold_train_files = []
    fold_val_files = []

    for train_indices, val_indices in stratified_kfold.split(unique_ids, [0] * len(unique_ids)):
        train_ids = [unique_ids[i] for i in train_indices]
        val_ids = [unique_ids[i] for i in val_indices]

        # Filter files based on train_ids and val_ids
        train_files = [file for file in file_names if get_unique_id([file])[0] in train_ids]
        val_files = [file for file in file_names if get_unique_id([file])[0] in val_ids]

        fold_train_files.append(train_files)
        fold_val_files.append(val_files)

    return fold_train_files, fold_val_files


def split_train_validation(file_names: list[Path], train_ratio: float, get_unique_id: get_id_func, seed):
    """ Split the data into training and validation sets at the participant level. """

    # Set the random seed if provided
    if seed is not None:
        random.seed(seed)

    # Get shuffled random ids from file_names
    unique_ids = get_unique_id(file_names)
    random.shuffle(unique_ids)

    # Separate training from validation ids
    train_ids = unique_ids[:int(len(unique_ids) * train_ratio)]
    val_ids = unique_ids[int(len(unique_ids) * train_ratio):]

    logger.debug(f'SPLIT DATA result:')
    logger.debug(f'train_names: {train_ids}')
    logger.debug(f'val_names: {val_ids}')

    train_files: List[str] = []
    val_files: List[str] = []
    for patient_file in file_names:
        name_id = patient_file.name[:5]
        if name_id in train_ids:
            train_files.append(patient_file)
        else:
            val_files.append(patient_file)
    return train_files, val_files
