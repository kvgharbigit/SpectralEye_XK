import random
from typing import List

from path import Path


def get_unique_id_by_filename(file_names: List[str]) -> List[str]:
    """ """

    return sorted(list(set([fname[:5] for fname in file_names])))



def split_train_validation(unique_ids: list[str], train_ratio: float, seed):
    """ Split the data into training and validation sets at the participant level. """

    # Set the random seed if provided
    rng = random.Random(seed)

    # Get shuffled ids
    rng.shuffle(unique_ids)

    # Separate training from validation ids
    train_ids = unique_ids[:int(len(unique_ids) * train_ratio)]
    val_ids = unique_ids[int(len(unique_ids) * train_ratio):]

    return train_ids, val_ids