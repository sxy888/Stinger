 != 0:
        x_min -= 1

    if max_value is None:
        while x_max % 5 != 0:
            x_max += 1
    else:
        x_max = max_value

    step = (x_max - x_min) // (num - 1)
    if max_value is not None:
        print("max_value:", max_value, "step:", step, "x_max:", x_max, "x_min:", x_min)
    while step % 5 != 0:
        step += 1
    new_array = [x_min + step * i for i in range(num)]
    return new_array, list(map(str, new_array))
                                                                 stinger-release/utils/path.py                                                                       0000644 0001762 0000033 00000000264 14737363242 015465  0                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   #!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: process.py
# Created: 2024-09-07
# Description: helper functions for processing data

import logging  # noqa
import torch
import typing as t # noqa
from keras.utils import to_categorical
from torch.utils.data import Dataset
from numpy import ndarray

logger = logging.getLogger(__name__)

def categorize(labels, dict_labels=None):
    possible_labels = list(set(labels))

    if not dict_labels:
        dict_labels = {}
        n = 0
        for label in possible_labels:
            dict_labels[label] = n
            n = n + 1

    new_labels = []
    for label in labels:
        new_labels.append(dict_labels[label])

    new_labels = to_categorical(new_labels)

    return new_labels


class NoDefDataSet(Dataset):
    def __init__(self, data: ndarray, label: ndarray):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item
import os


def get_project_root_path():
    """
    Returns the root path of the project.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

                                                                                                                                                                                                                                                                           