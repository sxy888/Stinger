#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: df.py
# Created: 2024-09-07
# Description: load the dataset of df


import logging  # noqa
import os
import typing as t # noqa
import numpy as np
import pickle

from constant import dataset
from constant.enum import DatasetChoice, DefenderChoice
from loader.base import AbstractLoader, Dataset, DatasetWrapper, Config

logger = logging.getLogger(__name__)


class DFLoader(AbstractLoader):

    def __init__(self, config: Config, use_for_defender=False) -> None:
        self.dataset_wrapper = None
        self.config = config
        self.use_for_defender = use_for_defender
        # load the target dataset
        if config.defender_choice == DefenderChoice.NO_DEF or use_for_defender:
            self.dataset_dir = dataset.DF_DATA_WITH_TIMESTAMP
        else:
            self.dataset_dir = os.path.join(
                dataset.DATA_DIR,
                config.defender_choice.value,
                DatasetChoice.DF.value,
                f"overhead_{confstinger-release/loader/__pycache__/base.cpython-312.pyc                                             0000644 0001762 0000033 00000011677 14737363242 022124  0                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   �
    ˚g�	  �                   ��   � d dl Z d dlZd dlZd dlmZ d dlmZm	Z	 d dl
mZmZ d dlmZmZ  e j$                  e�      Z ej*                  �       Z G d� d�      Z G d� d	�      Z G d
� d�      Z G d� de�      Zy)�    N)�Enum)�ABC�abstractmethod)�table�console)�DatasetChoice�DefenderChoicec            	       �V   � e Zd Zej                  ej                  ddfdedededefd�Zy)	�Configg�������?g�������?�dataset_choice�defender_choice�train_ratio�
test_ratioc                 �|   � ||z   dk\  rt        d�      �|| _        || _        || _        || _        d|z
  |z
  | _        y )N�   z,train_ratio + test_ratio must be less than 1)�
ValueErrorr   r   r   r   �
eval_ratio)�selfr   r   r   r   s        �)/home/zzh/projects/stinger/loader/base.py�__init__zConfig.__init__   sL   � � ��#�q�(��K�L�L�,���.���&���$����k�/�J�6���    N)	�__name__�
__module__�__qualname__r   �DFr	   �NO_DEF�floatr   � r   r   r   r      sC   � �(5�(8�(8�*8�*?�*?� ��	7�%�7�'�7� �7� �	7r   r   c                   �N   � e Zd Zdej                  dej                  ddfd�Zd� Zy)�Dataset�x�y�returnNc                 �    � || _         || _        y �N)r!   r"   )r   r!   r"   s      r   r   zDataset.__init__%   s   � ������r   c                 �d   � d| j                   zAWFLoader._get_dataset_path~   s�   � ��k�k�-�-���+�3�3�3��7�7�7��+�1�1�1��7�7�7��+�1�1�1��7�7�7��+�1�1�1��7�7�7��4�;�;�K�H�I�Ir   N)r   r   r   r   r2   r>   r
   rR   r=   �ndarrayrC   rE   �strr8   r   r   r   r/   r/   5   sM   � ��� �;�&�~� �6���� � �R�Z�Z� �
J�3� 
Jr   r/   )�logging�typingr,   �numpyr=   �enumr   �constantr   �keras.api.preprocessingr   �utils.processr   �loader.baser   r   r	   r
   �	getLoggerr   r9   r   �objectr   r/   r   r   r   �<module>rt      s\   �� � � � � � ,� $� N� N�	��	�	�8�	$���� �<�� <�4SJ�� SJr                                                                                                                                                                                                                                                                                                                        ig.defense_overhead}",
            )

    def load(self) -> None:
        dataset_dir = self.dataset_dir

        logger.info("loading DF dataset: {}".format(dataset_dir))
        x_length = 5000


        if os.path.exists(os.path.join(dataset_dir, dataset.TRACE_FILE)) or self.config.need_timestamp:
            with open(os.path.join(dataset_dir, dataset.TRACE_FILE), 'rb') as handle:
                data = pickle.load(handle, encoding='latin1')
                self.original_data = np.array(
                    self.__fill_trace_data(data, x_length)
                )
            if self.config.need_timestamp:
                with open(os.path.join(dataset_dir, dataset.TIMESTAMP_FILE), 'rb') as handle:
                    timestamp = pickle.load(handle, encoding='latin1')
                    self.original_timestamp = np.array(
                        self.__fill_trace_data(timestamp, x_length)
                    )
            with open(os.path.join(dataset_dir, dataset.SITES_FILE), 'rb') as handle:
                self.original_labels = np.array(pickle.load(handle, encoding='latin1'))
        else:
            # 如果是训练集、测试集、验证集分开保存的
            with open(os.path.join(dataset_dir, dataset.TRAIN_TRACE_FILE), 'rb') as handle:
                data = pickle.load(handle, encoding='latin1')
                train_x = np.array(
                    self.__fill_trace_data(data, x_length)
                )
            with open(os.path.join(dataset_dir, dataset.VALIDATE_TRACE_FILE), 'rb') as handle:
                data = pickle.load(handle, encoding='latin1')
                validate_x = np.array(
                    self.__fill_trace_data(data, x_length)
                )
            with open(os.path.join(dataset_dir, dataset.TEST_TRACE_FILE), 'rb') as handle:
                data = pickle.load(handle, encoding='latin1')
                test_x = np.array(
                    self.__fill_trace_data(data, x_length)
                )
            with open(os.path.join(dataset_dir, dataset.TRAIN_SITE_FILE), 'rb') as handle:
                train_y = np.array(pickle.load(handle, encoding='latin1'))

            with open(os.path.join(dataset_dir, dataset.VALIDATE_SITE_FILE), 'rb') as handle:
                validate_y = np.array(pickle.load(handle, encoding='latin1'))

            with open(os.path.join(dataset_dir, dataset.TEST_SITE_FILE), 'rb') as handle:
                test_y = np.array(pickle.load(handle, encoding='latin1'))

            self.dataset_wrapper = DatasetWrapper(
                name=DatasetChoice.DF,
                train_data=Dataset(train_x, train_y),
                test_data=Dataset(test_x, test_y),
                eval_data=Dataset(validate_x, validate_y)
            )
            self.dataset_wrapper.summary()
            self.original_data = np.concatenate((train_x, test_x, validate_x))
            self.original_labels = np.concatenate((train_y, test_y, validate_y))

        # 减小数据规模
        # if self.config.defender_choice == DefenderChoice.NO_DEF or self.use_for_defender:
        #     ratio = 0.3
        #     logger.info(f"缩减数据集大小至 {ratio}")
        #     fixed_length = int(len(self.original_data) * ratio)
        #     indices = np.random.choice(len(self.original_data), size=fixed_length, replace=False)
        #     self.original_data = self.original_data[indices]
        #     if self.config.need_timestamp:
        #         self.original_timestamp = self.original_timestamp[indices]
        #     self.original_labels = self.original_labels[indices]

        num_classes = len(np.unique(self.original_labels))
        self.num_classes = num_classes
        logger.info(f"The unique labels has {num_classes} kinds")

    def split(self) -> DatasetWrapper:
        """
        Split the dataset into train, test and eval sets.
        """
        if self.dataset_wrapper is no