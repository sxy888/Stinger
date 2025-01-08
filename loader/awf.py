#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: awf.py
# Created: 2024-09-07
# Description: load the dataset of awf
import os
import logging  # noqa
import typing as t # noqa
import numpy as np
import pickle
from enum import Enum
from constant import dataset
from constant.enum import DefenderChoice
from .base import Config

from loader.base import AbstractLoader, DatasetChoice, Dataset, DatasetWrapper

logger = logging.getLogger(__name__)


class AWFLoader(AbstractLoader):


    def __init__(self, config: Config, use_for_defender=False):
        self.config = config
        self.dataset_wrapper = None
        self.use_for_defender = use_for_defender
        if config.need_timestamp:
            raise ValueError("AWF dataset can't support timestamp")

        self.dataset_path, self.dataset_dir = None, None
        if config.defender_choice == DefenderChoice.NO_DEF or use_for_defender:
            self.dataset_path = dataset.AWF_100_WEBSITES_2500_TRACES
        else:
            self.dataset_dir = os.path.join(
                dataset.DATA_DIR,
                config.defender_choice.value,
                DatasetChoice.AWF.value,
                f"overhead_{config.defense_overhead}",
            )

    def load(self):
        if self.dataset_path is not None:
            self.load_no_def_data()
        else:
            self.load_defense_data()

        # 减小数据规模
        if self.config.defender_choice == DefenderChoice.NO_DEF.value or self.use_for_defender:
            ratio = 0.3
            fixed_length = int(len(self.original_data) * ratio)
            indices = np.random.choice(len(self.original_data), size=fixed_length, replace=False)
            self.original_data = self.original_data[indices]
            if self.config.need_timestamp:
                self.original_timestamp = self.original_timestamp[indices]
            self.original_labels = self.original_labels[indices]

        num_classes = len(np.unique(self.original_labels))
        self.num_classes = num_classes
        logger.info(f"The unique labels has {num_classes} kinds")

    def load_no_def_data(self):
        # load the target dataset
        path = self.dataset_path
        logger.info("Begin loading AWF dataset: {}".format(path))
        try:
            from numpy.lib.npyio import NpzFile
            dataset_file = np.load(path, allow_pickle=True) # type: NpzFile
        except Exception as e:
            logger.error("Failed to load dataset: {}".format(e))
            raise e

        if "data" not in dataset_file.files or "labels" not in dataset_file.files:
            raise ValueError("Invalid dataset format")

        data, labels = dataset_file["data"], dataset_file["labels"]
        # handle the data and labels
        self.original_data = data
        # convert string label to int
        idx, label_map = 0, {}
        convert_labels = []
        for l in labels:
            if l not in label_map:
                label_map[l] = idx
                idx += 1
            convert_labels.append(label_map[l])

        self.original_labels = np.array(convert_labels)

    def load_defense_data(self):
        dataset_dir = self.dataset_dir

        logger.info("loading DF dataset: {}".format(dataset_dir))
        x_length = 5000

        # 如果是训练集、测试集、验证集分开保存的
        if os.path.exists(os.path.join(dataset_dir, dataset.TRAIN_TRACE_FILE)):
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
                train_y = pickle.load(handle, encoding='latin1')

            with open(os.path.join(dataset_dir, dataset.VALIDATE_SITE_FILE), 'rb') as handle:
                validate_y = pickle.load(handle, encoding='latin1')

            with open(os.path.join(dataset_dir, dataset.TEST_SITE_FILE), 'rb') as handle:
                test_y = pickle.load(handle, encoding='latin1')

            self.dataset_wrapper = DatasetWrapper(
                name=DatasetChoice.DF,
                train_data=Dataset(train_x, train_y),
                test_data=Dataset(test_x, test_y),
                eval_data=Dataset(validate_x, validate_y)
            )
            self.dataset_wrapper.summary()
            self.original_data = np.concatenate((train_x, test_x, validate_x))
            self.original_labels = np.concatenate((train_y, test_y, validate_y))
        else:
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


    def split(self) -> DatasetWrapper:
        if self.dataset_wrapper is not None:
            return self.dataset_wrapper

        indices = np.arange(len(self.original_data))
        np.random.shuffle(indices)

        # use the indices to shuffle the data
        x = self.original_data[indices]
        y = self.original_labels[indices]

        # split the data into train, test and eval sets
        train_size = int(len(x) * self.config.train_ratio)
        test_size = int(len(x) * self.config.test_ratio)
        eval_size = len(x) - train_size - test_size

        train_data = Dataset(x[:train_size], y[:train_size])
        test_data = Dataset(x[train_size:train_size+test_size], y[train_size:train_size+test_size])
        eval_data = Dataset(x[-eval_size:], y[-eval_size:])

        self.dataset_wrapper = DatasetWrapper(
            name=DatasetChoice.AWF,
            train_data=train_data,
            test_data=test_data,
            eval_data=eval_data
        )
        self.dataset_wrapper.summary()

        return self.dataset_wrapper

    def __fill_trace_data(self, data: t.List[np.ndarray], length: int) -> t.List[np.ndarray]:
        """
        Fill the trace data with zero to make the length of each trace equal to the given length.
        """
        result = []
        for item in data:
            if len(item) >= length:
                result.append(item[:length])
            else:
                # fill with zero
                pad_length = length - len(item)
                fill_item = np.pad(item, (0, pad_length), 'constant', constant_values=(0))
                result.append(fill_item)
        return result
