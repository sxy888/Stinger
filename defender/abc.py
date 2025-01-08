#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: abc.py
# Created: 2024-09-08
# Description: abstract class for defender methods

import logging  # noqa
import os
import pickle
import typing as t # noqa
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from constant.enum import DefenderChoice, AttackChoice
from loader import AbstractLoader, Config
from loader.base import DatasetWrapper
from constant import dataset
from utils.common import get_memory_usage

logger = logging.getLogger(__name__)



class AbstractDefender(ABC):

    X: t.Union[None, np.ndarray] = None
    Y: t.Union[None, np.ndarray] = None
    Timestamp: t.Union[None, np.ndarray] = None
    dataset: t.Union[None, DatasetWrapper] = None

    def __init__(self, loader: t.Union[AbstractLoader, t.Type[AbstractLoader]]) -> None:
        if isinstance(loader, AbstractLoader):
            self.loader = loader
        else:
            self.loader = loader({})

    def run(self, defender_kwargs: t.Dict[str, any]) -> None:
        if not self.loader.config.load_data_by_self:
            self.loader.load()
            mem_rss,mem_vms = get_memory_usage()
            logger.info("After loader load data in Defender, memory usage: RSS=%s, VMS=%s" % (mem_rss, mem_vms))
            self.X = self.loader.original_data
            self.Y = self.loader.original_labels
            self.Timestamp = self.loader.original_timestamp
            self.dataset = self.loader.split()


        logger.info("%s begin running" % self.__class__.__name__)
        kwargs = defender_kwargs.copy()
        if self.loader.config.defender_choice == DefenderChoice.AWA:
            kwargs["use_same_dataset"] = False
        elif self.loader.config.defender_choice == DefenderChoice.AWA_SAME:
            kwargs["use_same_dataset"] = True

        if self.loader.config.defender_choice == DefenderChoice.STINGER_DF:
            kwargs["attack_method"] = AttackChoice.DF
        elif self.loader.config.defender_choice == DefenderChoice.STINGER_AWF_SDAE:
            kwargs["attack_method"] = AttackChoice.AWF_SDAE
        elif self.loader.config.defender_choice == DefenderChoice.STINGER_AWF_CNN:
            kwargs["attack_method"] = AttackChoice.AWF_CNN
        elif self.loader.config.defender_choice == DefenderChoice.STINGER_VAR_CNN:
            kwargs["attack_method"] = AttackChoice.VAR_CNN
        elif self.loader.config.defender_choice == DefenderChoice.STINGER_GandLAF:
            kwargs["attack_method"] = AttackChoice.GANDaLF

        n_x, n_timestamp, n_y, overhead = self.defense(**kwargs)


        logger.info("%s finished, now saving the generated data" % self.__class__.__name__)
        # save the generated data
        overhead = round(overhead * 100, 2)

        if defender_kwargs.get("dry_run"):
            # 不进行数据保存
            logger.info("dry run 模式，不保存生成数据集")
            return overhead
        config = self.loader.config
        target_dir = os.path.join(
            dataset.DATA_DIR,
            config.defender_choice.value,
            config.dataset_choice.value,
            "overhead_" + str(overhead),
        )
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(n_x, t.Tuple):
            # 训练集、测试集分开进行保存
            with open(os.path.join(target_dir, dataset.TRAIN_TRACE_FILE), 'wb') as f:
                pickle.dump(n_x[0], f)

            with open(os.path.join(target_dir, dataset.VALIDATE_TRACE_FILE), 'wb') as f:
                pickle.dump(n_x[1], f)

            with open(os.path.join(target_dir, dataset.TEST_TRACE_FILE), 'wb') as f:
                pickle.dump(n_x[2], f)

            with open(os.path.join(target_dir, dataset.TRAIN_SITE_FILE), 'wb') as f:
                pickle.dump(n_y[0], f)

            with open(os.path.join(target_dir, dataset.VALIDATE_SITE_FILE), 'wb') as f:
                pickle.dump(n_y[1], f)

            with open(os.path.join(target_dir, dataset.TEST_SITE_FILE), 'wb') as f:
                pickle.dump(n_y[2], f)
        else:
            with open(os.path.join(target_dir, dataset.TRACE_FILE), 'wb') as f:
                pickle.dump(n_x, f)

            with open(os.path.join(target_dir, dataset.SITES_FILE), 'wb') as f:
                pickle.dump(n_y, f)

        return overhead


    @classmethod
    def update_config(cls, config: Config) -> Config:
        return config

    @abstractmethod
    def defense(self, **kwargs) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        pass