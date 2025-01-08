#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: base.py
# Created: 2024-09-04
# Description: abstract class for data loader

import logging  # noqa
import typing as t # noqa
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from rich import table as r_table, console as r_console
from constant.enum import AttackChoice, DatasetChoice, DefenderChoice

logger = logging.getLogger(__name__)
console = r_console.Console()

class Config():
    def __init__(self,
        dataset_choice: DatasetChoice = DatasetChoice.DF,
        defender_choice: DefenderChoice = DefenderChoice.NO_DEF,
        defense_overhead: float = 0.0,
        attack_choice: t.Union[AttackChoice, None] = None,
        train_ratio: float = 0.8,
        test_ratio: float = 0.1,
        need_timestamp: bool = False,
        load_data_by_self: bool = False
    ):
        if train_ratio + test_ratio >= 1:
            raise ValueError("train_ratio + test_ratio must be less than 1")

        self.dataset_choice = dataset_choice
        self.defender_choice = defender_choice
        self.defense_overhead = defense_overhead
        self.attack_choice = attack_choice
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.eval_ratio = 1 - train_ratio - test_ratio
        self.need_timestamp = need_timestamp
        self.load_data_by_self = load_data_by_self

    def copy(self):
        return Config(
            dataset_choice = self.dataset_choice,
            defender_choice = self.defender_choice,
            defense_overhead = self.defense_overhead,
            attack_choice = self.attack_choice,
            train_ratio = self.train_ratio,
            test_ratio = self.test_ratio,
            need_timestamp = self.need_timestamp
        )
class Dataset():
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def unwrap(self):
        return self.x, self.y
    
    def __str__(self):
        return f"Dataset(x={self.x.shape}, y={self.y.shape})"

class DatasetWrapper():
    def __init__(
        self,
        name: DatasetChoice,
        train_data: Dataset,
        test_data: Dataset,
        eval_data: Dataset
    ):
        self.name = name
        self.train_data = train_data
        self.test_data = test_data
        self.eval_data = eval_data

    def summary(self):
        logger.info("Dataset of [%s] summary:" % self.name.value)
        table = r_table.Table(show_header=True, header_style="bold purple3")
        table.add_column("Dataset Type")
        table.add_column("X Shape")
        table.add_column("Y Shape")
        table.add_row("Train", str(self.train_data.x.shape), str(self.train_data.y.shape))
        table.add_row("Test", str(self.test_data.x.shape), str(self.test_data.y.shape))
        table.add_row("Eval", str(self.eval_data.x.shape), str(self.eval_data.y.shape))
        console.print(table)



class AbstractLoader(ABC):

    original_data: t.Union[None, np.ndarray] = None
    original_labels: t.Union[None, np.ndarray] = None
    original_timestamp: t.Union[None, np.ndarray] = None
    dataset_wrapper: t.Union[None, DatasetWrapper] = None
    config: Config = None
    num_classes: int = 95

    @abstractmethod
    def load(self, use_for_defender=False):
        pass

    @abstractmethod
    def split(self) -> DatasetWrapper:
        pass
