#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: __init__.py
# Created: 2024-09-04
# Description: the loader for loading original data or generated data (by some defenses)


import logging  # noqa
import typing as t
from .awf import AWFLoader
from .df import DFLoader
from .base import DatasetChoice, AbstractLoader, Config


class LoaderFactory(object):

    @classmethod
    def get(self, config: Config) -> t.Type[AbstractLoader]:
        choice = config.dataset_choice
        if choice == DatasetChoice.AWF:
            return AWFLoader
        elif choice == DatasetChoice.DF:
            return DFLoader

        raise ValueError(f"No loader class found: {choice}")