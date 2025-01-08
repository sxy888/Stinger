#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: factory.py
# Created: 2024-09-08
# Description: the factory of attack

import logging  # noqa
import os
import typing as t

from constant import dataset
from loader import LoaderFactory, Config
from defender.abc import AbstractDefender
from constant.enum import DefenderChoice
logger = logging.getLogger(__name__)



class DefenderFactory(object):
    defender_map = {
        DefenderChoice.TAMARAW.value: "defender.tamaraw.TamarawDefender",
        DefenderChoice.WTF_PAD.value: "defender.wtfpad.WTFPADDefender",
        DefenderChoice.FRONT.value: "defender.front.FrontDefender",
        DefenderChoice.MOCKING_BIRD.value: "defender.mockingbird.MockingBirdDefender",
        DefenderChoice.AWA.value: "defender.awa.AWADefender",
        DefenderChoice.AWA_SAME.value: "defender.awa.AWADefender",
        DefenderChoice.STINGER.value: "defender.stinger.StingerDefender",
        DefenderChoice.STINGER_DF.value: "defender.stinger.StingerDefender",
        DefenderChoice.STINGER_AWF_SDAE.value: "defender.stinger.StingerDefender",
        DefenderChoice.STINGER_AWF_CNN.value: "defender.stinger.StingerDefender",
        DefenderChoice.STINGER_VAR_CNN.value: "defender.stinger.StingerDefender",
        DefenderChoice.STINGER_GandLAF.value: "defender.stinger.StingerDefender",
        DefenderChoice.ALERT.value: "defender.alert.AlertDefender"
    }


    @classmethod
    def get_overhead_list(cls, config: Config) -> t.List[float]:
        if config.defender_choice == DefenderChoice.NO_DEF:
            raise ValueError("no defense data has no extra overhead")
        target_dir = os.path.join(
            dataset.DATA_DIR,
            config.defender_choice.value,
            config.dataset_choice.value
        )
        if not os.path.exists(target_dir):
            return []
        # traverse the sub dir
        overhead_list = []
        for entry in os.listdir(target_dir):
            full_path = os.path.join(target_dir, entry)
            if not os.path.isdir(full_path):
                logger.warning(f"{full_path} is not a directory")
                continue
            try:
                overhead = float(entry.replace("overhead_", ""))
                overhead_list.append(overhead)
            except:
                logger.warning(f"the overhead of {full_path} is not a float")
                continue
        return overhead_list

    @classmethod
    def get_defender(cls, config: Config) -> AbstractDefender:
        defender_name = config.defender_choice.value
        defender_cls = cls.defender_map.get(defender_name)
        if defender_cls is None:
            raise ValueError(f"defender_name: {defender_name} not supported")

        # import the defender class
        module_name, class_name = defender_cls.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        defender_cls = getattr(module, class_name) # type: t.Type[AbstractDefender]
        _config = config.copy()
        defender_cls.update_config(_config)

        loader_cls = LoaderFactory.get(_config)
        loader = loader_cls(_config, use_for_defender=True)
        return defender_cls(loader)