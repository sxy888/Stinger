#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: factory.py
# Created: 2024-09-08
# Description: the factory of attack

import logging  # noqa
import typing as t

from constant.enum import AttackChoice
from loader import LoaderFactory
from loader.base import Config # noqa
from attack.abc import AbstractAttack
logger = logging.getLogger(__name__)



class AttackFactory(object):

    attack_map = {
        AttackChoice.CUMUL.value: "attack.cumul.CumulAttack",
        AttackChoice.DF.value: "attack.df.DFAttack",
        AttackChoice.AWF_SDAE.value: "attack.awf.AWFSdaeAttack",
        AttackChoice.AWF_CNN.value: "attack.awf.AWFCnnAttack",
        AttackChoice.VAR_CNN.value: "attack.varcnn.VarCnnAttack",
        AttackChoice.GANDaLF.value: "attack.gandalf.GANDaLFAttack",
    }

    @classmethod
    def get_attack(self, config: Config) -> AbstractAttack:
        attack_name = config.attack_choice.value
        attack_cls = self.attack_map.get(attack_name)
        if attack_cls is None:
            raise ValueError(f"attack_name: {attack_name} not supported")

        # import the attack class
        module_name, class_name = attack_cls.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        attack_cls = getattr(module, class_name) # type: t.Type[AbstractAttack]

        loader_cls = LoaderFactory.get(config)
        loader = loader_cls(config)
        return attack_cls(loader)