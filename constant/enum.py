#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: enum.py
# Created: 2024-09-04
# Description:
from enum import Enum


class DatasetChoice(Enum):
    AWF = "AWF"
    DF = "DF"

class AttackChoice(Enum):
    CUMUL = "CUMUL"
    DF = "DF"
    AWF_SDAE = "AWF-SDAE"
    AWF_CNN = "AWF-CNN"
    VAR_CNN = "Var-CNN"
    GANDaLF = "GANDaLF"

class DefenderChoice(Enum):
    NO_DEF = "NoDef"
    TAMARAW = "Tamaraw"
    WTF_PAD = "WTF-PAD"
    FRONT = "FRONT"
    MOCKING_BIRD = "Mockingbird"
    AWA = "AWA"
    AWA_SAME = "AWA-same"
    STINGER = "Stinger"
    STINGER_DF = "Stinger-DF"
    STINGER_AWF_SDAE = "Stinger-AWF-SDAE"
    STINGER_AWF_CNN = "Stinger-AWF-CNN"
    STINGER_VAR_CNN = "Stinger-VAR-CNN"
    STINGER_GandLAF = "Stinger-GandLaf"
    ALERT = "ALERT"



