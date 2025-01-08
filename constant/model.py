#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: constant.py
# Created: 2024-09-04
# Description: the constant module for the models
import os
from .dataset import DATA_DIR

# MODEL_DIR means the directory of the dataset （relative to the project root）
MODEL_DIR = os.path.join(DATA_DIR, "models")
PIC_DIR = os.path.join(DATA_DIR, "pics")