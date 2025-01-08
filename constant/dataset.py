#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: constant.py
# Created: 2024-09-04
# Description: the constant module for the dataset
import os
from utils.path import get_project_root_path
from .enum import DatasetChoice

ROOT_PATH = get_project_root_path()
# DATA_DIR means the directory of the dataset （relative to the project root）
DATA_DIR = os.path.join(ROOT_PATH, "data")
NO_DEFENSE_DIR = "NoDef"
# AWF DATASET,  ref: https://github.com/DistriNet/DLWF
AWF_DATA_DIR = os.path.join(DATA_DIR, NO_DEFENSE_DIR, DatasetChoice.AWF.value)
# top 100 websites
AWF_100_WEBSITES_2500_TRACES = os.path.join(DATA_DIR, AWF_DATA_DIR, "tor_100w_2500tr.npz")
# top 200 websites
AWF_200_WEBSITES_2500_TRACES = os.path.join(DATA_DIR, AWF_DATA_DIR, "tor_200w_2500tr.npz")
# top 500 websites
AWF_500_WEBSITES_2500_TRACES = os.path.join(DATA_DIR, AWF_DATA_DIR, "tor_500w_2500tr.npz")
# top 900 websites ( total dataset )
AWF_900_WEBSITES_2500_TRACES = os.path.join(DATA_DIR, AWF_DATA_DIR, "tor_900w_2500tr.npz")


# DF DATASET, ref: https://github.com/deep-fingerprinting/df
DF_DATA_DIR = os.path.join(DATA_DIR, NO_DEFENSE_DIR, DatasetChoice.DF.value)
DF_DATA_WITH_TIMESTAMP = DF_DATA_DIR

# KNN DATASET
KNN_DATA_DIR = os.path.join(DATA_DIR, NO_DEFENSE_DIR, "knndata")

# file name
TRACE_FILE = "x_data.pkl"
TIMESTAMP_FILE = "x_timestamp.pkl"
SITES_FILE = "y_data.pkl"

TRAIN_TRACE_FILE = "x_train.pkl"
VALIDATE_TRACE_FILE = "x_validate.pkl"
TEST_TRACE_FILE = "x_test.pkl"

TRAIN_SITE_FILE = "y_train.pkl"
VALIDATE_SITE_FILE = "y_validate.pkl"
TEST_SITE_FILE = "y_test.pkl"