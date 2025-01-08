#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: log.py
# Created: 2024-08-17
# Description: 配置彩色日志和日志路径格式等信息
import os
import logging
import coloredlogs
import warnings
from tensorflow.python.util import deprecation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

class MyFormatter(coloredlogs.ColoredFormatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 定义路径名的颜色，例如使用青色
        self.path_color = "\033[36m"  # ANSI 转义序列中的青色
        self.reset_color = "\033[0m"   # 重置颜色到默认

    def format(self, record):
        log_record = super().format(record)
        if getattr(record, 'pathname', None) is not None:
            pathname: str = record.pathname
            short_pathname = record.pathname
            line_no = record.lineno
            # 文件路径进行省略
            if pathname.find("stinger") == -1:
                index = pathname.rfind("site-packages/")
                if index != -1:
                    short_pathname = pathname[index+14]
            else:
                short_pathname = pathname[pathname.rfind("stinger") :]
            colored_pathname = f"{self.path_color}{short_pathname}:{line_no}{self.reset_color}"

            return log_record.replace(pathname+f":{line_no}", colored_pathname)
        return log_record


def set_logging_formatter(level="INFO"):
    # configure the log format
    formatter = MyFormatter(
        fmt="%(asctime)s %(levelname)-8s %(pathname)s:%(lineno)d %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level), handlers=[
        handler,
    ])

    # set matplotlib log level
    logging.getLogger(
import os
import psutil
import typing as t
import numpy as np
from enum import Enum

def get