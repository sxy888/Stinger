#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: wtfpad.py
# Created: 2024-09-08
# Description:
import tqdm
import logging  # noqa
import typing as t # noqa
import numpy as np
import configparser
from defender.abc import AbstractDefender
from loader.base import AbstractLoader, Config
from .adaptive import AdaptiveSimulator
from .pparser import Trace, parse
from .constants import CONFIG_FILE

logger = logging.getLogger(__name__)

class WTFPADDefender(AbstractDefender):

    def __init__(self, loader: AbstractLoader) -> None:
        super().__init__(loader)
        self.ratio = 1.0

    @classmethod
    def update_config(cls, config: Config) -> Config:
        config.need_timestamp = True

    def defense(self, **kwargs) -> None:
        if kwargs.get("ratio"):
            self.ratio = kwargs["ratio"]

        conf_parser = configparser.RawConfigParser()
        conf_parser.read(CONFIG_FILE)
        # Get section in config file
        config = conf_parser._sections["normal_rcv"]
        # Use default values if not specified
        config = dict(config, **conf_parser._sections['normal_rcv'])

        wtfpad = AdaptiveSimulator(config, self.ratio)

        real_length = float(0)
        extra = 0
        trace_list = []
        time_list = []
        label = []
        for i in tqdm.tqdm(range(len(self.X))):
            timestamp = self.Timestamp[i]
            data = self.X[i]
            data = data[data != 0]
            sub_timestamp = timestamp[0: len(data)]
            real_length += len(data)  # compute the length of all traces
            trace = parse(data, sub_timestamp)
            simulated = wtfpad.simulate(Trace(trace))
            sub_trace = np.zeros(5000)
            direction_seq = np.array([j.direction for j in simulated])
            sub_timestamp = np.array([j.timestamp for j in simulated])
            tmp_extra = len(direction_seq) - len(data)  # overhead of each trace
            extra += tmp_extra
            if len(direction_seq) > 5000:
                direction_seq = direction_seq[0:5000]
                sub_timestamp = sub_timestamp[0:5000]

            sub_trace[0:len(direction_seq)] = direction_seq
            # timestamp[0:le0=ftn(sub_timestamp)] = sub_timestamp
            trace_list.append(sub_trace)
            time_list.append(sub_timestamp)
            label.append(self.Y[i])

        overhead = extra / real_length
        logger.info("the overhead is: %f" % (overhead))
        return np.array(trace_list), np.array(time_list), np.array(label), overhead