#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: tamaraw.py
# Created: 2024-09-08
# Description:


import logging  # noqa
import math
import os
import tqdm
import random
import typing as t # noqa
import numpy as np

from defender.abc import AbstractDefender
from loader.base import AbstractLoader, Config
from constant import TQDM_N_COLS, dataset


logger = logging.getLogger(__name__)

'''params'''
MON_SITE_NUM = 100
MON_INST_NUM = 90
DATA_SIZE = 800
DUMMY_CODE = 1


def f_sign(num):
    if num > 0:
        return 0
    else:
        return 1


def r_sign(num):
    if num == 0:
        return 1
    else:
        return abs(num) / num

class TamarawDefender(AbstractDefender):

    def __init__(self, loader) -> None:
        super().__init__(loader)
        # 用来控制插入数据的多少，两者之和影响 overhead 和防御成功率
        self.in_anoa_time = 0.008
        self.out_anoa_time = 0.04
        self.PadL = 10

    @classmethod
    def update_config(cls, config: Config) -> Config:
        config.load_data_by_self = True

    # @classmethod
    # def update_config(cls, config: Config):
    #     config.need_timestamp = True

    def defense(self, **kwargs) -> None:

        if kwargs.get("in_anoa_time") and isinstance(kwargs["in_anoa_time"], float):
            self.in_anoa_time = kwargs["in_anoa_time"]
        if kwargs.get("out_anoa_time") and isinstance(kwargs["out_anoa_time"], float):
            self.out_anoa_time = kwargs["out_anoa_time"]
        if kwargs.get("pad_l") is not None:
            self.PadL = kwargs["pad_l"]

        logger.info(f"Config info: in_anoa_time [{self.in_anoa_time}], out_anoa_time [{self.out_anoa_time}]")
        logger.info(f"Config info: PadL [{self.PadL}]")

        timestamp_set = self.Timestamp
        traces = []
        time = []
        real_length = float(1)
        extra = 0

        file_list = []  # type: t.List[str]
        for i in range(MON_SITE_NUM):
            for j in range(MON_INST_NUM):
                filename = os.path.join(dataset.KNN_DATA_DIR, f"{i}-{j}")
                file_list.append(filename)

        label = []
        for filename in tqdm.tqdm(file_list, ncols=TQDM_N_COLS):
            f = open(filename, 'r')
            lines = f.readlines()
            packets = []
            starttime = float(lines[0].split("\t")[0])
            for x in lines:
                x = x.split("\t")
                packets.append([float(x[0]) - starttime, int(x[1])])
                if len(packets) >= 2500:
                    break

            list2 = []
            parameters = [""]

            self.anoa(packets, list2, parameters)
            list2 = sorted(list2, key=lambda list2: list2[0])

            list3 = []
            append_packet = self.anoa_pad(list2, list3, self.PadL, 0)

            list3_data = []
            for x in packets:
                list3_data.append(x[1])
            for x in append_packet:
                list3_data.append(x[1])
            real_length += len(list2)
            extra += len(append_packet)

            trace = []
            for i in range(5000):
                if len(list3_data) > i:
                    if list3_data[i] == 0:
                        _packet = 0
                    elif list3_data[i] > 0:
                        _packet = 1
                    else:
                        _packet = -1
                    # _packet = list3_data[i]
                    trace.append(_packet)
                else:
                    trace.append(0)
            traces.append(trace)
            site = filename[filename.rfind('/')+1:filename.rfind('-')]
            label.append(int(site))

        logger.info("extra: %d, real_length: %d", extra, real_length)
        overhead = extra / real_length
        logger.info("the overhead is: %f" % overhead)
        logger.info(f"trace length: {len(traces)}")
        return np.array(traces), self.Timestamp, np.array(label), overhead


    def anoa(self, list1, list2, parameters):
        # Does NOT do padding, because ambiguity set analysis.
        # list1 WILL be modified! if necessary rewrite to temp ify list1.
        # list1: packets 原始数据
        start_time = list1[0][0]
        times = [start_time, start_time]  # last pos time, last neg time
        cur_time = start_time
        lengths = [0, 0]
        data_size = DATA_SIZE
        method = 0
        if method == 0:
            parameters[0] = "Constant packet rate: " + str(self.anoa_time([0, 0])) + ", " + str(self.anoa_time([1, 0])) + ". "
            parameters[0] += "Data size: " + str(data_size) + ". "
        if method == 1:
            parameters[0] = "Time-split varying bandwidth, split by 0.1 seconds. "
            parameters[0] += "Tolerance: 2x."

        list_ind = 0  # marks the next packet to send

        while list_ind < len(list1):
            # decide which packet to send
            anoa_out_time = times[0] + self.anoa_time([0, method, times[0] - start_time])
            anoa_in_time = times[1] + self.anoa_time([1, method, times[1] - start_time])
            if anoa_out_time < anoa_in_time:
                cur_sign = 0
            else:
                cur_sign = 1
            # 决定下一个包是入向的还是出向的
            times[cur_sign] += self.anoa_time([cur_sign, method, times[cur_sign] - start_time])
            cur_time = times[cur_sign]

            to_send = data_size
            # 如果当前下标的网络包时间小于要填充的时间
            # 并且网络包的方向刚好和现在的网络包方向相反
            # 并且 to_send 大于 0
            while list1[list_ind][0] <= cur_time and f_sign(list1[list_ind][1]) == cur_sign and to_send > 0:
                # 如果要发送的网络包大于当前下标的网络包大小
                if to_send >= abs(list1[list_ind][1]):
                    # 要发送的报文数据就 --
                    to_send -= abs(list1[list_ind][1])
                    list_ind += 1
                else:
                    # 原始数据中的网络包要减去剩余要发送的数据大小，r_sign 是保证原来的网络方向
                    list1[list_ind][1] = (abs(list1[list_ind][1]) - to_send) * r_sign(list1[list_ind][1])
                    to_send = 0
                if list_ind >= len(list1):
                    break
            if cur_sign == 0:
                list2.append([cur_time, data_size])
            else:
                list2.append([cur_time, -data_size])
            lengths[cur_sign] += 1

    def anoa_time(self, parameters):
        direction = parameters[0]  # 0 out, 1 in
        method = parameters[1]
        # !IMPORTANT  !!!!!! 两者之和决定 overhead
        if method == 0:
            if direction == 0:
                return self.out_anoa_time
            if direction == 1:
                return self.in_anoa_time

    def anoa_pad(self, list1, list2, padL, method):  #
        # idx 是外层循环的下标，不影响程序逻辑，只用来输出
        lengths = [0, 0]  # in out
        times = [0, 0]  # in out
        for x in list1:
            if x[1] > 0:  # 1 in
                lengths[0] += 1
                times[0] = x[0]
            else:  # 0 out
                lengths[1] += 1
                times[1] = x[0]
            list2.append(x)

        paddings = []

        for j in range(0, 2):
            cur_time = times[j]
            to_pad = -int(math.log(random.uniform(0.00001, 1), 2) - 1)  # 1/2 1, 1/4 2, 1/8 3, ... #check this 0.00001
            if method == 0:
                if padL == 0:
                    to_pad = 0
                else:
                    to_pad = (lengths[j] // padL + to_pad) * padL

            while lengths[j] < to_pad:
                cur_time += self.anoa_time([j, 0])
                if j == 0:
                    paddings.append([cur_time,  DATA_SIZE])
                else:
                    paddings.append([cur_time, - DATA_SIZE])
                lengths[j] += 1
        # list2.extend(paddings)
        return paddings