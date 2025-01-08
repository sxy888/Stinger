#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: front.py
# Created: 2024-10-19
# Description:

import datetime
import os
import logging
import typing as t
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm

from defender.abc import AbstractDefender
from loader.base import AbstractLoader, Config
from constant import dataset, TQDM_N_COLS

logger = logging.getLogger(__name__)
MON_SITE_NUM = 100
MON_INST_NUM = 90
UNMON_SITE_NUM = 9000

start_padding_time = 0
max_wnd = 8
min_wnd = 1


class FrontDefender(AbstractDefender):

    temp_dir: str = None

    def __init__(self, loader) -> None:
        super().__init__(loader)
        # IMPORTANT !!!  influence the overhead
        self.client_dummy_pkt_num = 600
        self.server_dummy_pkt_num = 1400
        self.client_min_dummy_pkt_num = 1
        self.server_min_dummy_pkt_num = 1

    @classmethod
    def update_config(cls, config: Config) -> Config:
        config.load_data_by_self = True


    def defense(self, **kwargs):
        if kwargs.get("client_dummy_pkt_num"):
            self.client_dummy_pkt_num = kwargs["client_dummy_pkt_num"]
        if kwargs.get("server_dummy_pkt_num"):
            self.server_dummy_pkt_num = kwargs["server_dummy_pkt_num"]

        logger.info(f"the client dummy pkt num is: {self.client_dummy_pkt_num}")
        logger.info(f"the server dummy pkt num is: {self.server_dummy_pkt_num}")

        self.temp_dir = self.init_temp_directory()
        file_list = self.load_data_file_list()
        for file in tqdm.tqdm(file_list, ncols=TQDM_N_COLS):
            self.simulate(file)

        data, label, overhead = self.combine_dump_trace()
        return data, None, label, overhead


    def init_temp_directory(self):
        temp_dir = os.path.join(dataset.KNN_DATA_DIR, "temp")
        if os.path.exists(temp_dir):
            # remove the last data
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
        temp_dir_obj = Path(temp_dir)
        temp_dir_obj.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def load_data_file_list(self):
        file_list = []
        for i in range(MON_SITE_NUM):
            for j in range(MON_INST_NUM):
                file_list.append(os.path.join(
                    dataset.KNN_DATA_DIR,
                    f"{i}-{j}"
                ))
        return file_list

    def load_trace(self, file_path: str):
        with open(file_path,'r') as f:
            tmp = f.readlines()
        t = (
            pd.Series(tmp)
            .str.slice(0,-1)
            .str.split('\t',expand = True)
            .astype('float')
        )
        return np.array(t)

    def simulate(self, file_path: str):
        if not os.path.exists(file_path):
            logger.warning(f"the file path not exist: {file_path}")
            return

        np.random.seed(datetime.datetime.now().microsecond)
        trace = self.load_trace(file_path)

        trace = self.RP(trace)
        fname = file_path.split('/')[-1]
        self.dump(trace, fname)

    def RP(self, trace):
        client_wnd = np.random.uniform(min_wnd, max_wnd)
        server_wnd = np.random.uniform(min_wnd, max_wnd)
        if self.client_min_dummy_pkt_num != self.client_dummy_pkt_num:
            client_dummy_pkt = np.random.randint(
                self.client_min_dummy_pkt_num, self.client_dummy_pkt_num
            )
        else:
            client_dummy_pkt = self.client_dummy_pkt_num
        if self.server_min_dummy_pkt_num != self.server_dummy_pkt_num:
            server_dummy_pkt = np.random.randint(self.server_min_dummy_pkt_num,self.server_dummy_pkt_num)
        else:
            server_dummy_pkt = self.server_dummy_pkt_num


        first_incoming_pkt_time = trace[np.where(trace[:,1] <0)][0][0]
        last_pkt_time = trace[-1][0]

        client_timetable = self.get_timestamps(client_wnd, client_dummy_pkt)
        client_timetable = client_timetable[np.where(start_padding_time+client_timetable[:,0] <= last_pkt_time)]

        server_timetable = self.get_timestamps(server_wnd, server_dummy_pkt)
        server_timetable[:,0] += first_incoming_pkt_time
        server_timetable = server_timetable[np.where(start_padding_time+server_timetable[:,0] <= last_pkt_time)]


        # print("client_timetable")
        # print(client_timetable[:10])
        client_pkts = np.concatenate((client_timetable, 888*np.ones((len(client_timetable),1))),axis = 1)
        server_pkts = np.concatenate((server_timetable, -888*np.ones((len(server_timetable),1))),axis = 1)


        noisy_trace = np.concatenate( (trace, client_pkts, server_pkts), axis = 0)
        noisy_trace = noisy_trace[ noisy_trace[:, 0].argsort(kind = 'mergesort')]
        return noisy_trace

    def get_timestamps(self, wnd, num):
        # timestamps = sorted(np.random.exponential(wnd/2.0, num))
        # print(wnd, num)
        # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
        timestamps = sorted(np.random.rayleigh(wnd,num))
        # print(timestamps[:5])
        # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
        return np.reshape(timestamps, (len(timestamps),1))

    def dump(self, trace, fname):
        with open(os.path.join(self.temp_dir, fname), 'w') as fo:
            for packet in trace:
                fo.write("{:.4f}".format(packet[0]) +'\t' + "{}".format(int(packet[1]))\
                    + "\n")


    def combine_dump_trace(self):
        def padding(trace):
            padding_num = 5000 - len(trace)
            for i in range(padding_num):
                trace.append(0)
            return trace

        data, label = [], []
        total_packet, append_packet = 0, 0
        for i in range(MON_SITE_NUM):
            for j in range(MON_INST_NUM):
                fname = f"{i}-{j}"
                label.append(i)
                with open(os.path.join(self.temp_dir, fname), 'r') as f:
                    tmp = f.readlines()
                t = (
                    pd.Series(tmp)
                    .str.slice(0,-1)
                    .str.split('\t',expand = True)
                    .astype('float')
                )
                trace = t[1].values.tolist()
                for index in range(len(trace)):
                    if abs(trace[index]) == 888:
                        append_packet += 1
                        if trace[index] > 0:
                            trace[index] = 1.0
                        else:
                            trace[index] = -1.0
                    else:
                        total_packet += 1
                if(len(trace) >=5000):
                    trace = trace[0:5000]
                else:
                    trace = padding(trace)
                data.append(np.array(trace))

        data = np.array(data)
        overhead = append_packet / total_packet
        logger.info("the overhead is: %f" % (overhead))
        return data, label, overhead