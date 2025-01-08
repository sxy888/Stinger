from . import constants as ct
from .constants import IN, OUT, TOR_CELL_SIZE

import logging

logger = logging.getLogger('ranpad')


import numpy as np



def create(num):
    t = np.random.exponential(0.0133529, num)
    timestamp = [0]
    tmp = 0
    for i in range(num):
        tmp += t[i]
        timestamp.append(tmp)
    return np.array(timestamp)


# def parse1(fpath):
#     '''Parse a file assuming format.'''
#     timestamp_list = []
#     for line in open(fpath):
#         try:
#             timestamp, length = line.strip().split(ct.TRACE_SEP)
#             timestamp_list.append(timestamp)
#         except ValueError:
#             logger.warn("Could not split line: %s in %s", line, fpath)
#             continue
#     return timestamp_list

# def parse(fpath):
#     '''Parse a file assuming format.'''
#     t = Trace()
#     for line in open(fpath):
#         try:
#             timestamp, length = line.strip().split(ct.TRACE_SEP)
#             direction = int(length) // abs(int(length))
#             t.append(Packet(float(timestamp), direction, abs(int(length))))
#         except ValueError:
#             logger.warn("Could not split line: %s in %s", line, fpath)
#             continue
#     return t
def parse(data, timestamp_list):
    '''Parse a file assuming format.'''
    t = Trace()
    # timestamp_list = create(len(data) - 1)
    for i in range(len(data)):
        t.append(Packet(float(timestamp_list[i]), int(data[i]), 1))
    return t


def dump(trace, fpath):
    '''Write trace packet into file `fpath`.'''
    with open(fpath, 'w') as fo:
        for packet in trace:
            fo.write("{:.5f}".format(packet.timestamp) + '\t' + "{}".format(packet.direction * packet.length) \
                     + ct.NL)


class Packet(object):
    """Define a packet.

    Direction is defined in the wfpaddef const.py.
    """
    payload = None

    def __init__(self, timestamp, direction, length, dummy=False):
        self.timestamp = timestamp
        self.direction = direction
        self.length = length
        self.dummy = dummy

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return '\t'.join(map(str, [self.timestamp, self.direction * self.length]))


class Trace(list):
    """Define trace as a list of Packets."""

    _index = 0

    def __init__(self, list_packets=None):
        if list_packets:
            for p in list_packets:
                self.append(p)

    def __getslice__(self, i, j):
        return Trace(list_packets=list.__getslice__(self, i, j))

    def __add__(self, new_packet):
        t = Trace(self.pcap)
        l = list.__add__(self, new_packet)
        for e in l:
            t.append(e)
        return t

    def __mul__(self, other):
        return Trace(list.__mul__(self, other))

    def get_next_by_direction(self, i, direction):
        # print(i,direction, len(self))
        flag = 0
        for j, p in enumerate(self[i + 1:]):
            if p.direction == direction:
                flag = 1
                return i + j + 1
        if flag == 0:
            return -1

    def next(self):
        try:
            i = self[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return i


class Flow(Trace):
    """Provides a structure to keep flow dependent variables."""

    def __init__(self, direction):
        """Initialize direction and state of the flow."""
        self.direction = direction
        self.expired = False
        self.timeout = 0.0
        self.state = ct.BURST
        Trace.__init__(self)