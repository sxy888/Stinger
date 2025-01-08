import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import tqdm
import pickle

def convert_trace_data_to_burst(train_x: np.ndarray, max_length: int):
    train_burst = []
    for trace in tqdm.tqdm(train_x.tolist(), ncols=100, desc='trace to burst'):
        burst = []
        i = 0
        while trace[i] == 0:
            i += 1
        tmp_dir, tmp_packet = trace[i], 1
        for j in range(i+1, len(trace)):
            cur_dir = trace[j]
            if cur_dir != tmp_dir:
                burst.append(tmp_packet * tmp_dir)
                tmp_packet = 1 if cur_dir != 0 else 0
                tmp_dir = cur_dir
            elif cur_dir == 0:
                burst.append(0)
            else:
                tmp_packet += 1
        burst.append(tmp_packet * tmp_dir)
        while len(burst) < max_length:
            burst.append(0)
        train_burst.append(burst[:max_length])

    train_burst = np.array(train_burst)
    return train_burst


def test_covert_burst():
    arr = [ 1, -1,  1, -1, -1, -1,  1,  1, -1,  1,]
    trace = np.array([arr])
    burst = convert_trace_data_to_burst(trace, 10)
    print(burst)
    import sys
    sys.exit(0)


if __name__ == "__main__":
    # test_covert_burst()
    # p_dir = "/home/tank06/Desktop/stinger/data/ALERT/DF/overhead_7.4"
    # p = f"{p_dir}/x_test.pkl"
    # g_test_x = np.array(pickle.load(open(p, 'rb'), encoding='latin1'))
    # p = f"{p_dir}/y_test.pkl"
    # g_test_y = np.array(pickle.load(open(p, 'rb'), encoding='latin1'))
    # print(g_test_x[g_test_y == 0][0][:10])

    p = "/home/tank06/Desktop/stinger/data/NoDef/ALERT/burst_data.npz"
    data = np.load(p)
    text_x, test_y = data['test_x'], data['test_y']
    print(text_x[test_y == 0][0][:10])


    p_dir = "/home/tank06/Desktop/stinger/data/NoDef/DF"
    p = f"{p_dir}/x_test.pkl"
    trace_test_x = np.array(pickle.load(open(p, 'rb'), encoding='latin1'))
    p = f"{p_dir}/y_test.pkl"
    trace_test_y = np.array(pickle.load(open(p, 'rb'), encoding='latin1'))
    print(trace_test_x[trace_test_y == 0][0][:10])
