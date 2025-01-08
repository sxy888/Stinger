import os
import sys
import tqdm
import pickle
import numpy as np
import random
import logging

from loader import AbstractLoader
from constant import TQDM_N_COLS
from constant.dataset import  DF_DATA_WITH_TIMESTAMP, AWF_DATA_DIR, DATA_DIR, NO_DEFENSE_DIR
from constant.enum import DatasetChoice

logger = logging.getLogger(__name__)
TRACE_LENGTH = 2000
num_samples_in_Gs_training = 400

def burst_to_seq(data, saveDir, name):
    finalArray = []
    for i in range(len(data)):
        midData = data[i]
        midData = np.round(midData)
        tmpData = np.array([])
        for j in range(len(midData)):
            x = np.full(int(abs(midData[j])), np.sign(midData[j]))
            tmpData = np.append(tmpData, x)
        if len(tmpData) >= 5000:
            tmpData = tmpData[0:5000]
        else:
            extra = np.zeros(5000 - len(tmpData))
            tmpData = np.append(tmpData, extra)
        finalArray.append(tmpData)
    finalArray = np.array(finalArray)
    np.save(saveDir + name, finalArray)

def dir_to_burst(dir_data):
    burst_data = np.zeros(dir_data.shape)
    logger.debug(f"begin convert dir data to burst data: {dir_data.shape}")
    for i in tqdm.tqdm(range(len(dir_data)), ncols=TQDM_N_COLS):
        k = 0
        dir_sign = dir_data[i][0]
        for j in range(TRACE_LENGTH):
            if dir_data[i][j] == 0:
                break
            if dir_data[i][j] != dir_sign:
                k += 1
                dir_sign = dir_data[i][j]
            burst_data[i][k] += dir_data[i][j]
    return burst_data


def LoadDataNoDefCW(loader: AbstractLoader):
    logger.info("Loading non-defended dataset for closed-world scenario")

    X_train = loader.dataset_wrapper.train_data.x
    y_train = loader.dataset_wrapper.train_data.y

    X_valid = loader.dataset_wrapper.eval_data.x
    y_valid = loader.dataset_wrapper.eval_data.y

    X_test = loader.dataset_wrapper.test_data.x
    y_test = loader.dataset_wrapper.test_data.y

    num_classes = loader.num_classes
    if loader.num_classes % 2 == 1:
        num_classes -= 1
        remove_site_idx = loader.num_classes  -1
        X_train = X_train[y_train != remove_site_idx]
        y_train = y_train[y_train != remove_site_idx]

        X_valid = X_valid[y_valid != remove_site_idx]
        y_valid = y_valid[y_valid != remove_site_idx]

        X_test = X_test[y_test != remove_site_idx]
        y_test = y_test[y_test != remove_site_idx]

    y_valid = np.eye(num_classes)[y_valid]
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]

    X_train = X_train[:, :, np.newaxis]
    X_valid = X_valid[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    return dir_to_burst(X_train), y_train, dir_to_burst(X_valid), y_valid, dir_to_burst(X_test), y_test


class DFDataLoader():
    """
    An image loader that uses just a few ImageNet-like images.
    """

    def __init__(self, loader: AbstractLoader):
        # if loader.config.dataset_choice == DatasetChoice.DF:
        #     burst_data_path = os.path.join(DF_DATA_WITH_TIMESTAMP, 'burst_data.npz')
        # else:
        #     burst_data_path = os.path.join(AWF_DATA_DIR, 'burst_data.npz')
        burst_data_path = os.path.join(
            DATA_DIR, "AWA", NO_DEFENSE_DIR, "burst_data.npz"
        )
        # judge if the burst data exists
        load_burst_data = os.path.isfile(burst_data_path)


        if not load_burst_data:
            load_result = LoadDataNoDefCW(loader)
            self.train_data, self.train_labels = load_result[0], load_result[1]
            self.validation_data, self.validation_labels = load_result[2], load_result[3]
            self.test_data, self.test_labels = load_result[4], load_result[5]
            logger.debug(f"save burst data to {burst_data_path}")
            np.savez(
                burst_data_path,
                train_x=self.train_data,
                train_y=self.train_labels,
                val_x=self.validation_data,
                val_y=self.validation_labels,
                test_x=self.test_data,
                test_y=self.test_labels
            )

        nb_classes = loader.num_classes
        # if nb_classes % 2 == 1:
        #     nb_classes -= 1

        burst_data = np.load(burst_data_path)
        train_data = burst_data['train_x'][:, :TRACE_LENGTH]
        train_labels = burst_data['train_y']
        self.validation_data  = burst_data['val_x'][:, :TRACE_LENGTH]
        self.validation_labels = burst_data['val_y']
        self.test_data  = burst_data['test_x'][:, :TRACE_LENGTH]
        self.test_labels = burst_data['test_y']

        transformer_train_x = None
        transformer_train_y = []
        user_train_x = None
        user_train_y = []
        for i in range(nb_classes):
            data_x = train_data[np.argmax(train_labels, axis=1) == i]
            if len(data_x) < 800:
                logger.warning(f"the site {i} data less than 800")
                sys.exit(1)
            data_y = train_labels[np.argmax(train_labels, axis=1) == i]
            if transformer_train_x is None:
                transformer_train_x = data_x[:400]
            else:
                transformer_train_x = np.concatenate((transformer_train_x, data_x[:400]))
            transformer_train_y.append(data_y[:400])
            if user_train_x is None:
                user_train_x = data_x[400:800]
            else:
                user_train_x = np.concatenate((user_train_x,data_x[400:800]))
            user_train_y.append(data_y[400:800])


        self.transformer_train_x = np.array(transformer_train_x).reshape(nb_classes * 400, TRACE_LENGTH, 1)
        self.transformer_train_y = np.array(transformer_train_y).reshape(nb_classes * 400, nb_classes)
        self.user_train_x = np.array(user_train_x).reshape(nb_classes * 400, TRACE_LENGTH, 1)
        self.user_train_y = np.array(user_train_y).reshape(nb_classes * 400, nb_classes)
        self.nb_classes = nb_classes

    def get_DF_data(self, data_type, batch_size=None, target_cls=None, src_cls=None):
        assert bool(target_cls != None) != bool(src_cls != None), "BAD func call"
        if data_type == 'transformer_train':
            data = self.transformer_train_x
            label = self.transformer_train_y
        elif data_type == 'user_train':
            data = self.user_train_x
            label = self.user_train_y
        elif data_type == 'validation':
            data = self.validation_data
            label = self.validation_labels
        elif data_type == 'test':
            data = self.test_data
            label = self.test_labels
        else:
            print('Please set the data_type variable')
        # 缩小数据规模
        # ratio = 0.2
        # ratio_length = int(len(data) * ratio )
        # data = data[0:ratio_length]
        # label = label[0:ratio_length]

        if target_cls != None:
            other_class_data = data[np.argmax(label, axis=1) != target_cls]
            other_class_labels = label[np.argmax(label, axis=1) != target_cls]
            if batch_size != None:
                sample = random.sample(list(np.arange(other_class_data.shape[0])), batch_size)
                return other_class_data[sample], other_class_labels[sample]
            return other_class_data, other_class_labels
        if src_cls != None:
            src_class_data = data[np.argmax(label, axis=1) == src_cls]
            src_class_labels = label[np.argmax(label, axis=1) == src_cls]
            if batch_size != None:
                sample = random.sample(list(np.arange(src_class_data.shape[0])), batch_size)
                return src_class_data[sample], src_class_labels[sample]
            return src_class_data, src_class_labels