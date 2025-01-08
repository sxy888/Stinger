import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Lambda, Conv2DTranspose

logger = logging.getLogger(__name__)
NB_CLASSES = 95  # number of outputs = number of classes

def loss_plot(g1_l, g2_l, d1_l, d2_l=None, path=None, ind=None):
    x = np.arange(len(g1_l))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss graph")
    plt.plot(x, np.array(g1_l), label="G1_Loss")
    plt.plot(x, g2_l, label="G2_Loss")
    if d1_l != None:
        plt.plot(x, d1_l, label="D_Loss")
    if d2_l != None:
        plt.plot(x, d2_l, label="D2_Loss")
    plt.legend()
    if path == None:
        plt.show()
    else:
        plt.savefig(path + 'loss_' + str(ind))
    plt.close()


import PIL.Image


def visualize(trace_len, src=None, trg=None, g1=None, g2=None, path=None, ind=None):
    src = np.round(np.sum(src, axis=0) / len(src)).reshape(1, trace_len, 1)
    trg = np.round(np.sum(trg, axis=0) / len(trg)).reshape(1, trace_len, 1)
    g1 = np.round(np.sum(g1, axis=0) / len(g1)).reshape(1, trace_len, 1)
    g2 = np.round(np.sum(g2, axis=0) / len(g2)).reshape(1, trace_len, 1)
    plt.plot(src[0], '.', color='blue')
    plt.plot(trg[0], '.', color='red')
    if True:
        plt.plot(g1[0], '.', color='red')
        plt.plot(g2[0], '.', color='blue')
    plt.plot(np.zeros(src.shape)[0], '.', color='black')
    plt.ylabel('value')
    plt.xlabel('ind')
    if True:
        plt.legend(['src', 'trg', 'g1', 'g2'], loc='lower right')

    #   plt.legend([str(c1),str(c2),'zero'], loc='lower right')
    if path == None:
        plt.show()
    else:
        plt.savefig(path + 'vis_' + str(ind))
    plt.close()


def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_overhead(name="", original_data=None, manipulated_data=None, pr=0, flush=False):
    oh = np.round((np.sum(np.abs(manipulated_data) - np.abs(original_data)) / np.sum(np.abs(original_data))) * 100, 2)
    if pr == 1:
        if flush:
            print("", flush=True)
        logger.info(f"{name} overhead: {oh}")
    return oh


def oh_acc_plot(oh1, oh2, oh3, oh4, acc_list, path=None, ind=None):
    x_oh = np.array(oh1)[:, 1]
    if len(acc_list) > 0:
        x_acc = np.array(acc_list)[:, 1]
    plt.xlabel("Step")
    plt.title("Overhead vs Accuracy under 70%")
    plt.plot(x_oh, np.array(oh1)[:, 0], label="G1_OH_train")
    plt.plot(x_oh, np.array(oh2)[:, 0], label="G1_OH_test")
    plt.plot(x_oh, np.array(oh3)[:, 0], label="G2_OH_train")
    plt.plot(x_oh, np.array(oh4)[:, 0], label="G2_OH_test")
    plt.plot(x_oh, np.ones(len(x_oh)) * 20, label="Baseline", color='red')
    if len(acc_list) > 0:
        plt.plot(x_acc, np.array(acc_list)[:, 0], label="Acc < 70%", marker='o')
    plt.legend()
    if path == None:
        plt.show()
    else:
        plt.savefig(path + 'OH_Acc_' + str(ind))
    plt.close()


class ct1d():
    @staticmethod
    def Conv1DTranspose(model, filters=16, kernel_size=3, strides=2, padding='same'):
        model.add(Lambda(lambda x: tf.expand_dims(x, axis=2)))
        model.add(Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding))
        model.add(Lambda(lambda x: tf.squeeze(x, axis=2)))


ct = ct1d()


class data_class():

    def set_information(self, src_cls, trg_cls, oh_src_train, oh_trg_train):
        self.src_cls = src_cls
        self.trg_cls = trg_cls

        self.oh_src_train = oh_src_train
        self.oh_trg_train = oh_trg_train

    def set_data(self, clean_src_trans_train, g_src_trans_train, clean_trg_trans_train, g_trg_trans_train,
                 clean_src_user_train, g_src_user_train, clean_trg_user_train, g_trg_user_train,
                 clean_src_valid, g_src_valid, clean_trg_valid, g_trg_valid, clean_src_test, g_src_test, clean_trg_test,
                 g_trg_test):
        self.clean_src_trans_train = clean_src_trans_train
        self.g_src_trans_train = g_src_trans_train
        self.clean_trg_trans_train = clean_trg_trans_train
        self.g_trg_trans_train = g_trg_trans_train

        self.clean_src_user_train = clean_src_user_train
        self.g_src_user_train = g_src_user_train
        self.clean_trg_user_train = clean_trg_user_train
        self.g_trg_user_train = g_trg_user_train

        self.clean_src_valid = clean_src_valid
        self.g_src_valid = g_src_valid
        self.clean_trg_valid = clean_trg_valid
        self.g_trg_valid = g_trg_valid

        self.clean_src_test = clean_src_test
        self.g_src_test = g_src_test
        self.clean_trg_test = clean_trg_test
        self.g_trg_test = g_trg_test
