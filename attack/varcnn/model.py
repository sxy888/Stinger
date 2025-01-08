from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Activation, ZeroPadding1D, \
    GlobalAveragePooling1D, Add, Concatenate, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import Input

import numpy as np
parameters = {'kernel_initializer': 'he_normal'}

# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def dilated_basic_1d(filters, suffix, stage=0, block=0, kernel_size=3,
                     numerical_name=False, stride=None,
                     dilations=(1, 1)):
    """A one-dimensional basic residual block with dilations.

    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param dilations: tuple representing amount to dilate first and second conv layers
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if block > 0 and numerical_name:
        block_char = 'b{}'.format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = Conv1D(filters, kernel_size, padding='causal', strides=stride,
                   dilation_rate=dilations[0], use_bias=False,
                   name='res{}{}_branch2a_{}'.format(
                       stage_char, block_char, suffix), **parameters)(x)
        y = BatchNormalization(epsilon=1e-5,
                               name='bn{}{}_branch2a_{}'.format(
                                   stage_char, block_char, suffix))(y)
        y = Activation('relu',
                       name='res{}{}_branch2a_relu_{}'.format(
                           stage_char, block_char, suffix))(y)

        y = Conv1D(filters, kernel_size, padding='causal', use_bias=False,
                   dilation_rate=dilations[1],
                   name='res{}{}_branch2b_{}'.format(
                       stage_char, block_char, suffix), **parameters)(y)
        y = BatchNormalization(epsilon=1e-5,
                               name='bn{}{}_branch2b_{}'.format(
                                   stage_char, block_char, suffix))(y)

        if block == 0:
            shortcut = Conv1D(filters, 1, strides=stride, use_bias=False,
                              name='res{}{}_branch1_{}'.format(
                                  stage_char, block_char, suffix),
                              **parameters)(x)
            shortcut = BatchNormalization(epsilon=1e-5,
                                          name='bn{}{}_branch1_{}'.format(
                                              stage_char, block_char,
                                              suffix))(shortcut)
        else:
            shortcut = x

        y = Add(name='res{}{}_{}'.format(stage_char, block_char, suffix))(
            [y, shortcut])
        y = Activation('relu',
                       name='res{}{}_relu_{}'.format(stage_char, block_char,
                                                     suffix))(y)

        return y

    return f


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def basic_1d(filters, suffix, stage=0, block=0, kernel_size=3,
             numerical_name=False, stride=None, dilations=(1, 1)):
    """A one-dimensional basic residual block without dilations.

    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param dilations: tuple representing amount to dilate first and second conv layers
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    dilations = (1, 1)

    if block > 0 and numerical_name:
        block_char = 'b{}'.format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = Conv1D(filters, kernel_size, padding='same', strides=stride,
                   dilation_rate=dilations[0], use_bias=False,
                   name='res{}{}_branch2a_{}'.format(stage_char, block_char,
                                                     suffix), **parameters)(x)
        y = BatchNormalization(epsilon=1e-5,
                               name='bn{}{}_branch2a_{}'.format(
                                   stage_char, block_char, suffix))(y)
        y = Activation('relu',
                       name='res{}{}_branch2a_relu_{}'.format(
                           stage_char, block_char, suffix))(y)

        y = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   dilation_rate=dilations[1],
                   name='res{}{}_branch2b_{}'.format(
                       stage_char, block_char, suffix), **parameters)(y)
        y = BatchNormalization(epsilon=1e-5,
                               name='bn{}{}_branch2b_{}'.format(
                                   stage_char, block_char, suffix))(y)

        if block == 0:
            shortcut = Conv1D(filters, 1, strides=stride, use_bias=False,
                              name='res{}{}_branch1_{}'.format(
                                  stage_char, block_char, suffix),
                              **parameters)(x)
            shortcut = BatchNormalization(epsilon=1e-5,
                                          name='bn{}{}_branch1_{}'.format(
                                              stage_char, block_char,
                                              suffix))(shortcut)
        else:
            shortcut = x

        y = Add(name='res{}{}_{}'.format(stage_char, block_char, suffix))(
            [y, shortcut])
        y = Activation('relu',
                       name='res{}{}_relu_{}'.format(stage_char, block_char,
                                                     suffix))(y)

        return y

    return f


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def ResNet18(inputs, suffix, blocks=None, block=None, numerical_names=None):
    """Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of
        `keras_resnet.blocks.basic_2d`)
    :param numerical_names: list of bool, same size as blocks, used to
        indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`)
        or classification output (if `include_top=True`)
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]
    if block is None:
        block = dilated_basic_1d
    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = ZeroPadding1D(padding=3, name='padding_conv1_' + suffix)(inputs)
    x = Conv1D(64, 7, strides=2, use_bias=False, name='conv1_' + suffix)(x)
    x = BatchNormalization(epsilon=1e-5, name='bn_conv1_' + suffix)(x)
    x = Activation('relu', name='conv1_relu_' + suffix)(x)
    x = MaxPooling1D(3, strides=2, padding='same', name='pool1_' + suffix)(x)

    features = 64
    outputs = []

    for stage_id, iterations in enumerate(blocks):
        x = block(features, suffix, stage_id, 0, dilations=(1, 2),
                  numerical_name=metadata
            metadata_output = BatchNormalization()(metadata_output)
            metadata_output = Activation('relu')(metadata_output)

        # Forms input and output lists and possibly add final dense layer
        input_params = []
        concat_params = []
        if use_dir:
            input_params.append(dir_input)
            concat_params.append(dir_output)
        if use_time:
            input_params.append(time_input)
            concat_params.append(time_output)
        if use_metadata:
            input_params.append(metadata_input)
            concat_params.append(metadata_output)

        if len(concat_params) == 1:
            combined = concat_params[0]
        else:
            combined = Concatenate()(concat_params)

        # Better to have final fc layer if combining multiple models
        if len(concat_params) > 1:
            combined = Dense(1024)(combined)
            combined = BatchNormalization()(combined)
            combined = Activation('relu')(combined)
            backs = [lr_reducer, early_stopping, model_checkpoint]
        return model, callbacks                                                                                                                                                                                                                                                                                                                                                                                                                                          stinger-release/attack/abc.py                                                                       0000644 0001762 0000033 00000002523 14737363242 015365  0                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   #!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: abc.py
# Created: 2024-09-08
# Description: abstract class for attack methods

import logging  # noqa
import typing as t # noqa
import numpy as np
from abc import ABC, abstractmethod

from loader import AbstractLoader, Config
from loader.base import DatasetWrapper

logger = logging.getLogger(__name__)



class AbstractAttack(ABC):

    X: t.Union[None, np.ndarray] = None
    Y: t.Union[None, np.ndarray] = None
    dataset: t.Union[None, DatasetWrapper] = None

    def __init__(self, loader: t.Union[AbstractLoader, t.Type[AbstractLoader]]) -> None:
        if isinstance(loader, AbstractLoader):
            self.loader = loader
        else:
            self.loader = loader({})

    def run(self) -> float:
        self.loader.load()
        original_x = self.loader.original_data
        original_y = self.loader.original_labels
        self.X, self.Y = self.process_data(original_x, original_y)
        self.dataset = self.loader.split()

        # attack
        logger.info("%s begin running" % self.__class__.__name__)
        acc = self.attack()
        if not acc:
            return 0
        else:
            return acc

    @abstractmethod
    def process_data(self, x: np.ndarray, y: np.ndarray):
        pass


    @abstractmethod
    def attack(self, **kwargs) -> None:
        pass                                                                                                                                                                             stinger-release/attack/__pycache__/                                                                 0000755 0001762 0000033 00000000000 14737363242 016474  5                                                                                                    ustar   tank06                          sudo                                                                                                                                                                                                                   stinger-release/attack/__pycache__/factory.cpython-312.pyc                            