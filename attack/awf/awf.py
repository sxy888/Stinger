#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: awf.py
# Created: 2024-09-08
# Description:
import logging  # noqa
import typing as t
import numpy as np
from numpy import ndarray # noqa
from keras.utils import to_categorical
from keras.optimizers import SGD, RMSprop

from attack import AbstractAttack
from .model import AWFCnnNet, AWFSdaeNet
from .data import DataGenerator
logger = logging.getLogger(__name__)

class AWFCnnAttack(AbstractAttack):

    def process_data(self, x: ndarray, y: ndarray):
        return x, y

    def attack(self, **kwargs) -> None:
        batch_size = 256

        x_train = self.dataset.train_data.x
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        y_train = self.dataset.train_data.y

        x_test = self.dataset.test_data.x
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        y_test = self.dataset.test_data.y

        x_valid = self.dataset.eval_data.x
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
        y_valid = self.dataset.eval_data.y

        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)
        y_test = to_categorical(y_test)

        train_gen = DataGenerator(batch_size=batch_size).generate(
            x_train, y_train, np.arange(y_train.shape[0])
        )
        valid_gen = DataGenerator(batch_size=batch_size).generate(
            x_valid, y_valid, np.arange(y_valid.shape[0])
        )

        model = AWFCnnNet.build(self.loader.num_classes)
        optimizer = RMSprop(
            lr=0.0011,
            decay=0.0,
        )
        metrics = ['accuracy']
        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=metrics
        )
        train_steps = x_train.shape[0] // batch_size
        valid_steps = x_valid.shape[0] // batch_size
        history = model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_steps,
            validation_data=valid_gen,
            validation_steps=valid_steps,
            epochs=30,
        )
        score_test = model.evaluate(x_test, y_test)
        logger.info(f'Test Score: {score_test}')
        return score_test[1]


class AWFSdaeAttack(AbstractAttack):

    def process_data(self, x: ndarray, y: ndarray):
        return x, y

    def attack(self, **kwargs) -> None:
        x_train = self.dataset.train_data.x
        y_train = self.dataset.train_data.y

        x_test = self.dataset.test_data.x
        y_test = self.dataset.test_data.y

        x_valid = self.dataset.eval_data.x
        y_valid = self.dataset.eval_data.y

        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)
        y_test = to_categorical(y_test)

        batch_size = 32
        train_gen = DataGenerator(batch_size=batch_size).generate(
            x_train, y_train, np.arange(y_train.shape[0])
        )
        valid_gen = DataGenerator(batch_size=batch_size).generate(
            x_valid, y_valid, np.arange(y_valid.shape[0])
        )

        model = AWFSdaeNet.build(self.loader.num_classes)
        optimizer = SGD(
            lr=0.001,
            decay=0.0,
            momentum=0.9,
            nesterov=True,
        )
        metrics = ['accuracy']
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=metrics
        )
        train_steps = x_train.shape[0] // batch_size
        valid_steps = x_valid.shape[0] // batch_size
        history = model.fit_generator(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=valid_gen,
            validation_steps=valid_steps,
            epochs=30,
        )
        score_test = model.evaluate(x_test, y_test)
        logger.info(f'Test Score: {score_test}')
        return score_test[1]