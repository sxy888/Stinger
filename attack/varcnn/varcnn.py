#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: df.py
# Created: 2024-09-08
# Description:
import logging  # noqa
import time
import typing as t
import numpy as np
from numpy import ndarray # noqa
from keras import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Activation, Concatenate, Dropout
from keras.layers.normalization import BatchNormalization


from attack import AbstractAttack
from loader.base import AbstractLoader, DatasetWrapper
from .model import ResNet18, basic_1d, dilated_basic_1d
from . import generator as data_generator

logger = logging.getLogger(__name__)


class VarCnnAttack(AbstractAttack):

    def __init__(self, loader) -> None:
        super().__init__(loader)
        self.num_mon_sites = self.loader.num_classes
        self.num_mon_inst_train = 800
        self.num_mon_inst_test = 100
        self.num_mon_inst = self.num_mon_inst_train + self.num_mon_inst_test
        self.num_unmon_sites_train = 0
        self.num_unmon_sites_test = 0
        self.num_unmon_sites = 0

        self.model_name = "var-cnn"
        self.batch_size = 25
        self.seq_length = 5000
        self.epochs = 30
        self.base_patience = 5

        self.dir_dilations = True
        self.time_dilations = True
        self.inter_time = False
        self.scale_metadata = False
        self.use_timestamp = False

    def process_data(self, x: ndarray, y: ndarray):
        return x, y

    def attack(self, **kwargs) -> None:

        model, callbacks = self.get_model()

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

        train_steps = x_train.shape[0] // self.batch_size
        valid_steps = x_valid.shape[0] // self.batch_size

        train_data = data_generator.generate(
            self.batch_size, 'training_data',
            x_train, y_train,
        )
        validate_data = data_generator.generate(
            self.batch_size, 'validation_data',
            x_valid, y_valid
        )
        logger.info("Start training")
        model.fit_generator(
            train_data,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=validate_data,
            validation_steps=valid_steps,
            shuffle=True,
        )
        logger.info("Finished training")

        score_test = model.evaluate(x_test, y_test)
        logger.info(f'Test Score: {score_test}')
        return score_test[1]

    def evaluate(self, model) -> float:
        test_size = self.num_mon_sites * self.num_mon_inst_test + self.num_unmon_sites_test
        test_steps = test_size // self.batch_size
        # predictions = model.predict_generator(
        #     test_data,
        #     steps=test_steps if test_size % self.batch_size == 0 else test_steps + 1,
        #     verbose=0,
        # )

    def get_model(self):
        """
        Returns Var-CNN model
        """
        dir_input = Input(shape=(self.seq_length, 1), name='dir_input')
        if self.dir_dilations:
            dir_output = ResNet18(dir_input, 'dir', block=dilated_basic_1d)
        else:
            dir_output = ResNet18(dir_input, 'dir', block=basic_1d)

        if self.use_timestamp:
            time_input = Input(shape=(self.seq_length, 1,), name='time_input')
            if self.time_dilations:
                time_output = ResNet18(time_input, 'time', block=dilated_basic_1d)
            else:
                time_output = ResNet18(time_input, 'time', block=basic_1d)

        input_params = []
        concat_params = []

        input_params.append(dir_input)
        concat_params.append(dir_output)

        if self.use_timestamp:
            input_params.append(time_input)
            concat_params.append(time_output)

        if len(concat_params) == 1:
            combined = concat_params[0]
        else:
            combined = Concatenate()(concat_params)

        # Better to have final fc layer if combining multiple models
        if len(concat_params) > 1:
            combined = Dense(1024)(combined)
            combined = BatchNormalization()(combined)
            combined = Activation('relu')(combined)
            combined = Dropout(0.5)(combined)

        # output_classes = self.num_mon_sites if self.num_unmon_sites == 0 else self.num_mon_sites + 1
        model_output = Dense(
            units=self.loader.num_classes,
            activation='softmax',
            name='model_output'
        )(combined)

        model = Model(inputs=input_params, outputs=model_output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(0.001),
            metrics=['accuracy']
        )
        lr_reducer = ReduceLROnPlateau(
            monitor='val_acc',
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=self.base_patience,
            min_lr=1e-5,
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_acc',
            patience=2 * self.base_patience
        )
        # model_checkpoint = ModelCheckpoint('model_weights.h5', monitor='val_acc',
        #                                    save_best_only=True,
        #                                    save_weights_only=True, verbose=1)

        # callbacks = [lr_reducer, early_stopping, model_checkpoint]
        callbacks = [lr_reducer, early_stopping]
        # model.summary()
        # exit(0)
        return model, callbacks