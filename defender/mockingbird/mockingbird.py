#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: mockingbird.py
# Created: 2024-10-19
# Description:
import logging
import os
import random
import copy
import time
import typing as t
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from numpy import ndarray
from keras import backend as K
from keras.optimizers import  Adamax, RMSprop
import tqdm

import constant
from constant.enum import DefenderChoice
import constant.model
from defender.abc import AbstractDefender
from loader.base import Config
from . import mockingbird_utility as mb_utility

logger = logging.getLogger(__name__)

VERBOSE = 2
VALIDATION_SPLIT = 0.1

tf.set_random_seed(1234)
rng = np.random.RandomState([2020, 11, 1])
learning_rate = 0.002
prob_threshold = 0.01
confidence_threshold = 0.0001
num_bursts = feat = 750

class MockingBirdDefender(AbstractDefender):

    def __init__(self, loader) -> None:
        super().__init__(loader)
        # IMPORTANT !!!!  和 overhead 正比关系
        self.alpha = 5


    def get_detector_model(self, detector, X_train, Y_train):
        if detector == 'DF':
            logger.info('Using DF Attack')  # Sirinam's CNN
            NB_EPOCH = 15
            BATCH_SIZE = 128
            input_shape = (1, num_bursts, 1)
            OPTIMIZER = Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            K.image_data_format()
            model = mb_utility.ConvNet.build(input_shape=input_shape, classes=self.loader.num_classes)
            model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                                validation_split=VALIDATION_SPLIT)

        if detector == 'AWF':
            logger.info('Using AWF Attack')  # Rimmer's CNN.
            input_shape = (1, num_bursts, 1)
            BATCH_SIZE = 256
            NB_EPOCH = 2
            OPTIMIZER = RMSprop(lr=0.0008, decay=0.0)
            K.image_data_format()
            model = mb_utility.AWFConvNet.build(input_shape=input_shape, classes=self.loader.num_classes)
            model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                                validation_split=VALIDATION_SPLIT)
        logger.info("Detector model train finished!")
        return model


    def defense(self, **kwargs):
        if kwargs.get("alpha"):
            self.alpha = kwargs["alpha"]
        data_type = kwargs.get("data_type", "full_duplex")
        detector = kwargs.get("detector", "DF")
        target_pool = kwargs.get("target_pool", 30)
        iterations = kwargs.get("num_iteration", 100)
        cases = [0]

        X_OriTrain = mb_utility.dir_to_burst(self.dataset.train_data.x)
        y_train = self.dataset.train_data.y
        X_train = X_OriTrain[:, :num_bursts]  # pick the first 'number of bursts' in each trace

        X_train, y_train, xmax = mb_utility.fixme((X_train, y_train))

        X_train = X_train.reshape(
        (X_train.shape[0], 1, X_train.shape[1], 1))  # reshape the data to be compatible with CNN input

        # we do the same thing on the part 2
        X_Oritest_org = self.dataset.test_data.x
        X_Oritest_org = mb_utility.dir_to_burst(X_Oritest_org)
        y_test = self.dataset.test_data.y
        X_test_org = X_Oritest_org[:, :num_bursts]

        # Here, the test is basically the second part of the data.
        # Mockingbird generates the adversarial traces of these data
        X_test, y_test, _ = mb_utility.fixme((X_test_org, y_test), xmax=xmax)
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], 1))

        # change the labels to the categorical format (one-hot encoding)
        # for using in CNN with categorical-cross-entropy loss function
        Y_train = np_utils.to_categorical(y_train, self.loader.num_classes)
        Y_test = np_utils.to_categorical(y_test, self.loader.num_classes)

        logger.info('X_train shape {0} x_test.shape:{1}'.format(X_train.shape, X_test.shape))
        logger.info('y_train shape {0} y_test.shape:{1}'.format(Y_train.shape, Y_test.shape))
        # Detector Model Parameters
        model_path = os.path.join(
            constant.model.MODEL_DIR,
            DefenderChoice.MOCKING_BIRD.value,
            data_type,
        )
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        K.clear_session()
        K.image_data_format()
        tf.reset_default_graph()
        # create a session
        sess = tf.Session()

        model = self.get_detector_model(
            detector, X_train, Y_train,
        )

        # the placeholder for the source samples
        x_tf = tf.placeholder(tf.float32, shape=(None, 1, X_train.shape[2], 1))

        # The placeholder of the target samples
        target_tf = tf.placeholder(tf.float32, shape=(None, 1, X_train.shape[2], 1))

        # The distance function between target and source samples
        dist_tf = tf.sqrt(tf.reduce_sum(tf.square(x_tf - target_tf), list(range(1, len(X_train.shape)))))

        # The gredient of the distance function w.r.t the source samples' placeholder
        grad = tf.gradients(dist_tf, x_tf)

        generated_dav_list = []
        for case in cases:
            X_adv = []  # a list for keeping the generated defended samples for this config
            Y_adv = []  # contains the labels of the generated defended trace

             # We loop through each site, and consider it as the source class,
            # and work on the traces from that class to modify them

            for source in tqdm.tqdm(range(self.loader.num_classes), ncols=constant.TQDM_N_COLS, position=0, leave=True):
                X_source = mb_utility.get_class_samples(X_train, Y_train, source)

                if case == 0:
                    X_others, _ = mb_utility.exclute_class(X_train, Y_train, source)

                # for each sample in source class
                for i in tqdm.tqdm(range(len(X_source)), ncols=constant.TQDM_N_COLS, position=1, leave=False):
                    X_sample = X_source[i: i + 1]  # contains the source sample
                    Y_sample = source  # contains the label for the source sample
                    ind = np.random.randint(0, high=len(X_others), size=target_pool)
                    X_others_samples = X_others[ind]  # contains the selected target samples
                    X_sample_new = copy.copy(X_sample)

                    cnt = 1  # a counter

                    # Distance computation
                    dist = mb_utility.distance(X_sample_new, X_others_samples,
                                    feat)  # distance between the source and selected target samples
                    min_dist = np.argmin(dist)  # contains the index of the minimum distance between the source trace and all the target traces
                    max_dist = np.argmax(dist)  # contains the index the maximum distance between the source trace and all the target traces

                    # we pick the target trace that have the minimum distance to the source trace,
                    X_nearest_sample = X_others_samples[min_dist:min_dist + 1]

                    steps = 0

                    for k in range(iterations):
                        steps += 1

                        # Compute the gredient of the distance function
                        feed_dict = {x_tf: X_sample_new, target_tf: X_nearest_sample}
                        derivative, d = sess.run([grad, dist_tf], feed_dict)

                        # multiply with -1 to get the direction toward the minimum.
                        derivative = -1 * derivative[0]

                        # Get the indices where -1*gredient is negative,
                        # we don't want to decrease the burst's size
                        ind = np.where(derivative >= 0)

                        # Keep a copy of the current version of source sample,
                        # later we want to now how much we change it.
                        x1 = copy.copy(X_sample_new)

                        # Change to the source traces values according to 'derivative'.
                        # We scale 'derivative' with 'alpha'
                        X_sample_new[ind] = X_sample_new[ind] * (1 + self.alpha * derivative[ind])

                        # Get how our model predict the modified source traces, and how much its confidence in the source class
                        ypredict, source_prob = mb_utility.test_classification(model, detector, X_sample_new, source, sess=sess)

                        # How much we change the source trace in this iteration.
                        change_applied = np.sum(X_sample_new - x1)

                        if change_applied < confidence_threshold and (steps % 10 == 0):  # drop the target and pick a new one
                            # refill the target traces with new ones
                            ind = np.random.randint(0, high=len(X_others), size=target_pool)
                            X_others_samples = X_others[ind]

                            dist = mb_utility.distance(X_sample_new, X_others_samples, feat)
                            min_dist = np.argmin(dist)  # contains the index of min distance
                            max_dist = np.argmax(dist)  # contains the index of max distance

                            # Pick the target traces in index min_dist
                            X_nearest_sample = X_others_samples[min_dist:min_dist + 1]

                        # Overhead applied to the source trace so far
                        overhead = np.sum(X_sample_new - X_sample) / np.sum(X_sample)

                        if source != ypredict:
                            cnt += 1

                        if (source != ypredict and source_prob < prob_threshold and
                            change_applied < 2 * confidence_threshold) or cnt > \
                                (iterations * .7):
                            break

                            # Add the modified source trace to the list
                    X_adv.append(X_sample_new.reshape((1, num_bursts, 1)))
                    Y_adv.append(source)
            # Compute the overhead
            overhead = (np.sum(np.abs(X_adv)) - np.sum(np.abs(X_train))) / np.sum(np.abs(X_train))
            logger.info(f"the overhead is: {overhead}")
            generated_dav_list.append((X_adv, Y_adv))
        sess.close()

        r_x, r_y = self.convert_dav_to_dataset(generated_dav_list, X_OriTrain)
        return r_x, None, r_y, overhead
        # return super().defense(**kwargs)


    def convert_dav_to_dataset(self,  dav_list, X_OriTrain):
        for X_adv, Y_adv in dav_list:
            # Y_adv = np_utils.to_categorical(Y_adv, num_classes)  # change numerical Y values to categrical
            X_adv = np.array(X_adv)

            X_adv_rescaled = mb_utility.rescale_X(X_adv, feat, X_OriTrain)

            # input_length is the size of the input to the model after converting the bursts to the packets
            input_length = 5000
            # Expand bursts to packets
            X_adv_expand = mb_utility.expand_me(X_adv_rescaled, input_length=input_length)
            # reshape it to be compatible to CNN
            X_adv_expand = X_adv_expand.reshape((len(X_adv_expand), 1, input_length, 1))


        return np.squeeze(X_adv_expand), np.squeeze(Y_adv)

