#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: cumul.py
# Created: 2024-09-08
# Description:

import logging  # noqa
import typing as t
import numpy as np
from numpy import ndarray # noqa
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from scipy import interpolate
import tqdm

from attack import AbstractAttack
import constant

logger = logging.getLogger(__name__)


class CumulAttack(AbstractAttack):

    def process_data(self, x: ndarray, y: ndarray):
        return x, y

    def extract(self, trace_record):
        # trace_record: list of packet sizes
        # first 4 features

        in_size = 0
        outsize = 0
        in_packet = 0
        out_packet = 0

        for i in range(0, len(trace_record)):
            if trace_record[i] < 0:
                outsize += abs(trace_record[i])
                out_packet += 1
            else:
                in_size += trace_record[i]
                in_packet += 1
        features = [in_size, outsize, in_packet, out_packet]

        # 100 interpolates

        n = 100  # number of linear interpolates
        cumulative_representation = [0.]
        cumulative_sum = 0
        for i in range(len(trace_record)):
            cumulative_sum += trace_record[i]
            cumulative_representation.append(cumulative_sum)
        x = np.linspace(start=0, stop=len(cumulative_representation), num=len(cumulative_representation), endpoint=False)
        cumulative_representation = np.array(cumulative_representation, dtype=float)
        liner_piecewise_func = interpolate.interp1d(x, cumulative_representation)
        stepwise = float(len(cumulative_representation) - 1) / 100.
        for i in range(n - 1):
            features.append(float(liner_piecewise_func(stepwise * (i + 1))))
        features.append(cumulative_representation[-1])
        return features

    def attack(self, **kwargs) -> None:
        x_train_list, x_test_list = [], []
        iterator = tqdm.tqdm(
            range(len(self.dataset.train_data.x)),
            ncols=constant.TQDM_N_COLS,
            desc="Process Train Data"
        )
        for i in iterator:
            data = self.dataset.train_data.x[i]
            data = data[data != 0]
            data = -data
            feature = self.extract(data)
            x_train_list.append(feature)
        x_train = np.array(x_train_list, dtype=float)

        iterator = tqdm.tqdm(
            range(len(self.dataset.test_data.x)),
            ncols=constant.TQDM_N_COLS,
            desc="Process Test Data"
        )
        for i in iterator:
            data = self.dataset.test_data.x[i]
            data = data[data != 0]
            data = -data
            feature = self.extract(data)
            x_test_list.append(feature)
        x_test = np.array(x_test_list, dtype=float)

        scaler = preprocessing.MinMaxScaler()
        x_train = scaler.fit_transform(x_train).tolist()
        x_test = scaler.transform(x_test).tolist()

        # clf = self.grid_search(self.X, self.Y)
        # C = clf.best_params_['C']
        # gamma = clf.best_params_['gamma']
        C = 8192
        gamma = 8.00

        model = SVC(C=C, gamma=gamma, kernel='rbf', verbose=False)
        logger.debug("cumul model fit")
        model.fit(x_train, self.dataset.train_data.y)
        logger.debug("cumul model fit finished")

        # logger.info("train score %s" % model.score(x_train, self.dataset.train_data.y))
        test_score = model.score(x_test, self.dataset.test_data.y)
        logger.info("test score %s" % test_score)
        return test_score


    def grid_search(self, train_x: ndarray, train_y: ndarray ):
        #find the optimal gamma
        param_grid = [{
            'C': [2**11,2**13,2**15,2**17],
            'gamma' : [2**-3,2**-1,2**1,2**3]
        }]
        my_scorer = "accuracy"
        clf = GridSearchCV(
            estimator = SVC(kernel = 'rbf'),
            param_grid = param_grid,
            scoring = my_scorer,
            cv = 5,
            verbose = 0,
            n_jobs = -1
        )
        clf.fit(train_x, train_y)
        logger.info('Best estimator:%s'%clf.best_estimator_)
        logger.info('Best_score_:%s'%clf.best_score_)
        return clf

    def score_func(MON_SITE_NUM, ground_truths, predictions):
        tp, wp, fp, p, n = 0, 0, 0, 0 ,0
        for truth,prediction in zip(ground_truths, predictions):
            if truth != MON_SITE_NUM:
                p += 1
            else:
                n += 1
            if prediction != MON_SITE_NUM:
                if truth == prediction:
                    tp += 1
                else:
                    if truth != MON_SITE_NUM:
                        wp += 1
                        # logger.info('Wrong positive:%d %d'%(truth, prediction))
                    else:
                        fp += 1
                        # logger.info('False positive:%d %d'%(truth, prediction))
        # logger.info('%4d %4d %4d %4d %4d'%(tp, wp, fp, p, n))
        tps += tp
        wps += wp
        fps += fp
        ps += p
        ns += n
        try:
            r_precision = tp*n / (tp*n+wp*n+r*p*fp)
        except:
            r_precision = 0.0
        # logger.info('r-precision:%.4f',r_precision)
        # return r_precision
        return tp/p