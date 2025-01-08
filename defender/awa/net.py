import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.layers import (
    Conv1D, Activation, BatchNormalization,
    Flatten, Dense, ELU, MaxPooling1D,
)
from tensorflow_core.python.keras.models import (
    Model, Sequential
)
from tensorflow_core.python.keras.initializers import glorot_uniform
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

from constant.model import MODEL_DIR
from loader.base import AbstractLoader
from .utils import ct
from .dataload import DFDataLoader

logger = logging.getLogger(__name__)
batch_size = 128
disc_weight = 1e2
oh_weight = 1e3
# oh_weight = 1e2
logits_weight = 1e3
TRACE_LENGTH = 2000
num_samples_in_Gs_training = 400
batch_size = 100



def AC_train(
    data, file_name, input_shape, classes,
    num_epochs=50, batch_size=128, train_temp=1, init=None,
    test_flag=0, load_flag=1,
    verbose=0, model_name=None
):
    model = Sequential()

    filter_num = ['None', 32, 32, 64, 64]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                     strides=conv_stride_size[1], padding='same',
                     input_shape=input_shape,
                     name='ACblock1_conv1'))
    model.add(ELU(alpha=1.0, name='ACblock1_adv_act1'))

    model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                     strides=conv_stride_size[2], padding='same',
                     name='ACblock2_conv1'))
    model.add(ELU(alpha=1.0, name='ACblock1_adv_act2'))
    model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                           padding='same', name='ACblock1_pool'))

    model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='ACblock3_conv1'))
    model.add(ELU(alpha=1.0, name='ACblock2_adv_act1'))

    model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='ACblock4_conv1'))
    model.add(ELU(alpha=1.0, name='ACblock2_adv_act2'))
    model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                           padding='same', name='ACblock2_pool'))

    model.add(Flatten(name='ACflatten'))
    model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='ACfc1'))
    model.add(Activation('relu', name='ACfc1_act'))
    model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='ACfc2'))
    model.add(Activation('relu', name='ACfc2_act'))
    model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='ACfc3'))
    model.add(Activation('softmax', name="softmax"))

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=correct,
            logits=predicted / train_temp
        )

    OPTIMIZER = tf.compat.v1.train.AdamOptimizer(
        learning_rate=0.0002, beta1=0.9, beta2=0.999, epsilon=1e-08
    )  # Optimizer

    model_dir = os.path.join(MODEL_DIR, "AWA")
    model_path = os.path.join(model_dir, model_name + '.weights.h5')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    best_model_save = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
    )
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

    if load_flag == 1 and os.path.exists(model_path):
        if os.path.exists(model_path):
            logger.info("Loading AC Model!!!")
            model.load_weights(model_path)
        if test_flag == 1:
            score_test = model.evaluate(data.test_data, data.test_labels, verbose=verbose)
            y_pred = np.argmax(model.predict(data.test_data), axis=1)
            print('Confusion Matrix')
            print(confusion_matrix(np.argmax(data.test_labels, axis=1), y_pred))
            print("Testing accuracy:", score_test[1])
            print(classification_report(np.argmax(data.test_labels, axis=1), y_pred))
        return model

    dataset_shape = "train: {}, valid: {}, test: {}".format(
       data.transformer_train_x.shape,
       data.transformer_train_x.shape,
       data.test_data.shape,
    )
    logger.info(f"AC train data shape: {dataset_shape}")

    if test_flag == 1:
        model.fit(data.transformer_train_x, data.transformer_train_y, batch_size=batch_size,
                            epochs=num_epochs, verbose=verbose,
                            validation_data=(data.transformer_train_x, data.transformer_train_y),
                            callbacks=[best_model_save])
        model.save(model_path)
    else:
        model.fit(
            data.transformer_train_x, data.transformer_train_y,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=verbose,
            validation_data=(data.transformer_train_x, data.transformer_train_y)
        )
        model.save(model_path)

    if test_flag == 1:
        print("Load Best Model")
        model.load_weights(model_path)

        # Start evaluating model with testing data
        score_test = model.evaluate(data.test_data, data.test_labels, verbose=verbose)
        y_pred = np.argmax(model.predict(data.test_data), axis=1)
        logger.info('Confusion Matrix')
        logger.info(confusion_matrix(np.argmax(data.test_labels, axis=1), y_pred))
        logger.info("Testing accuracy: {}".format(score_test[1]))
        logger.info(classification_report(np.argmax(data.test_labels, axis=1), y_pred))

    if file_name != None:
        model.save(file_name)

    return model


def AC_layers(layer_names, loader: AbstractLoader):
    """ Creates a AC model that returns a list of intermediate output values."""
    # Load our model. Load pretrained AC, trained on WF data
    Trace_loader = DFDataLoader(loader)
    nb_class = loader.num_classes
    # if nb_class % 2 == 1:
    #     nb_class -= 1

    AC = AC_train(
        Trace_loader,
        file_name=None,
        input_shape=(TRACE_LENGTH, 1),
        classes=nb_class,
        num_epochs=30,
        verbose=2,
        test_flag=0,
        load_flag=1,
        model_name='AC',
    )
    AC.trainable = False
    # layer_names.append('softmax')
    outputs = [AC.get_layer(name).output for name in layer_names]

    model = Model([AC.input], outputs)
    return model


def cal_logit_loss(logit_outputs, src=None):
    logit_loss = logits_weight * tf.reduce_mean(tf.maximum(logit_outputs[:, src], 0))
    return logit_loss


class LogitsModel(tf.keras.models.Model):
    def __init__(self, logit_layer, loader: AbstractLoader):
        super(LogitsModel, self).__init__()
        self.AC = AC_layers(logit_layer, loader)
        self.logit_layer = logit_layer
        self.AC.trainable = False

    def call(self, inputs):
        outputs = self.AC(inputs)
        return outputs

class AWA_Class:
    def __init__(
            self,
            loader: AbstractLoader,
            trace_len=None,
            logit_layer=None,
            AWA_type = 'NUAWA',
            tau_high = 0.30,
            tau_low = 0.05,
        ):
        self.tau_high = tau_high
        self.tau_low =  tau_low

        self.session = tf.compat.v1.InteractiveSession()
        self.max_burst_trace_len = trace_len
        sign_vector = np.ones([trace_len, 1]) * -1
        sign_vector[::2] = 1
        self.sign_vector = tf.convert_to_tensor(sign_vector, dtype=tf.float64)
        self.max_trace_len = tf.convert_to_tensor(trace_len, dtype=tf.float64)

        with tf.name_scope('placeholders'):
            self.cls_i_data = tf.compat.v1.placeholder(tf.float32, [None, self.max_burst_trace_len, 1], name="x_class_i")
            self.cls_j_data = tf.compat.v1.placeholder(tf.float32, [None, self.max_burst_trace_len, 1], name="x_class_j")
            self.noise_i = tf.compat.v1.placeholder(tf.float32, [None, self.max_burst_trace_len, 1], name="noise_i")
            self.noise_j = tf.compat.v1.placeholder(tf.float32, [None, self.max_burst_trace_len, 1], name="noise_j")
            self.src_1 = tf.compat.v1.placeholder(tf.int32, shape=(), name="src_1")
            self.src_2 = tf.compat.v1.placeholder(tf.int32, shape=(), name="src_2")

        self.logit_extractor = LogitsModel(logit_layer, loader)

        self.generator1 = self.make_generator1()
        self.generator2 = self.make_generator2()
        self.discriminator = self.make_discriminator()

        if AWA_type == 'UAWA':
            # 模型的输出
            self.generated_1 = self.generator1(self.noise_i)
            self.generated_2 = self.generator2(self.noise_j)
        elif AWA_type == 'NUAWA':
            # IMPORTANT !!! 我们用这个
            self.generated_1 = self.generator1(self.cls_i_data)
            self.generated_2 = self.generator2(self.cls_j_data)

        # 一个公式对应的结果
        self.adjusted_generated_1 = self.adjust_WF_data(self.cls_i_data, self.generated_1)
        self.adjusted_generated_2 = self.adjust_WF_data(self.cls_j_data, self.generated_2)
        # 模型输出
        self.d_class_i = self.discriminator(self.adjusted_generated_1)
        self.d_class_j = self.discriminator(self.adjusted_generated_2)
        self.d_loss = self.discriminator_loss(self.d_class_i, self.d_class_j)

        self.padded_adjusted_generated_1 = tf.pad(self.adjusted_generated_1, tf.constant(
            [[0, 0], [0, TRACE_LENGTH - self.max_burst_trace_len], [0, 0]]), mode='CONSTANT', constant_values=0)
        self.logit_outputs_1 = self.logit_extractor(self.padded_adjusted_generated_1)
        self.logit_loss_1 = cal_logit_loss(self.logit_outputs_1, src=self.src_1)
        self.g1_loss_org, self.g1_oh_loss = self.generator1_loss(self.adjusted_generated_1, self.cls_i_data,
                                                                 self.d_class_i)
        self.g1_loss = self.g1_loss_org + self.g1_oh_loss + self.logit_loss_1

        self.padded_adjusted_generated_2 = tf.pad(self.adjusted_generated_2, tf.constant(
            [[0, 0], [0, TRACE_LENGTH - self.max_burst_trace_len], [0, 0]]), mode='CONSTANT', constant_values=0)
        self.logit_outputs_2 = self.logit_extractor(self.padded_adjusted_generated_2)
        self.logit_loss_2 = cal_logit_loss(self.logit_outputs_2, src=self.src_2)
        self.g2_loss_org, self.g2_oh_loss = self.generator2_loss(self.adjusted_generated_2, self.cls_j_data,
                                                                 self.d_class_j)
        self.g2_loss = self.g2_loss_org + self.g2_oh_loss + self.logit_loss_2

        with tf.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
            g1_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='generator1')
            self.g1_train = optimizer.minimize(self.g1_loss, var_list=g1_vars)
            g2_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='generator2')
            self.g2_train = optimizer.minimize(self.g2_loss, var_list=g2_vars)
            d_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
            self.d_train = optimizer.minimize(self.d_loss, var_list=d_vars)

        tf.compat.v1.global_variables_initializer().run()

    def adjust_WF_data(self, x=None, perturbation=None, sign_vector=None):

        # adding the perturbation into the original trace.
        mask = tf.expand_dims(
            tf.tile(tf.cast(tf.minimum(tf.sign(x[:, 0]), 0) * -1, tf.float32), multiples=[1, self.max_burst_trace_len]),
            2)
        pert_rolled = tf.roll(tf.pad(perturbation, [[0, 0], [0, 1], [0, 0]]), shift=1, axis=1)[:,
                      :self.max_burst_trace_len]
        return x + ((mask * pert_rolled + (1 - mask) * perturbation) * tf.sign(x))

    def discriminator_loss(self, d_class_i, d_class_j):
        d_loss_i = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_class_i, labels=tf.ones_like(d_class_i)))
        d_loss_j = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_class_j, labels=tf.zeros_like(d_class_j)))
        d_loss = d_loss_i + d_loss_j
        return d_loss

    def generator1_loss(self, perturbation_i=None, x_class_i=None, d_class_i=None):
        g1_oh_loss = tf.maximum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_i - perturbation_i)),
                                                              tf.reduce_sum(tf.abs(x_class_i)))) - self.tau_high, 0)
        g1_oh_loss_new = tf.minimum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_i - perturbation_i)),
                                                                  tf.reduce_sum(tf.abs(x_class_i)))) - self.tau_low, 0) * -1
        g1_loss_org = -1 * (
                tf.reduce_mean(1 / 2 * tf.math.log(tf.math.sigmoid(d_class_i) + 0.0000001)) + tf.reduce_mean(
            1 / 2 * tf.math.log(1 - tf.math.sigmoid(d_class_i) + 0.0000001)))
        return disc_weight * g1_loss_org, oh_weight * (g1_oh_loss + g1_oh_loss_new)

    def generator2_loss(self, perturbation_j=None, x_class_j=None, d_class_j=None):
        g2_oh_loss = tf.maximum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_j - perturbation_j)),
                                                              tf.reduce_sum(tf.abs(x_class_j)))) - self.tau_high, 0)
        g2_oh_loss_new = tf.minimum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_j - perturbation_j)),
                                                                  tf.reduce_sum(tf.abs(x_class_j)))) - self.tau_low, 0) * -1
        g2_loss_org = -1 * (
                tf.reduce_mean(1 / 2 * tf.math.log(tf.math.sigmoid(d_class_j) + 0.0000001)) + tf.reduce_mean(
            1 / 2 * tf.math.log(1 - tf.math.sigmoid(d_class_j) + 0.0000001)))
        return disc_weight * g2_loss_org, oh_weight * (g2_oh_loss + g2_oh_loss_new)

    def make_generator1(self):
        with tf.compat.v1.variable_scope('generator1'):
            model = tf.keras.Sequential()
            # ---------------------------------------------------------------------
            # c3s1-8
            model.add(
                Conv1D(filters=8, kernel_size=3, strides=1, padding='same', input_shape=(self.max_burst_trace_len, 1)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))
            # -----------------------------------------------------------------
            # d16
            model.add(Conv1D(filters=16, kernel_size=3, strides=2, padding='same'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

            # -------------------------------------------------------------------
            # d32
            model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='same'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

            # ---------------------------------------------------------------------
            # four r32 blocks
            for _ in range(8):
                model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
                model.add(BatchNormalization(momentum=0.8))
                model.add(ELU(alpha=2.0))

            # -----------------------------------------------------------------------
            # u16
            ct.Conv1DTranspose(model, filters=16, kernel_size=3, strides=2, padding='same')
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))
            # this below line is for the cases which our input shape is odd !!
            # G = Conv1D(filters=16, kernel_size=2, strides=1, padding='valid')(G)

            # ----------------------------------------------------------------------------
            # u8
            ct.Conv1DTranspose(model, filters=8, kernel_size=3, strides=2, padding='same')
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

            # -----------------------------------------------------------------------------
            # c3s1-3
            model.add(Conv1D(filters=1, kernel_size=3, strides=1, padding='same'))
            model.add(Activation('relu'))
            return model

    def make_generator2(self):
        with tf.compat.v1.variable_scope('generator2'):
            model = tf.keras.Sequential()
            # ---------------------------------------------------------------------
            # c3s1-8
            model.add(
                Conv1D(filters=8, kernel_size=3, strides=1, padding='same', input_shape=(self.max_burst_trace_len, 1)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))
            # -----------------------------------------------------------------
            # d16
            model.add(Conv1D(filters=16, kernel_size=3, strides=2, padding='same'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

            # -------------------------------------------------------------------
            # d32
            model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='same'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

            # ---------------------------------------------------------------------
            # four r32 blocks
            for _ in range(8):
                model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
                model.add(BatchNormalization(momentum=0.8))
                model.add(ELU(alpha=2.0))

            # -----------------------------------------------------------------------
            # u16
            ct.Conv1DTranspose(model, filters=16, kernel_size=3, strides=2, padding='same')
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))
            # this below line is for the cases which our input shape is odd !!
            # G = Conv1D(filters=16, kernel_size=2, strides=1, padding='valid')(G)

            # ----------------------------------------------------------------------------
            # u8
            ct.Conv1DTranspose(model, filters=8, kernel_size=3, strides=2, padding='same')
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

            # -----------------------------------------------------------------------------
            # c3s1-3
            model.add(Conv1D(filters=1, kernel_size=3, strides=1, padding='same'))
            model.add(Activation('relu'))
            return model

    def make_discriminator(self):
        with tf.compat.v1.variable_scope('discriminator'):
            model = tf.keras.Sequential()
            filter_num = ['None', 32, 32, 64, 64]
            kernel_size = ['None', 8, 8, 8, 8]
            conv_stride_size = ['None', 1, 1, 1, 1]
            pool_stride_size = ['None', 4, 4, 4, 4]
            pool_size = ['None', 8, 8, 8, 8]

            model.add(
                Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=(self.max_burst_trace_len, 1),
                       strides=conv_stride_size[1], padding='same',
                       name='dblock1_conv1'))
            model.add(ELU(alpha=1.0, name='dblock1_adv_act1'))

            model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                             strides=conv_stride_size[2], padding='same',
                             name='dblock2_conv1'))
            model.add(ELU(alpha=1.0, name='dblock1_adv_act2'))
            model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                                   padding='same', name='dblock1_pool'))

            model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                             strides=conv_stride_size[3], padding='same',
                             name='dblock3_conv1'))
            model.add(ELU(alpha=1.0, name='dblock2_adv_act1'))

            model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                             strides=conv_stride_size[4], padding='same',
                             name='dblock4_conv1'))
            model.add(ELU(alpha=1.0, name='dblock2_adv_act2'))
            model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                                   padding='same', name='dblock2_pool'))

            model.add(Flatten(name='dflatten'))
            model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='dfc1'))
            model.add(Activation('relu', name='dfc1_act'))
            model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='dfc2'))
            model.add(Activation('relu', name='dfc2_act'))
            model.add(Dense(1, kernel_initializer=glorot_uniform(seed=0), name='dfc3'))
            return model
