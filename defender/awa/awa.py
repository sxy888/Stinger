import datetime
import logging  # noqa
import typing as t
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from tensorflow.python.keras.backend import clear_session

from loader.base import AbstractLoader, Config, DatasetWrapper
from defender import AbstractDefender
from .utils import data_class, print_overhead
from .dataload import DFDataLoader, burst_to_seq
from .net import AWA_Class

logger = logging.getLogger(__name__)
AWA_TYPE = 'NUAWA'  # UAWA or NUAWA
iterations = 300     # the iteration of trace
batch_size = 100    # the batch size
d_iteration = 2     # d iteration
g_iteration = 2     # g iteration


class AWADefender(AbstractDefender):

    dataset: t.Union[None, DatasetWrapper] = None

    def __init__(self, loader) -> None:
        super().__init__(loader)
        self.OH = 0.5 # overhead
        self.tau_high = 0.30
        self.tau_low = 0.05
        self.AWA_TYPe = 'NUAWA'


    @classmethod
    def update_config(cls, config: Config) -> Config:
        config.load_data_by_self = False

    def defense(self, **kwargs):
        self.OH = kwargs.get("OH", self.OH)
        self.tau_high = kwargs.get("tau_high", self.tau_high)
        self.tau_low = kwargs.get("tau_low", self.tau_low)
        use_same_dataset = kwargs.get("use_same_dataset", False),

        logger.info("模型使用 tau 区间为 %.2f ~ %.2f, 目标 overhead: %.2f" % (self.tau_low, self.tau_high, self.OH))
        logger.info("same 属性值： %s" % use_same_dataset)

        real_length = 0
        extra_length = 0

        logits_layer = ['ACfc3']  # the name of logits layers: for fool the AC

        number_classes = self.loader.num_classes
        # 保证是偶数
        if number_classes % 2 == 1:
            number_classes -= 1
        cls_list = np.arange(number_classes)
        keys = np.random.permutation(cls_list)
        if len(keys) % 2 != 0:
            raise ValueError('The number of classes must be even')

        key1, key2 = keys[:number_classes//2], keys[number_classes//2:]

        X_train,X_validate, X_test = [], [], []
        Y_train, Y_validate, Y_test = [], [], []

        trace_loader = DFDataLoader(self.loader)

        for cld_itr, cls_index in enumerate(range(len(key1))):
            g1_loss_plot = []
            g2_loss_plot = []
            best_result = 0
            d_loss_plot = []
            acc_list = []
            oh_src_test = []
            oh_src_train = []
            oh_trg_test = []
            oh_trg_train = []
            sub_X_train, sub_X_validate, sub_X_test = [], [], []
            sub_Y_train, sub_Y_validate, sub_Y_test = [], [], []

            # create list for retraining.
            export_data = data_class()
            src_cls = key1[cls_index]
            trg_cls = key2[cls_index]
            logger.info(f"current src cls: {src_cls}, target cls: {trg_cls}")


            src_trans_train_data, src_trans_train_labels = trace_loader.get_DF_data(
                data_type='transformer_train', src_cls=src_cls
            )
            src_user_train_data, src_user_train_labels = trace_loader.get_DF_data(
                data_type='user_train', src_cls=src_cls
            )
            src_valid_data, src_valid_labels = trace_loader.get_DF_data(
                data_type='validation', src_cls=src_cls
            )
            src_test_data, src_test_labels = trace_loader.get_DF_data(
                data_type='test', src_cls=src_cls
            )
            trg_trans_train_data, trg_trans_train_labels = trace_loader.get_DF_data(
                data_type='transformer_train', src_cls=trg_cls
            )
            trg_user_train_data, trg_user_train_labels = trace_loader.get_DF_data(
                data_type='user_train', src_cls=trg_cls
            )
            trg_valid_data, trg_valid_labels = trace_loader.get_DF_data(
                data_type='validation', src_cls=trg_cls
            )
            trg_test_data, trg_test_labels = trace_loader.get_DF_data(
                data_type='test', src_cls=trg_cls
            )

            max_burst_trace_len = 2000  # max([max(np.array([ np.count_nonzero(i) for i in src_trans_train_data])),max(np.array([ np.count_nonzero(i) for i in trg_trans_train_data]))])

            src_trans_train_data = src_trans_train_data[:, :max_burst_trace_len]
            src_user_train_data = src_user_train_data[:, :max_burst_trace_len]
            src_valid_data = src_valid_data[:, :max_burst_trace_len]
            src_test_data = src_test_data[:, :max_burst_trace_len]
            trg_trans_train_data = trg_trans_train_data[:, :max_burst_trace_len]
            trg_user_train_data = trg_user_train_data[:, :max_burst_trace_len]
            trg_valid_data = trg_valid_data[:, :max_burst_trace_len]
            trg_test_data = trg_test_data[:, :max_burst_trace_len]

            clear_session()

            awa = AWA_Class(
                self.loader,
                trace_len=max_burst_trace_len,
                logit_layer=logits_layer,
                AWA_type=AWA_TYPE,
                tau_high=self.tau_high,
                tau_low=self.tau_low,
            )

            transformers_selecting_flag = 0
            for i in tqdm.tqdm(range(iterations), ncols=60, position=0, desc="iteration", leave=False):
                t_g1_lor = 0
                t_g2_lor = 0
                t_d_l = 0
                for d_i in range(d_iteration):
                    sample_index = np.random.randint(len(src_trans_train_data), size=batch_size)
                    batch_src_x = src_trans_train_data[sample_index]
                    batch_src_noise = np.random.normal(size=batch_src_x.shape)
                    sample_index = np.random.randint(len(trg_trans_train_data), size=batch_size)
                    batch_trg_x = trg_trans_train_data[sample_index]
                    batch_trg_noise = np.random.normal(size=batch_trg_x.shape)
                    _, d_l = awa.session.run([awa.d_train, awa.d_loss],
                                            feed_dict={awa.cls_i_data: batch_src_x, awa.cls_j_data: batch_trg_x,
                                                        awa.noise_i: batch_src_noise, awa.noise_j: batch_trg_noise})
                    t_d_l += d_l
                d_loss_plot.append(t_d_l / d_iteration)

                for g1_i in tqdm.tqdm(range(g_iteration), ncols=60, desc="g1 iteration", position=1, leave=False):
                    sample_index = np.random.randint(len(src_trans_train_data), size=batch_size)
                    batch_src_x = src_trans_train_data[sample_index]
                    batch_src_noise = np.random.normal(size=batch_src_x.shape)

                    _, g1_l, g1_lor, g1_loh, l1_loss, d1_out, pert, adj_new_data = awa.session.run(
                        [awa.g1_train, awa.g1_loss, awa.g1_loss_org, awa.g1_oh_loss, awa.logit_loss_1, awa.d_class_i,
                        awa.generated_1, awa.adjusted_generated_1],
                        feed_dict={awa.cls_i_data: batch_src_x, awa.src_1: src_cls, awa.noise_i: batch_src_noise})
                    assert np.sum(pert < 0) == 0, "Health issue in perturbation"
                    assert np.sum((np.abs(adj_new_data) - np.abs(batch_src_x)) < 0) == 0, "Health issue in sign"
                    assert np.array_equal(np.array([np.sum(np.abs(np.sign(i))) for i in adj_new_data]), np.array(
                        [np.sum(np.abs(np.sign(i))) for i in batch_src_x])), "Health issue in size of trace"
                    t_g1_lor += (g1_lor / g_iteration)
                g1_loss_plot.append(t_g1_lor)

                for d_i in tqdm.tqdm(range(d_iteration), ncols=60, desc="g iteration", position=2, leave=False):
                    sample_index = np.random.randint(len(src_trans_train_data), size=batch_size)
                    batch_src_x = src_trans_train_data[sample_index]
                    batch_src_noise = np.random.normal(size=batch_src_x.shape)
                    sample_index = np.random.randint(len(trg_trans_train_data), size=batch_size)
                    batch_trg_x = trg_trans_train_data[sample_index]
                    batch_trg_noise = np.random.normal(size=batch_trg_x.shape)

                    _, d_l = awa.session.run([awa.d_train, awa.d_loss],
                                            feed_dict={awa.cls_i_data: batch_src_x, awa.cls_j_data: batch_trg_x,
                                                        awa.noise_i: batch_src_noise, awa.noise_j: batch_trg_noise})

                for g2_i in tqdm.tqdm(range(g_iteration), ncols=60, desc="g2 iteration", position=3, leave=False):
                    sample_index = np.random.randint(len(trg_trans_train_data), size=batch_size)
                    batch_trg_x = trg_trans_train_data[sample_index]
                    batch_trg_noise = np.random.normal(size=batch_trg_x.shape)

                    _, g2_l, g2_lor, g2_loh, l2_loss, d2_out, pert, adj_new_data = awa.session.run(
                        [awa.g2_train, awa.g2_loss, awa.g2_loss_org, awa.g2_oh_loss, awa.logit_loss_2, awa.d_class_j,
                        awa.generated_2, awa.adjusted_generated_2],
                        feed_dict={awa.cls_j_data: batch_trg_x, awa.src_2: trg_cls, awa.noise_j: batch_trg_noise})
                    assert np.sum(pert < 0) == 0, "Health issue in perturbation"
                    assert np.sum((np.abs(adj_new_data) - np.abs(batch_trg_x)) < 0) == 0, "Health issue in sign"
                    assert np.array_equal(np.array([np.sum(np.abs(np.sign(i))) for i in adj_new_data]), np.array(
                        [np.sum(np.abs(np.sign(i))) for i in batch_trg_x])), "Health issue in size of trace"
                    t_g2_lor += (g2_lor / g_iteration)
                g2_loss_plot.append(t_g2_lor)

                if (i + 1) % 50 == 0 or (i + 1) % iterations == 0:
                    generated_src_trans_train = awa.adjusted_generated_1.eval(feed_dict={awa.cls_i_data: src_trans_train_data, awa.noise_i: np.random.normal(size=src_trans_train_data.shape) })
                    assert np.sum((np.abs(generated_src_trans_train) - np.abs(src_trans_train_data)) < 0) == 0, "Health issue in sign1"
                    generated_trg_trans_train = awa.adjusted_generated_2.eval(feed_dict={awa.cls_j_data: trg_trans_train_data, awa.noise_j: np.random.normal(size=trg_trans_train_data.shape) })
                    assert np.sum((np.abs(generated_trg_trans_train) - np.abs(trg_trans_train_data)) < 0) == 0, "Health issue in sign2"

                    _print = 0
                    oh_src_train.append([print_overhead("g1_train",src_trans_train_data,generated_src_trans_train,pr = _print, flush=True),i])
                    oh_trg_train.append([print_overhead("g2_train",trg_trans_train_data,generated_trg_trans_train,pr = _print),i])

                    if (
                        (oh_src_train[-1][0] < self.OH * 100 and oh_trg_train[-1][0] < self.OH * 100)
                        or
                        ( (i + 1) % iterations == 0 and transformers_selecting_flag == 0)
                    ):
                        transformers_selecting_flag = 1
                        generated_src_user_train = awa.adjusted_generated_1.eval(feed_dict={awa.cls_i_data: src_user_train_data, awa.noise_i: np.random.normal(size=src_user_train_data.shape) })
                        assert np.sum((np.abs(generated_src_user_train) - np.abs(src_user_train_data)) < 0) == 0, "Health issue in sign11"
                        generated_trg_user_train = awa.adjusted_generated_2.eval(feed_dict={awa.cls_j_data: trg_user_train_data, awa.noise_j: np.random.normal(size=trg_user_train_data.shape) })
                        assert np.sum((np.abs(generated_trg_user_train) - np.abs(trg_user_train_data)) < 0) == 0, "Health issue in sign22"

                        oh_src_train.append([print_overhead("g1_user_train",src_user_train_data,generated_src_user_train,pr = 1),i])
                        oh_trg_train.append([print_overhead("g2_user_train",trg_user_train_data,generated_trg_user_train,pr = 1),i])

                        generated_src_valid = awa.adjusted_generated_1.eval(feed_dict={awa.cls_i_data: src_valid_data, awa.noise_i: np.random.normal(size=src_valid_data.shape) })
                        assert np.sum((np.abs(generated_src_valid) - np.abs(src_valid_data)) < 0) == 0, "Health issue in sign3"
                        generated_trg_valid = awa.adjusted_generated_2.eval(feed_dict={awa.cls_j_data: trg_valid_data, awa.noise_j: np.random.normal(size=trg_valid_data.shape) })
                        assert np.sum((np.abs(generated_trg_valid) - np.abs(trg_valid_data)) < 0) == 0, "Health issue in sign4"

                        generated_src_test = awa.adjusted_generated_1.eval(feed_dict={awa.cls_i_data: src_test_data, awa.noise_i: np.random.normal(size=src_test_data.shape) })
                        assert np.sum((np.abs(generated_src_test) - np.abs(src_test_data)) < 0) == 0, "Health issue in sign5"
                        generated_trg_test = awa.adjusted_generated_2.eval(feed_dict={awa.cls_j_data: trg_test_data, awa.noise_j: np.random.normal(size=trg_test_data.shape) })
                        assert np.sum((np.abs(generated_trg_test) - np.abs(trg_test_data)) < 0) == 0, "Health issue in sign6"

                        sub_real_length = np.sum(np.abs(src_user_train_data))
                        sub_extra_length = np.sum(np.abs(generated_src_trans_train)) - sub_real_length
                        real_length += sub_real_length
                        extra_length += sub_extra_length

                        sub_X_train = [generated_src_trans_train, generated_trg_trans_train]
                        sub_Y_train = [
                            np.argmax(src_trans_train_labels, axis=1),
                            np.argmax(trg_user_train_labels, axis=1),
                        ]

                        sub_X_validate = [generated_src_valid, generated_trg_valid]
                        sub_Y_validate = [
                            np.argmax(src_valid_labels, axis=1),
                            np.argmax(trg_valid_labels, axis=1),
                        ]

                        sub_X_test = [generated_src_test, generated_trg_test]
                        sub_Y_test = [
                            np.argmax(src_test_labels, axis=1),
                            np.argmax(trg_test_labels, axis=1)
                        ]

            X_train.extend(sub_X_train)
            X_validate.extend(sub_X_validate)
            X_test.extend(sub_X_test)
            Y_train.extend(sub_Y_train)
            Y_validate.extend(sub_Y_validate)
            Y_test.extend(sub_Y_test)

            awa.session.close()
            if cld_itr >= 3:
                break

        X_train = np.reshape(np.array(X_train), (-1, max_burst_trace_len))
        X_validate = np.reshape(np.array(X_validate), (-1, max_burst_trace_len))
        X_test = np.reshape(np.array(X_test), (-1, max_burst_trace_len))
        Y_train = np.reshape(np.array(Y_train), (-1))
        Y_validate = np.reshape(np.array(Y_validate), (-1))
        Y_test = np.reshape(np.array(Y_test), (-1))
        f = "{}, {}, {},".format(X_train.shape, X_validate.shape, X_test.shape)
        f += "{}, {}, {}".format(Y_train.shape, Y_validate.shape, Y_test.shape)
        logger.info(f"generated shape: {f}")


        _overhead = np.round(extra_length / real_length, 2)
        logger.info(f"the overhead is: {_overhead}")
        return (X_train, X_validate, X_test), None, (Y_train, Y_validate, Y_test), _overhead