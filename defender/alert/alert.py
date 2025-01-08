from collections import defaultdict
import datetime
import os
import logging  # noqa
import typing as t
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from defender import AbstractDefender
from loader.base import DatasetWrapper, Dataset
from constant import TQDM_N_COLS
from constant.model import MODEL_DIR, PIC_DIR, DATA_DIR
from constant.enum import DefenderChoice
from utils.process import NoDefDataSet
from .model import Generator, Discriminator


logger = logging.getLogger(__name__)


def get_random_value_excluding_m(n, m):
    # 生成从1到n的列表
    numbers = list(range(0, n))

    # 从列表中移除m
    numbers = [num for num in numbers if num != m]

    # 从更新后的列表中随机选择一个值
    return random.choice(numbers)


def convert_trace_data_to_burst(train_x: np.ndarray, max_length: int):
    train_burst = []
    for trace in tqdm.tqdm(train_x.tolist(), ncols=TQDM_N_COLS, desc='trace to burst'):
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


def convert_burst_to_trace_data(burst_data: np.ndarray, max_length: int):
    trace = []
    for burst in burst_data:
        if burst == 0:
            trace.append(0)
            continue
        packet = 1 if burst > 0  else -1
        packets = [packet for _ in range(abs(int(burst)))]
        trace.extend(packets)

    while len(trace) < max_length:
        trace.append(0)

    # if len(trace) > max_length:
    #     logger.warning(f"trace 长度大于 {max_length}，截断")
    return trace[:max_length]



class AlertDefender(AbstractDefender):


    def __init__(self, loader) -> None:
        super().__init__(loader)
        self.batch_size = 64
        # 控制 loss per 的 overhead 阈值
        self.oh_max_threshold = 0.50
        self.oh_min_threshold = 0.10
        # loss 函数值的三个系数
        self.loss_alpha, self.loss_beta, self.loss_gamma = 1.0, 1.0, 1.0
        # burst 最长长度
        self.max_length = 2000
        self.num_epochs = 60
        # 正态分布噪声的均值和标准差
        self.noise_param = (0, 1)
        # 记录 loss 值
        self.loss_record = defaultdict(lambda: defaultdict(list))
        # 临时调试设置的站点阈值
        self.debug_limit_site_idx = 1

    @classmethod
    def update_config(cls, config):
        config.load_data_by_self = True
        return config

    def defense(self, **kwargs):
        self.oh_min_threshold = kwargs.get("oh_min_threshold", self.oh_min_threshold) + 0.06
        self.oh_max_threshold = kwargs.get("oh_max_threshold", self.oh_max_threshold)

        self.load_burst_data()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        number_classes = self.loader.num_classes

        train_x, train_y = self.dataset.train_data.unwrap()
        test_x, test_y = self.dataset.test_data.unwrap()
        eval_x, eval_y = self.dataset.eval_data.unwrap()

        # convert the trace data to burst data
        self.train_burst = train_x
        logger.info(self.train_burst.shape)

        self.discriminator = Discriminator(self.loader.num_classes).to(self.device)
        self.train_discriminator(self.train_burst, train_y)
        logger.info("DF 训练完成，即将训练 Generator")

        dummy_length, real_length = 0, 0
        generated_train_x, generated_train_y = [], []
        generated_test_x, generated_test_y = [], []
        generated_eval_x, generated_eval_y = [], []

        # 遍历每一个站点
        for site_idx, site_o in enumerate(range(number_classes)):
            generator = self.train_generator(site_o)
            site_burst_data = train_x[train_y == site_o]
            site_burst_test = test_x[test_y == site_o]
            site_burst_eval = eval_x[eval_y == site_o]


            # 转换成正常的 trace
            for original_burst, generated_x, generated_y in zip(
                [site_burst_data, site_burst_test, site_burst_eval],
                [generated_train_x, generated_test_x, generated_eval_x],
                [generated_train_y, generated_test_y, generated_eval_y]
            ):
                original_burst_tensor = torch.from_numpy(original_burst).to(self.device).float()
                generated_data_tensor = self.add_noise(original_burst_tensor, generator)
                generated_data = generated_data_tensor.detach().cpu().numpy()
                for i, burst_data in enumerate(generated_data):
                    trace = convert_burst_to_trace_data(burst_data, 5000)
                    dummy_length += np.sum(np.abs(trace))
                    generated_x.append(trace)
                generated_y.extend([site_o for _ in range(generated_data.shape[0])])
                # 记录原始数据长度
                for trace in original_burst:
                    for item in trace:
                        real_length += abs(int(item))

            if self.debug_limit_site_idx is not None and site_idx >= self.debug_limit_site_idx:
                break

        # 绘制 loss 变化曲线
        self.draw_loss_plot()
        overhead = (dummy_length - real_length) / real_length
        logger.info(f" overhead is {round(overhead * 100, 2)}%")
        return (
            (generated_train_x, generated_eval_x, generated_test_x),
            None,
            (generated_train_y, generated_eval_y, generated_test_y),
            overhead
        )


    def train_generator(self, site_o: int):
        site_burst_data = self.train_burst[self.dataset.train_data.y == site_o]

        # 随机选择一个其他站点不等于 o
        num_candidates = self.loader.num_classes
        if self.debug_limit_site_idx is not None:
            num_candidates = self.debug_limit_site_idx + 1
        target_site_t = get_random_value_excluding_m(num_candidates, site_o)
        logger.info(f"current site: {site_o}, target_site: {target_site_t}")
        target_burst_data = self.train_burst[self.dataset.train_data.y == target_site_t]

        train_loader = DataLoader(
            NoDefDataSet(
                site_burst_data,
                self.dataset.train_data.y[self.dataset.train_data.y == site_o]
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # init the generator
        generator = Generator(self.max_length, self.max_length).to(self.device)
        optimizer = optim.Adam(generator.parameters(), lr=1e-5)


        # self.num_epochs = 1
        pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS + 20,
            desc=f'Site: {site_o}, Epoch',
            position=0,
            total=self.num_epochs,
            ascii=True,
            leave=True
        )
        batch_pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS + 20,
            desc='Batch',
            position=1,
            total=len(train_loader),
            ascii=True,
            leave=False
        )

        for epoch in range(self.num_epochs):
            generator.train()
            sample = random.sample(list(np.arange(target_burst_data.shape[0])), self.batch_size)
            target_sample_burst = torch.from_numpy(target_burst_data[sample]).to(self.device).float()

            for (site_sample_batch, _) in train_loader:
                batch_size = site_sample_batch.shape[0]
                if batch_size != self.batch_size:
                    # logger.warning(f"batch  size too small: {batch_size}")
                    batch_pbr.update(1)
                    continue

                # logger.debug(f"current site: {site_o} with data shape {site_sample_batch.shape}, target site: {target_site_t} with data {target_burst_data.shape}")
                site_sample_batch = site_sample_batch.to(self.device)
                site_perturbation = self.add_noise(site_sample_batch, generator)
                # logger.info(sign_generated_noise.detach().cpu().numpy()[0, :10])

                # 计算损失值

                # 相似度损失值
                # 使用 DF 输出来进行计算

                _, predict_site = self.discriminator(torch.unsqueeze(site_perturbation, 1))
                # predict_target, _ = self.discriminator(torch.unsqueeze(target_sample_burst, 1))
                loss_sim = F.cross_entropy(predict_site, torch.tensor([target_site_t] * batch_size).to(self.device)) / 11

                # cosine_similarities = F.cosine_similarity(predict_site, predict_target, dim = 1)
                # loss_sim = loss_sim_all.mean()

                # 置信度损失值
                _, site_batch_to_target = self.discriminator(torch.unsqueeze(site_perturbation, 1))
                # 归一化
                site_batch_to_target_norm = F.normalize(site_batch_to_target, p=2, dim=1)
                # site_batch_to_target = torch.softmax(site_batch_to_target_original, dim=1).to(self.device)
                # logger.info(f"第一行置信度 {site_batch_to_target.shape}")
                # 获取当前 perturbation 被识别为站点 o 的置信度
                perturbation_at_o = site_batch_to_target_norm[:, site_o:site_o + 1]
                # logger.info(f"被识别为站点 {site_o} 的置信度 shape: {perturbation_at_t.shape}")
                site_batch_expect_site_o = torch.cat([
                    site_batch_to_target_norm[:,:site_o],
                    site_batch_to_target_norm[:,site_o+1:],
                ], dim=1)
                max_except_o = torch.max(site_batch_expect_site_o, dim=1).values

                # logger.info(f"被识别为其他站点的最大置信度， shape: {max_except_o.shape}")
                adversarial = perturbation_at_o - max_except_o
                adversarial_result = torch.max(adversarial, torch.full_like(adversarial, 0)).to(self.device)
                loss_adv = adversarial_result.mean()

                # overhead 损失值
                perturbed_bandwidth = torch.norm(site_perturbation, p = 1, dim = 1).to(self.device)
                site_bandwidth = torch.norm(site_sample_batch, p = 1, dim = 1).to(self.device)
                overhead = ((perturbed_bandwidth - site_bandwidth) / site_bandwidth)
                # logger.info(overhead.detach().cpu().numpy())

                overhead_greater = (overhead - self.oh_max_threshold)
                loss_per1 = torch.where(overhead_greater < 0, torch.tensor(0.).to(self.device), overhead_greater).mean()

                overhead_less = (self.oh_min_threshold - overhead)
                loss_per2 = torch.where(overhead_less < 0, torch.tensor(0.).to(self.device), overhead_less).mean()

                loss_per = loss_per1 + loss_per2

                loss = (
                    self.loss_alpha * loss_sim +
                    self.loss_beta * loss_adv +
                    self.loss_gamma * loss_per
                ) # type: torch.Tensor

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_pbr.update(1)
                batch_pbr.set_postfix(dict(
                    loss_sim=round(loss_sim.item(), 3),
                    loss_adv=round(loss_adv.item(), 3),
                    loss_per=round(loss_per.item(), 3),
                ))

            batch_pbr.reset()
            pbr.update(1)
            pbr.set_postfix(dict(loss=round(loss.item(), 3)))
            self.record_loss_value(site_o, [
                loss_sim, loss_adv, loss_per, loss
            ])

        batch_pbr.close()
        pbr.close()
        print("")

        return generator

    def train_discriminator(self, x_train, y_train, load_model=True):
        """ 训练识别器模型 """
        lr = 0.01
        momentum = 0.5
        batchSize = 64
        epochs = 30
        weight_decay = 1e-7
        save_weight_dir = os.path.join(
            MODEL_DIR, DefenderChoice.ALERT.value
        )
        if not os.path.exists(save_weight_dir):
            os.mkdir(save_weight_dir)

        weight_path = os.path.join(
            save_weight_dir, "df_weights.pth"
        )
        if load_model and os.path.exists(weight_path):
            self.discriminator.load_state_dict(torch.load(weight_path))
            logger.info("加载 discriminator 模型权重，跳过训练")
            return

        train_loader = DataLoader(
            NoDefDataSet(x_train, y_train),
            batch_size=batchSize,
            shuffle=True,
        )

        opt = optim.SGD(
            [{'params': self.discriminator.parameters()}],
            lr=lr, momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )

        pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS, desc='Epoch', position=0,
            total=epochs, ascii=True, leave=True
        )
        batch_pbr = tqdm.tqdm(
            ncols=TQDM_N_COLS, desc='Batch', position=1,
            total=len(train_loader), ascii=True, leave=False
        )

        for epoch in range(epochs):
            self.discriminator.train()
            totalLoss = 0
            correct = 0
            predict_list = []
            Label_list = []
            for batchID, (flow, target) in enumerate(train_loader):

                flow = flow.to(self.device)
                target = target.to(self.device)
                flow, target = Variable(flow), Variable(target)
                flow = flow.unsqueeze(1)

                opt.zero_grad()
                _, prediction = self.discriminator(flow)
                if batchID == 0:
                    logger.info(f"prediction shape: {prediction.shape}")
                    logger.info(f"target shape: {target.shape}")
                loss = F.cross_entropy(prediction, target.long())
                loss.backward()
                opt.step()
                totalLoss += loss
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += target.eq(pred.view_as(target)).cpu().sum().item()
                if epoch == epochs - 1:
                    predict_list += list(prediction.cpu().data.numpy().tolist())
                    Label_list += list(target.cpu().data.numpy().tolist())
                batch_pbr.update()

            loss = round(totalLoss.item(), 2)
            correct = round(correct * 100 / len(train_loader.dataset), 2)
            pbr.update()
            pbr.set_description(f'loss: {loss}, acc: {correct}%')
            batch_pbr.reset()

        pbr.close()
        batch_pbr.close()
        print("")
        torch.save(self.discriminator.state_dict(), weight_path)

    def add_noise(self, burst_data: torch.Tensor, generator: nn.Module) -> torch.Tensor:
        batch_size = burst_data.shape[0]
        mean_noise, std_noise = self.noise_param
        # 生成噪音
        noise = (
            torch.normal(
                mean=mean_noise,
                std=std_noise,
                size=(batch_size, self.max_length)
            ).to(self.device)
        )
        generated_noise = generator(noise) # type: torch.Tensor
        # 将处理过的噪音和原始数据相加生成 perturbed sequences
        sign_generated_noise = torch.where(
            burst_data > 0,
            generated_noise, torch.where(burst_data < 0, -generated_noise, 0 * generated_noise)
        )
        perturbation = burst_data + sign_generated_noise
        return perturbation


    def load_burst_data(self):
        """ 加载聚合后的数据 """
        burst_data_dir = os.path.join(
            DATA_DIR, DefenderChoice.NO_DEF.value, DefenderChoice.ALERT.value
        )
        if not os.path.exists(burst_data_dir):
            os.mkdir(burst_data_dir)

        burst_data_path = os.path.join(
            burst_data_dir, "burst_data.npz"
        )
        if os.path.exists(burst_data_path):
            logger.info("加载 burst_data.npz")
            burst_data = np.load(burst_data_path)
            train_data = Dataset(burst_data['train_x'], burst_data['train_y'])
            test_data = Dataset(burst_data['test_x'], burst_data['test_y'])
            eval_data = Dataset(burst_data['eval_x'], burst_data['eval_y'])
            self.dataset = DatasetWrapper(
                self.loader.config.dataset_choice,
                train_data, test_data, eval_data,
            )
            self.dataset.summary()
        else:
            self.loader.load()
            dataset = self.loader.split()
            dataset.train_data.x = convert_trace_data_to_burst(dataset.train_data.x, self.max_length)
            dataset.test_data.x = convert_trace_data_to_burst(dataset.test_data.x, self.max_length)
            dataset.eval_data.x = convert_trace_data_to_burst(dataset.eval_data.x, self.max_length)
            self.dataset = dataset
            np.savez(
                burst_data_path,
                train_x=dataset.train_data.x,
                train_y=dataset.train_data.y,
                test_x=dataset.test_data.x,
                test_y=dataset.test_data.y,
                eval_x=dataset.eval_data.x,
                eval_y=dataset.eval_data.y
            )



    def record_loss_value(self, site, loss_list: t.List[torch.tensor]):
        for loss_type, loss_value in zip([
            "loss_sim", "loss_adv", "loss_per", "loss"
        ], loss_list):
            self.loss_record[site][loss_type].append(round(loss_value.item(), 3))


    def draw_loss_plot(self):
        import matplotlib.pyplot as plt

        for site_o, loss_dic in self.loss_record.items():
            fig, ax = plt.subplots()

            for loss_type, loss_value in loss_dic.items():
                avg_loss = []
                total_loss = 0
                for item in loss_value:
                    total_loss += item
                    avg_loss.append(total_loss / (len(avg_loss) + 1))
                ax.plot(avg_loss, label=loss_type)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            pic_path = os.path.join(PIC_DIR, f'{site_o}.png')
            plt.savefig(pic_path)