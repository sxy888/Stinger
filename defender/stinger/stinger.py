import logging  # noqa
import sys
import typing as t
import numpy as np
from numpy import ndarray # noqa
from keras.utils import to_categorical
from keras.optimizers import Adamax
import torch
import random
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import tqdm
from loader.base import Config, DatasetWrapper, Dataset
from .model import DF, BD, AS, AC, VC, GANDALF
from constant.enum import AttackChoice
from defender.abc import AbstractDefender
logger = logging.getLogger(__name__)


lr = 0.01
momentum = 0.5
batchSize = 128
epochs = 30
weight_decay = 1e-7

attack_method_map = {
    AttackChoice.DF.value: DF,
    AttackChoice.AWF_SDAE.value:  AS,
    AttackChoice.AWF_CNN.value: AC,
    AttackChoice.VAR_CNN.value: VC,
    AttackChoice.GANDaLF.value: GANDALF,
} # type: t.Dict[str, t.Type[torch.nn.Module]]

class NoDefDataSet(Dataset):
    def __init__(self, data: ndarray, label: ndarray):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

class StingerDefender(AbstractDefender):

    dataset: t.Union[None, DatasetWrapper] = None

    def defense(self, **kwargs):
        attack_method = kwargs.get("attack_method", "DF")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if isinstance(attack_method, AttackChoice):
            attack_method = attack_method.value

        attack_model_cls = attack_method_map.get(attack_method)
        logger.info(f"使用模型 {attack_method}: {attack_model_cls} 进行对抗训练")
        if attack_model_cls is None:
            logger.error(f"对抗模型方法 {attack_method} 不存在")
            sys.exit(1)

        self.attack_model = attack_model_cls(self.loader.num_classes).to(self.device)
        self.bd = BD(self.loader.num_classes).to(self.device)

        x_train = self.dataset.train_data.x
        y_train = self.dataset.train_data.y

        x_test = self.dataset.test_data.x
        y_test = self.dataset.test_data.y

        x_valid = self.dataset.eval_data.x
        y_valid = self.dataset.eval_data.y

        index_train = random.randint(0, 5000 - 256)
        index_test  = random.randint(0, 5000 - 256)
        opt = optim.SGD([
                { 'params': self.attack_model.parameters() },
                { 'params': self.bd.parameters() }
            ],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
        train_loader = DataLoader(NoDefDataSet(x_train, y_train), batch_size=batchSize, shuffle=True)
        test_loader = DataLoader(NoDefDataSet(x_test, y_test), batch_size=batchSize, shuffle=True)
        valid_loader = DataLoader(NoDefDataSet(x_valid, y_valid), batch_size=batchSize, shuffle=True)

        predict_list_B, Label_list_B = self.train(index_train, train_loader, epochs, opt)

        self.test(index_test,  test_loader)
        self.test(index_train, valid_loader)

        traces, overhead_list = [], []
        generated_train, train_overhead = self.bd_insert(index_train, x_train, y_train, 0)
        generated_valid, valid_overhead = self.bd_insert(index_train, x_valid, y_valid, 0)
        generated_test, test_overhead = self.bd_insert(index_test, x_test, y_test, 1)

        traces.extend(generated_train)
        traces.extend(generated_valid)
        traces.extend(generated_test)

        overhead = sum([
            train_overhead * len(generated_train) / len(self.X),
            valid_overhead * len(generated_valid) / len(self.X),
            test_overhead * len(generated_test) / len(self.X),
        ])

        logger.info("the final overhead is %.2f", overhead)
        return (generated_train, generated_valid, generated_test), self.Timestamp, (y_train, y_valid, y_test), overhead




    def train(self, index, train_loader, epochs, optimizer):
        logger.info("Begin training Stinger attack model")
        pbr = tqdm.tqdm(
            ncols=100, desc='Epoch', position=0,
            total=epochs, ascii=True, leave=True
        )
        batch_pbr = tqdm.tqdm(
            ncols=100, desc='Batch', position=1,
            total=len(train_loader), ascii=True, leave=False
        )
        for epoch in range(epochs):
            self.attack_model.train()
            totalLoss = 0
            correct = 0
            predict_list = []
            Label_list = []

            for batchID, (flow, flowLabel) in enumerate(train_loader):
                bd_C = torch.eye(self.loader.num_classes)[flowLabel, :]
                bd_C = bd_C.to(self.device)
                flow = flow.to(self.device)
                # target = target.to(device)
                flowLabel = flowLabel.to(self.device)
                # bd_C, flow, target = Variable(bd_C), Variable(flow), Variable(target)
                bd_C, flow, flowLabel = Variable(bd_C), Variable(flow), Variable(flowLabel)

                backdoor = self.bd(bd_C)
                sub1, sub2 = torch.split(flow, dim=1, split_size_or_sections=[index, 5000 - index])

                flow_ = torch.cat((sub1, backdoor, sub2), dim=1)
                flow_ = flow_[:, 0:5000]
                flow_ = flow_.unsqueeze(1)

                optimizer.zero_grad()

                # TODO, 切换 DF 模型为别的攻击算法
                bd_p = self.attack_model(flow_)
                # logger.info(f"bd_p size: {bd_p.size()}, flowLabel size: {flowLabel.size()}")
                loss = F.cross_entropy(bd_p, flowLabel)
                loss.backward()
                optimizer.step()
                totalLoss += loss

                pred = bd_p.data.max(1, keepdim=True)[1]
                correct += flowLabel.eq(pred.view_as(flowLabel)).cpu().sum().item()
                if epoch == epochs - 1:
                    predict_list += list(bd_p.cpu().data.numpy().tolist())
                    Label_list += list(flowLabel.cpu().data.numpy().tolist())
                batch_pbr.update()
            loss = round(totalLoss.item(), 2)
            correct = round(correct * 100 / len(train_loader.dataset), 2)
            pbr.update()
            pbr.set_description(f'loss: {loss}, acc: {correct}%')
            batch_pbr.reset()
        pbr.close()
        batch_pbr.close()
        return predict_list, Label_list

    def test(self, index, test_loader):
        self.attack_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batchID, (flow, flowLabel) in enumerate(test_loader):
                target = torch.from_numpy(np.random.randint(0, self.loader.num_classes, np.shape(flow.numpy())[0])).to(self.device)
                bd_C = torch.eye(self.loader.num_classes).to(self.device)[target, :]
                bd_C = bd_C.to(self.device)
                # print(data.shape)
                flow = flow.to(self.device)
                target = target.to(self.device)
                bd_C, flow, target = Variable(bd_C), Variable(flow), Variable(target)
                # # add
                # bd_C = bd_C.unsqueeze(1)

                backdoor = self.bd(bd_C)
                # index = random.randint(0, 5000 - 256)
                sub1, sub2 = torch.split(flow, dim=1, split_size_or_sections=[index, 5000 - index])
                flow_ = torch.cat((sub1, backdoor, sub2), dim=1)
                flow_ = flow_[:, 0:5000]
                flow_ = flow_.unsqueeze(1)
                bd_p = self.attack_model(flow_)
                loss = F.cross_entropy(bd_p, target.long())
                test_loss += loss
                pred = bd_p.data.max(1, keepdim=True)[1]
                correct += target.eq(pred.view_as(target)).cpu().sum().item()

            # print('test loss ',test_loss.item(),'correct ',str(correct/len(test_loader.dataset)*100)+'%')
            logger.info('test_loss:{}, correct:{}%'.format(test_loss.item(), round(correct * 100 / len(test_loader.dataset), 2)))


    def bd_insert(self, index,  dataset, labelName, flag,):
        pad_data = []
        total = 0
        extra = 0
        backSet = np.array([i for i in range(0, self.loader.num_classes)])
        for i in range(len(dataset)):
            data = dataset[i]
            data = data[data != 0]
            real_length = len(data)
            total += real_length
            fin_data = np.zeros(6000)
            fin_data[0:real_length] = data
            # label = labelSet[i]
            label = random.choice(backSet)
            tmp = 0
            # flag=1 represent test dataset
            if flag:
                # label = random.choice(backSet)
                index = random.randint(0, min(real_length, 4999))
            label = torch.eye(self.loader.num_classes)[label, :]
            label = label.to(self.device)
            backdoor = self.bd(label)
            backdoor = backdoor.cpu().numpy()
            # index = random.randint(0, min(real_length, 4999))
            fin_data = np.insert(fin_data, index, backdoor)
            tmp += 256
            pad_data.append(fin_data[0:5000])
            extra += tmp
        overhead = extra / total

        return pad_data, overhead