# import torch
#
# print(torch.cuda.is_available())
# print(torch.__version__)
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DF(nn.Module):

    def __init__(self, classes_num):
        super(DF, self).__init__()

        classes = classes_num
        filter_num = ['None', 32, 64, 128, 256]
        kernel_size = ['None', 8, 8, 8, 8]
        pool_stride_size = ['None', 4, 4, 4, 4]
        pool_size = ['None', 8, 8, 8, 8]
        self.block1 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),  # pytroch 不支持 padding='same' 手动填充
            nn.Conv1d(in_channels=1, out_channels=filter_num[1], kernel_size=kernel_size[1]),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1]),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(),
            nn.ConstantPad1d((2, 2), 0),
            nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1]),
            nn.Dropout(0.1)
        )

        self.block2 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),  # pytroch 不支持 padding='same' 手动填充
            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2]),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2]),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 3), 0),
            nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2]),
            nn.Dropout(0.1)
        )

        self.block3 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),  # pytroch 不支持 padding='same' 手动填充
            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3]),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3]),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3]),
            nn.Dropout(0.1)
        )

        self.block4 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),  # pytroch 不支持 padding='same' 手动填充
            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4]),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4]),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(),
            nn.ConstantPad1d((2, 3), 0),
            nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4]),
            nn.Dropout(0.1)
        )

        self.fc_block = nn.Sequential(
            nn.Linear(5120, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, classes)
        )

    def forward(self, input):
        """
        X: [batch_size, sequence_length]
        """
        batch_size = input.shape[0]
        block1 = self.block1(input)  # [batch_size, output_channel*1*1]
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        flatten = block4.view(batch_size, -1)
        fc = self.fc_block(flatten)
        return fc


class BD(nn.Module):
    def __init__(self, bd_num):
        super(BD, self).__init__()

        classes = bd_num

        self.block0 = nn.Sequential(
            # nn.Linear(1, 30),
            nn.Linear(bd_num, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            # nn.Tanh()
            # nn.Linear(128, 64),
            nn.Sigmoid()
        )
        # self.block0 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8),
        #     nn.BatchNorm1d(32),
        #     nn.ELU(),
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8),
        #     nn.BatchNorm1d(32),
        #     nn.ELU(),
        #     nn.MaxPool1d(kernel_size=8, stride=4),
        #     nn.Dropout(0.1)
        # )
        # self.fc_block = nn.Sequential(
        #         nn.Linear(608, 128),
        #         nn.Tanh(),
        #         nn.Linear(128, 256),
        #         nn.Sigmoid()
        # )

    def forward(self, input):
        """
        X: [batch_size, sequence_length]
        """
        # # add
        # batch_size = input.shape[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        block0 = self.block0(input)
        # # add
        # flatten = block0.view(batch_size, -1)
        # bd = self.fc_block(flatten)
        # bd = torch.where(bd > 0.5, torch.tensor(1.0).to(device), torch.tensor(-1.0).to(device))

        bd = torch.where(block0 > 0.5, torch.tensor(1.0).to(device), torch.tensor(-1.0).to(device))

        return bd


class AC(nn.Module):  # AWF_CNN
    def __init__(self, classes_num):
        super(AC, self).__init__()

        classes = classes_num
        self.Conv1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.flatten = nn.Flatten()
        self.block1 = nn.Sequential(
            nn.Linear(2432, 2432),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2432, classes)
            # nn.Softmax()
        )

    def forward(self, input):
        """
        X: [batch_size, sequence_length]
        """
        # block0 = self.block0(input)  # [batch_size, output_channel*1*1]
        x = self.Conv1(input)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.flatten(x)
        block1 = self.block1(x)
        return block1


class AS(nn.Module):  # AWF_SDAE

    def __init__(self, classes_num: int):
        super(AS, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300, classes_num)
        )

    def forward(self, input):
        x = self.block1(input)
        x1 = torch.squeeze(x)
        # print("=========", x.shape, x1.shape)
        return x1


class VC(nn.Module):
    def __init__(self, classes_num):
        super(VC, self).__init__()

        classes = classes_num

        self.pre = nn.Sequential(
            nn.ConstantPad1d((3, 3), 0),
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConstantPad1d((0, 1), 0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.block1 = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, bias=False),  # **parameters
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConstantPad1d((4, 0), 0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, bias=False),
            # , **parameters
            nn.BatchNorm1d(64)

        )
        self.shortCut1 = nn.Sequential(
            # shortcut the input is X
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(64)
        )
        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.ConstantPad1d((8, 0), 0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=4, bias=False),
            # , **parameters
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConstantPad1d((16, 0), 0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=8, bias=False),  # , **parameters
            nn.BatchNorm1d(64)
        )
        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.ConstantPad1d((1, 0), 0),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, dilation=1, bias=False),
            # , **parameters
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConstantPad1d((4, 0), 0),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, dilation=2, bias=False),  # , **parameters
            nn.BatchNorm1d(128)
        )
        self.shortCut2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(128)
        )
        self.block4 = nn.Sequential(
            nn.ReLU(),
            nn.ConstantPad1d((8, 0), 0),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=4, bias=False),
            # , **parameters
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConstantPad1d((16, 0), 0),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, dilation=8, bias=False),  # , **parameters
            nn.BatchNorm1d(128)
        )
        self.block5 = nn.Sequential(
            nn.ReLU(),
            nn.ConstantPad1d((2, 0), 0),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, dilation=1, bias=False),
            # , **parameters
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConstantPad1d((4, 0), 0),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, dilation=2, bias=False),  # , **parameters
            nn.BatchNorm1d(256)

        )
        self.shortCut3 = nn.Sequential(
            # shortcut
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(256)
        )
        self.block6 = nn.Sequential(
            nn.ReLU(),
            nn.ConstantPad1d((8, 0), 0),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=4, bias=False),
            # , **parameters
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConstantPad1d((16, 0), 0),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, dilation=8, bias=False),  # , **parameters
            nn.BatchNorm1d(256)
        )
        self.block7 = nn.Sequential(
            nn.ReLU(),
            nn.ConstantPad1d((2, 0), 0),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, dilation=1, bias=False),
            # , **parameters
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConstantPad1d((4, 0), 0),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, bias=False),  # , **parameters
            nn.BatchNorm1d(512)
        )
        self.shortCut4 = nn.Sequential(
            # shortcut
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(512)
            # nn.Add()  ?merge
        )
        self.block8 = nn.Sequential(
            nn.ReLU(),
            nn.ConstantPad1d((8, 0), 0),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dilation=4, bias=False),
            # , **parameters
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConstantPad1d((16, 0), 0),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=8, bias=False),  # , **parameters
            nn.BatchNorm1d(512)
        )
        self.globalAvgPool = nn.Sequential(
            nn.ReLU()
            # nn.AdaptiveAvgPool1d(512)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, classes)
        )

    def forward(self, input):
        """
        X: [batch_size, sequence_length]
        """
        z = self.pre(input)
        x = self.block1(z)
        y = self.shortCut1(z)
        z = torch.add(x, y)
        x = self.block2(z)
        z = torch.add(x, z)

        x = self.block3(z)
        y = self.shortCut2(z)
        z = torch.add(x, y)
        x = self.block4(z)
        z = torch.add(x, z)

        x = self.block5(z)
        y = self.shortCut3(z)
        z = torch.add(x, y)
        x = self.block6(z)
        z = torch.add(x, z)

        x = self.block7(z)
        y = self.shortCut4(z)
        z = torch.add(x, y)
        x = self.block8(z)
        z = torch.add(x, z)

        x = self.globalAvgPool(z)
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x


class GANDALF(nn.Module):

    def __init__(self, classes_num: int):
        super(GANDALF, self).__init__()

        self.head_model = nn.Sequential(
            nn.Linear(5000, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, classes_num)
        )
    def forward(self, input):
        x = self.head_model(input)
        x1 = torch.squeeze(x)
        return x1