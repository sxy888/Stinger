import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        # self.scale_factor = 20.0
        #  input_size
        self.layers = nn.Sequential(
            # 重复五次的结构
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),

            # 最后再接的部分
            nn.Linear(1024, output_size),
            nn.ReLU()  # 合适的激活函数
        )

    def forward(self, x):
        x = self.layers(x)
        return x



class Discriminator(nn.Module):

    # use DF as the Discriminator
    def __init__(self, classes_num):
        super(Discriminator, self).__init__()

        self.classes_num = classes_num
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
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
        )
        
        self.fc_block1 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, classes),
        )
        

    def forward(self, input):
        """
        X: [batch_size, sequence_length]
        """
        # logger.info(input.shape)
        batch_size = input.shape[0]
        block1 = self.block1(input)  # [batch_size, output_channel*1*1]
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        flatten = block4.view(batch_size, -1)
        fc = self.fc_block(flatten)
        fc1 = self.fc_block1(fc)
        return fc, fc1


