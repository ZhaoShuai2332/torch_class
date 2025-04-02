import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一层卷积层：输入1通道（灰度图像），输出6通道，卷积核大小为5x5
        self.conv_layer1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第一层池化层：最大池化，窗口大小为2x2，步幅为2
        self.pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二层卷积层：输入6通道，输出16通道，卷积核大小为5x5
        self.conv_layer2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二层池化层：最大池化，窗口大小为2x2，步幅为2
        self.pool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一层全连接层：输入维度16*4*4，输出维度120
        self.fc_layer1 = nn.Linear(16 * 4 * 4, 120)
        # 第二层全连接层：输入维度120，输出维度84
        self.fc_layer2 = nn.Linear(120, 84)
        # 第三层全连接层：输入维度84，输出维度10（对应10个类别）
        self.fc_layer3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播函数
        # 输入经过第一层卷积层、ReLU激活函数，然后经过第一层池化层
        x = self.pool_layer1(F.relu(self.conv_layer1(x)))
        # 输入经过第二层卷积层、ReLU激活函数，然后经过第二层池化层
        x = self.pool_layer2(F.relu(self.conv_layer2(x)))
        # 将多维特征图展平为一维向量，以便输入全连接层
        x = x.view(-1, 16 * 4 * 4)
        # 输入经过第一层全连接层、ReLU激活函数
        x = torch.relu(self.fc_layer1(x))
        # 输入经过第二层全连接层、ReLU激活函数
        x = torch.relu(self.fc_layer2(x))
        # 输入经过第三层全连接层，输出最终分类结果
        x = self.fc_layer3(x)
        return x
    
    # 定义优化器，使用随机梯度下降（SGD）
    def get_optimizer(model, learning_rate=0.01, momentum=0.9):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        return optimizer
    
    # def predict(model, inputs):
    #     # 将模型设置为评估模式
    #     model.eval()
    #     # 前向传播，获取模型输出
    #     with torch.no_grad():
    #         outputs = model(inputs)
    #     # 获取预测结果
    #     _, predicted = torch.max(outputs.data, 1)
    #     return predicted


