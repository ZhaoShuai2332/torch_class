import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SGDLinear(nn.Module):
    def __init__(self):
        # 初始化父类（Module）
        super(SGDLinear, self).__init__()
        # 创建一个线性层，输入维度为784（28x28像素），输出维度为10（数字0-9）
        self.linear = nn.Linear(784, 10)
    
    def forward(self, x):
        # 定义前向传播过程
        # x: 输入数据
        # 返回: 模型输出
        return self.linear(x)
    
    def get_optimizer(self, learning_rate=0.02, momentum=0.9):
        # 获取优化器，使用随机梯度下降（SGD）算法
        # model: 模型参数
        # learning_rate: 学习率，默认为0.01
        # momentum: 动量参数，默认为0.9
        return optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
    
    def get_loss_function(self):
        # 获取损失函数，使用交叉熵损失
        # 适用于分类问题
        return nn.CrossEntropyLoss()
    
    def get_scheduler(self, optimizer):
        # 获取学习率调度器，使用阶梯式学习率衰减
        # optimizer: 优化器
        # num_epochs: 训练轮数，默认为10
        # 每10个epoch学习率降为原来的0.1倍
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    
