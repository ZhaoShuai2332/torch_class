import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 通过随机梯度下降实现线性回归
class SGDLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SGDLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
    def backward(self, x, y):
        # 前向传播
        y_pred = self.forward(x)
        # 计算损失
        loss = F.mse_loss(y_pred, y)
        # 清空梯度
        self.zero_grad()
        # 反向传播
        loss.backward()
        return loss.item()
    
    def update(self, learning_rate):
        # 更新参数
        for param in self.parameters():
            param.data -= learning_rate * param.grad.data
        return self.linear.weight.data, self.linear.bias.data
    
    def train_model(self, x_train, y_train, learning_rate=0.001, epochs=100):
        # 训练模型
        losses = []
        for epoch in range(epochs):
            loss = self.backward(x_train, y_train)
            self.update(learning_rate)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        return losses
    
    def predict(self, x):
        # 预测
        with torch.no_grad():
            return self.forward(x).numpy()
        
    def plot_loss(self, losses):
        # 绘制损失曲线
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.show()
    
