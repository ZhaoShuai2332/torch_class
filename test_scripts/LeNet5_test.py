from models import LeNet5
from data import load_data
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
import os

def preprocess_data(train_features, train_labels, test_features, test_labels):
    # 定义数据转换操作，包括随机裁剪、水平翻转和标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 创建训练集和测试集
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train_model(train_loader, num_epochs=10):
    # 初始化 LeNet5 模型
    model = LeNet5.LeNet5()
    # 将模型移动到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 获取模型的优化器
    optimizer = LeNet5.get_optimizer(model)
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 初始化列表，用于存储每个 epoch 的训练损失和准确率
    train_loss = []
    train_accuracy = []
    
    # 遍历指定的训练轮数（epoch）
    for epoch in range(num_epochs):
        # 将模型设置为训练模式
        model.train()
        # 初始化变量，用于跟踪当前 epoch 的损失和准确率
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 遍历训练数据
        for inputs, labels in train_loader:
            # 将输入和标签数据移动到 GPU（如果可用）
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            
            # 累加损失
            running_loss += loss.item()
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 累加总样本数
            total += labels.size(0)
            # 累加正确预测的样本数
            correct += (predicted == labels).sum().item()
        
        # 计算当前 epoch 的平均损失
        epoch_loss = running_loss / len(train_loader)
        # 计算当前 epoch 的准确率
        epoch_accuracy = 100 * correct / total
        # 将损失和准确率添加到列表中
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        
        # 打印当前 epoch 的损失和准确率
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    # 保存训练好的模型
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(script_dir, 'saved_models', 'lenet5.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return train_loss, train_accuracy

def test_model(test_loader):
    # 加载保存好的模型
    model = LeNet5.LeNet5()
    model.load_state_dict(torch.load('saved_models/lenet5.pth'))
    # 将模型移动到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 给出预测值
    model.eval()
    correct = 0
    total = 0
    predictions = []
    # 禁用梯度计算（测试时不需要计算梯度，节省内存和加速）
    with torch.no_grad():
        # 遍历测试数据集
        for inputs, labels in test_loader:
            # 将输入和标签数据移动到 GPU（如果可用）
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播，获取模型输出
            outputs = model(inputs)
            # 获取预测结果（取输出的最大值对应的类别）
            _, predicted = torch.max(outputs.data, 1)
            # 累加测试样本总数
            total += labels.size(0)
            # 累加正确预测的样本数
            correct += (predicted == labels).sum().item()
            # 保存预测结果
            predictions.extend(predicted.cpu().numpy())
    # 计算测试集的准确率
    accuracy = 100 * correct / total
    # 打印测试集的准确率
    print(f'Test Accuracy: {accuracy:.2f}%')
    # 返回测试集的准确率和预测结果
    return accuracy, predictions

# if __name__ == '__main__':
#     torch.manual_seed(42)  # 设置随机种子以确保可重复性
#     # 加载数据集
#     train_features, train_labels, test_features, test_labels = load_data.load_data()
#     # 定义数据转换操作


