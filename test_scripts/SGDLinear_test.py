import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.SGDLinear import SGDLinear
from data import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data(train_features, train_labels, test_features, test_labels):
    # 如果输入是多维的，将每个样本重塑为一维向量
    if train_features.dim() > 2:
        # 获取第一维的大小（样本数量）
        batch_size = train_features.size(0)
        # 将剩余维度展平为一维
        train_features = train_features.view(batch_size, -1)
    
    if test_features.dim() > 2:
        batch_size = test_features.size(0)
        test_features = test_features.view(batch_size, -1)
    
    # 确保特征是浮点类型，这对于神经网络训练很重要
    train_features = train_features.float()
    test_features = test_features.float()
        
    # 创建训练集和测试集
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader

def train_model(train_loader, num_epochs=10):
    # 初始化模型
    model = SGDLinear()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 获取优化器
    optimizer = model.get_optimizer()
    # 获取损失函数
    criterion = model.get_loss_function()
    # 获取学习率调度器
    scheduler = model.get_scheduler(optimizer)

    # 初始化列表，用于存储每个 epoch 的训练损失和准确率
    train_loss = []
    train_accuracy = []
    
    # 训练模型
    for epoch in range(num_epochs):
        # 将模型设置为训练模式
        model.train()
        # 初始化变量，用于跟踪当前 epoch 的损失和准确率
        running_loss = 0.0
        correct = 0
        total = 0

        # 遍历训练数据加载器
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # 将输入数据移动到 GPU（如果可用）
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 更新损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()    

        # 计算当前 epoch 的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total

        # 将当前 epoch 的损失和准确率添加到列表中
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # 打印训练信息
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    # 保存模型
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(script_dir, 'saved_models', 'SGDLinear.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    return train_loss, train_accuracy

def test_model(test_loader):
    model = SGDLinear()
    # 使用绝对路径加载模型
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, 'saved_models', 'SGDLinear.pth')
    model.load_state_dict(torch.load(model_path))
    # 将模型设置为评估模式
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 初始化变量，用于跟踪测试损失和准确率
    correct = 0
    total = 0
    predictions = []

    # 禁用梯度计算（测试时不需要计算梯度，节省内存和加速）
    with torch.no_grad():
        # 遍历测试数据加载器
        for i, (inputs, labels) in enumerate(test_loader, 0):
            # 将输入数据移动到 GPU（如果可用）
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 前向传播，获取模型输出
            outputs = model(inputs)
            # 获取预测结果（取输出的最大值对应的类别）
            _, predicted = torch.max(outputs.data, 1)
            # 更新预测结果
            predictions.extend(predicted.cpu().numpy())
            # 更新准确率
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算测试集的准确率
        accuracy = 100 * correct / total
        # 打印测试集的准确率
        print(f'Test Accuracy: {accuracy:.2f}%')
        # 返回测试集的准确率和预测结果
        return accuracy, predictions

if __name__ == '__main__':
    torch.manual_seed(42)  # 设置随机种子以确保可重复性
    # 加载数据集
    train_features, train_labels, test_features, test_labels = load_data.fetch_mnist_data()
    # print(train_features.size())
    # print(train_labels.size())
    # print(test_features.size())
    # print(test_labels.size())
    # 预处理数据
    preprocess_data(train_features, train_labels, test_features, test_labels)
    train_loader, test_loader = preprocess_data(train_features, train_labels, test_features, test_labels)
    # 训练模型
    train_loss, train_accuracy = train_model(train_loader, num_epochs=10)
    # 测试模型
    accuracy, predictions = test_model(test_loader)
    # 绘制训练损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 打印前十个预测于真实值的对比结果
    # 设置支持中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题  
    # 创建一个2x5的子图布局，用于显示10张图片
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # 设置子图间距
    # 遍历前10个测试样本
    for i in range(10):
        row, col = i // 5, i % 5  # 计算行列索引
        ax = axes[row, col]
        
        # 在对应的子图中显示图片
        ax.imshow(test_features[i].reshape(28, 28), cmap=plt.cm.gray)
        ax.set_title(f"实际为：{test_labels[i].item()}，预测为：{predictions[i]}")
        ax.axis('off')  # 隐藏坐标轴
    plt.tight_layout()
    plt.show()
    
