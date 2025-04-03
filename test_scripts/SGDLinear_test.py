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
        batch_size = train_features.size(0)
        train_features = train_features.view(batch_size, -1)
    
    if test_features.dim() > 2:
        batch_size = test_features.size(0)
        test_features = test_features.view(batch_size, -1)
    
    train_features = train_features.float()
    test_features = test_features.float()
        
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader

def train_model(train_loader, num_epochs=10):
    model = SGDLinear()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = model.get_optimizer()
    criterion = model.get_loss_function()
    scheduler = model.get_scheduler(optimizer)

    train_loss = []
    train_accuracy = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()    

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    # 保存模型
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(script_dir, 'saved_models', 'SGDLinear.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    return train_loss, train_accuracy

def test_model(test_loader):
    model = SGDLinear()
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, 'saved_models', 'SGDLinear.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy, predictions

if __name__ == '__main__':
    torch.manual_seed(42)  # 设置随机种子以确保可重复性
    # 加载数据集
    train_features, train_labels, test_features, test_labels = load_data.fetch_mnist_data()
    # print(train_features.size())
    # print(train_labels.size())
    # print(test_features.size())
    # print(test_labels.size())
    preprocess_data(train_features, train_labels, test_features, test_labels)
    train_loader, test_loader = preprocess_data(train_features, train_labels, test_features, test_labels)
    train_loss, train_accuracy = train_model(train_loader, num_epochs=10)
    accuracy, predictions = test_model(test_loader)
    
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
    
