import torch
from torch.utils.data import TensorDataset, DataLoader
from models.LeNet5 import LeNet5
from data.load_data import fetch_mnist_data

# 加载数据
train_feature, train_label, test_feature, test_label = fetch_mnist_data()

# 创建数据集
train_dataset = TensorDataset(train_feature, train_label)
test_dataset = TensorDataset(test_feature, test_label)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
model = LeNet5()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = model.get_optimizer(model, learning_rate=0.01, momentum=0.9)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f'Accuracy: {100 * correct / total:.2f}%') 