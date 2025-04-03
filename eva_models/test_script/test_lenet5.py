import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.LeNet5 import LeNet5
from data.load_data import fetch_mnist_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Use Path for more reliable path handling
script_dir = Path(__file__).parent.parent.parent

def load_model():
    model_path = script_dir / 'saved_models' / 'lenet5.pth'
    model = LeNet5()
    model.load_state_dict(torch.load(str(model_path)))
    return model

def load_data():
    test_data_path = script_dir / 'data' / 'test_data'
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    images = []
    labels = []
    
    for file_name in os.listdir(test_data_path):
        if file_name.endswith('.png'):
            label = int(file_name.split('_')[-1][0])  # Extract label from the last character before the extension
            img_path = test_data_path / file_name
            img = torchvision.io.read_image(str(img_path))
            print(img.shape)
            # Convert to RGB if the image has an alpha channel
            # if img.shape[0] == 4:  # Check if the image has 4 channels (RGBA)
            #     img = img[:3]  # Take only the RGB channels
            # print(img.shape)
            # img = transform(img)
            images.append(img)
            labels.append(label)
    
    images = torch.stack(images).view(-1, 1, 28, 28)  # Convert to 4D tensor
    labels = torch.tensor(labels)
    
    return images, labels

if __name__ == '__main__':
    # load_data()
    images, labels = load_data()
    print(len(images))
    print(len(labels))
    print(script_dir)
    # test_dataset = torch.utils.data.TensorDataset(images, labels)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    # model = load_model()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # model.eval()
    # with torch.no_grad():
    #     # 创建一个大图，用于显示30张图片
    #     fig, axes = plt.subplots(5, 6, figsize=(15, 12))
    #     axes = axes.flatten()
        
    #     # 计数器，最多显示30张图片
    #     count = 0
        
    #     for data, target in test_loader:
    #         # 获取模型预测结果
    #         output = model(data)
    #         pred = output.argmax(dim=1, keepdim=True)
            
    #         # 显示图片和预测结果
    #         img = data.squeeze().numpy()
    #         pred_label = pred.item()
    #         true_label = target.item()
            
    #         # 在子图中显示图片
    #         axes[count].imshow(img, cmap='gray')
    #         axes[count].set_title(f'预测: {pred_label}, 真实: {true_label}')
    #         axes[count].axis('off')
            
    #         count += 1
            
    #         # 如果已经显示了30张图片，就退出循环
    #         if count >= 30:
    #             break
        
    #     # 显示图片
    #     plt.tight_layout()
    #     plt.show()
