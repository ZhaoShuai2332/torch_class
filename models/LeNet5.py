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
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)
    
    def get_optimizer(model, learning_rate=0.01, momentum=0.9):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        return optimizer
    
    # def predict(model, inputs):
    #     model.eval()
    #     with torch.no_grad():
    #         outputs = model(inputs)
    #     _, predicted = torch.max(outputs.data, 1)
    #     return predicted


