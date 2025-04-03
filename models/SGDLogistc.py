import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SGDLogistic(nn.Module):
    def __init__(self):
        super(SGDLogistic, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(784, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.linear(x)
    
    def get_optimizer(self, learning_rate=0.025, momentum=0.9, weight_decay=1e-4):
        return optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    def get_loss_function(self):
        return nn.CrossEntropyLoss()
    
    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        
