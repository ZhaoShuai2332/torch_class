import torch.nn as nn
import torch.optim as optim

class SGDLogistic(nn.Module):
    def __init__(self):
        super(SGDLogistic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_optimizer(self, learning_rate=0.04, momentum=0.9, weight_decay=1e-4):
        return optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    def get_loss_function(self):
        return nn.CrossEntropyLoss()  # Placeholder
    
    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

