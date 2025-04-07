import torch.nn as nn
import torch.optim as optim

class DecisionTree(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(DecisionTree, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, learning_rate=0.01, momentum=0.9, weight_decay=1e-4):
        return optim.SGD(self.parameters(), lr=learning_rate,
                         momentum=momentum, weight_decay=weight_decay)

    def get_loss_function(self):
        return nn.CrossEntropyLoss()  # Placeholder for decision tree loss

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


