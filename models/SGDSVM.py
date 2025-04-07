import torch
import torch.nn as nn
import torch.optim as optim

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, targets):
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        targets_one_hot = targets_one_hot * 2 - 1  # 转换为 {-1, 1}

        margins = 1 - outputs * targets_one_hot
        losses = torch.clamp(margins, min=0)

        return torch.mean(losses)

# 核函数为linear核的支持向量机
class SGDSVM(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(SGDSVM, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )
        # self.model = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, learning_rate=0.1, momentum=0.9, weight_decay=1e-4):
        return optim.SGD(self.parameters(), lr=learning_rate,
                         momentum=momentum, weight_decay=weight_decay)

    def get_loss_function(self):
        return HingeLoss()

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
