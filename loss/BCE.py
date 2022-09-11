"""

二值交叉熵损失函数
"""

import torch.nn as nn


class BCELoss(nn.Module):  # Binary CrossEntropy loss 用于二分类的交叉熵损失函数

    def __init__(self):
        super().__init__()  # python3可以直接写成super().__init__(); python2必须写成super(本类名,self).__init__()

        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        return self.bce_loss(pred, target)
