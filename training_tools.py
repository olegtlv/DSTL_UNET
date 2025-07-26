import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, logits=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            inputs = torch.sigmoid(inputs)

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(self, gamma=0.85, weight_dice=0.25):
        super(CombinedLoss, self).__init__()
        self.criterion_BCE = nn.BCEWithLogitsLoss()  # ✅ safer for AMP
        self.criterion_Dice = DiceLoss(logits=True)  # ✅ we'll fix this below
        self.gamma = gamma
        self.weight_dice = weight_dice

    #         self.dice_step = dice_step

    def get_optimizer_and_scheduler(self, model_parameters):
        optimizer = optim.Adam(model_parameters)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return optimizer, scheduler

    def forward(self, outputs, targets, epoch):
        loss_BCE = self.criterion_BCE(outputs, targets)
        loss_Dice = self.criterion_Dice(outputs, targets)
        loss_comb = loss_Dice * self.weight_dice + \
                    loss_BCE * (1 - self.weight_dice)

        return loss_comb