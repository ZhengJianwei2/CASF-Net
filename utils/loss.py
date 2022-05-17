import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
# from opt import opt

"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""BCE + DICE Loss + IoULoss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        #self.bce = BCELoss(weight, size_average)
        self.fl = FocalLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, pred, target):
        fcloss = self.fl(pred, target)
        diceloss = self.dice(pred, target)
        iouloss = self.iou(pred, target)
        loss = fcloss + diceloss + iouloss     #Use the obmination of loss

        return loss


""" Deep Supervision Loss"""


def DeepSupervisionLoss(pred, gt):
    d0, d1, d2 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0, gt)
    # gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    loss1 = criterion(d1, gt)
    # gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    loss2 = criterion(d2, gt)
    # gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    # loss3 = criterion(d3, gt)
    # gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    # loss4 = criterion(d4, gt)

    return 0.2*loss0+0.2*loss1+0.6*loss2


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=4,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        # print("input, target",input.dtype, target.dtype)
        ce_loss =  F.binary_cross_entropy_with_logits(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU




# Code from PraNet

class IoULossW(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULossW, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        pred = torch.sigmoid(inputs)
        weit = 1 + 5*torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
        inter = ((pred * targets)* weit).sum(dim=(2, 3))
        #print(inter.shape)
        union = ((pred + targets)* weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        wiou = wiou.mean()*1e-5
        return wiou
