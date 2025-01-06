from torch import nn
from basicts.losses import masked_mae
import torch

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def megacrn_loss(prediction, target,query1, pos1, neg1, query2, pos2, neg2, null_val):
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
    # criterion = masked_mae_loss

    loss1 = masked_mae_loss(prediction, target)
    # lossb1 = separate_loss(query, pos.detach(), neg.detach())
    # lossb2 = compact_loss(query, pos.detach())
    lossc1 = separate_loss(query1, pos1.detach(), neg1.detach())
    lossc2 = compact_loss(query1, pos1.detach())
    lossd1 = separate_loss(query2, pos2.detach(), neg2.detach())
    lossd2 = compact_loss(query2, pos2.detach())
    loss = loss1 + 0.01 * (lossc1 + lossd1) + 0.01*(lossc2 + lossd2)

    return loss
