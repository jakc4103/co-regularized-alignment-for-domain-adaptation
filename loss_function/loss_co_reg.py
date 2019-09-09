import torch
import torch.nn as nn

def diverse_loss(feat1, feat2, margin):
    loss = torch.sqrt(torch.sum((torch.mean(feat1 - feat2, 0)) ** 2))
    if loss > margin:
        return 0
    return loss


def agreement_loss(prob1, prob2):
    #return torch.abs(torch.mean(torch.sum(prob1 - prob2, -1)))
    return torch.mean(torch.sum(torch.abs(prob1 - prob2), -1))


def cross_entropy_loss(logits):
    prob = nn.functional.softmax(logits, 1)
    return torch.mean(torch.sum(- prob * torch.log(prob), 1))