import torch

def diverse_loss(feat1, feat2, margin):
    loss = torch.sqrt(torch.sum((torch.mean(feat1 - feat2, 0)) ** 2))
    if loss > margin:
        return 0
    return loss


def agreement_loss(prob1, prob2):
    return torch.abs(torch.mean(torch.sum(prob1 - prob2, -1)))