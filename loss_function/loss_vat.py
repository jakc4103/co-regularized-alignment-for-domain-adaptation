"""!
Script source:
https://github.com/lyakaap/VAT-pytorch

"""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def cross_entropy(trg_pred, pred):
    return torch.sum(- trg_pred * torch.log(pred)) / trg_pred.size(0)


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1, src_weight=1.0, trg_weight=1e-2):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.src_weight = src_weight
        self.trg_weight = trg_weight


    def forward(self, model, x):
        model.zero_grad()
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        #d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.normal(mean=torch.zeros_like(x), std=1).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat, _, _ = model(x + self.xi * d)
                #logp_hat = F.log_softmax(pred_hat, dim=1)
                #adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = cross_entropy(pred, p_hat)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat, _, _ = model(x + r_adv)
            # logp_hat = F.log_softmax(pred_hat, dim=1)
            # lds = F.kl_div(logp_hat, pred, reduction='batchmean')
            p_hat = F.softmax(pred_hat, dim=1)
            lds = cross_entropy(pred, p_hat)

        return lds, x + r_adv
