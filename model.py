import torch.nn as nn
from collections import OrderedDict
from layers import ConvBlock, Noise, GlobalPool, Reshape, InstanceNorm

def build_small_model(instance_norm=False, get_feat=False):
    if instance_norm:
        feature_extractor = OrderedDict([
            ('instance_norm', InstanceNorm(3)),
            ('ConvBlock_1', ConvBlock(3, 64)),
            ('ConvBlock_2', ConvBlock(64, 64)),
            ('ConvBlock_3', ConvBlock(64, 64)),
            ('max_pool_1', nn.MaxPool2d(2, 2)),
            ('drop_1', nn.Dropout2d(p=0.5)),
            ('Noise_1', Noise()),
            ('ConvBlock_4', ConvBlock(64, 64)),
            ('ConvBlock_5', ConvBlock(64, 64)),
            ('ConvBlock_6', ConvBlock(64, 64)),
            ('max_pool_2', nn.MaxPool2d(2, 2)),
            ('drop_2', nn.Dropout2d(p=0.5)),
            ('Noise_2', Noise())
        ])
    else:
        feature_extractor = OrderedDict([
            ('ConvBlock_1', ConvBlock(3, 64)),
            ('ConvBlock_2', ConvBlock(64, 64)),
            ('ConvBlock_3', ConvBlock(64, 64)),
            ('max_pool_1', nn.MaxPool2d(2, 2)),
            ('drop_1', nn.Dropout2d(p=0.5)),
            ('Noise_1', Noise()),
            ('ConvBlock_4', ConvBlock(64, 64)),
            ('ConvBlock_5', ConvBlock(64, 64)),
            ('ConvBlock_6', ConvBlock(64, 64)),
            ('max_pool_2', nn.MaxPool2d(2, 2)),
            ('drop_2', nn.Dropout2d(p=0.5)),
            ('Noise_2', Noise())
        ])

    category_classifier = OrderedDict([
        ('ConvBlock_7', ConvBlock(64, 64)),
        ('ConvBlock_8', ConvBlock(64, 64)),
        ('ConvBlock_9', ConvBlock(64, 64)),
        ('GlobalPool', GlobalPool()),
        ('classifier', nn.Linear(64, 10))
    ])

    domain_discriminator = OrderedDict([
        ('reshape', Reshape()),
        ('linear_1', nn.Linear(8*8*64, 1000)),
        ('relu', nn.ReLU()),
        ('discriminator', nn.Linear(1000, 1))
    ])

    FE = nn.Sequential(feature_extractor)
    CC = nn.Sequential(category_classifier)
    DD = nn.Sequential(domain_discriminator)

    net = SmallNet(FE, CC, DD, get_feat)

    return net


class SmallNet(nn.Module):
    def __init__(self, FE, CC, DD, get_feat=False):
        super(SmallNet, self).__init__()
        self.FE = FE
        self.CC = CC
        self.DD = DD
        self.get_feat = get_feat


    def forward(self, x):
        feats = self.FE(x)
        logits = self.CC(feats)
        dom_logits = self.DD(feats)

        if not self.get_feat:
            return logits, dom_logits
        else:
            return logits, dom_logits, feats