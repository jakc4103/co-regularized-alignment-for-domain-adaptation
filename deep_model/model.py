import torch.nn as nn
from collections import OrderedDict
from deep_model.layers import ConvBlock, Noise, GlobalPool, Reshape

def build_small_model(instance_norm=False, get_feat=False):
    if instance_norm:
        feature_extractor = OrderedDict([
            ('instance_norm', nn.InstanceNorm2d(3, affine=True))])
    else:
        feature_extractor = OrderedDict()
    
    feature_extractor.update(OrderedDict([
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
        ]))

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

    FE = Sequential(feature_extractor)
    CC = Sequential(category_classifier)
    DD = Sequential(domain_discriminator)

    net = SmallNet(FE, CC, DD, get_feat)

    return net


# sequantial module that support updata_stats param
class Sequential(nn.Sequential):
    def forward(self, input, update_stats=True):
        for module in self._modules.values():
            if hasattr(module, 'update_stats'):
                input = module(input, update_stats)
            else:
                input = module(input)
        return input


class SmallNet(nn.Module):
    def __init__(self, FE, CC, DD, get_feat=False):
        super(SmallNet, self).__init__()
        self.FE = FE
        self.CC = CC
        self.DD = DD
        self.get_feat = get_feat


    def forward(self, x, update_stats=False):
        feats = self.FE(x, update_stats)
        logits = self.CC(feats, update_stats)
        dom_logits = self.DD(feats)

        if not self.get_feat:
            return logits, dom_logits
        else:
            return logits, dom_logits, feats