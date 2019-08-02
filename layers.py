import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, channels=3, affine=True):
        super(InstanceNorm, self).__init__()
        self.affine = affine
        self.weight = torch.nn.Parameter(torch.ones(channels, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(channels, dtype=torch.float))
        self.cuda = False

    def forward(self, x):
        if not self.cuda and x.is_cuda:
            self.cuda = True
            self.weight = self.weight.cuda()
            self.bias = self.bias.cuda()

        mean = x.view(x.size(1), -1).mean(-1).view(1, -1, 1, 1)
        var = x.view(x.size(1), -1).var(-1).view(1, -1, 1, 1)

        x = (x - mean) / var
        x = (self.weight.view(1, -1, 1, 1) * x) + self.bias.view(1, -1, 1, 1)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, pad=1, dilate=1, group=1, leak=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, dilate, group)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.99)
        #self.lrelu = nn.LeakyReLU(leak)
        self.lrelu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class GlobalPool(nn.Module):
    def __init__(self):
        super(GlobalPool, self).__init__()
        #self.axis = axis
    def forward(self, x):
        out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
        return out


class Noise(nn.Module):
    def __init__(self, std=1):
        super(Noise, self).__init__()
        self.std = float(std)

    def forward(self, x):
        if self.training:
            #return x + torch.normal(means=torch.zeros(x.size()), std=self.std).cuda()#(0., self.std)
            if x.is_cuda:
                return x + torch.randn(x.size()).cuda()
            else:
                return x + torch.randn(x.size())
        else:
            return x
        # return x


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)