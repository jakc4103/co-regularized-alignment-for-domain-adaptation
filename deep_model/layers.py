import torch
import torch.nn as nn
import torch.nn.functional as F

class TrgBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input, update_stats=True):
        self._check_input_dim(input)

        if self.momentum is None or not update_stats:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats and update_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None and update_stats:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                elif update_stats:  # use exponential moving average
                    exponential_average_factor = self.momentum
                else:
                    exponential_average_factor = 0.0

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, pad=1, dilate=1, group=1, leak=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, dilate, group)
        self.bn = TrgBatchNorm2d(out_ch, momentum=0.99)
        self.lrelu = nn.LeakyReLU(leak)
        self.update_stats = True
        #self.lrelu = nn.ReLU()

    def forward(self, x, update_stats=True):
        x = self.conv(x)
        x = self.bn(x, update_stats)
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