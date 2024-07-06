import functools
from models.ArbVSR.init import init_fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.conv2(self.relu(self.conv1(x)))
        output = torch.add(output, x)
        return output

class ResBlocks(nn.Module):
    def __init__(self, input_channels, num_resblocks, num_channels):
        super(ResBlocks, self).__init__()
        self.input_channels = input_channels
        self.first_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

        modules = []
        for _ in range(num_resblocks):
            modules.append(ResBlock(in_channels=num_channels, mid_channels=num_channels, out_channels=num_channels))
        self.resblocks = nn.Sequential(*modules)

        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

    def forward(self, h):
        shallow_feature = self.first_conv(h)
        new_h = self.resblocks(shallow_feature)
        return new_h

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, style_ch=82, demod=True, stride=1, padding=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        self.to_style = nn.Linear(style_ch, in_chan)
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    # def _get_same_padding(self, size, kernel, dilation, stride):
    #     return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, scale):
        b, c, h, w = x.shape

        y = self.to_style(scale)
        w1 = y[:, None, :, None, None]
        # print('w1--', w1.shape)
        w2 = self.weight[None, :, :, :, :]
        # print('w2--',w2.shape)
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        # padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        # print(x.shape)

        x = x.reshape(-1, self.filters, h, w)
        return x

class ResBlocks_mod(nn.Module):
    def __init__(self, input_channels, num_resblocks, num_channels):
        super(ResBlocks_mod, self).__init__()
        self.input_channels = input_channels
        self.first_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.num_resblocks = num_resblocks
        modules = []
        for _ in range(num_resblocks):
            modules.append(ResBlock(in_channels=num_channels, mid_channels=num_channels, out_channels=num_channels))
        self.resblocks = nn.Sequential(*modules)
        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

        modc = []
        for _ in range(num_resblocks // 5):
            modc.append(Conv2DMod(num_channels, num_channels, 3))
        self.mbconv = nn.Sequential(*modc)

    def forward(self, h, scale):
        new_h = self.first_conv(h)
        for i in range(self.num_resblocks//5):
            new_h = self.mbconv[i](new_h, scale)
            new_h = self.resblocks[i*5:(i+1)*5](new_h)
        return new_h

class D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(D, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        self.convs = nn.Sequential(*layers)

        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

    def forward(self, x):
        x = self.convs(x)
        return x
