import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
from torch.nn.functional import normalize
import torch.nn.functional as F
import math

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Downsample(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        # print (g)

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class ADISTS(torch.nn.Module):
    def __init__(self, window_size=21):
        super(ADISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), Downsample(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), Downsample(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), Downsample(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), Downsample(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.windows = nn.ParameterList()
        self.window_size = window_size
        for k in range(len(self.chns)):
            self.windows.append(self.create_window(self.window_size, self.window_size / 3, self.chns[k]))

    def compute_prob(self, x, k):

        theta = [[0, 0],
                 [1.0, 0.29],
                 [2.0, 0.52],
                 [2.95, 0.56],
                 [0.97, 0.25],
                 [0.21, 0.10]]

        ps = 1 / (1 + torch.exp(-(x - theta[k][0]) / theta[k][1]))
        pt = 1 - ps
        return ps, pt

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, window_sigma, channel):
        _1D_window = self.gaussian(window_size, window_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return nn.Parameter(window, requires_grad=False)

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        if len(self.chns) == 6:
            h = self.stage5(h)
            h_relu5_3 = h
            outs = [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        else:
            outs = [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
        return outs

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            feats_x = self.forward_once(x)
            feats_y = self.forward_once(y)
        else:
            with torch.no_grad():
                feats_x = self.forward_once(x)
                feats_y = self.forward_once(y)

        D = 0
        c1 = 1e-6
        c2 = 1e-6
        pad = nn.ReflectionPad2d(0)
        for k in range(len(self.chns) - 1, -1, -1):
            try:
                x_mean = F.conv2d(pad(feats_x[k]), self.windows[k], stride=1, padding=0, groups=self.chns[k])
                y_mean = F.conv2d(pad(feats_y[k]), self.windows[k], stride=1, padding=0, groups=self.chns[k])
                x_var = F.conv2d(pad(feats_x[k] ** 2), self.windows[k], stride=1, padding=0,
                                 groups=self.chns[k]) - x_mean ** 2
                y_var = F.conv2d(pad(feats_y[k] ** 2), self.windows[k], stride=1, padding=0,
                                 groups=self.chns[k]) - y_mean ** 2
                xy_cov = F.conv2d(pad(feats_x[k] * feats_y[k]), self.windows[k], stride=1, padding=0,
                                  groups=self.chns[k]) - x_mean * y_mean
            except:
                x_mean = feats_x[k].mean([2, 3], keepdim=True)
                y_mean = feats_y[k].mean([2, 3], keepdim=True)
                x_var = ((feats_x[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
                y_var = ((feats_y[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
                xy_cov = (feats_x[k] * feats_y[k]).mean([2, 3], keepdim=True) - x_mean * y_mean

            T = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            S = (2 * xy_cov + c2) / (x_var + y_var + c2)

            if k > 0:
                ratio = torch.mean(x_var / (x_mean + 1e-12), dim=1, keepdim=True)
                ps, pt = self.compute_prob(ratio, k)

            D_map = pt * T + ps * S
            # D = D + D_map.mean([2,3]).mean(1)/len(self.chns)
            D = D + D_map.mean([2, 3]).sum(1) / sum(self.chns)

        if as_loss:
            return 1 - D.mean()
        else:
            return 1 - D


def prepare_image(image, resize=False):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


# if __name__ == '__main__':
#     from PIL import Image
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ref', type=str, default='./images/gt.png')
#     parser.add_argument('--dist', type=str, default='./images/shan08.png')
#     args = parser.parse_args()
#
#     ref = prepare_image(Image.open(args.ref).convert("RGB"))
#     dist = prepare_image(Image.open(args.dist).convert("RGB"))
#     assert ref.shape == dist.shape
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ADISTS().to(device)
#     ref = ref.to(device)
#     dist = dist.to(device)
#     score = model(ref, dist, as_loss=False)
#     print(score.item())
#     # score: 0.662