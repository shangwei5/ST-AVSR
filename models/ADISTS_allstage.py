import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
from torch.nn.functional import normalize
import torch.nn.functional as F
import math
from PIL import Image
import argparse
import cv2

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

        # self.chns = [3, 64]  #[3, 64, 128, 256, 512, 512]
        # self.windows = nn.ParameterList()
        # self.window_size = window_size
        # for k in range(len(self.chns)):
        #     self.windows.append(self.create_window(self.window_size, self.window_size / 3, self.chns[k]))

    def forward_once(self, x):
        h_list = []
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        _, _, H, W = h.size()
        h_list.append(h)
        h = self.stage2(h)
        h_list.append(F.interpolate(h, (H, W), mode="bilinear"))
        h = self.stage3(h)
        h_list.append(F.interpolate(h, (H, W), mode="bilinear"))
        h = self.stage4(h)
        h_list.append(F.interpolate(h, (H, W), mode="bilinear"))
        h = self.stage5(h)
        h_list.append(F.interpolate(h, (H, W), mode="bilinear"))

        return torch.cat(h_list, dim=1) #B, 64+12+256+512+512, H, W

    def std_(self, img, window_size=3):
        assert window_size % 2 == 1
        pad = window_size // 2

        # calculate std on the mean image of the color channels
        # img = torch.mean(img, dim=1, keepdim=True)
        N, C, H, W = img.shape
        img = nn.functional.pad(img, [pad] * 4, mode='reflect')
        img = nn.functional.unfold(img, kernel_size=window_size)
        img = img.view(N, C, window_size * window_size, H, W)
        img = img - torch.mean(img, dim=2, keepdim=True)
        # print(img.shape)
        img = img * img
        img = torch.mean(img, dim=2, keepdim=True)
        # print(img.shape)
        img = torch.sqrt(img)
        # print(img.shape)
        img = img.squeeze(2)
        return img

    def generate_alpha(self,input, lower=1, upper=5):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, C, H, W)) * 0.5
        input_std = self.std_(input)
        ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
        ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
        ratio = ratio  #.detach()

        return ratio

    def forward(self, x):

        # with torch.no_grad():
        feats_x = self.forward_once(x)

        # print(feats_x.shape)
        # feats_x = self.generate_alpha(feats_x*255)
        # pad = nn.ReflectionPad2d(0)
        # ps_x = self.compute_prob(feats_x)
        # # print(ps_x[1].shape)
        # weight = []
        # for k in range(0, len(self.chns)):
        #     weight.append(self.entropy(feats_x[k]))
        # weight = torch.concat(weight, dim=1)
        # # print(weight.shape)
        #
        # weight = weight / weight.sum(dim=(1, 2), keepdim=True)
        # weight_mean = weight.mean(dim=(1, 2), keepdim=True)
        # weight_std = torch.sqrt(((weight - weight_mean) ** 2).mean(dim=(1, 2), keepdim=True))
        # weight = weight.clamp(min=weight_mean - 0.5 * weight_std, max=weight_mean + 0.5 * weight_std)
        # weight = weight / weight.sum(dim=(1, 2), keepdim=True)
        # weight_list = torch.split(weight, self.chns, dim=1)
        # # print(weight_list[1].shape)
        #
        # # for k in range(len(self.chns) - 1, -1, -1):
        # feat_x = F.normalize(feats_x[1], dim=(2, 3))
        # try:
        #     x_mean = F.conv2d(pad(feat_x), self.windows[1], stride=1, padding=0, groups=self.chns[1])
        #     # y_mean = F.conv2d(pad(feat_y), self.windows[k], stride=1, padding=0, groups=self.chns[k])
        #     x_var = F.conv2d(pad(feat_x ** 2), self.windows[1], stride=1, padding=0,
        #                      groups=self.chns[1]) - x_mean ** 2
        #     # y_var = F.conv2d(pad(feat_y ** 2), self.windows[k], stride=1, padding=0,
        #     #                  groups=self.chns[k]) - y_mean ** 2
        #     xy_cov = F.conv2d(pad(feat_x * feat_x), self.windows[1], stride=1, padding=0, groups=self.chns[1]) \
        #              - x_mean * x_mean
        # except:
        #     x_mean = feat_x.mean([2, 3], keepdim=True)
        #     # y_mean = feat_y.mean([2, 3], keepdim=True)
        #     x_var = ((feat_x - x_mean) ** 2).mean([2, 3], keepdim=True)
        #     # y_var = ((feat_y - y_mean) ** 2).mean([2, 3], keepdim=True)
        #     xy_cov = (feat_x * feat_x).mean([2, 3], keepdim=True) - x_mean * x_mean
        #
        # T = (2 * x_mean * x_mean + 1e-6) / (x_mean ** 2 + x_mean ** 2 + 1e-6)
        # S = (2 * xy_cov + 1e-6) / (x_var + x_var + 1e-6)
        # S = F.interpolate(S, size=(ps_x[1].shape[-2], ps_x[1].shape[-1]), mode='bilinear', align_corners=True)
        # T = F.interpolate(T, size=(ps_x[1].shape[-2], ps_x[1].shape[-1]), mode='bilinear', align_corners=True)
        # # print(S.shape, T.shape, ps_x[1].shape)
        # ps = ps_x[1]  #.expand(x_mean.shape[0], x_mean.shape[1], -1, -1)
        # pt = 1 - ps
        # D_map = (pt * T + ps * S) * weight_list[1].unsqueeze(3)
        # print(D_map.mean([2, 3]))
        return feats_x

# def prepare_image(image, resize=True):
#     if resize and min(image.size) > 256:
#         image = transforms.functional.resize(image, 256)
#     image = transforms.ToTensor()(image)
#     return image.unsqueeze(0)
#
#
# if __name__ == '__main__':
#     from PIL import Image
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ref', type=str, default='/data1/shangwei/dataset/video/REDS/val/val_sharp_bicubic/X4/002/00000002.png')
#     parser.add_argument('--dist', type=str, default='/data1/shangwei/dataset/video/REDS/val/val_sharp_bicubic/X4/002/00000003.png')
#     args = parser.parse_args()
#
#     ref = prepare_image(Image.open(args.ref).convert("RGB"), resize=False)
#     dist = prepare_image(Image.open(args.dist).convert("RGB"), resize=False)
#     assert ref.shape == dist.shape
#     # print(ref.shape)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ADISTS().to(device)
#     ref = ref.to(device)
#     dist = dist.to(device)
#     map = model(ref).mean(1)
#     # print(score.item())
#     print(map.shape)
#     # print(map.min(), map.max())
#     save = map.cpu().numpy().squeeze() * 255.
#     save = save.astype(np.uint8)
#     cv2.imwrite("/data1/shangwei/dataset/video/REDS/val/002_00002_lr_ratio_w21_prob_flat255.png", save)

