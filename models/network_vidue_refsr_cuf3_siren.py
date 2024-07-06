import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.blocks as blocks
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

def make_coord(shape):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # v0, v1 = -1, 1

        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    ret = torch.stack(torch.meshgrid(coord_seqs), dim=-1)
    return ret

class DynAgg(ModulatedDeformConv2d):
    '''
    Use other features to generate offsets and masks.
    Intialized the offset with precomputed non-local offset.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 extra_offset_mask=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, deform_groups)
        self.extra_offset_mask = extra_offset_mask
        channels_ = self.deform_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset):
        '''
            Args:
                pre_offset: precomputed_offset. Size: [b, 2, h, w]
        '''
        # pre_offset

        if self.extra_offset_mask:
            # x = [input, features]
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset_ = torch.cat((o1, o2), dim=1)

        offset = offset_ + pre_offset.flip(1).repeat(1, offset_.size(1) // 2, 1, 1)

        # # repeat pre_offset along dim1, shape: [b, 9*groups, h, w, 2]
        # pre_offset = pre_offset.repeat([1, self.deform_groups, 1, 1, 1])
        # # the order of offset is [y, x, y, x, ..., y, x]
        # pre_offset_reorder = torch.zeros_like(offset)
        # # add pre_offset on y-axis
        # pre_offset_reorder[:, 0::2, :, :] = pre_offset[:, :, :, :, 1]
        # # add pre_offset on x-axis
        # pre_offset_reorder[:, 1::2, :, :] = pre_offset[:, :, :, :, 0]
        # offset = offset + pre_offset_reorder
        # print(offset.size())
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset_))
        if offset_mean > 100:
            print('Offset mean is {}, larger than 100.'.format(offset_mean))
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.deform_groups)


class MRAPAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.
    Args:
        nf (int): Number of the channels of middle features.
            Default: 64.
        ref_nf (int): Number of the channels of middle features.
            Default: 256.
    """

    def __init__(self,
                 nf=64,
                 ref_nf=256):
        super().__init__()

        # multi-ref attention (before fusion conv)
        self.patch_size = 3
        channels = ref_nf
        self.conv_emb1 = nn.Sequential(
            nn.Conv2d(nf, channels, 1),
            nn.PReLU())
        self.conv_emb2 = nn.Sequential(
            nn.Conv2d(ref_nf, channels,
                      self.patch_size, 1, self.patch_size // 2),
            nn.PReLU())
        self.conv_ass = nn.Conv2d(ref_nf, channels * 2,
                                  self.patch_size, 1, self.patch_size // 2)
        self.scale = channels ** -0.5
        self.feat_fusion = nn.Conv2d(
            nf + channels * 2, nf, 1)

        # spatial attention (after fusion conv)
        self.spatial_attn = nn.Conv2d(
            nf + channels * 2, channels * 2, 1)
        self.spatial_attn_mul1 = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=1)
        self.spatial_attn_mul2 = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=1)
        self.spatial_attn_add1 = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=1)
        self.spatial_attn_add2 = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def spatial_padding(self, feats):
        _, _, h, w = feats.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        feats = F.pad(feats, [0, pad_w, 0, pad_h], mode='reflect')
        return feats

    def forward(self, target, refs):
        n, _, h_input, w_input = target.size()
        t = len(refs)

        target = self.spatial_padding(target)
        refs = torch.stack(refs, dim=1).flatten(0, 1)
        refs = self.spatial_padding(refs)
        # multi-ref attention
        embedding_target = self.conv_emb1(target) * self.scale  # (n, c, h, w)
        embedding_target = embedding_target.permute(0, 2, 3, 1).contiguous().unsqueeze(3)  # (n, h, w, 1, c)
        embedding_target = embedding_target.contiguous().flatten(0, 2)  # (n*h*w, 1, c)
        emb = self.conv_emb2(refs).unflatten(0, (n, t))  # (n, t, c, h, w)
        emb = emb.permute(0, 3, 4, 2, 1).contiguous()  # (n, h, w, c, t)
        emb = emb.flatten(0, 2)  # (n*h*w, c, t)
        ass = self.conv_ass(refs).unflatten(0, (n, t))  # (n, t, c*2, h, w)
        ass = ass.permute(0, 3, 4, 1, 2).contiguous()  # (n, h, w, t, c*2)
        ass = ass.flatten(0, 2)  # (n*h*w, t, c*2)

        corr_prob = torch.matmul(embedding_target, emb)  # (n*h*w, 1, t)
        corr_prob = F.softmax(corr_prob, dim=2)
        refs = torch.matmul(corr_prob, ass).squeeze(1)  # (n*h*w, c*2)
        refs = refs.unflatten(0, (n, *target.shape[-2:]))  # (n, h, w, c*2)
        refs = refs.permute(0, 3, 1, 2).contiguous()  # (n, c*2, h, w)

        del embedding_target, emb, ass, corr_prob

        # spatial attention
        attn = self.lrelu(self.spatial_attn(torch.cat([target, refs], dim=1)))
        attn_mul = self.spatial_attn_mul2(self.lrelu(self.spatial_attn_mul1(attn)))
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn_mul = torch.sigmoid(attn_mul)

        # after initialization, * 2 makes (attn_mul * 2) to be close to 1.
        refs = refs * attn_mul * 2 + attn_add

        # fusion
        feat = self.lrelu(self.feat_fusion(torch.cat([target, refs], dim=1)))
        return feat[:, :, :h_input, :w_input]

class SynBlock(nn.Module):
    def __init__(self, nf, ks, groups=8):
        super(SynBlock, self).__init__()
        self.xN = 5
        self.offset_conv1 = nn.Conv2d(
            nf*2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dyn_agg = DynAgg(
            nf,
            nf,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=groups,
            extra_offset_mask=True)
        self.head = MRAPAFusion(nf=nf, ref_nf=nf)
        self.body = blocks.ResBlock(nf, nf, kernel_size=ks, stride=1, se=False)
        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, pre_offset_list, img_ref_feat_list, size):  #fea, flow_list, vec, out_size):  #

        swapped_feat_list = []
        for pre_offset, img_ref_feat in zip(pre_offset_list, img_ref_feat_list):
            _, _, H, W = pre_offset.shape
            TH, TW = size
            scale = int(TW / W)
            # print(scale, TW, W)
            pre_offset = F.interpolate(pre_offset, size=size, mode='bilinear') * scale
            offset = torch.cat([x, img_ref_feat], 1)
            offset = self.lrelu(self.offset_conv1(offset))
            offset = self.lrelu(self.offset_conv2(offset))
            swapped_feat = self.lrelu(
                self.dyn_agg([img_ref_feat, offset], pre_offset))
            swapped_feat_list.append(swapped_feat)

        h = self.head(x, swapped_feat_list)
        h = self.body(h) + x
        x = self.tail(h)
        return x
