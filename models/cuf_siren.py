import numpy as np

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from models.SIREN import Siren


import models
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        coord_x = -1+(2*i+1)/W
        coord_y = -1+(2*i+1)/H
        normalize to (-1, 1)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class PositionEncoder(nn.Module):
    def __init__(
        self,
        posenc_type='sinusoid',
        complex_transform=False,
        posenc_scale=10,
        gauss_scale=1,
        in_dims=2,
        enc_dims=64,
        hidden_dims=64,
        out_dims=32,
        head=8,
        gamma=1
    ):
        super().__init__()

        self.posenc_type = posenc_type
        self.complex_transform = complex_transform
        self.posenc_scale = posenc_scale
        self.gauss_scale = gauss_scale
        self.out_dims = out_dims
        self.in_dims = in_dims
        self.enc_dims = enc_dims
        self.hidden_dims = hidden_dims
        self.head = head
        self.gamma = gamma

        self.define_parameter()

    def define_parameter(self):
        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            self.b_vals = 2.**torch.linspace(
                0, self.posenc_scale, self.enc_dims // 4
            ) - 1  # -1 -> (2 * pi)
            self.b_vals = torch.stack([self.b_vals, torch.zeros_like(self.b_vals)], dim=-1)
            self.b_vals = torch.cat([self.b_vals, torch.roll(self.b_vals, 1, -1)], dim=0)
            self.a_vals = torch.ones(self.b_vals.shape[0])
            self.proj = nn.Linear(self.enc_dims, self.out_dims) #self.head)

        elif self.posenc_type == 'learn':
            self.Wr = nn.Linear(self.in_dims, self.hidden_dims // 2, bias=False)
            self.mlp = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(nn.GELU(), nn.Linear(self.enc_dims, self.head))
            self.init_weight()

        elif self.posenc_type == 'dpb':
            self.mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dims),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.enc_dims, self.head)
            )

    def init_weight(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, positions, cells=None):

        if self.posenc_type is None:
            return positions

        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            self.b_vals = self.b_vals.to(positions.device) #.cuda()
            self.a_vals = self.a_vals.to(positions.device) #.cuda()

            # b, q, 1, c (x -> c/2, y -> c/2)
            sin_part = self.a_vals * torch.sin(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            )
            cos_part = self.a_vals * torch.cos(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            )

            if self.posenc_type == 'ipe':
                # b, q, 2
                cell = cells.clone()
                cell_part = torch.sinc(
                    torch.matmul((1 / np.pi * cell), self.b_vals.transpose(-2, -1))
                )

                sin_part = sin_part * cell_part
                cos_part = cos_part * cell_part

            if self.complex_transform:
                pos_enocoding = torch.view_as_complex(torch.stack([cos_part, sin_part], dim=-1))
            else:
                pos_enocoding = torch.cat([sin_part, cos_part], dim=-1)
                # pos_bias = self.proj(pos_enocoding)

        elif self.posenc_type == 'learn':
            projected_pos = self.Wr(positions)

            sin_part = torch.sin(projected_pos)
            cos_part = torch.cos(projected_pos)

            if self.complex_transform:
                pos_enocoding = 1 / np.sqrt(self.hidden_dims) * torch.view_as_complex(
                    torch.stack([cos_part, sin_part], dim=-1)
                )
            else:
                pos_enocoding = 1 / np.sqrt(self.hidden_dims
                                           ) * torch.cat([sin_part, cos_part], dim=-1)
                pos_enocoding = self.mlp(pos_enocoding)

        elif self.posenc_type == 'dpb':
            pos_enocoding = self.mlp(positions)

        pos_bias = None if self.complex_transform else self.proj(pos_enocoding)

        return pos_enocoding, pos_bias

class KerP(nn.Module):
    def __init__(
        self,
        feat_dim,
        sq_factor=4,
        r=1,
    ):
        super().__init__()

        # self.dim = base_dim
        self.r = r
        self.pb_encoder1 = PositionEncoder()
        self.pb_encoder2 = PositionEncoder()
        self.pb_encoder3 = PositionEncoder()
        self.r_area = (2 * self.r + 1)**2
        imnet_in_dim = 32 * 3
        self.imnets = Siren(in_features=imnet_in_dim, out_features=feat_dim, hidden_features=[feat_dim//sq_factor, feat_dim//sq_factor, feat_dim//sq_factor],
                                  hidden_layers=2, outermost_linear=True)   #MLP(imnet_in_dim, feat_dim)

    def forward(self, feat, sample_coord, cell):
        bs = feat.shape[0]
        r = self.r
        coord_lr = make_coord(feat.shape[-2:], flatten=False).to(sample_coord.device).permute(2, 0, 1).contiguous(). \
            unsqueeze(0).expand(bs, 2, *feat.shape[-2:])

        # b, 2, h_in, w_in -> b, 2, h_t, w_t -> b, h_t, w_t, 2  -> b, q, 2
        sample_coord_k = F.grid_sample(
            coord_lr, sample_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1).contiguous().view(bs, -1, 2).contiguous()

        rel_coord = sample_coord.view(bs, -1, 2).contiguous() - sample_coord_k
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        dh = torch.linspace(-r, r, 2 * r + 1).to(sample_coord.device)  # * rh
        dw = torch.linspace(-r, r, 2 * r + 1).to(sample_coord.device)  # * rw
        # 1, r_area, 2
        delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, -1, 2).repeat(bs, 1, 1).contiguous()

        # b, 2 -> b, q, 2
        rel_cell = cell.unsqueeze(1).repeat(1, sample_coord_k.shape[1], 1)
        rel_cell[..., 0] *= feat.shape[-2]
        rel_cell[..., 1] *= feat.shape[-1]

        _, pb1 = self.pb_encoder1(rel_cell)  # b, q, 32
        _, pb2 = self.pb_encoder2(delta)  # b, r_area, 32
        _, pb3 = self.pb_encoder3(rel_coord)  # b, q, 32
        # print(pb1.shape, pb2.shape,pb3.shape)
        pb1 = pb1.unsqueeze(1).repeat(1, self.r_area, 1, 1).view(-1, sample_coord_k.shape[1], 32)
        pb2 = pb2.unsqueeze(2).repeat(1, 1, sample_coord_k.shape[1], 1).view(-1, sample_coord_k.shape[1],
                                                                             32)  # b*r_area, q, 32
        pb3 = pb3.unsqueeze(1).repeat(1, self.r_area, 1, 1).view(-1, sample_coord_k.shape[1], 32)
        # print(pb1.shape, pb2.shape, pb3.shape)
        feat_in = torch.cat([pb1, pb2, pb3], dim=-1)

        pred = self.imnets(feat_in)  # b*r_area, q, feat_dim
        pred = pred.view(bs, self.r_area, sample_coord_k.shape[1], feat.shape[1]).contiguous().permute(0, 3, 1,
                                                                                                       2).contiguous()
        kernel = pred.view(bs, feat.shape[1] * self.r_area, -1).contiguous()
        return kernel

class CUF(nn.Module):
    def __init__(
        self,
        feat_dim,
        r=1,
    ):
        super().__init__()

        self.r = r
        self.r_area = (2 * self.r + 1)**2
        self.unfold = torch.nn.Unfold((2 * r + 1, 2 * r + 1), 1, 1)
        self.imnets = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(feat_dim, 3, 3, 1, 1)
        )

    def forward(self, feat, kernel, sample_coord, inp):
        '''
        feat: B, C, H_in, W_in
        smaple_coord: B, H_target, W_target, 2
        cell: B, 2
        inp:  B, C, H_in, W_in
        '''

        bs, H_t, W_t, _ = sample_coord.shape

        feat_unf = self.unfold(feat)  # B, C*r_area, H_in*W_in
        feat_unf = F.grid_sample(feat_unf.view(bs, -1, feat.shape[-2], feat.shape[-1]).contiguous(), sample_coord.flip(-1), mode='bilinear', align_corners=False).view(bs,feat.shape[1]*self.r_area,-1).contiguous()

        pred = feat_unf * kernel  # B, C*r_area, H_target * W_target

        Fold = torch.nn.Fold(sample_coord.shape[1:3], (2 * self.r + 1, 2 * self.r + 1), 1, 1)
        pred_ = Fold(pred)
        pred = self.imnets(pred_)

        pred_f = pred + F.grid_sample(inp, sample_coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
        return pred_f

