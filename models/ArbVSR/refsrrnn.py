from models.ArbVSR.components import ResBlocks, D
from models.ArbVSR.pytorch_pwc.extract_flow import extract_flow_torch
from models.ArbVSR.pytorch_pwc.pwc import PWCNet
import torch
import torch.nn as nn
from models.ArbVSR.warp import warp
import torch.nn.functional as F
from models.cuf_siren import KerP, CUF
from models.network_vidue_refsr_cuf3_siren import SynBlock

class RefsrRNN(nn.Module):
    def __init__(self, img_channels=3, num_resblocks=15, num_channels=64, count=2):
        super(RefsrRNN, self).__init__()
        self.num_channels = num_channels
        self.pwcnet = PWCNet()
        self.forward_rnn = ResBlocks(input_channels=img_channels + num_channels, num_resblocks=num_resblocks, num_channels=num_channels)
        self.forward_rnn2 = ResBlocks(input_channels=img_channels + num_channels, num_resblocks=num_resblocks, num_channels=num_channels)
        self.d = D(in_channels=num_channels * 2, mid_channels=num_channels * 2, out_channels=num_channels)
        self.kernel_predict = KerP(num_channels)
        self.upsample = CUF(num_channels)
        self.predict = SynBlock(num_channels, ks=3)
        self.count = count
        for param in self.pwcnet.parameters():
            param.requires_grad = False

    # def trainable_parameters(self):
    #     return [{'params':self.forward_rnn.parameters()}, {'params':self.d.parameters()}, {'params':self.kernel_predict.parameters()}, {'params':self.upsample.parameters()}]

    def forward(self, images, scale, coord, cell, isTrain=True):
        seqn_not_pad = torch.stack(images, dim=1)
        N, T, C, H, W = seqn_not_pad.shape
        seqn_not_pad_ = F.interpolate(seqn_not_pad.view(-1, C, H, W).contiguous(), size=(coord.shape[-3:-1]),
                                      mode='bicubic')
        seqn_not_pad_ = seqn_not_pad_.view(N, T, C, coord.shape[-3], coord.shape[-2]).contiguous()
        seqdn = torch.empty_like(seqn_not_pad_)
        del seqn_not_pad_

        # border_queue = []
        # size_h, size_w = int(H * self.border_ratio), int(W * self.border_ratio)

        # reflect pad seqn and noise_level_map
        seqn = torch.empty((N, T + self.count, C, H, W), device=seqn_not_pad.device)
        seqn[:, :T] = seqn_not_pad
        for i in range(self.count):
            seqn[:, T + i] = seqn_not_pad[:, T - 2 - i]

        # flatten_map = generate_alpha(seqn.reshape(-1, C, H, W).contiguous()*255.).view(N, T+self.count*2, C, H, W).contiguous()
        # flatten_map = self.adists(seqn.reshape(-1, C, H, W).contiguous()).view(N, T+self.count, self.num_channels, H, W).contiguous()

        inp1, inp2 = seqn[:, 1:].reshape(-1, C, H, W).contiguous(), seqn[:, :-1].reshape(-1, C, H, W).contiguous()
        flow = extract_flow_torch(self.pwcnet, inp1, inp2)
        flow_queue = flow.view(N, T + self.count - 1, 2, H, W).contiguous()

        hidden_list = []
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        h_ = self.forward_rnn(torch.cat((seqn[:, 0], init_forward_h), dim=1))
        hidden_list.append(h_)
        arb_up_kernel = self.kernel_predict(h_, coord, cell)
        for j in range(1, self.count):
            h_, _ = warp(h_, flow_queue[:, j - 1])
            h_ = self.forward_rnn(torch.cat((seqn[:, j], h_), dim=1))
            hidden_list.append(h_)

        for i in range(self.count, T + self.count):

            h_, _ = warp(hidden_list[-1], flow_queue[:, i - 1])
            h_ = self.forward_rnn(torch.cat((seqn[:, i], h_), dim=1))
            hidden_list.append(h_)
            if i > self.count:
                hidden_list.pop(0)

            assert len(hidden_list) == self.count + 1

            refsr_flow_list = []
            for j in range(1, self.count + 1):
                refsr_flow_list.append(
                    extract_flow_torch(self.pwcnet, seqn[:, i - self.count], seqn[:, i - self.count + j]))

            assert len(refsr_flow_list) == self.count

            hidden = hidden_list[0]
            fusion_h = self.predict(hidden, refsr_flow_list, hidden_list[1:], hidden.size()[-2:])

            h = self.forward_rnn2(torch.cat((seqn[:, i - self.count], fusion_h), dim=1))
            res = self.d(torch.cat([h, hidden], dim=1))
            seqdn[:, i - self.count] = self.upsample(res, arb_up_kernel, coord, seqn[:, i - self.count])

        return seqdn

    def test_forward(self, images, arb_up_kernel, coord):
        seqn_not_pad = torch.stack(images, dim=1)
        N, T, C, H, W = seqn_not_pad.shape
        seqn_not_pad_ = F.interpolate(seqn_not_pad.view(-1, C, H, W).contiguous(), size=(coord.shape[-3:-1]),
                                      mode='bicubic')
        seqn_not_pad_ = seqn_not_pad_.view(N, T, C, coord.shape[-3], coord.shape[-2]).contiguous()
        seqdn = torch.empty_like(seqn_not_pad_)
        del seqn_not_pad_

        # reflect pad seqn and noise_level_map
        seqn = torch.empty((N, T + self.count, C, H, W), device=seqn_not_pad.device)
        seqn[:, :T] = seqn_not_pad
        for i in range(self.count):
            seqn[:, T + i] = seqn_not_pad[:, T - 2 - i]


        inp1, inp2 = seqn[:, 1:].reshape(-1, C, H, W).contiguous(), seqn[:, :-1].reshape(-1, C, H, W).contiguous()
        flow = extract_flow_torch(self.pwcnet, inp1, inp2)
        flow_queue = flow.view(N, T + self.count - 1, 2, H, W).contiguous()

        hidden_list = []
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        h_ = self.forward_rnn(torch.cat((seqn[:, 0], init_forward_h), dim=1))
        hidden_list.append(h_)
        for j in range(1, self.count):
            h_, _ = warp(h_, flow_queue[:, j - 1])
            h_ = self.forward_rnn(torch.cat((seqn[:, j], h_), dim=1))
            hidden_list.append(h_)

        for i in range(self.count, T + self.count):
            h_, _ = warp(hidden_list[-1], flow_queue[:, i - 1])
            h_ = self.forward_rnn(torch.cat((seqn[:, i], h_), dim=1))
            hidden_list.append(h_)
            if i > self.count:
                hidden_list.pop(0)

            assert len(hidden_list) == self.count + 1

            refsr_flow_list = []
            for j in range(1, self.count + 1):
                refsr_flow_list.append(
                    extract_flow_torch(self.pwcnet, seqn[:, i - self.count], seqn[:, i - self.count + j]))

            assert len(refsr_flow_list) == self.count

            hidden = hidden_list[0]
            fusion_h = self.predict(hidden, refsr_flow_list, hidden_list[1:], hidden.size()[-2:])

            h = self.forward_rnn2(torch.cat((seqn[:, i - self.count], fusion_h), dim=1))
            res = self.d(torch.cat([h, hidden], dim=1))
            seqdn[:, i - self.count] = self.upsample(res, arb_up_kernel, coord, seqn[:, i - self.count])

        return seqdn
