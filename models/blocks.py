import torch.nn as nn
import torch
import torch.nn.functional as F
###############################
# common
###############################
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # self.fusion = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, eve):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #b, c, h*w/4
        g_x = g_x.permute(0, 2, 1)

        theta_eve = self.theta(eve).view(batch_size, self.inter_channels, -1)   #b, c, h*w
        theta_eve = theta_eve.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)   #b, c, h*w/4
        f = torch.matmul(theta_eve, phi_x)   # (b, h*w, c)  X  (b, c, h*w/4)   =  (b, h*w, h*w/4)
        N = f.size(-1)
        f_div_C = f / N
        f_div_C = self.softmax(f_div_C)

        y = torch.matmul(f_div_C, g_x)    # (b, h*w, h*w/4) X (b, h*w/4, c) = (b, h*w, c)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])   # b,c,h,w
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.se = SEBlock(oup_dim, 4)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, pair=None):
        if pair is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            h, c = pair
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)

        h = self.relu(self.se(h))
        return h, [h, c]

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

###############################
# ResNet
###############################
# class ResBlock(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bn=False, se=True):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
#                                padding=get_same_padding(kernel_size, dilation), dilation=dilation)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
#                                padding=get_same_padding(kernel_size, dilation), dilation=dilation)
#         self.relu = nn.LeakyReLU(0.1, inplace=True)
#         if se:
#             self.se = SEBlock(planes, 4)
#         if bn:
#             self.bn1 = nn.InstanceNorm2d(planes)
#             self.bn2 = nn.InstanceNorm2d(planes)
#         self.se_ = se
#         self.bn_ = bn
#         self.res_translate = None
#         if not inplanes == planes or not stride == 1:
#             self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
#
#     def forward(self, x):
#         residual = x
#         if self.bn_:
#             out = self.relu(self.bn1(self.conv1(x)))
#         else:
#             out = self.relu(self.conv1(x))
#         if self.bn_:
#             out = self.bn2(self.conv2(out))
#         else:
#             out = self.conv2(out)
#         if self.se_:
#             out = self.se(out)
#         if self.res_translate is not None:
#             residual = self.res_translate(residual)
#         out += residual
#
#         return out

class CoralLayer(torch.nn.Module):
    """ Implements CORAL layer described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    -----------
    size_in : int
        Number of input features for the inputs to the forward method, which
        are expected to have shape=(num_examples, num_features).

    num_classes : int
        Number of classes in the dataset.


    """
    def __init__(self, size_in, num_classes):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        self.coral_bias = torch.nn.Parameter(
             torch.zeros(num_classes-1).float())

    def forward(self, x):
        """
        Computes forward pass.

        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.

        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        return self.coral_weights(x) + self.coral_bias

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bn=False, se=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        if se:
            self.se = SEBlock(planes, 4)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.se_ = se
        self.bn_ = bn
        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        if self.bn_:
            out = self.relu(self.bn1(self.conv1(x)))
        else:
            out = self.relu(self.conv1(x))
        if self.bn_:
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(out)
        if self.se_:
            out = self.se(out)
        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out

class GResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1,group=1):
        super(GResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=group)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=group)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=group)
        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                # nn.BatchNorm2d(features),
                nn.LeakyReLU(0.1)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            # nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            # nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            # nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                # nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class Unet(nn.Module):
#
#     def __init__(self, in_dim=3*4, out_dim=3*3, block=2, feats=32, kernel_size=3):
#         super(Unet, self).__init__()
#         InBlock = []
#
#         InBlock.extend([nn.Sequential(
#             nn.Conv2d(in_dim, feats, kernel_size=7, stride=1,
#                       padding=7 // 2),
#             nn.LeakyReLU(0.1,inplace=True)
#         )])
#
#         InBlock.extend([ResBlock(feats, feats, kernel_size=kernel_size, stride=1)
#                         for _ in range(block)])
#
#         # encoder1
#         Encoder_first = [nn.Sequential(
#             nn.Conv2d(feats, feats*2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
#             nn.LeakyReLU(inplace=True)
#         )]
#         Encoder_first.extend([ResBlock(feats*2, feats*2, kernel_size=kernel_size, stride=1)
#                               for _ in range(block)])
#         # encoder2
#         Encoder_second = [nn.Sequential(
#             nn.Conv2d(feats*2, feats*4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
#             nn.LeakyReLU(inplace=True)
#         )]
#         Encoder_second.extend([ResBlock(feats * 4, feats * 4, kernel_size=kernel_size, stride=1)
#                                for _ in range(block)])
#
#         # decoder2
#         Decoder_second = [ResBlock(feats * 4, feats * 4, kernel_size=kernel_size, stride=1)
#                           for _ in range(block)]
#         Decoder_second.append(nn.Sequential(
#             nn.ConvTranspose2d(feats * 4, feats * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.LeakyReLU(inplace=True)
#         ))
#         # decoder1
#         Decoder_first = [ResBlock(feats * 2, feats * 2, kernel_size=kernel_size, stride=1)
#                          for _ in range(block)]
#         Decoder_first.append(nn.Sequential(
#             nn.ConvTranspose2d(feats * 2, feats, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.LeakyReLU(inplace=True)
#         ))
#
#         OutBlock = [ResBlock(feats, feats, kernel_size=kernel_size, stride=1)
#                     for _ in range(block)]
#         OutBlock.append(nn.Conv2d(feats, out_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
#
#
#         self.inBlock = nn.Sequential(*InBlock)
#         self.encoder_first = nn.Sequential(*Encoder_first)
#         self.encoder_second = nn.Sequential(*Encoder_second)
#         self.decoder_second = nn.Sequential(*Decoder_second)
#         self.decoder_first = nn.Sequential(*Decoder_first)
#         self.outBlock = nn.Sequential(*OutBlock)
#
#     def forward(self, x):
#
#         x = torch.cat(x, dim=1)
#
#         first_scale_inblock = self.inBlock(x)
#         first_scale_encoder_first = self.encoder_first(first_scale_inblock)
#         first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
#
#         return first_scale_inblock, first_scale_encoder_first, first_scale_encoder_second


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction, kernel_dim):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        # self.kernel = nn.Sequential(
        #     nn.Linear(kernel_dim, kernel_dim//2, bias=False),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(kernel_dim//2, channels_in * self.kernel_size * self.kernel_size, bias=False)  #3
        # )
        # self.conv = nn.Conv2d(channels_in, channels_out, 1, padding=0)
        self.ca = CA_layer(kernel_dim, channels_out, reduction)

        # self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, vec):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C  #4,256
        '''
        # print(x.size(), vec.size())
        # b, c, h, w = x.size()
        # # print(x[0].size())
        # # branch 1
        # kernel = self.kernel(vec).view(-1, 1, self.kernel_size, self.kernel_size)  #4,128*3*3   512,1,3,3
        # # print(kernel.size())
        # out = self.relu(F.conv2d(x.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        # out = self.conv(out.view(b, -1, h, w))
        # print(out.size())
        # branch 2
        out = x
        out = out + self.ca(x, vec)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x,vec):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(vec[:, :, None, None])

        return x * att

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
# nn.InstanceNorm2d
# b=torch.tensor([
#     [
#         [[1,0,1],[0,1,0],[1,0,1]],[[1,0,1],[0,1,0],[1,0,1]]
#     ],
#     [
#         [[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
#     ]
#     ])

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class Sty_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(Sty_layer, self).__init__()
        self.norm = ChanNorm(channels_out)
        self.conv_du = nn.Sequential(
            nn.Linear(channels_in, channels_in//reduction, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels_in // reduction, channels_out, bias=False),
            # nn.Sigmoid()
        )
        self.conv = Conv2DMod(channels_out, channels_out, 3)

    def forward(self, x, vec):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        norm_x = self.norm(x)
        sty = self.conv_du(vec)
        out = self.conv(norm_x, sty)

        return out + x


class Ada_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(Ada_layer, self).__init__()
        self.conv_alpha = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv_belta = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False)
        )

    def forward(self, x,vec):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        alpha = self.conv_alpha(vec[:, :, None, None])
        belta = self.conv_belta(vec[:, :, None, None])

        return x * alpha + belta

class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size, kernel_dim, reduction=8):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction, kernel_dim)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction, kernel_dim)
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2))
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2))

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, inp):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        x, vec = inp
        out = self.relu(self.da_conv1(x,vec))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2(out, vec))
        out = self.conv2(out) + x

        return out,vec

class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, nf//2, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(nf//2, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, nf//2, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(nf//2, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class SFT_Residual_Block(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Residual_Block, self).__init__()
        self.sft1 = SFT_Layer(nf=nf, para=para)
        self.sft2 = SFT_Layer(nf=nf, para=para)
        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, para_maps):
        fea1 = F.relu(self.sft1(feature_maps, para_maps))
        fea2 = F.relu(self.sft2(self.conv1(fea1), para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)
