import functools
import torch
from torch.nn import init


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    if net_type == 'vidue':
        from models.network_vidue import ResNet as net
        netG = net(n_inputs=opt_net['n_inputs'], blocks=opt_net['blocks'], feats=opt_net['feats'])

    elif net_type == 'stgtn_cuf3_siren':
        from models.network_stgtn_cuf3_siren import HybridT as net
        netG = net()
        
    elif net_type == 'hat_refsr_cuf_siren':
        from models.network_hat_refsr_cuf_siren import ResNet as net
        netG = net(n_inputs=opt_net['n_inputs'], blocks=opt_net['blocks'], feats=opt_net['feats'],
                   optical_flow_path=opt_net['flow_path'],pretrained_extractor_path=opt_net['pretrained_extractor_path'])

    elif net_type == 'vidue_bsrpp_align_refsr_cuf3_siren':
        from models.network_vidue_bsrpp_align_refsr_cuf3_siren import ResNet as net
        netG = net(n_inputs=opt_net['n_inputs'], blocks=opt_net['blocks'], feats=opt_net['feats'],
                   optical_flow_path=opt_net['flow_path'])

    elif net_type == 'refsrrnn_cuf_siren':
        from models.ArbVSR.refsrrnn import RefsrRNN as net
        netG = net()

    elif net_type == 'refsrrnn_cuf_siren_adists_fgda_only_future':
        from models.ArbVSR.refsrrnn_adists_fgda_only_future import RefsrRNN as net
        netG = net(count=opt_net['count'])

    elif net_type == 'refsrrnn_cuf_siren_adists_prob_fgda_only_future':
        from models.ArbVSR.refsrrnn_adists_prob_fgda_only_future import RefsrRNN as net
        netG = net(count=opt_net['count'])

    elif net_type == 'refsrrnn_cuf_siren_adists_prob_fgda_only_future_new':
        from models.ArbVSR.refsrrnn_adists_prob_fgda_only_future_new import RefsrRNN as net
        netG = net(count=opt_net['count'])

    elif net_type == 'refsrrnn_cuf_siren_adists_onlyop':
        from models.ArbVSR.refsrrnn_adists_onlyop import RefsrRNN as net
        netG = net()

    elif net_type == 'refsrrnn_cuf_siren_adists_concat_fusion':
        from models.ArbVSR.refsrrnn_adists_concat_fusion import RefsrRNN as net
        netG = net()

    elif net_type == 'refsrrnn_cuf_siren_adists_mfca':
        from models.ArbVSR.refsrrnn_adists_mfca import RefsrRNN as net
        netG = net()

    elif net_type == 'refsrrnn_cuf_siren_adists_texture_filter':
        from models.ArbVSR.refsrrnn_adists_texture_filter import RefsrRNN as net
        netG = net()

    elif net_type == 'refsrrnn_cuf_siren_texture_v1':
        from models.ArbVSR.refsrrnn_texture_v1 import RefsrRNN as net
        netG = net()

    elif net_type == 'refsrrnn_cuf_siren_texture_v2':
        from models.ArbVSR.refsrrnn_texture_v2 import RefsrRNN as net
        netG = net()

    elif net_type == 'bvsrpp_cuf_siren':
        from models.ArbVSR.basicvsr_pp import BasicVSRPlusPlus as net
        netG = net()

    elif net_type == 'swinir':
        from models.network_swinir import SwinIR as net
        netG =  net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'],
                   talking_heads=opt_net['talking_heads'],
                   use_attn_fn=opt_net['attn_fn'],
                   head_scale=opt_net['head_scale'],
                   on_attn=opt_net['on_attn'],
                   use_mask=opt_net['use_mask'],
                   mask_ratio1=opt_net['mask_ratio1'],
                   mask_ratio2=opt_net['mask_ratio2'],
                   mask_is_diff=opt_net['mask_is_diff'],
                   type=opt_net['type'],
                   opt=opt_net,
                   )
    elif net_type == 'EQSR':
        from models.SOTA.EQSR.archs.hat_ModMBFormer_Sim_arch import EQSR as net
        netG = net()

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # # ----------------------------------------
    # # initialize weights
    # # ----------------------------------------
    # if opt['is_train']:
    #     init_weights(netG,
    #                  init_type=opt_net['init_type'],
    #                  init_bn_type=opt_net['init_bn_type'],
    #                  gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
