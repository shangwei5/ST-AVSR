from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW, Adamax

from models.select_network import define_G
from models.model_plain import ModelPlain
from models.restormer_lr_scheduler import CosineAnnealingRestartCyclicLR
from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from torch.autograd import Variable

from utils import utils_image as util
import os
import random
try:
    from .loss import *
except:
    from loss import *

class ModelMFAE(ModelPlain):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelMFAE, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.patch_size = self.opt['datasets']['train']['patch_size']
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt)
        self.fix_iter = self.opt_train.get('fix_iter', 0)
        self.fix_keys = self.opt_train.get('fix_keys', [])
        self.fix_unflagged = True
        self.index = 0


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------

    def define_loss(self):
        # G_lossfn_type = self.opt_train['G_lossfn_type']
        ratios, losses = loss_parse(self.opt_train['G_lossfn_type'])
        self.losses_name = losses
        self.ratios = ratios
        self.losses = []
        for loss in losses:
            loss_fn = eval('{}(self.opt_train)'.format(loss))
            self.losses.append(loss_fn)

        self.G_lossfn_weight = self.ratios   #self.opt_train['G_lossfn_weight']


    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        self.fix_keys = self.opt_train.get('fix_keys', [])
        if self.opt_train.get('fix_iter', 0) and len(self.fix_keys) > 0:
            fix_lr_mul = self.opt_train['fix_lr_mul']
            print(f'Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.')
            if fix_lr_mul == 1:
                G_optim_params = self.netG.parameters()
            else:  # separate flow params and normal params for different lr
                normal_params = []
                flow_params = []
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        flow_params.append(param)
                        print(name)
                    else:
                        normal_params.append(param)
                G_optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': self.opt_train['G_optimizer_lr']
                    },
                    {
                        'params': flow_params,
                        'lr': self.opt_train['G_optimizer_lr'] * fix_lr_mul
                    },
                ]
        else:
            G_optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    G_optim_params.append(v)
                else:
                    print('Params [{:s}] will not optimize.'.format(k))
        # self.params = G_optim_params
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            self.G_optimizer = AdamW(G_optim_params, lr=self.opt_train['G_optimizer_lr'])
        elif self.opt_train['G_optimizer_type'] == 'adamax':
            self.G_optimizer = Adamax(G_optim_params, lr=self.opt_train['G_optimizer_lr'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        if self.fix_iter:
            if self.fix_unflagged and current_step < self.fix_iter:
                print(f'Fix keys: {self.fix_keys} for the first {self.fix_iter} iters.')
                self.fix_unflagged = False
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_step == self.fix_iter:
                print(f'Train all the parameters from {self.fix_iter} iters.')
                self.netG.requires_grad_(True)

        super(ModelMFAE, self).optimize_parameters(current_step)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingRestartCyclicLR':
            self.schedulers.append(CosineAnnealingRestartCyclicLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        # self.L = data['L'].to(self.device)
        # if need_H:
        #     self.H = data['H'].to(self.device)
        self.L, self.input_path, self.sw, self.sh, self.H, self.coord, self.cell = data   #rs_imgs, gs_imgs, fl_imgs, prior_imgs, time_rsc, out_paths, input_path
        self.L = [inp.to(self.device) for inp in self.L]
        self.H = [inp.to(self.device) for inp in self.H]
        self.sw, self.sh = self.sw.to(self.device), self.sh.to(self.device)
        self.sw, self.sh = self.sw.unsqueeze(-1), self.sh.unsqueeze(-1)
        if isinstance(self.coord, list):
            self.coord = [crd.to(self.device) for crd in self.coord]
        else:
            self.coord = self.coord.to(self.device)
        if isinstance(self.cell, list):
            self.cell = [cll.to(self.device) for cll in self.cell]
        else:
            self.cell = self.cell.to(self.device)
        # print("input:", self.input_path, 'output:', self.out_path)
        # print(self.sw.dtype, self.sh.dtype, self.H.dtype)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def pad(self, img, ratio=32):
        if len(img.shape) == 5:
            b, n, c, h, w = img.shape
            img = img.reshape(b * n, c, h, w)
            ph = ((h - 1) // ratio + 1) * ratio
            pw = ((w - 1) // ratio + 1) * ratio
            padding = (0, pw - w, 0, ph - h)
            img = F.pad(img, padding, mode='replicate')
            img = img.reshape(b, n, c, ph, pw)
            return img
        elif len(img.shape) == 4:
            n, c, h, w = img.shape
            ph = ((h - 1) // ratio + 1) * ratio
            pw = ((w - 1) // ratio + 1) * ratio
            padding = (0, pw - w, 0, ph - h)
            img = F.pad(img, padding)  #, mode='replicate'
            return img
        elif len(img.shape) == 6:
            b, n1, n2, c, h, w = img.shape
            img_list = []
            for i in range(n1):
                img1 = img[:, i].reshape(b * n2, c, h, w)
                ph = ((h - 1) // ratio + 1) * ratio
                pw = ((w - 1) // ratio + 1) * ratio
                padding = (0, pw - w, 0, ph - h)
                img1 = F.pad(img1, padding, mode='replicate')
                img1 = img1.reshape(b, n2, c, ph, pw)
                img_list.append(img1)
            img = torch.stack(img_list, dim=1)
            return img

    def netG_forward(self, is_train=True):
        if is_train:

            # ori_h, ori_w = h, w
            # if ori_h % 4 != 0 or ori_w % 4 != 0:
            #     self.L = self.pad(self.L)

            out = self.netG(self.L, (self.sw, self.sh), self.coord, self.cell)   #

            # if ori_h % 4 != 0 or ori_w % 4 != 0:
            #     out = out[:, :, :ori_h, :ori_w]

            self.E = out
        else:

            # ori_h, ori_w = h, w
            # if ori_h % 8 != 0 or ori_w % 8 != 0:
            #     L = self.pad(L, 8)
            #
            # self.L = [L[:, i, :, :, :] for i in range(L.shape[1])]

            out = self.netG(self.L, (self.sw, self.sh), self.coord, self.cell, isTrain=False)  # b, num_frames*3, h, w   -----   b, num_frames*2*2, h, w

            # if ori_h % 8 != 0 or ori_w % 8 != 0:
            #     out = out[:, :, :self.H.shape[-2], :self.H.shape[-1]]
            #     self.L = self.L[:, :, :ori_h, :ori_w]

            self.E = out
            self.L = [F.interpolate(L, (self.H[0].shape[-2], self.H[0].shape[-1]), mode='bicubic') for L in self.L]
            self.L = torch.stack(self.L, dim=1)
            self.H = torch.stack(self.H, dim=1)


    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        self.netG_forward()

        # b, num_gs, c, h, w = self.H.shape  # (b, 3, 3, 256, 256)
        # self.H = self.H.reshape(b, num_gs * c, h, w).detach()
        H = torch.stack(self.H, dim=1)
        b, t, c, h, w = H.shape
        E = self.E.view(-1, c, h, w).contiguous()
        H = H.view(-1, c, h, w).contiguous()
        # assert fc == 12, fc
        # print([flow.size() for flow in self.flows[0]])
        losses = {}
        loss_all = None
        for i in range(len(self.losses)):
            if self.losses_name[i].lower().startswith('epe'):
                loss_sub = self.losses[i](self.flows[0], self.H_flows, 1)
                for flow in self.flows[1:]:
                    loss_sub += self.losses[i](flow, self.H_flows, 1)
                loss_sub = self.ratios[i] * loss_sub.mean()
            elif self.losses_name[i].lower().startswith('census'):
                loss_sub = self.ratios[i] * [self.losses[i](self.E.reshape(b * int(c // 3), 3, h, w), self.H.reshape(b * int(c // 3), 3, h, w)).mean()]
            elif self.losses_name[i].lower().startswith('variation'):
                # loss_sub = self.losses[i](self.flows[0].reshape(fb * int(fc // 2), 2, fh, fw), mean=True)
                # for flow in self.flows[1:]:
                #     loss_sub += self.losses[i](flow.reshape(fb * int(fc // 2), 2, fh, fw), mean=True)
                # loss_sub = self.ratios[i] * loss_sub
                loss_sub = self.ratios[i] * self.losses[i](self.flow, mean=True)
            elif self.losses_name[i].lower().startswith('freq'):
                loss_sub = self.ratios[i] * (self.losses[i](self.E, self.H, self.mask))  #self.ratios[i] * (self.losses[i](self.E, self.L) +
            elif self.losses_name[i].lower().startswith('perceptual'):
                loss_sub = self.ratios[i] * (self.losses[i](E, H))
            else:
                loss_sub = self.ratios[i] * (self.losses[i](E, H))
            losses[self.losses_name[i]] = loss_sub
            self.log_dict[self.losses_name[i]] = loss_sub.item()
            if loss_all == None:
                loss_all = loss_sub
            else:
                loss_all += loss_sub
        G_loss = loss_all
        # with torch.autograd.detect_anomaly():
        G_loss.backward()
        # for name, param in self.netG.named_parameters():
        #     if param.grad.isnan():
        #         print(name)  #module.net_scale.conv1.weight
            # break
        # print(self.netG.module.net_scale.conv1.weight, self.netG.module.net_scale.conv1.weight.grad)

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward(False)
        # self.H = torch.chunk(self.H, chunks=self.H.size()[1]//3, dim=1)
        # b,c,h,w = self.E.shape
        # self.E = self.E.reshape(b, c//3, 3, h, w)
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        # out_dict['L_t2b'] = self.L_t2b.detach()[0].float().cpu()
        # out_dict['L_b2t'] = self.L_b2t.detach()[0].float().cpu()
        # out_dict['E0'] = self.E[:, 0].detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        # out_dict['L0'] = self.L[:, 0].detach()[0].float().cpu()
        # out_dict['L1'] = self.L[:, 1].detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
