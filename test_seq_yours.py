import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from models.ArbVSR.refsrrnn_adists_fgda_only_future import RefsrRNN
import torch.nn.functional as F
# import torch.nn.parallel as P
import torch.nn as nn
# from thop import profile

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

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

class Inference:
    def __init__(self, args):
        self.args = args
        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        # self.n_seq = args.n_sequence
        self.device = 'cuda'
        self.GPUs = args.n_GPUs
        self.scale = args.space_scale

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        self.input_path = self.data_path

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        # self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = RefsrRNN()
        self.net.load_state_dict(torch.load(self.model_path))  # , strict=False
        self.net = self.net.to(self.device)
        if args.n_GPUs > 1:
            self.net = nn.DataParallel(self.net, range(args.n_GPUs))
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()

    def infer(self):
        print(self.scale)
        with torch.no_grad():
            total_psnr = {}
            total_ssim = {}
            total_t = {}
            # total_num = 0
            videos = sorted(os.listdir(self.input_path))
            scale_h = torch.ones(1).to(self.device) * (1/float(self.scale[0]))
            scale_w = torch.ones(1).to(self.device) * (1/float(self.scale[1]))
            hs, hw = 1. / scale_h, 1. / scale_w
            hs, hw = hs.unsqueeze(-1), hw.unsqueeze(-1)
            # kernel = None

            for v in videos:
                kernel = None
                video_psnr = []
                video_ssim = []
                total_time = 0
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*")))

                start_time = time.time()
                inputs = [imageio.imread(p) for p in input_frames]
                h, w, c = inputs[0].shape
                hr_coord = make_coord((h*self.scale[0], w*self.scale[1])).unsqueeze(0).to(self.device)

                cell = torch.ones(2).unsqueeze(0).to(self.device)
                cell[:, 0] *= 2. / h
                cell[:, 1] *= 2. / w

                in_tensor = self.numpy2tensor(inputs, self.device)

                torch.cuda.synchronize()
                preprocess_time = time.time()

                if kernel is None:
                    print("Computing the upsampling kernnel...")
                    res = torch.zeros((1, self.net.num_channels, in_tensor[0].shape[-2], in_tensor[0].shape[-1]), device=in_tensor[0].device)
                    kernel = self.net.kernel_predict(res, hr_coord, cell)

                output = self.net.test_forward(in_tensor, kernel, hr_coord).squeeze(0)  #T,C,H,W

                torch.cuda.synchronize()
                forward_time = time.time()

                # if i >= self.net.count:
                output_img = self.tensor2numpy(output)
                gt = inputs
                print(len(gt), len(output_img))

                # if i != 0:
                    total_time = (forward_time - preprocess_time)

                total_time = total_time / (len(input_frames))
                total_psnr[v] = video_psnr
                total_ssim[v] = video_ssim
                total_t[v] = total_time
                self.logger.write_log('> {} model_inference_time:{:.5}s'.format(v, total_time))
            sum_psnr = 0.
            sum_ssim = 0.
            n_img = 0
            
            self.logger.write_log(
                "# Total AVG-Inference_time={:.5}s".format(sum(total_t.values()) / len(total_t)))

    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[1:1+half]
            img_list_temp.reverse()
            img_list_temp.extend(img_list)
            end_list = img_list[-half - 1:-1]
            end_list.reverse()
            img_list_temp.extend(end_list)
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list, img_list

    def numpy2tensor(self, input_seq, device='cuda', rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor.unsqueeze(0).to(device))
        # stacked = torch.stack(tensor_list).unsqueeze(0)
        return tensor_list

    def tensor2numpy(self, tensors, rgb_range=1.):
        img_list = []
        for tensor in tensors:
            rgb_coefficient = 255 / rgb_range
            img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
            img = img.data
            img_list.append(np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        return img_list

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArbVSR')

    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--border', action='store_true', default=False, help='restore border images of video if true')

    # parser.add_argument('--default_data', type=str, default='GOPRO',
    #                     help='quick test, optional: Adobe, GOPRO')
    # parser.add_argument('--data_path', type=str, default='/data1/shangwei/dataset/video/Vid4_val/Vid4',
    #                     help='the path of test data')
    parser.add_argument('--data_path', type=str, default='/data1/shangwei/dataset/video/REDS/val/val_sharp',
                        help='the path of test data')
    parser.add_argument('--model_path', type=str, default='./refsrrnn_cuf_siren_adists_allstage_only_future_t2.pth',
                        help='the path of pretrain model')
    # parser.add_argument('--result_path', type=str,
    #                     default='/data1/shangwei/dataset/video/Vid4_val/results_verify/refsrrnn_cuf_siren_adists_allstage_only_future_t2/Vid4_val_X2.5_3.5',
    #                     help='the path of deblur result')
    parser.add_argument('--result_path', type=str,
                        default='/data1/shangwei/dataset/video/REDS/results_verify_/refsrrnn_cuf_siren_adists_allstage_only_future_t2/REDS_val_X8',
                        help='the path of deblur result')
    parser.add_argument('--space_scale', type=str, default="8,8", help="upsampling space scale")
    args = parser.parse_args()
    args.space_scale = args.space_scale.split(',')
    args.n_GPUs = 1
    Infer = Inference(args)
    Infer.infer()
