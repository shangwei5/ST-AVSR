import cv2
import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import utils.utils as utils
import imageio
import math
import random
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

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

class VIDEODATA(data.Dataset):
    def __init__(self, path, args, name='GOPRO', train=True):
        self.args = args
        self.name = name
        self.train = train

        self.n_frames_video = []

        self._set_filesystem(path)

        # if self.train:
        self.images_input = self._scan()   #
        # else:
        #     self.images_input, self.images_gt = self._scan()
        self.n_seq = args['n_seq']
        self.num_video = len(self.images_input)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        if self.train:
            self.dir_input = os.path.join(self.apath, 'train_sharp')
            print("DataSet INPUT path:", self.dir_input)
        else:
        #     self.dir_gt = os.path.join(self.apath, 'gt')
            self.dir_input = os.path.join(self.apath, 'val_sharp')
        #     print("DataSet GT path:", self.dir_gt)
            print("DataSet INPUT path:", self.dir_input)

    def _scan(self):
        # if self.train:
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        images_input = []
        for vid_input_name in vid_input_names:  #
            input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
            images_input.append(input_dir_names)
            self.n_frames_video.append(len(input_dir_names))
        return images_input
        # else:
        #     vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        #     vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        #     images_input = []
        #     images_gt = []
        #     for (vid_input_name, vid_gt_name) in zip(vid_input_names, vid_gt_names):  #
        #         gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
        #         input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
        #         images_input.append(input_dir_names)
        #         images_gt.append(gt_dir_names)
        #         self.n_frames_video.append(len(input_dir_names))
        #
            # return images_input, images_gt


    def __getitem__(self, idx):
        # if self.train:
        inputs, filenames = self._load_file(idx)   #

        inputs_concat = np.concatenate(inputs, axis=2)
        if self.train:
            scale = random.uniform(2, 6), random.uniform(2, 6)  # random.uniform(1, 4), random.uniform(1, 4) # random.uniform(1, 8), random.uniform(1, 8)
        else:
            scale = 4, 4

        inputs_concat = self.get_patch(inputs_concat, scale=scale)
        inputs_list = [inputs_concat[:, :, i * self.args['n_colors']:(i + 1) * self.args['n_colors']] for i in
                       range(self.n_seq)]
        inputs = np.array(inputs_list)
        gt_tensors = utils.np2Tensor(*inputs, rgb_range=1, n_colors=3)
        if self.train:
            input_tensors = [F.interpolate(gt_tensor.unsqueeze(0), (self.args['patch_size'], self.args['patch_size']), mode='bicubic').squeeze(0) for gt_tensor in gt_tensors]
        else:
            input_tensors = [F.interpolate(gt_tensor.unsqueeze(0), (gt_tensor.shape[-2]//scale[0], gt_tensor.shape[-1]//scale[1]), mode='bicubic').squeeze(0) for gt_tensor in gt_tensors]

        scale = gt_tensors[0].shape[-2] / input_tensors[0].shape[-2], gt_tensors[0].shape[-1] / input_tensors[0].shape[
            -1]
        # print(scale)
        hr_coord = make_coord(gt_tensors[0].shape[-2:])
        cell = torch.ones(2)
        cell[0] *= 2. / gt_tensors[0].shape[-2]
        cell[1] *= 2. / gt_tensors[0].shape[-1]
        # cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        if self.train:
            x0 = random.randint(0, gt_tensors[0].shape[-2] - self.args['patch_size'])
            y0 = random.randint(0, gt_tensors[0].shape[-1] - self.args['patch_size'])
            sample_coord = hr_coord[x0: x0 + self.args['patch_size'], y0: y0 + self.args['patch_size'], :]
            gt_tensors = [gt_tensor[:, x0: x0 + self.args['patch_size'], y0: y0 + self.args['patch_size']] for gt_tensor in gt_tensors]
        else:
            sample_coord = hr_coord

        return input_tensors, filenames, torch.from_numpy(np.array(scale[0])).float(), torch.from_numpy(np.array(scale[1])).float(), gt_tensors, sample_coord, cell   #


    def __len__(self):
        if self.train:
            return self.num_frame    #  * self.repeat
        else:
            return 40 // 40   #self.num_frame   #  // 10

    def _get_index(self, idx):
        if self.train:
            return idx  #% self.num_frame
        else:
            return idx * 40

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)
        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        # print(idx, n_poss_frames)
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        # if self.train:
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        if self.train:
            temp_flip = random.random() < 0.5
            if temp_flip:
                f_inputs = f_inputs[::-1]
        inputs = [np.array(imageio.imread(lr_name)) for lr_name in f_inputs]
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0] for name in f_inputs]
                     #
        return inputs, filenames
        # else:
        #     f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        #     inputs = np.array([imageio.imread(lr_name) for lr_name in f_inputs])
        #     f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        #     gts = np.array([imageio.imread(hr_name) for hr_name in f_gts])
        #     filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
        #                  for name in f_inputs]
        #
        #     return inputs, gts, filenames

    def get_patch(self, input, size_must_mode=1, scale=(1,1)):   #
        if self.train:
            input = utils.get_single_scale_patch(input, patch_size=(int(self.args['patch_size']*scale[0]), int(self.args['patch_size']*scale[1])))   #
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input = input[:new_h, :new_w, :]   #, gt[:new_h, :new_w, :], bm[:new_h, :new_w, :]   #, gt, bm
            if not self.args['no_augment']:
                input = utils.single_data_augment(input)   #, gt, bm  , gt, bm
        else:
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input = input[:new_h, :new_w, :]   #, gt[:new_h, :new_w, :], bm[:new_h, :new_w, :]   #, gt, bm
        return input   #, gt, bm


# if __name__ == "__main__":

