import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, unpaired_paths_from_lmdb
from basicsr.data.transforms import oct_augment, unpaired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class UnpairedOCTDataset(data.Dataset):
    """
    Read unpaired reference images, i.e., source (src) and target (tgt),
    """

    def __init__(self, opt):
        super(UnpairedOCTDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.gt_paths, self.lq_paths = unpaired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
        #     self.gt_paths, self.lq_paths = unpaired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #                                                   self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.gt_paths, self.lq_paths = unpaired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        # use self.gt_paths and self.lq_paths to form self.paths. gt_paths and lq_paths should be randomly grouped
        if opt.get("ratios"):
            ratio_gt, ratio_lq = opt["ratios"]
            self.gt_paths *= ratio_gt
            self.lq_paths *= ratio_lq

        merged_gt = list(self.gt_paths)
        random.shuffle(merged_gt)
        self.gt_paths[:]= zip(*merged_gt)
        # TODO: deal with the situation when gt_paths and lq_paths does not have the same length


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
        lq_path = self.lq_paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, flag='grayscale', float32=True)
        # augmentation for training
        # TODO: Need to adjust it based on the feature of OCT images
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            if self.opt['lq_size'] is None:
                # random crop
                img_gt, img_lq = unpaired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
                # flip, rotation
                img_gt, img_lq = oct_augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            else:
                lq_size = self.opt['lq_size']
                # random crop
                img_gt, img_lq = unpaired_random_crop(img_gt, img_lq, gt_size, lq_size, gt_path)
                # flip, rotation
                img_gt, img_lq = oct_augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if len(img_gt) == 3 & (img_gt.shape(2) == 1):
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        else:
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}


    def __len__(self):
        return len(self.gt_paths)
