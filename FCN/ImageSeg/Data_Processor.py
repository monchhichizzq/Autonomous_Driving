# -*- coding: utf-8 -*-
# @Time    : 2020/9/12 19:32
# @Author  : Zeqi@@
# @FileName: Data_Processor.py
# @Software: PyCharm

import os
import re
import cv2
import random
import numpy as np
import scipy.misc
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


class Data_Generator(Dataset):
    def __init__(self,data_folder,**kwargs):
        self.image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        self.label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                   for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        self.transform = kwargs.get('transform', None)
        self.input_shape = kwargs.get('input_shape', (160, 608))
        self.is_train = kwargs.get('is_train', True)

    def bright_adjust(self,img, a, b):
        if random.uniform(0, 1) > 0.5:
            return img
        else:
            img = img.astype(np.int)
            img = img * a + b
            return img

    def crop_resize_data(self, img, cropped_height, cropped_width):
        '''裁剪掉图像上面一小部分区域
        param:
        img：输入图像'''
        cropped_top = np.shape(img)[0] - cropped_height*2
        cropped_right = np.shape(img)[1] - cropped_width*2
        roi_image = img[cropped_top:, :(np.shape(img)[1] - cropped_right)]
        return roi_image

    def common_process(self, index, **kwargs):
        self.is_gt = kwargs.get('gt', False)
        if self.is_gt:
            image_file = self.label_paths[os.path.basename(self.image_paths[index])]
        else:
            image_file = self.image_paths[index]
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Cut the top of image, less training
        roi_img = self.crop_resize_data(img, self.input_shape[0], self.input_shape[1])
        image_shape = (np.shape(roi_img)[1] // 2, (np.shape(roi_img)[0]) // 2)
        image = cv2.resize(roi_img, image_shape)
        return image

    def data_augmentation(self, image):
        # #Contrast augmentation and Brightness augmentation
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        contr = random.uniform(0.85, 1.15)  # Contrast augmentation
        bright = random.randint(-45, 30)  # Brightness augmentation
        image = self.bright_adjust(image, contr, bright)
        # Input normalization
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        return image/255

    def gt_mask(self, gt_image):
        background_color = np.array([255, 0, 0])
        gt_background = np.all(gt_image == background_color, axis=2)  # output True or False
        gt_background = np.expand_dims(gt_background, axis=-1)

        # Concatenate two gt backgrounds, one is for road, the other is for the others
        gt_image = np.concatenate((gt_background, np.invert(gt_background)), axis=2)
        # gt_image = gt_background
        gt_image = gt_image.transpose([2, 0, 1])  # The torch gt size should be (2, 135, 621)
        return gt_image


    def __getitem__(self,index):
        # (375, 1242, 3)
        if self.is_train:
            # Input Image processing
            image =self.common_process(index)
            image = self.data_augmentation(image)
            # Ground Truth processing
            gt_image = self.common_process(index, gt=True)
            gt_mask = self.gt_mask(gt_image)
            gt = torch.FloatTensor(gt_mask)
            image = transforms.ToTensor()(image).to(dtype=torch.float)
            return image, gt
        else:
            # Input Image processing
            image = self.common_process(index)
            torch_image = transforms.ToTensor()(image).to(dtype=torch.float)
            return torch_image, image

    def __len__(self):
        return len(self.image_paths)

if __name__ == '__main__':
    for i in range(500):
        training_dataset = Data_Generator(data_folder='../road/data_road/training').__getitem__(i)
    trainloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
