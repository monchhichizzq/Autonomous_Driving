# -*- coding: utf-8 -*-
# @Time    : 2020/9/12 19:32
# @Author  : Zeqi@@
# @FileName: Model_Loader.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.resnet import resnet18, resnet34

import numpy as np

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

def load_model():
    '''Download the pretrained model'''
    model = resnet34(pretrained=True).to('cuda')
    # summary(model, input_size=(3, 224, 224))
    #print('Number of Residual blocks: ', len(list(model.children())))
    # print('\n', list(model.children())[:-4])
    return model


class fcn(nn.Module):
    def __init__(self,pretrained_net, num_classes, mode = 'FCN8'):
        super(fcn,self).__init__()
        """
            Find the layers where the feature map size is 1/8 (1/16, 1/32) of the original input
            The input image size: 224,224,3
            list(resnet18_model.children())[-1] ---> Classifier
            list(resnet18_model.children())[-2] ---> Global Average Pooling
            list(resnet18_model.children())[-3] ---> output: 7,7,512 ==>> 1/32
            list(resnet18_model.children())[-4] ---> output: 14,14,256 ==>> 1/16
            list(resnet18_model.children())[-5] ---> output: 28,28,128 ==>> 1/8
            
            
        """
        self.mode = mode

        # Part one，output 28, 28, 128
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        # Part two，output 14, 14, 256
        self.stage2 = list(pretrained_net.children())[-4]
        # Part three，output 7, 7, 512
        self.stage3 = list(pretrained_net.children())[-3]

        self.scores3 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.scores2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.scores1 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)

        # Upsampling heat map 2x,  conv kernel initialized with the bilinear weights
        self.upsample_2x = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes,num_classes,4)
        # Upsampling heat map 4x, conv kernel initialized with the bilinear weights
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        # Upsampling heat map 8x, conv kernel initialized with the bilinear weights
        self.upsample_8x = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=16,stride=8,padding=4,bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes,num_classes,16)

        # 不确定
        # Upsampling heat map 16x, conv kernel initialized with the bilinear weights
        self.upsample_16x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_16x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        # Upsampling heat map 32x, conv kernel initialized with the bilinear weights
        self.upsample_32x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32, padding=0,bias=False)
        self.upsample_32x.weight.data = bilinear_kernel(num_classes, num_classes, 32)

    def forward(self,x):
        if self.mode == 'FCN8':
            # Feature extraction
            x = self.stage1(x)
            s1 = x
            x = self.stage2(x)
            s2 = x
            x = self.stage3(x)
            s3 = x

            # 2x upsampling
            s3 = self.scores3(s3)
            s3 = self.upsample_2x(s3)
            s2 = self.scores2(s2)
            s2 = s2+s3

            # 4x upsampling
            s1 = self.scores1(s1)
            s2 = self.upsample_4x(s2)
            s1 = s2+s1

            # 8x upsampling
            s = self.upsample_8x(s1)
            return s

        if self.mode == 'FCN16':
            # Feature extraction
            x = self.stage1(x)
            s1 = x
            x = self.stage2(x)
            s2 = x
            x = self.stage3(x)
            s3 = x

            # 2x upsampling
            s3 = self.scores3(s3)
            s3 = self.upsample_2x(s3)
            s2 = self.scores2(s2)
            s2 = s2 + s3

            # 16x upsampling
            s = self.upsample_16x(s2)

        if self.mode == 'FCN32':
            # Feature extraction
            x = self.stage1(x)
            s1 = x
            x = self.stage2(x)
            s2 = x
            x = self.stage3(x)
            s3 = x

            # 32x upsampling
            s3 = self.scores3(s3)
            s = self.upsample_32x(s3)
            return s

if __name__=='__main__':
    # load_model()
    resnet34 = load_model()
    resnet34_fcn = fcn(resnet34, 2, mode = 'FCN8')
    summary(resnet34_fcn.to('cuda'), input_size=(3, 160, 608))
