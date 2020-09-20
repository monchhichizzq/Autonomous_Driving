# -*- coding: utf-8 -*-
# @Time    : 2020/9/20 3:39
# @Author  : Zeqi@@
# @FileName: Test.py
# @Software: PyCharm

import os
import copy
import time
import cv2
import numpy as np
import torch
from Model_Loader import fcn, load_model
from torchsummary import summary
from torch.nn import CrossEntropyLoss, BCELoss, Softmax
from torch.optim import Adam
from Data_Processor import Data_Generator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_model(device, best_model, trainloader, testloader, image_shape=(160,608), validation=True):
    best_model.eval()
    plt.figure()
    if validation:
        for i, datas in enumerate(testloader):
            # Pretrained network performs on the testset
            image, ori_image = datas
            image = image.to(device)
            ori_image = np.array(ori_image[0])
            # print('orig',np.shape(ori_image))
            output = best_model(image)
            output = output[0]
            pred = Softmax(dim=0)(output)
            road_map = np.array(pred[1, :, :].to('cpu') > 0.5) * 255

            seg = np.zeros((*image_shape, 3)).astype('uint8')
            seg[:,:,1] = road_map
            print('seg', np.shape(seg))

            add_image = cv2.addWeighted(seg,0.3,ori_image,0.7,0)
            result = np.concatenate([ori_image, add_image], axis = 0)
            plt.imshow(result)
            plt.pause(2)
    else:
        for i, datas in enumerate(trainloader):
            # The prediction of the pretrained network compared with ground truth
            image, labels = datas
            ori_image = image.permute(0,2,3,1)
            ori_image = np.array(ori_image.to('cpu')[0]).astype('uint8')*255

            image = image.to(device)
            print('orig', np.shape(ori_image))
            output = best_model(image)
            output = output[0]
            pred = Softmax(dim=0)(output)
            road_map = np.array(pred[1, :, :].to('cpu') > 0.5) * 255
            road_map_truth = np.array(labels[0][1, :, :].to('cpu') > 0.5) * 255

            seg = np.zeros((*image_shape, 3)).astype('uint8')
            seg_truth = np.zeros((*image_shape, 3)).astype('uint8')
            seg[:, :, 1] = road_map
            seg_truth[:, :, 2] = road_map_truth
            #print(np.shape(ori_image), type(ori_image[0,0,0]), np.shape(seg_truth), type(seg_truth[0,0,0]))
            add_image_truth = cv2.addWeighted(seg_truth, 0.8, ori_image, 0.2, 0)
            add_image = cv2.addWeighted(seg, 0.3, ori_image, 0.7, 0)
            result = np.concatenate([add_image_truth, add_image], axis=0)
            plt.imshow(result)
            plt.pause(0.5)

if __name__ == '__main__':
    testing_dataset = Data_Generator(data_folder='../road/data_road/testing', is_train=False)
    testloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)
    training_dataset = Data_Generator(data_folder='../road/data_road/training', is_train=True)
    trainloader = DataLoader(training_dataset, batch_size=1, shuffle=False)

    pretrained_model = torch.load('saved_models/IOU_98.95.pth')
    summary(pretrained_model, input_size=(3, 160, 608))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model(device, pretrained_model, trainloader, testloader, image_shape=(160,608), validation=True)

