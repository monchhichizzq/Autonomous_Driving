# -*- coding: utf-8 -*-
# @Time    : 2020/9/12 23:59
# @Author  : Zeqi@@
# @FileName: Trainer.py
# @Software: PyCharm

import os
import copy
import time
import numpy as np
import torch
from Model_Loader import fcn, load_model
from torchsummary import summary
from torch.nn import CrossEntropyLoss, BCELoss, Softmax
from torch.optim import Adam
from Data_Processor import Data_Generator
from torch.utils.data import DataLoader


def train_model(device,
                model,
                dataloader,
                testloader,
                batch_size,
                criterion,
                optimizer,
                scheduler,
                save_path,
                num_epoch):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_IOU = 0.0
    step_message = "step {:05d},  loss = {:6.4f}, IOU = {:6.4f} --- {:5.2f} ms/batch"
    epoch_message = "-" * 30 + "\n" + \
                    "learning rate : {:6.4f}" + "\n" + \
                    "overall loss : {:<6.4f}" + "\n" + \
                    "overall IOU : {:<6.4f}" + "\n" + \
                    "-" * 30
    best_result_message = "-" * 30 + "\n" + \
                    "Best IOU : {:<6.4f}" + "\n" + \
                    "-" * 30

    for epoch in range(num_epoch):
        # container for IOU
        # we only concern for road
        iou_record = np.zeros(2, dtype=int)
        running_loss = 0
        since = time.time()
        print('****epoch {}/{}****'.format(epoch + 1, num_epoch))

        for i, datas in enumerate(dataloader):
            images, labels = datas
            images = images.to(device)
            labels = labels.to(device)
            # init grad
            optimizer.zero_grad()
            # forward pass
            outputs = model(images)

            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, 2)
            labels = labels.permute(0, 2, 3, 1).reshape(-1, 2)
            outputs = Softmax(dim=1)(outputs)
            loss = criterion(outputs, labels)

            pred = (outputs > 0.5)[:, 1].type(torch.uint8)
            labels = (labels> 0.5)[:, 1].type(torch.uint8)
            # pred correctly
            iou_record[0] += torch.sum((pred == 0) & (labels == 0)).item()
            # all gt pixel belongs to road and all pred pixel belongs to road
            iou_record[1] += torch.sum(labels == 0).item() + torch.sum(pred == 0).item()

            # update parameters
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * batch_size

            if i % 10 == 9:
                time_eplased = time.time()
                iou = iou_record[0] / (iou_record[1] - iou_record[0])
                step = (i + 1) // 10

                print(step_message.format(i + 1, loss / (10 * step), iou, (time_eplased - since) * 100))
                # update
                since = time.time()

        iou = iou_record[0] / (iou_record[1] - iou_record[0])
        if iou > best_IOU:
            best_IOU = iou
            best_model_wts = copy.deepcopy(model.state_dict())
            print(best_result_message .format(iou))
            torch.save(model, os.path.join(save_path, 'IOU_'+str(np.round(iou*100, 2))+'.pth'))
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(epoch_message.format(current_lr, running_loss, iou))
        # valid_model(device,
        #             model,
        #             testloader,
        #             criterion)
    model.load_state_dict(best_model_wts)
    return model


def valid_model(device,
                model,
                dataloader,
                criterion):

    best_IOU = 0.0
    step_message = "val_loss = {:6.4f}, val_IOU = {:6.4f} --- {:5.2f} ms/batch"
    best_result_message = "-" * 30 + "\n" + \
                    "Best IOU : {:<6.4f}" + "\n" + \
                    "-" * 30

    # container for IOU
    # we only concern for road
    iou_record = np.zeros(2, dtype=int)
    running_loss = 0
    since = time.time()

    for i, datas in enumerate(dataloader):
        images, labels = datas
        images = images.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(images)

        outputs = outputs.permute(0, 2, 3, 1).reshape(-1, 2)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 2)
        outputs = Softmax(dim=1)(outputs)
        loss = criterion(outputs, labels)

        pred = (outputs > 0.5)[:, 1].type(torch.uint8)
        labels = (labels> 0.5)[:, 1].type(torch.uint8)
        # pred correctly
        iou_record[0] += torch.sum((pred == 0) & (labels == 0)).item()
        # all gt pixel belongs to road and all pred pixel belongs to road
        iou_record[1] += torch.sum(labels == 0).item() + torch.sum(pred == 0).item()

        running_loss += loss.item()

    time_eplased = time.time()
    iou = iou_record[0] / (iou_record[1] - iou_record[0])
    print(step_message.format(running_loss, iou, (time_eplased - since) * 100))

    iou = iou_record[0] / (iou_record[1] - iou_record[0])
    if iou > best_IOU:
        best_IOU = iou
        # best_model_wts = copy.deepcopy(model.state_dict())
        print(best_result_message .format(best_IOU))
        # torch.save(model, os.path.join(save_path, 'IOU_'+str(np.round(iou*100, 2))+'.pth'))



def trainer():
    # warnings.filterwarnings('ignore')

    # Set the hyperparameters
    num_classes = 2
    image_shape = (160, 608)
    batch_size = 4
    lr = 0.001
    num_epoch = 200
    save_path = 'saved_models'
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_weighted = torch.tensor((0.75, 0.25)).to(device)

    # Load the train/test data
    training_dataset = Data_Generator(data_folder='../road/data_road/training', is_train=True)
    trainloader = DataLoader(training_dataset, batch_size, shuffle=True)
    testing_dataset = Data_Generator(data_folder='../road/data_road/testing', is_train=False)
    testloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)


    # Load the model
    pretrained_net = load_model()
    fcn_net = fcn(pretrained_net, 2, mode = 'FCN8')
    fcn_net = fcn_net.to('cuda')
    summary(fcn_net, input_size=(3, 160, 608))

    # model = fcn(resnet34, 2).to('cuda')

    # criterion = CrossEntropyLoss(sample_weighted)
    criterion = BCELoss(sample_weighted)
    optimizer = Adam(fcn_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.95)

    best_model = train_model(device,
                            fcn_net,
                            trainloader,
                            testloader,
                            batch_size,
                            criterion,
                            optimizer,
                            scheduler,
                            save_path,
                            num_epoch)






if __name__ == '__main__':
    trainer()