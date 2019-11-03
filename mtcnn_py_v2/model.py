# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 11:26:49
@LastEditTime: 2019-11-03 19:14:52
@Update: 
'''
import torch
import torch.cuda as cuda
import torch.nn as nn


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 10, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 16, 3, 1)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(32, 15, 1, 1)    # 1 + 4 + 10

    def forward(self, x):
        """
        Params:
            x: {tensor(N, 3, H, W)}
        Returns:
            x: {tensor(N, 15, H // 12, W // 12)}
        """
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.prelu3(x)
        
        x = self.conv4(x)

        return x


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 28, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(28, 48, 3, 1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(48, 64, 2, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Linear(64*2*2, 128)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Linear(128,  15)    # 1 + 4 + 10

    def forward(self, x):
        """
        Params:
            x: {tensor(N, 3, 24, 24)}
        Returns:
            x: {tensor(N, 15)}
        """

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.prelu3(x)

        x = x.view(x.shape[0], -1)
        
        x = self.conv4(x)
        x = self.prelu4(x)

        x = self.conv5(x)

        return x


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.conv1 = nn.Conv2d( 3,  32, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(32,  64, 3, 1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(64,  64, 3, 1)
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 2, 1)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Linear(256,  15)    # 1 + 4 + 10

    def forward(self, x):
        """
        Params:
            x: {tensor(N, 3, 48, 48)}
        Returns:
            x: {tensor(N, 15)}
        """
        
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.prelu4(x)

        x = x.view(x.shape[0], -1)
        
        x = self.conv5(x)
        x = self.prelu5(x)

        x = self.conv6(x)

        return x

class MtcnnLoss(nn.Module):

    def __init__(self, cls, bbox, landmark, ohem=0.7):
        super(MtcnnLoss, self).__init__()

        self.cls = cls
        self.bbox = bbox
        self.landmark = landmark
        self.ohem = ohem
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, labels, offsets, landmarks):
        """
        Params:
            pred:      {tensor(N, 15)}
            labels:    {tensor(N)}
            offsets:   {list[tensor(0,) or tensor(4,)]}
            landmarks: {list[tensor(0,) or tensor(10,)]}
        """
        n_samples = pred.shape[0]
        if pred.dim() > 2:
            pred = pred.view(n_samples, -1)

        # 分类
        cls_pred = torch.sigmoid(pred[:, 0].squeeze())
        cls_gt   = torch.where((labels == 1)^(labels == -2), 
                                torch.ones_like(labels), torch.zeros_like(labels))
        mask     = labels >= 0
        cls_pred = torch.masked_select(cls_pred, mask)
        cls_gt   = torch.masked_select(cls_gt, mask)
        
        loss_cls = self.bce(cls_pred, cls_gt)
        n_keep = int(self.ohem * loss_cls.shape[0])
        loss_cls = torch.mean(torch.topk(loss_cls, n_keep)[0])

        # 位置
        idx = (labels == 1)^(labels == -1)
        if idx.sum() == 0:
            loss_offset = 0
        else:
            offset_pred = pred[idx, 1: 5]
            offset_gt   = torch.stack([offsets[i] for i in range(n_samples) if idx[i] == 1], dim=0)
            loss_offset = torch.mean(self.mse(offset_pred, offset_gt), dim=1)
            n_keep = int(self.ohem * loss_offset.shape[0])
            loss_offset = torch.mean(torch.topk(loss_offset, n_keep)[0])

        # 关键点
        idx = labels == -2
        if idx.sum() == 0:
            loss_landmark = 0
        else:
            landmark_pred = pred[idx, 5:]
            landmark_gt   = torch.stack([landmarks[i] for i in range(n_samples) if idx[i] == 1], dim=0)
            loss_landmark = torch.mean(self.mse(landmark_pred, landmark_gt), dim=1)
            n_keep = int(self.ohem * loss_landmark.shape[0])
            loss_landmark = torch.mean(torch.topk(loss_landmark, n_keep)[0])

        loss_total = self.cls*loss_cls + self.bbox*loss_offset + self.landmark*loss_landmark
        return loss_total, loss_cls, loss_offset, loss_landmark

# if __name__ == "__main__":
    
#     net = PNet()
#     x = torch.rand(4, 3, 12, 12)
#     y = net(x)

#     net = RNet()
#     x = torch.rand(4, 3, 24, 24)
#     y = net(x)

#     net = ONet()
#     x = torch.rand(4, 3, 48, 48)
#     y = net(x)