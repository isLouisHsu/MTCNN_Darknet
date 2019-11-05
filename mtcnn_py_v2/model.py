# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 11:26:49
@LastEditTime: 2019-11-05 09:47:43
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
            pred:        {tensor(N, 15)}
            gt_label:    {tensor(N)}
            gt_bbox:     {tensor(N,  4)}
            gt_landmark: {tensor(N, 10)}
        """
        n_samples = pred.shape[0]
        pred = pred.squeeze()

        # 分类
        cls_pred = torch.sigmoid(pred[:, 0])
        cls_gt   = torch.where((labels == 1)^(labels == -2), 
                                torch.ones_like(labels), torch.zeros_like(labels))
        mask     = labels >= 0
        cls_pred = torch.masked_select(cls_pred, mask)
        cls_gt   = torch.masked_select(cls_gt,   mask)
        
        loss_cls = self.bce(cls_pred, cls_gt)
        n_keep = int(self.ohem * loss_cls.shape[0])
        loss_cls = torch.mean(torch.topk(loss_cls, n_keep)[0])

        # 位置
        idx = (labels == 1)^(labels == -1)
        if idx.sum() == 0:
            loss_offset = 0
        else:
            offset_pred = pred[idx, 1: 5]
            offset_gt   = offsets[idx]
            loss_offset = torch.mean(self.mse(offset_pred, offset_gt), dim=1)
            n_keep = int(self.ohem * loss_offset.shape[0])
            loss_offset = torch.mean(torch.topk(loss_offset, n_keep)[0])

        # 关键点
        idx = labels == -2
        if idx.sum() == 0:
            loss_landmark = 0
        else:
            landmark_pred = pred[idx, 5:]
            landmark_gt   = landmarks[idx]
            loss_landmark = torch.mean(self.mse(landmark_pred, landmark_gt), dim=1)
            n_keep = int(self.ohem * loss_landmark.shape[0])
            loss_landmark = torch.mean(torch.topk(loss_landmark, n_keep)[0])

        loss_total = self.cls*loss_cls + self.bbox*loss_offset + self.landmark*loss_landmark
        return loss_total, loss_cls, loss_offset, loss_landmark


class LossFn(nn.Module):
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        super(LossFn, self).__init__()
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.landmark_factor = landmark_factor
        self.loss_cls = nn.BCELoss()  # binary cross entropy
        self.loss_box = nn.MSELoss()  # mean square error
        self.loss_landmark = nn.MSELoss()

    def cls_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.loss_cls(valid_pred_label, valid_gt_label) * self.cls_factor

    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        # get the mask element which != 0
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)
        #mask = gt_label != 0
        # convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        # only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor

    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label, -2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark) * self.land_factor

    def forward(self, pred, gt_label, gt_bbox, gt_landmark):
        """
        Params:
            pred:        {tensor(N, 15)}
            gt_label:    {tensor(N)}
            gt_bbox:     {tensor(N,  4)}
            gt_landmark: {tensor(N, 10)}
        """
        pred = pred.view(pred.shape[0], 15)

        cls_pred        = torch.sigmoid(pred[:, 0])
        box_offset_pred = pred[:, 1: 5]
        landmark_offset_pred = pred[:, 5:]
            
        cls_loss = self.cls_loss(gt_label, cls_pred)

        box_offset_loss = 0
        if self.box_factor != 0:
            box_offset_loss = self.box_loss(gt_label, gt_bbox, box_offset_pred)

        landmark_loss = 0
        if self.landmark_factor != 0:
            landmark_loss = self.landmark_loss(gt_label, gt_landmark, landmark_offset_pred)
            
        total_loss = cls_loss + box_offset_loss + landmark_loss
        return total_loss, cls_loss, box_offset_loss, landmark_loss


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