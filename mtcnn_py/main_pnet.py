# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 11:52:23
@LastEditTime: 2019-11-06 09:24:13
@Update: 
'''
import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from config import configer
from dataset import MtcnnData
from model import PNet
from model import MtcnnLoss, LossFn
from trainer import MtcnnTrainer

net = PNet()
# state = torch.load('ckptdir/PNet_0025.pkl', map_location='cpu')['net_state']; net.load_state_dict(state)

params = net.parameters()
trainset = MtcnnData(configer.datapath, 12, 'train', save_in_memory=False)
validset = MtcnnData(configer.datapath, 12, 'valid', save_in_memory=False)
testset  = MtcnnData(configer.datapath, 12, 'test',  save_in_memory=False)
# criterion = MtcnnLoss(1.0, 0.5, 0.0)
criterion = LossFn(1.0, 0.5, 0.5)
optimizer = optim.Adam
lr_scheduler = lr_scheduler.ExponentialLR

trainer = MtcnnTrainer(configer, net, params, trainset, validset, testset, criterion, optimizer, lr_scheduler)
trainer.train()
