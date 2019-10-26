# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 11:52:23
@LastEditTime: 2019-10-26 12:51:15
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
from model import MtcnnLoss
from trainer import MtcnnTrainer

net = PNet(); params = net.parameters()
trainset = MtcnnData(configer.datapath, 12, 'train', save_in_memory=False)
validset = MtcnnData(configer.datapath, 12, 'valid', save_in_memory=False)
criterion = MtcnnLoss(1.0, 0, 0) # TODO:
optimizer = optim.SGD
lr_scheduler = lr_scheduler.MultiStepLR

trainer = MtcnnTrainer(configer, net, params, trainset, validset, criterion, optimizer, lr_scheduler)
trainer.train()