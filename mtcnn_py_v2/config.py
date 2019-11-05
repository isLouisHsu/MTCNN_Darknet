# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 10:43:04
@LastEditTime: 2019-11-04 16:18:38
@Update: 
'''
from easydict import EasyDict

configer = EasyDict()

configer.datapath = '../data/'
configer.logdir = './logs/'
configer.ckptdir = './ckptdir/'

configer.batchsize = 512
configer.n_epoch = 20

configer.lrbase = 1e-2
configer.adjstep = [12, 16]
configer.gamma = 1e-1

configer.cuda = True

