# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 10:43:04
@LastEditTime: 2019-10-29 11:55:58
@Update: 
'''
from easydict import EasyDict

configer = EasyDict()

configer.datapath = '../data/'
configer.logdir = './logs/'
configer.ckptdir = './ckptdir/'

configer.batchsize = 2**12
configer.n_epoch = 500

configer.lrbase = 1e-3
configer.adjstep = [500,]
configer.gamma = 1e-1

configer.cuda = True

