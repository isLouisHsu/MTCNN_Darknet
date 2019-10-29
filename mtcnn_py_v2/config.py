# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 10:43:04
@LastEditTime: 2019-10-27 11:44:38
@Update: 
'''
from easydict import EasyDict

configer = EasyDict()

configer.datapath = '../data/'
configer.logdir = './logs/'
configer.ckptdir = './ckptdir/'

configer.batchsize = 2**12
configer.n_epoch = 250

configer.lrbase = 5e-4
configer.adjstep = [200,]
configer.gamma = 1e-1

configer.cuda = True

