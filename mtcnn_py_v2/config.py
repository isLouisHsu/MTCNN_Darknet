# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 10:43:04
@LastEditTime: 2019-10-26 12:20:01
@Update: 
'''
from easydict import EasyDict

configer = EasyDict()

configer.datapath = '../data/'
configer.logdir = './logs/'
configer.ckptdir = './ckptdir/'

configer.batchsize = 8 # TODO:
configer.n_epoch = 1000

configer.lrbase = 1e-3
configer.adjstep = [750, 900]
configer.gamma = 1e-1

configer.cuda = True

