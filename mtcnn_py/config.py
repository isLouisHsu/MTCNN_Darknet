# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 10:43:04
@LastEditTime: 2019-11-10 11:26:00
@Update: 
'''
from easydict import EasyDict

configer = EasyDict()

configer.datapath = '../data/'
configer.logdir = './logs/'
configer.ckptdir = './ckptdir/'

configer.batchsize = 512
configer.n_epoch = 80

configer.lrbase = 1e-2
# configer.adjstep = [12, 16]
configer.gamma = 0.95

configer.cuda = True

