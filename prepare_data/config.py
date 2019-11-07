# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-25 11:06:37
@LastEditTime: 2019-11-07 19:38:58
@Update: 
'''
from easydict import EasyDict

configer = EasyDict()

configer.images      = '../data/'
configer.annotations = '../data/annotations.txt'

configer.splitratio = [0.6, 0.15, 0.25]       # 训练：验证：测试
configer.augment   = True
configer.sideMin   = 40                     # min(w, h) < minSide 的框被忽略
configer.iouThresh = [0.3, 0.4, 0.65]       # 分别为 neg, part, pos 阈值
configer.label     = {'neg': 0, 'pos': 1, 'part': -1, 'landmark': -2}

configer.pImage      = '../data/12x12/'
configer.pAnno       = ['../data/12x12.txt', 
                        '../data/12x12_train.txt', 
                        '../data/12x12_valid.txt', 
                        '../data/12x12_test.txt']
configer.pNums       = [5, 5, 10, 10, 50]   # 分别表示每张图片先采样满足个数的负样本(与框个数有关)，在框附近进行采样的`neg`, `part`, `pos`, `landmark`样本个数

configer.rDets       = '../data/rDets.mat'  # 利用PNet生成的候选框
configer.rImage      = '../data/24x24/'
configer.rAnno       = ['../data/24x24.txt', 
                        '../data/24x24_train.txt', 
                        '../data/24x24_valid.txt', 
                        '../data/24x24_test.txt']
configer.rNums       = [15, -1, -1, 50]   # 分别表示每张图片`neg`, `part`, `pos`, `landmark`样本个数，由PNet结果决定

configer.oDets       = '../data/oDets.mat'  # 利用PNet生成的候选框
configer.oImage      = '../data/48x48/'
configer.oAnno       = ['../data/48x48.txt', 
                        '../data/48x48_train.txt', 
                        '../data/48x48_valid.txt', 
                        '../data/48x48_test.txt']
configer.oNums       = [10, -1, -1, 50]   # 分别表示每张图片`neg`, `part`, `pos`, `landmark`样本个数，由PNet结果决定
