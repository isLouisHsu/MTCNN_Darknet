# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 09:50:41
@LastEditTime: 2019-10-26 09:59:51
@Update: 
'''
import numpy as np
from numpy import random as npr
from config import configer

# 划分数据集
with open(configer.pAnno[0], 'r') as f:
    lines = f.readlines()()
    
n_samples = len(lines)
n_train   = int(n_samples * configer.splitratio[0])
n_valid   = int(n_samples * configer.splitratio[1])
n_test    = n_samples - n_train - n_valid
idx       = np.arange(n_samples)
idx_train = npr.choice(idx, n_train)
idx_valid_test = list(filter(lambda x: x not in idx_train, idx))
idx_valid = npr.choice(idx_valid_test, n_valid)
idx_test  = list(filter(lambda x: x not in idx_valid, idx_valid_test))

print("Train: {} | Valid: {} | Test: {} || {}: {}: {}".\
            format(n_train, n_valid, n_test, 
                    n_train / n_samples, n_valid / n_samples, n_test / n_samples))

with open(configer.pAnno[1], 'w') as f:
    f.writelines(lines[idx_train])
with open(configer.pAnno[2], 'w') as f:
    f.writelines(lines[idx_valid])
with open(configer.pAnno[3], 'w') as f:
    f.writelines(lines[idx_test])