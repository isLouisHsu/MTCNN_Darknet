# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-25 12:40:40
@LastEditTime: 2019-11-03 20:20:42
@Update: 
'''
import os
import numpy as np
from config import configer

with open(configer.pAnno[0], 'r') as f:
    annotations = f.readlines()
annotations = list(map(lambda x: x.strip(), annotations))
n_images = len(annotations)

types = np.array(list(map(lambda x: int(x.split(' ')[1]), annotations)))
n_pos = types[types ==  configer.label['pos']].shape[0]
n_neg = types[types ==  configer.label['neg']].shape[0]
n_part = types[types == configer.label['part']].shape[0]
n_landmark = types[types == configer.label['landmark']].shape[0]

print("Totally {} images".format(n_images))
print("Pos({}): Neg({}): Part({}): Landmark({}) = {}: {}: {}: {}".\
        format(configer.label['pos'], configer.label['neg'], configer.label['part'], configer.label['landmark'],
                n_pos / n_images, n_neg / n_images, n_part / n_images, n_landmark / n_images))

print("Cls      | Pos({}): Neg({}) = {}".format(configer.label['pos'], configer.label['neg'], n_pos / n_neg))
print("Offset   | [Pos({}) + Part({})] : n_samples = {}".format(configer.label['pos'], configer.label['part'], (n_pos + n_part) / n_images))
print("Landmark | Landmark({})         : n_samples = {}".format(configer.label['landmark'], n_landmark / n_images))