# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-25 12:40:40
@LastEditTime: 2019-10-29 11:31:43
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

print("Pos({}): Neg({}): Part({}): Landmark({}) = {}: {}: {}: {}".\
        format(configer.label['neg'], configer.label['part'], configer.label['landmark'], configer.label['pos'],
                n_pos / n_images, n_neg / n_images, n_part / n_images, n_landmark / n_images))

print("Cls | Pos / Neg = {}".format((n_pos + n_landmark) / (n_images - n_pos - n_landmark)))
print("Offset | n_offset / n_samples = {}".format((n_pos + n_part) / n_images))
print("Landmark | n_landmark / n_samples = {}".format(n_landmark / n_images))