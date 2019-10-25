# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-25 12:40:40
@LastEditTime: 2019-10-25 17:05:40
@Update: 
'''
import os
import numpy as np
from config import configer

with open(configer.annotations, 'r') as f:
    annotations = f.readlines()
annotations = list(map(lambda x: x.strip(), annotations))
n_images = len(annotations)

n_faces = 0
for anno in annotations:
    _type = anno.split('/')[0]
    if _type == 'Align':
        n_faces += 1
    else:
        n_faces += len(anno.split('jpg')[-1].strip().split(' ')) // 4

types = list(map(lambda x: x.split('/')[0], annotations))
isWider = list(map(lambda x: x == 'WIDER', types))
n_wider = np.array(isWider).sum()
n_point = n_images - n_wider

print("Number of images: {} | Number of faces: {}".format(n_images, n_faces))
print("Number of WIDER:  {}({} faces) | Number of POINT: {}({} faces)".\
                    format(n_wider, n_faces - n_point, n_point, n_point))

print("neg({}): {} | part({}): {} | pos({}): {} | landmark({}): {}".\
                    format(configer.label['neg'], n_wider * (configer.pNums[0] + configer.pNums[1]),
                            configer.label['part'], n_wider * (configer.pNums[2]),
                            configer.label['pos'], n_wider * (configer.pNums[3]),
                            configer.label['landmark'], n_point * (configer.pNums[4])
                    ))