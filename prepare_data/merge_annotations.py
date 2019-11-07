# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-25 11:29:06
@LastEditTime: 2019-10-25 12:24:26
@Update: 
'''
import os
import numpy as np
from config import configer

def read_wider_face_bbx_gt_txt(txtfile):
    mode = txtfile.split('_')[2]
    prefix = 'WIDER/WIDER_{}/images/'.format(mode)
    
    with open(txtfile, 'r') as f:
        anno = f.readlines()
    anno = list(map(lambda x: x.strip(), anno))

    # jpg
    isFile = list(map(lambda x: x.split('.')[-1] == 'jpg', anno))
    index  = list(np.where(np.array(isFile) == True)[0])
    n_index = len(index)

    # list
    annolist = []
    for i_index in range(n_index):
        i = index[i_index]
        j = -1 if i_index == n_index - 1 else index[i_index + 1]
        anno_i = anno[i: j]
        del anno_i[1]
        anno_i = [anno_i[0]] + list(map(lambda x: ' '.join(x.split(' ')[:4]), anno_i[1:]))
        anno_i = ' '.join(anno_i)
        annolist += [anno_i]
    
    annolist = list(map(lambda x: prefix + x, annolist))
    return annolist

def read_align_image_list_txt(txtfile):
    prefix = 'Align/'
    
    with open(txtfile, 'r') as f:
        annolist = f.readlines()
    annolist = list(map(lambda x: x.strip(), annolist))
    annolist = list(map(lambda x: prefix + x.replace('\\', '/'), annolist))
    
    return annolist

widerTrain = read_wider_face_bbx_gt_txt('../data/WIDER/wider_face_train_bbx_gt.txt')
widerVal   = read_wider_face_bbx_gt_txt('../data/WIDER/wider_face_val_bbx_gt.txt'  )
wider = widerTrain + widerVal

alignTrain = read_align_image_list_txt('../data/Align/trainImageList.txt')
alignTest  = read_align_image_list_txt('../data/Align/testImageList.txt' )
align = alignTrain + alignTest

annotations = wider + align
annotations = list(map(lambda x: x + '\n', annotations))
with open(configer.annotations, 'w') as f:
    f.writelines(annotations)