# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-26 10:43:43
@LastEditTime: 2019-11-04 20:56:18
@Update: 
'''
import os
import cv2
import time
import numpy as np

import torch
from torch.utils.data   import Dataset, DataLoader
from torchvision.transforms import ToTensor

from utils import getTime

def collate_fn(batch):

    images = np.stack(list(map(lambda x: np.transpose(x[0], [2, 0, 1]), batch)), axis=0)
    labels = list(map(lambda x: x[1], batch))
    offsets   = list(map(lambda x: x[2], batch))
    landmarks = list(map(lambda x: x[3], batch))
    
    images = torch.from_numpy(images).float()
    labels = torch.tensor(labels).float()
    offsets = torch.tensor(offsets)
    landmarks = torch.tensor(landmarks)

    return images, labels, offsets, landmarks

class MtcnnData(Dataset):

    def __init__(self, datapath, imsize, mode, save_in_memory=False):

        with open(os.path.join(datapath, '{}x{}_{}.txt'.format(imsize, imsize, mode)), 'r') as f:
            lines = f.readlines()

        self.samples_ = np.array(list(map(lambda x: self._parse_line(x), lines)))
        self.save_in_memory = save_in_memory

        if save_in_memory:
            self.images_ = list(map(lambda x: cv2.imread(x[0], cv2.IMREAD_COLOR), self.samples_))
            print("Data loaded!")
        
        self.n_samples = self.samples_.shape[0]
        self.image_size = [3, imsize, imsize]

    def _parse_line(self, line):

        line = line.strip().split(' ')
        filename = line[0]
        label    = int(line[1])
        if label == 0:      # neg
            offset   = np.zeros(4)
            landmark = np.zeros(10)
        elif label == 1:    # pos
            offset   = list(map(float, line[2:]))
            landmark = np.zeros(10)
        elif label == -1:   # part
            offset   = list(map(float, line[2:]))
            landmark = np.zeros(10)
        elif label == -2:   # landmark
            offset   = np.zeros(4)
            landmark = list(map(float, line[2:]))

        return filename, label, offset, landmark

    def __getitem__(self, index):

        filename, label, offset, landmark = self.samples_[index]
        
        if self.save_in_memory:
            image = self.images_[index]
        else:
            image = cv2.imread(filename, cv2.IMREAD_COLOR)

        image = image.astype('float') / 255.

        return image, label, offset, landmark
        
    def __len__(self):

        return self.n_samples

if __name__ == '__main__':

    d = MtcnnData('../data/', 12, 'train')
    l = DataLoader(d, 2, collate_fn=collate_fn)
    for dd in l:
        pass