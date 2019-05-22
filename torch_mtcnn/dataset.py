import os
import cv2
import time
import numpy as np

import torch as t
from torch.utils.data   import Dataset
from torchvision.transforms import ToTensor

from utiles import getTime

ANNO = '../data/imglists/{}/{}_{}.txt'

class MtcnnData(Dataset):
    """ load MTCNN data
    
    Params:
        net: {str} 'PNet', 'RNet', 'ONet'
        mode:{str} 'train', 'valid'
    """

    def __init__(self, net, mode, save_in_memory=False):
        
        self.save_in_memory = save_in_memory

        start_time = time.time()
        print("{} || loading data ...".format(getTime()))

        anno = ANNO.format(net, net.lower(), mode)
        with open(anno, 'r') as f:
            annolist = f.readlines()

        self.samplelist = [self._parse(anno) for anno in annolist]

        print("{} || loaded: {}min".\
            format(getTime(), (time.time()-start_time)/60))

    def _parse(self, anno):
        """
        Params:
            anno: {str} 'path label bbox landmark'
        """

        anno = anno.strip().split(' ')
        image = anno[0]

        anno = list(map(float, anno[1: ]))

        if self.save_in_memory:
            image = cv2.imread(image, cv2.IMREAD_ANYCOLOR)

        return image, anno

    def __getitem__(self, index):
        
        image, anno = self.samplelist[index]

        if not self.save_in_memory:
            image = cv2.imread(image, cv2.IMREAD_ANYCOLOR)
        
        image = ToTensor()(image)
        anno = t.from_numpy(np.array(anno))

        return image, anno
    
    def __len__(self):
        
        return len(self.samplelist)

