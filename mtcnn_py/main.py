import os

import torch as t
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from dataset import MtcnnData
from model import model_dict, loss_coef, MtcnnLoss
from trainer import MtcnnTrainer

def train_mtcnn(configer):

    net = configer.net

    model = model_dict[net]()
    trainset = MtcnnData(net, 'train', True)
    validset = MtcnnData(net, 'valid', True)

    coef = loss_coef[net]
    criterion = MtcnnLoss(coef[0], coef[1], coef[2])
    # optimizer = lambda params, lrbase: Adam(params, lrbase)
    optimizer = lambda params, lrbase: SGD(params, lrbase)
    lr_scheduler = lambda optimizer, adjstep, gamma: MultiStepLR(optimizer, adjstep, gamma)

    trainer = MtcnnTrainer(configer, model, trainset, validset, 
                        criterion, optimizer, lr_scheduler, resume=configer.resume)
    trainer.train()




if __name__ == "__main__":
    
    for net in ['PNet', 'RNet', 'ONet']:
        
        configer.net = net
        if configer.net == 'PNet':
            configer.inputsize = (3, 12, 12)
        elif configer.net == 'RNet':
            configer.inputsize = (3, 24, 24)
        elif configer.net == 'ONet':
            configer.inputsize = (3, 48, 48)

        train_mtcnn(configer)

    
