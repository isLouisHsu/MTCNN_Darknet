# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-10 11:50:12
@LastEditTime: 2019-11-10 11:52:22
@Update: 
'''
import os
import torch
from model import *

""" see https://github.com/marvis/pytorch-caffe-darknet-convert """

# --------------------------------------------------------------------
def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)

def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def save_prelu(fp, prelu_model):
    prelu_model.weight.data.numpy().tofile(fp)

# --------------------------------------------------------------------
def convert2darknet(model, filename):
    """
    Params:
        model: pytorch model
        filename: {str} weight file
    """
    fp = open(filename, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)

    for layer in model.children():
        if type(layer) == torch.nn.Conv2d:
            print(layer)
            save_conv(fp, layer)
        elif type(layer) == torch.nn.Linear:
            print(layer)
            save_fc(fp, layer)
        elif type(layer) == torch.nn.PReLU:
            print(layer)
            save_prelu(fp, layer)
    fp.close()

# --------------------------------------------------------------------
def extract_pnet():

    net = PNet()
    state = torch.load("./ckptdir/PNet.pkl", map_location='cpu')['net_state']
    net.load_state_dict(state)
    convert2darknet(net, '../mtcnn_c/weights/PNet.weights')

def extract_rnet():

    net = RNet()
    state = torch.load("./ckptdir/RNet.pkl", map_location='cpu')['net_state']
    net.load_state_dict(state)
    convert2darknet(net, '../mtcnn_c/weights/RNet.weights')

def extract_onet():

    net = ONet()
    state = torch.load("./ckptdir/ONet.pkl", map_location='cpu')['net_state']
    net.load_state_dict(state)
    convert2darknet(net, '../mtcnn_c/weights/ONet.weights')

# --------------------------------------------------------------------
def main():
    extract_pnet()
    extract_rnet()
    extract_onet()

if __name__ == "__main__":
    main()