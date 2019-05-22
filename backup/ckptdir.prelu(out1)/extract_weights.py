import os
import torch
from model import *

""" see https://github.com/marvis/pytorch-caffe-darknet-convert """

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

def save_pkl(net, state):
    checkpoint_state = {
        'save_time': 0,

        'cur_epoch': 0,
        'cur_batch': 0,
        'elapsed_time': 0,
        'valid_loss': 0,
            
        'net_state': state,
        'optimizer_state': None,
        'lr_scheduler_state': None,
    }
    checkpoint_path = os.path.join("./ckptdir/{}.pkl".format(net))
    torch.save(checkpoint_state, checkpoint_path)

def extract_pnet():

    prenet = PNet_(is_train=False, use_cuda=False)
    prestate = torch.load("../backup/ckptdir.prelu(out1)/pnet_epoch.pt", map_location='cpu')

    net = PNet()
    state = net.state_dict()

    state['conv1.bias']   = prestate['pre_layer.0.bias']
    state['conv1.weight'] = prestate['pre_layer.0.weight']
    state['prelu1.weight']= prestate['pre_layer.1.weight']

    state['conv2.bias']   = prestate['pre_layer.3.bias']
    state['conv2.weight'] = prestate['pre_layer.3.weight']
    state['prelu2.weight']= prestate['pre_layer.4.weight']

    state['conv3.bias']   = prestate['pre_layer.5.bias']
    state['conv3.weight'] = prestate['pre_layer.5.weight']
    state['prelu3.weight']= prestate['pre_layer.6.weight']

    state['conv4.bias']   = torch.cat([prestate['conv4_1.bias'], prestate['conv4_2.bias'], prestate['conv4_3.bias']], dim=0)
    state['conv4.weight'] = torch.cat([prestate['conv4_1.weight'], prestate['conv4_2.weight'], prestate['conv4_3.weight']], dim=0)

    prenet.load_state_dict(prestate)
    net.load_state_dict(state)

    save_pkl('PNet', state)
    convert2darknet(net, '../mtcnn/weights/PNet.weights')

def extract_rnet():

    prenet = RNet_(is_train=False, use_cuda=False)
    prestate = torch.load("../backup/ckptdir.prelu(out1)/rnet_epoch.pt", map_location='cpu')

    net = RNet()
    state = net.state_dict()

    state['conv1.bias']   = prestate['pre_layer.0.bias']
    state['conv1.weight'] = prestate['pre_layer.0.weight']
    state['prelu1.weight']= prestate['pre_layer.1.weight']

    state['conv2.bias']   = prestate['pre_layer.3.bias']
    state['conv2.weight'] = prestate['pre_layer.3.weight']
    state['prelu2.weight']= prestate['pre_layer.4.weight']

    state['conv3.bias']   = prestate['pre_layer.6.bias']
    state['conv3.weight'] = prestate['pre_layer.6.weight']
    state['prelu3.weight']= prestate['pre_layer.7.weight']

    state['conv4.bias']   = prestate['conv4.bias']
    state['conv4.weight'] = prestate['conv4.weight']
    state['prelu4.weight']= prestate['prelu4.weight']

    state['conv5.bias']   = torch.cat([prestate['conv5_1.bias'], prestate['conv5_2.bias'], prestate['conv5_3.bias']], dim=0)
    state['conv5.weight'] = torch.cat([prestate['conv5_1.weight'], prestate['conv5_2.weight'], prestate['conv5_3.weight']], dim=0)

    prenet.load_state_dict(prestate)
    net.load_state_dict(state)

    save_pkl('RNet', state)
    convert2darknet(net, '../mtcnn/weights/RNet.weights')

def extract_onet():

    prenet = ONet_(is_train=False, use_cuda=False)
    prestate = torch.load("../backup/ckptdir.prelu(out1)/onet_epoch.pt", map_location='cpu')

    net = ONet()
    state = net.state_dict()

    state['conv1.bias']   = prestate['pre_layer.0.bias']
    state['conv1.weight'] = prestate['pre_layer.0.weight']
    state['prelu1.weight']= prestate['pre_layer.1.weight']

    state['conv2.bias']   = prestate['pre_layer.3.bias']
    state['conv2.weight'] = prestate['pre_layer.3.weight']
    state['prelu2.weight']= prestate['pre_layer.4.weight']

    state['conv3.bias']   = prestate['pre_layer.6.bias']
    state['conv3.weight'] = prestate['pre_layer.6.weight']
    state['prelu3.weight']= prestate['pre_layer.7.weight']

    state['conv4.bias']   = prestate['pre_layer.9.bias']
    state['conv4.weight'] = prestate['pre_layer.9.weight']
    state['prelu4.weight']= prestate['pre_layer.10.weight']

    state['conv5.bias']   = prestate['conv5.bias']
    state['conv5.weight'] = prestate['conv5.weight']
    state['prelu5.weight']= prestate['prelu5.weight']

    state['conv6.bias']   = torch.cat([prestate['conv6_1.bias'], prestate['conv6_2.bias'], prestate['conv6_3.bias']], dim=0)
    state['conv6.weight'] = torch.cat([prestate['conv6_1.weight'], prestate['conv6_2.weight'], prestate['conv6_3.weight']], dim=0)

    prenet.load_state_dict(prestate)
    net.load_state_dict(state)

    save_pkl('ONet', state)
    convert2darknet(net, '../mtcnn/weights/ONet.weights')

def main():
    extract_pnet()
    extract_rnet()
    extract_onet()

if __name__ == "__main__":
    main()