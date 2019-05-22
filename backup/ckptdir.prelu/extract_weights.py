import os
import torch
from model import model_dict
from config import configer

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
    fp.close()

def extract_mtcnn(net):
    """
    Params:
        net:    {str}
    """

    model = model_dict[net]()
    checkpoint_path = os.path.join(configer.ckptdir, "{}.pkl".format(net))
    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_state['net_state'])

    weigths_path = "../mtcnn/weights/{}.weights".format(model._get_name())
    convert2darknet(model, weigths_path)


def main():
    ## extract weights
    extract_mtcnn("PNet")
    extract_mtcnn("RNet")
    extract_mtcnn("ONet")

if __name__ == "__main__":
    main()