import os
import time
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from processbar import ProcessBar
from utiles import getTime

def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple) or isinstance(input_size, list):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = (total_params_size + total_output_size + total_input_size) / 1024

    print("================================================================")
    print("Total params:                    {0:,}".format(total_params))
    print("Trainable params:                {0:,}".format(trainable_params))
    print("Non-trainable params:            {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB):                 %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB):                %0.2f" % total_params_size)
    print("Estimated Total Size (GB):       %0.2f" % total_size)
    print("----------------------------------------------------------------")
# return summary

class MtcnnTrainer(object):
    """ Train 
    """

    def __init__(self, configer, net, trainset, validset, criterion, optimizer, lr_scheduler, resume=False):

        self.configer = configer

        self.net = net
        if configer.cuda and cuda.is_available(): self.net.cuda()
        
        ## directory for log and checkpoints
        self.logdir = os.path.join(configer.logdir, self.net._get_name())
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = configer.ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        
        ## datasets
        self.trainset = trainset
        self.validset = validset
        self.trainloader = DataLoader(trainset, configer.batchsize, True)
        self.validloader = DataLoader(validset, configer.batchsize, True)

        ## for optimization
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters(), configer.lrbase)
        self.lr_scheduler = lr_scheduler(self.optimizer, configer.adjstep, configer.gamma)
        self.writer = SummaryWriter(configer.logdir)

        dummy_input = torch.rand([1] + list(configer.inputsize))
        if configer.cuda and cuda.is_available(): dummy_input = dummy_input.cuda()
        # self.writer.add_graph(self.net, dummy_input)

        ## initialize
        self.valid_loss = float('inf')
        self.elapsed_time = 0
        self.cur_epoch = 0
        self.cur_batch = 0

        ## if resume
        if resume:
            self.load_checkpoint()

        ## print information
        # summary(self.net, configer.inputsize, configer.batchsize, device="cuda" if cuda.is_available() else "cpu")

        print("==============================================================================================")
        print("model:           {}".format(self.net._get_name()))
        print("logdir:          {}".format(self.logdir))
        print("ckptdir:         {}".format(self.ckptdir))
        print("train samples:   {}".format(len(trainset)/1000))
        print("valid samples:   {}".format(len(validset)/1000))
        print("batch size:      {}".format(configer.batchsize))
        print("batch per epoch: {}".format(len(trainset)/configer.batchsize))
        print("epoch:           [{:4d}]/[{:4d}]".format(self.cur_epoch, configer.n_epoch))
        print("learing rate:    {}".format(configer.lrbase))
        print("==============================================================================================")
        


    def train(self):
        
        n_epoch = self.configer.n_epoch - self.cur_epoch
        print("Start training! current epoch: {}, remain epoch: {}".format(self.cur_epoch, n_epoch))

        bar = ProcessBar(n_epoch)

        for i_epoch in range(n_epoch):

            if self.configer.cuda and cuda.is_available(): cuda.empty_cache()

            self.cur_epoch += 1
            bar.step(self.cur_epoch)

            self.lr_scheduler.step(self.cur_epoch)
            cur_lr = self.lr_scheduler.get_lr()[-1]
            self.writer.add_scalar('{}/lr'.format(self.net._get_name()), cur_lr, self.cur_epoch)

            loss_train = self.train_epoch()
            # print("----------------------------------------------------------------------------------------------")
            loss_valid = self.valid_epoch()
            # print("----------------------------------------------------------------------------------------------")

            self.writer.add_scalars('{}/loss'.format(self.net._get_name()), 
                                {'train': loss_train, 'valid': loss_valid}, self.cur_epoch)

            # print_log = "{} || Elapsed: {:.4f}h || Epoch: [{:3d}]/[{:3d}] || lr: {:.6f},| train loss: {:4.4f}, valid loss: {:4.4f}".\
            #         format(getTime(), self.elapsed_time/3600, self.cur_epoch, self.configer.n_epoch, 
            #             cur_lr, loss_train, loss_valid)
            # print(print_log)
            
            if loss_valid < self.valid_loss:
                self.valid_loss = loss_valid
                self.save_checkpoint()
                
            # print("==============================================================================================")


    def train_epoch(self):
        
        self.net.train()
        avg_loss = []
        start_time = time.time()
        n_batch = len(self.trainset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.trainloader):

            self.cur_batch += 1

            X = Variable(X.float()); y = Variable(y.float())
            if self.configer.cuda and cuda.is_available(): 
                X = X.cuda(); y = y.cuda()
            
            y_pred = self.net(X)
            total_i, cls_i, bbox_i, landmark_i = self.criterion(y_pred, y)

            self.optimizer.zero_grad()
            total_i.backward()
            self.optimizer.step()

            avg_loss += [total_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/train/total_i'.format(self.net._get_name()), total_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/train/cls_i'.format(self.net._get_name()), cls_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/train/bbox_i'.format(self.net._get_name()), bbox_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/train/landmark_i'.format(self.net._get_name()), landmark_i, self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()
            self.elapsed_time += duration_time
            # total_time = duration_time * self.configer.n_epoch * len(self.trainset) // self.configer.batchsize
            # left_time = total_time - self.elapsed_time

            # print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] | cur: [{:3d}] || lr: {:.6f}, loss: {:4.4f}".\
            #     format(getTime(), self.elapsed_time/3600, left_time/3600, self.configer.batchsize / duration_time,
            #         self.cur_epoch, self.configer.n_epoch, i_batch, n_batch, self.cur_batch,
            #         self.lr_scheduler.get_lr()[-1], total_i
            #     )
            # print(print_log)
        
        avg_loss = np.mean(np.array(avg_loss))
        return avg_loss


    def valid_epoch(self):
        
        self.net.eval()
        avg_loss = []
        start_time = time.time()
        n_batch = len(self.validset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.validloader):

            X = Variable(X.float()); y = Variable(y.float())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            y_pred = self.net(X)
            total_i, cls_i, bbox_i, landmark_i = self.criterion(y_pred, y)

            avg_loss += [total_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/valid/total_i'.format(self.net._get_name()), total_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/valid/cls_i'.format(self.net._get_name()), cls_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/valid/bbox_i'.format(self.net._get_name()), bbox_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/valid/landmark_i'.format(self.net._get_name()), landmark_i, self.cur_epoch*n_batch + i_batch)

            # duration_time = time.time() - start_time
            # start_time = time.time()

            # print_log = "{} || FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || loss: {:4.4f}".\
            #     format(getTime(), self.configer.batchsize / duration_time,
            #         self.cur_epoch, self.configer.n_epoch, i_batch, n_batch, total_i
            #     )
            # print(print_log)
        
        avg_loss = np.mean(np.array(avg_loss))
        return avg_loss
    

    def save_checkpoint(self):
        
        checkpoint_state = {
            'save_time': getTime(),

            'cur_epoch': self.cur_epoch,
            'cur_batch': self.cur_batch,
            'elapsed_time': self.elapsed_time,
            'valid_loss': self.valid_loss,
            
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
        }

        checkpoint_path = os.path.join(self.ckptdir, "{}.pkl".format(self.net._get_name()))
        torch.save(checkpoint_state, checkpoint_path)

        # print("checkpoint saved at {}".format(checkpoint_path))


    def load_checkpoint(self):
        
        checkpoint_path = os.path.join(self.ckptdir, "{}.pkl".format(self.net._get_name()))
        checkpoint_state = torch.load(checkpoint_path)
        
        self.cur_epoch = checkpoint_state['cur_epoch']
        self.cur_batch = checkpoint_state['cur_batch']
        self.elapsed_time = checkpoint_state['elapsed_time']
        self.valid_loss = checkpoint_state['valid_loss']

        self.net.load_state_dict(checkpoint_state['net_state'])
        self.optimizer.load_state_dict(checkpoint_state['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler_state'])

        # print("load checkpoint from {}, last save time: {}".\
        #                         format(checkpoint_path, checkpoint_state['save_time']))

