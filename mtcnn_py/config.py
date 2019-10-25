from easydict import EasyDict

configer = EasyDict()

configer.logdir = './logs'
configer.ckptdir = './ckptdir'


configer.net = 'PNet'	# choose net to train
configer.resume = False # resume

if configer.net == 'PNet':
    configer.inputsize = (3, 12, 12)
elif configer.net == 'RNet':
    configer.inputsize = (3, 24, 24)
elif configer.net == 'ONet':
    configer.inputsize = (3, 48, 48)


configer.batchsize = 2**10
configer.n_epoch = 1000

configer.lrbase = 1e-6
configer.adjstep = [750, 900]
configer.gamma = 1e-3

configer.cuda = True

