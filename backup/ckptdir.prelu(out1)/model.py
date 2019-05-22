import torch
import torch.cuda as cuda
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class PNet_(nn.Module):
    ''' PNet '''

    def __init__(self, is_train=False, use_cuda=True):
        super(PNet_, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()  # PReLU3
        )
        # detection
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

        # weight initiation with xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        label = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        # landmark = self.conv4_3(x)

        if self.is_train is True:
            # label_loss = LossUtil.label_loss(self.gt_label,torch.squeeze(label))
            # bbox_loss = LossUtil.bbox_loss(self.gt_bbox,torch.squeeze(offset))
            return label, offset
        #landmark = self.conv4_3(x)
        return label, offset


class RNet_(nn.Module):
    ''' RNet '''

    def __init__(self, is_train=False, use_cuda=True):
        super(RNet_, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()  # prelu3

        )
        self.conv4 = nn.Linear(64 * 2 * 2, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        # landmark = self.conv5_3(x)

        if self.is_train is True:
            return det, box
        #landmard = self.conv5_3(x)
        return det, box


class ONet_(nn.Module):
    ''' RNet '''

    def __init__(self, is_train=False, use_cuda=True):
        super(ONet_, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Linear(128 * 2 * 2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Linear(256, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        det = torch.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        if self.is_train is True:
            return det, box, landmark
        #landmard = self.conv5_3(x)
        return det, box, landmark









class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 10, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 16, 3, 1)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(32, 15, 1, 1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.prelu3(x)
        
        x = self.conv4(x)

        return x


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 28, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(28, 48, 3, 1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(48, 64, 2, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Linear(64*2*2, 128)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Linear(128,  15)

    def forward(self, x):

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.prelu3(x)

        x = x.view(x.shape[0], -1)
        
        x = self.conv4(x)
        x = self.prelu4(x)

        x = self.conv5(x)

        return x


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.conv1 = nn.Conv2d( 3,  32, 3, 1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(32,  64, 3, 1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(64,  64, 3, 1)
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 2, 1)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Linear(256,  15)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.prelu4(x)

        x = x.view(x.shape[0], -1)
        
        x = self.conv5(x)
        x = self.prelu5(x)

        x = self.conv6(x)

        return x


model_dict = {
	'PNet': PNet,
	'RNet': RNet,
	'ONet': ONet,
}

class MtcnnLoss(nn.Module):

    def __init__(self, cls, bbox, landmark, ohem=0.7):
        super(MtcnnLoss, self).__init__()

        self.cls = cls
        self.bbox = bbox
        self.landmark = landmark
        self.ohem = ohem
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, gt):
        """
        Params:
            pred:   {tensor(N, n) or tensor(N, n, 1, 1)}
            gt:     {tensor(N, n)}
        
        Notes:
            y_true
        """
        N = pred.shape[0]
        pred = pred.view(N, -1)

        ## origin label
        gt_labels = gt[:, 0]

        ## pos -> 1, neg -> 0, others -> 0
        pred_cls = pred[:, 0]
        gt_cls = gt_labels.clone(); gt_cls[gt_labels!=1.0] = 0.0
        loss_cls = self.bce(pred_cls, gt_cls)
        # ohem
        n_keep = int(self.ohem * loss_cls.shape[0])
        loss_cls = torch.mean(torch.topk(loss_cls, n_keep)[0])

        ## label=1 or label=-1 then do regression
        idx = (gt_labels==1)^(gt_labels==-1)
        pred_bbox = pred[idx, 1: 5]
        gt_bbox = gt[idx, 1: 5]
        loss_bbox = self.mse(pred_bbox, gt_bbox)
        loss_bbox = torch.mean(loss_bbox, dim=1)
        # ohem
        n_keep = int(self.ohem * loss_bbox.shape[0])
        loss_bbox = torch.mean(torch.topk(loss_bbox, n_keep)[0])    

        ## keep label =-2  then do landmark detection
        idx = gt_labels==-2
        pred_landmark = pred[idx, 5:]
        gt_landmark = gt[idx, 5:]
        loss_landmark = self.mse(pred_landmark, gt_landmark)
        loss_landmark = torch.mean(loss_landmark, dim=1)
        # ohem
        n_keep = int(self.ohem * loss_landmark.shape[0])
        loss_landmark = torch.mean(torch.topk(loss_landmark, n_keep)[0])

        ## total loss
        loss_total = self.cls*loss_cls + self.bbox*loss_bbox + self.landmark*loss_landmark

        return loss_total, loss_cls, loss_bbox, loss_landmark


loss_coef = {
    'PNet': [1.0, 0.5, 0.5],
    'RNet': [1.0, 0.5, 0.5],
    'ONet': [1.0, 0.5, 1.0],
}
