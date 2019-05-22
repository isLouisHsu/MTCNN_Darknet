import torch
import torch.cuda as cuda
import torch.nn as nn
from utiles import finetune

class PNet(nn.Module):

    def __init__(self, pretrained=True):
        super(PNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 10, 3, 1)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 16, 3, 1)
        self.prelu2 = nn.PReLU(16)

        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.prelu3 = nn.PReLU(32)

        self.conv4 = nn.Conv2d(32, 15, 1, 1)

        if pretrained:
            pretrained = torch.load('./pretrained/weights/PNet_features.pkl')
            totrain = finetune(pretrained, self.state_dict())
            self.load_state_dict(totrain)

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

    def __init__(self, pretrained=True):
        super(RNet, self).__init__()

        self.conv1 = nn.Conv2d( 3, 28, 3, 1)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(28, 48, 3, 1)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(48, 64, 2, 1)
        self.prelu3 = nn.PReLU(64)

        self.conv4 = nn.Linear(64*2*2, 128)
        self.prelu4 = nn.PReLU(128)

        self.conv5 = nn.Linear(128,  15)

        if pretrained:
            pretrained = torch.load('./pretrained/weights/RNet_features.pkl')
            totrain = finetune(pretrained, self.state_dict())
            self.load_state_dict(totrain)

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

    def __init__(self, pretrained=True):
        super(ONet, self).__init__()

        self.conv1 = nn.Conv2d( 3,  32, 3, 1)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(32,  64, 3, 1)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(64,  64, 3, 1)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 2, 1)
        self.prelu4 = nn.PReLU(128)

        self.conv5 = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU(256)

        self.conv6 = nn.Linear(256,  15)

        if pretrained:
            pretrained = torch.load('./pretrained/weights/ONet_features.pkl')
            totrain = finetune(pretrained, self.state_dict())
            self.load_state_dict(totrain)

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

if __name__ == "__main__":
    
    from torchstat import stat
    
    stat(PNet(False), (3, 16, 16))
    stat(RNet(False), (3, 24, 24))
    stat(ONet(False), (3, 48, 48))

    PNet()
    RNet()
    ONet()
