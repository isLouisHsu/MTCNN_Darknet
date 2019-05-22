import os
import cv2
import numpy as np

from torchstat import stat
import torch
from torchvision.transforms import ToTensor

from model import PNet, RNet, ONet
from utiles import *

sigmoid = lambda x: 1 / (1 + np.e**(-x))

class MtcnnDetector(object):
    """ mtcnn detector

    Params:
        prefix: {str} checkpoint
    Attributes:

    Content:

    """
    def __init__(self, min_face=20, thresh=[0.6, 0.7, 0.7], scale=0.79, stride=2, cellsize=12):
        
        self.min_face = min_face
        self.thresh = thresh
        self.scale  = scale
        self.stride = stride
        self.cellsize = cellsize

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        
        self._load_state(self.pnet)
        self._load_state(self.rnet)
        self._load_state(self.onet)

        # stat(self.pnet, (3, 12, 12))
        # stat(self.rnet, (3, 24, 24))
        # stat(self.onet, (3, 48, 48))

    def _load_state(self, net):
        
        ckpt = './ckptdir/{}.pkl'.format(net._get_name())
        if not os.path.exists(ckpt): return
        ckpt = torch.load(ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(ckpt['net_state'])
    
    def detect_image(self, image):
        """ Detect face over single image
        Params:
            image:    {ndarray(H, W, C)}
        """

        boxes, boxes_c, _ = self._detect_pnet(image)

    def _detect_pnet(self, image):
        """
        Params:
            image:      {ndarray(1, C, H, W)}
        Returns:
            boxes:    {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            boxes_c:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            landmark: None
        """
        NETSIZE = 12

        def _resize_image(image, scale):
            """ resize image according to scale
            Params:
                image:  {ndarray(h, w, c)}
                scale:  {float}
            """
            h, w, c = image.shape
            hn = int(h*scale); wn = int(w*scale)
            resized = cv2.resize(image, (wn, hn), interpolation=cv2.INTER_LINEAR)
            return resized
        
        def _generate_box(cls_map, reg_map, thresh, scale):
            """ generate boxes
            Params:
                cls_map: {ndarray(h, w)}
                reg_map: {ndarray(4, h, w)}
                thresh:  {float}
                scale:   {float}
            Returns:
                bboxes:  {ndarray(n_boxes, 9)} x1, y1, x2, y2, score, offsetx1, offsety1, offsetx2, offsety2
            """
            idx = np.where(cls_map>thresh)

            if idx[0].size == 0:
                return np.array([])

            x1 = np.round(self.stride * idx[1] / scale)
            y1 = np.round(self.stride * idx[0] / scale)
            x2 = np.round((self.stride * idx[1] + self.cellsize) / scale)
            y2 = np.round((self.stride * idx[0] + self.cellsize) / scale)

            # print("current scale: {} current size: {}".format(scale, self.cellsize/scale))

            score = cls_map[idx[0], idx[1]]
            reg = np.array([reg_map[i, idx[0], idx[1]] for i in range(4)])

            boxes = np.vstack([x1, y1, x2, y2 ,score, reg]).T

            return boxes

        # ======================= generate boxes ===========================
        cur_scale = NETSIZE / self.min_face
        cur_img = _resize_image(image, cur_scale)
        all_boxes = None

        while min(cur_img.shape[:-1]) >= NETSIZE:

            ## forward network
            X = ToTensor()(cur_img).unsqueeze(0)
            y_pred = self.pnet(X)[0].detach().numpy()

            ## generate bbox
            cls_map = sigmoid(y_pred[0,:,:])
            reg_map = y_pred[1:5,:,:]
            boxes = _generate_box(cls_map, reg_map, self.thresh[0], cur_scale)

            ## update scale
            cur_scale *= self.scale
            cur_img = _resize_image(image, cur_scale)
            if boxes.size == 0: continue
            
            ## nms
            # boxes = boxes[self._nms(boxes[:, :5], 0.6, 'Union')]
            # show_bbox(image.copy(), boxes[:, :5])

            ## save bbox
            if all_boxes is None:
                all_boxes = boxes
            else:
                all_boxes = np.concatenate([all_boxes, boxes], axis=0)

        # ====================================================================

        if all_boxes is None: 
            return np.array([]), np.array([]), None

        ## nms
        all_boxes = all_boxes[self._nms(all_boxes[:, 0:5], 0.6, 'Union')]

        ## parse
        boxes  = all_boxes[:, :4]                   # (n_boxes, 4)
        score  = all_boxes[:,  4].reshape((-1, 1))  # (n_boxes, 1)
        offset = all_boxes[:, 5:]                   # (n_boxes, 4)
        
        # refine bbox
        boxes_c = self._cal_box(boxes, offset)
        
        ## concat
        boxes = np.concatenate([boxes, score], axis=1)
        boxes_c = np.concatenate([boxes_c, score], axis=1)

        ## landmark
        landmark = None

        return boxes, boxes_c, landmark

    def _detect_rnet(self, image, bboxes):
        """
        Params:
            image: {ndarray(H, W, C)}
            bboxes:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
        Returns:
            boxes:    {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            boxes_c:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            landmark: None
        """
        NETSIZE = 24

        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), None

        bboxes = self._square(bboxes)
        patches = self._crop_patch(image, bboxes, NETSIZE)
        
        ## forward network
        X = torch.cat(list(map(lambda x: ToTensor()(x).unsqueeze(0), patches)), dim=0)
        y_pred = self.rnet(X).detach().numpy()  # (n_boxes, 15)
        scores = sigmoid(y_pred[:, 0])          # (n_boxes,)
        offset = y_pred[:, 1: 5]                # (n_boxes, 4)
        landmark = y_pred[:, 5:]                # (n_boxes, 10)

        ## update score
        bboxes[:, -1] = scores

        ## filter
        idx = scores > self.thresh[1]
        bboxes = bboxes[idx]                        # (n_boxes, 5)
        offset = offset[idx]                        # (n_boxes, 4)
        landmark = landmark[idx]                    # (n_boxes, 10)
        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), None

        ## nms
        idx = self._nms(bboxes, 0.5)
        bboxes = bboxes[idx]
        offset = offset[idx]
        landmark = landmark[idx]

        ## landmark
        landmark = self._cal_landmark(bboxes[:, :-1], landmark)

        bboxes_c = self._cal_box(bboxes[:,:-1], offset)
        bboxes_c = np.concatenate([bboxes_c, bboxes[:, -1].reshape((-1, 1))], axis=1)

        return bboxes, bboxes_c, landmark
    
    def _detect_onet(self, image, bboxes):
        """
        Params:
            image: {ndarray(H, W, C)}
            bboxes:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
        Returns:
            boxes:    {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            boxes_c:  {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
            landmark: None
        """
        NETSIZE = 48

        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        bboxes = self._square(bboxes)
        patches = self._crop_patch(image, bboxes, NETSIZE)
        
        ## forward network
        X = torch.cat(list(map(lambda x: ToTensor()(x).unsqueeze(0), patches)), dim=0)
        y_pred = self.onet(X).detach().numpy()  # (n_boxes, 15)
        scores = sigmoid(y_pred[:, 0])          # (n_boxes,)
        offset = y_pred[:, 1: 5]                # (n_boxes, 4)
        landmark = y_pred[:, 5:]                # (n_boxes, 10)
        
        ## update score
        bboxes[:, -1] = scores

        ## filter
        idx = scores > self.thresh[2]
        bboxes = bboxes[idx]                        # (n_boxes, 5)
        offset = offset[idx]                        # (n_boxes, 4)
        landmark = landmark[idx]                    # (n_boxes, 10)
        if bboxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        ## nms
        idx = self._nms(bboxes, 0.5, mode='Minimum')
        bboxes = bboxes[idx]
        offset = offset[idx]
        landmark = landmark[idx]
        
        ## landmark
        landmark = self._cal_landmark(bboxes[:, :-1], landmark)

        bboxes_c = self._cal_box(bboxes[:,:-1], offset)
        bboxes_c = np.concatenate([bboxes_c, bboxes[:, -1].reshape((-1, 1))], axis=1)

        return bboxes, bboxes_c, landmark

    @classmethod
    def _cal_box(self, boxes, offset):
        """ refine boxes
        Params:
            boxes:  {ndarray(n_boxes, 4)} unrefined boxes
            offset: {ndarray(n_boxes, 4)} boxes offset
        Returns:
            boxes_c:{ndarray(n_boxes, 4)} refined boxes
        Notes:
            offset = (gt - square) / size of square box
             => gt = square + offset * size of square box (*)
            where
                - `offset`, `gt`, `square` are ndarrays
                - `size of square box` is a number
        """
        ## square boxes' heights and widths
        x1, y1, x2, y2 = np.hsplit(boxes, 4)        # (n_boxes, 1)
        w = x2 - x1 + 1; h = y2 - y1 + 1            # (n_boxes, 1)
        bsize = np.hstack([w, h]*2)                 # (n_boxes, 4)
        bbase = np.hstack([x1, y1, x2, y2])         # (n_boxes, 4)
        ## refine
        boxes_c = bbase + offset*bsize
        return boxes_c
    
    @classmethod
    def _cal_landmark(self, boxes, offset):
        """ calculate landmark
        Params:
            boxes:  {ndarray(n_boxes,  4)} unrefined boxes
            offset: {ndarray(n_boxes, 10)} landmark offset
        Returns:
            landmark:{ndarray(n_boxes, 10)} landmark location
        Notes:
            offset_x = (gt_x - square_x1) / size of square box
             => gt_x = square_x1 + offset_x * size of square box (*)
            offset_y = (gt_y - square_y1) / size of square box
             => gt_y = square_y1 + offset_y * size of square box (*)
            where
                - `offset_{}`, `gt_{}`, `square_{}1` are ndarrays
                - `size of square box` is a number
        """
        ## square boxes' heights and widths
        x1, y1, x2, y2 = np.hsplit(boxes, 4)        # (n_boxes, 1)
        w = x2 - x1 +1; h = y2 - y1 + 1             # (n_boxes, 1)
        bsize = np.hstack([w, h]*5)                 # (n_boxes, 10)
        bbase = np.hstack([x1, y1]*5)               # (n_boxes, 10)
        ## refine
        landmark = bbase + offset*bsize
        return landmark

    @classmethod
    def _nms(self, dets, thresh, mode="Union"):
        """
        Params:
            dets:   {ndarray(n_boxes, 5)} x1, y1, x2, y2 score
            thresh: {float} retain overlap <= thresh
            mode:   {str} 'Union' or 'Minimum'
        Returns:
            idx:   {list[int]} indexes to keep
        Notes:
            greedily select boxes with high confidence
            idx boxes overlap <= thresh
            rule out overlap > thresh

            if thresh==1.0, keep all
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        idx = []
        while order.size > 0:
            i = order[0]
            idx.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            if mode == "Union":
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == "Minimum":
                ovr = inter / np.minimum(areas[i], areas[order[1:]])

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return idx
    
    @classmethod
    def _square(self, bbox):
        """ convert rectangle bbox to square bbox
        Params:
            bbox: {ndarray(n_boxes, 5)}
        Returns:
            bbox_s: {ndarray(n_boxes, 5)}
        """
        ## rectangle boxes' heights and widths
        x1, y1, x2, y2, score = np.hsplit(bbox, 5)  # (n_boxes, 1)
        w = x2 - x1 +1; h = y2 - y1 + 1             # (n_boxes, 1)
        maxsize = np.maximum(w, h)                  # (n_boxes, 1)

        ## square boxes' heights and widths
        x1 = x1 + w/2 - maxsize/2
        y1 = y1 + h/2 - maxsize/2
        x2 = x1 + maxsize - 1
        y2 = y1 + maxsize - 1

        bbox_s = np.hstack([x1, y1, x2, y2, score])
        return bbox_s

    @classmethod
    def _crop_patch(self, image, bbox_s, size):
        """ crop patches from image
        Params:
            image: {ndarray(H, W, C)}
            bbox_s: {ndarray(n_boxes, 5)} squared bbox
        Returns:
            patches: {list[ndarray(h, w, c)]}
        """

        def locate(bbox, imh, imw):
            """ 
            Params:
                bbox:       {ndarray(n_boxes, 5)} x1, y1, x2, y2, score
                imh, imw:   {float} size of input image
            Returns:
                oriloc, dstloc: {ndarray(n_boxes, 4)} x1, y1, x2, y2
            """
            ## origin boxes' heights and widths
            x1, y1, x2, y2, score = np.hsplit(bbox_s, 5)# (n_boxes, 1)
            x1, y1, x2, y2 = list(map(lambda x: x.astype('int').reshape(-1), [x1, y1, x2, y2]))
            w = x2 - x1 + 1; h = y2 - y1 + 1            # (n_boxes, 1)

            ## destinate boxes
            xx1 = np.zeros_like(x1)
            yy1 = np.zeros_like(y1)
            xx2 = w.copy() - 1
            yy2 = h.copy() - 1

            ## left side out of image
            i = x1 < 0
            xx1[i] = 0 + (0 - x1[i])
            x1 [i] = 0
            ## top side out of image
            i = y1 < 0
            yy1[i] = 0 + (0 - y1[i])
            y1 [i] = 0
            ## right side out of image
            i = x2 > imw - 1
            xx2[i] = (w[i]-1) + (imw-1 - x2[i])
            x2 [i] = imw - 1
            ## bottom side out of image
            i = y2 > imh - 1
            yy2[i] = (h[i]-1) + (imh-1 - y2[i])
            y2 [i] = imh - 1

            return [x1, y1, x2, y2, xx1, yy1, xx2, yy2]

        imh, imw, _ = image.shape
        n_boxes = bbox_s.shape[0]

        x1, y1, x2, y2, score = np.hsplit(bbox_s, 5)    
        pw = x2 - x1 +1; ph = y2 - y1 + 1
        pshape = np.hstack([ph, pw, 3*np.ones(shape=(score.shape[0], 1))]).astype('int')   # (n_boxes, 2)

        x1, y1, x2, y2, xx1, yy1, xx2, yy2 = locate(bbox_s, imh, imw) # (n_boxes, 1)

        patches = []
        for i_boxes in range(n_boxes):
            patch = np.zeros(shape=pshape[i_boxes], dtype='uint8')
            patch[yy1[i_boxes]: yy2[i_boxes], xx1[i_boxes]: xx2[i_boxes]] = image[y1[i_boxes]: y2[i_boxes], x1[i_boxes]: x2[i_boxes]]
            patch = cv2.resize(patch, (size, size))
            patches += [patch]
        
        return patches


def show_bbox(image, bbox, landmark=None, show_score=False):
    """
    Params: 
        image:  {ndarray(H, W, C)}
        bbox:   {ndarray(n_box, 5)} x1, y1, x2, y2, score
    """
    n_box = bbox.shape[0]
    for i_box in range(n_box):
        score = str(bbox[i_box, -1])
        x1, y1, x2, y2 = bbox[i_box, :-1].astype(np.int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255))
        if show_score:
            cv2.putText(image, str(score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if landmark is not None:
        for i_land in range(landmark.shape[0]):
            land = landmark[i_land].reshape((5, -1)).astype(np.int)
            for i_pt in range(land.shape[0]):
                cv2.circle(image, tuple(land[i_pt]), 1, (255, 255, 255), 2)

    cv2.imshow("", image)
    cv2.waitKey(0)


def test_24(net):
    
    FILE = "/home/louishsu/Desktop/patches/24/{:d}.jpg"
    
    outs = []
    for i in range(184):
        file = FILE.format(i)
        img = ToTensor()(cv2.imread(file)).unsqueeze(0)
        out = ' '.join(map(str, list(net(img).squeeze().detach().numpy()))) + '\n'
        outs += [out]

    with open("/home/louishsu/Desktop/patches/p_24.txt", 'w') as f:
        f.writelines(outs)


def test_48(net):
    
    FILE = "/home/louishsu/Desktop/patches/48/{:d}.jpg"
    
    outs = []
    for i in range(1):
        file = FILE.format(i)
        img = ToTensor()(cv2.imread(file)).unsqueeze(0)
        out = ' '.join(map(str, list(net(img).squeeze().detach().numpy()))) + '\n'
        outs += [out]

    with open("/home/louishsu/Desktop/patches/p_48.txt", 'w') as f:
        f.writelines(outs)

def test():
    detector = MtcnnDetector(min_face=12, thresh=[0.9, 0.7, 0.7], scale=0.79, stride=2, cellsize=12)

    test_24(detector.rnet)
    test_48(detector.onet)

def main():

    import sys
    
    detector = MtcnnDetector(min_face=12, thresh=[0.8, 0.6, 0.7], scale=0.79, stride=2, cellsize=12)
    
    imgfile = sys.argv[1]
    # imgfile = "../images/test.jpg"

    image = cv2.imread(imgfile)
    
    boxes, boxes_c, landmark = detector._detect_pnet(image)
    show_bbox(image.copy(), boxes, None, False)
    show_bbox(image.copy(), boxes_c, landmark, False)

    boxes, boxes_c, landmark = detector._detect_rnet(image, boxes_c)
    show_bbox(image.copy(), boxes, None, True)
    show_bbox(image.copy(), boxes_c, landmark, True)

    boxes, boxes_c, landmark = detector._detect_onet(image, boxes_c)
    show_bbox(image.copy(), boxes, None, True)
    show_bbox(image.copy(), boxes_c, landmark, True)
    
if __name__ == "__main__":
    # test()
    main()