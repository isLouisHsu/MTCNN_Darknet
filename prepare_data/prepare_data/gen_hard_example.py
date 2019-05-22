#coding:utf-8
import sys
#sys.path.append("../")
from prepare_data.utils import convert_to_square

sys.path.insert(0,'..')
import numpy as np
import argparse
import os
import pickle as pickle
import cv2
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from train_models.MTCNN_config import config
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from prepare_data.utils import *
from prepare_data.data_utils import *

#net : 24(RNet)/48(ONet)
#data: dict()
def save_hard_example(net, data_dir):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    if net == "RNet":
        image_size = 24
    if net == "ONet":
        image_size = 48

    data = read_annotation(data_dir, './prepare_data/wider_face_train_bbx_gt.txt')
    im_idx_list = data['images']
    # print(images[0])
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    print("processing %d images in total" % num_of_images)

    
    # save files
    label_file_dir = data_dir + "/%d" % image_size
    if not os.path.exists(label_file_dir): os.mkdir(label_file_dir)
    neg_dir  = data_dir + "/%d/negative" % (image_size)
    if not os.path.exists(neg_dir): os.mkdir(neg_dir)
    pos_dir  = data_dir + "/%d/positive" % (image_size)
    if not os.path.exists(pos_dir): os.mkdir(pos_dir)
    part_dir = data_dir + "/%d/part"     % (image_size)
    if not os.path.exists(part_dir): os.mkdir(part_dir)
    neg_label_file  = data_dir + "/%d/neg_%d.txt"   % (image_size, image_size)
    pos_label_file  = data_dir + "/%d/pos_%d.txt"   % (image_size, image_size)
    part_label_file = data_dir + "/%d/part_%d.txt"  % (image_size, image_size)
    neg_file = open(neg_label_file, 'w')
    part_file = open(part_label_file, 'w')
    pos_file = open(pos_label_file, 'w')

    #read detect result
    det_boxes = pickle.load(open(data_dir + '/%s/detections.pkl' % net, 'rb'))
    # print(len(det_boxes), num_of_images)
    print(len(det_boxes))
    print(num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    #im_idx_list image index(list)
    #det_boxes detect result(list)
    #gt_boxes_list gt(list)
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        #change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3            
            if np.max(Iou) < 0.3 and neg_num < 60:
                #save the examples
                save_file = neg_dir + "/%s.jpg" % n_idx
                # print(save_file)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = pos_dir + "/%s.jpg" % p_idx
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = part_dir + "/%s.jpg" % d_idx
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


def t_net(data_dir,
            prefix=['./modelfiles/PNet/PNet', 
                    './modelfiles/RNet/RNet', 
                    './modelfiles/ONet/ONet'], 
            epoch=[18, 14, 16], batch_size=[2048, 256, 16], 
            test_mode="PNet", thresh=[0.6, 0.6, 0.7], min_face_size=25,
            stride=2, slide_window=False, shuffle=False, vis=False):

    detectors = [None, None, None]
    print("Test model: ", test_mode)
    #PNet-echo
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    print(model_path[0])
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        print("==================================", test_mode)
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet
    # load onet model
    if test_mode == "ONet":
        print("==================================", test_mode)
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet
        
    # anno_file
    # read anotation(type:dict), include 'images' and 'bboxes'
    data = read_annotation(data_dir, './prepare_data/wider_face_train_bbx_gt.txt')
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    print("==================================")
    print('load test data')
    test_data = TestLoader(data['images'])
    print ('finish loading')
    
    save_net = 'RNet'
    if test_mode == "PNet":
        save_net = "RNet"
        image_size = 24
    elif test_mode == "RNet":
        save_net = "ONet"
        image_size = 48
    
    # save detect result
    save_path = os.path.join(data_dir, save_net)
    print ('save_path is :')
    print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, "detections.pkl")
    
    print ('start detecting....')
    if not os.path.exists(save_file):
        detections, _ = mtcnn_detector.detect_face(test_data)
        with open(save_file, 'wb') as f:
            pickle.dump(detections, f,1)
    print ('finish detecting ')

    print("%s测试完成开始OHEM" % image_size)


def gen_hard_example(data_dir, net):
    if net == "RNet":
        test_net = "PNet"
    elif net == "ONet":
        test_net = "RNet"
    else:
        print("Net type error! ")
        return 
    
    t_net(data_dir, 
            prefix=['./modelfiles/PNet/PNet', 
                    './modelfiles/RNet/RNet', 
                    './modelfiles/ONet/ONet'], 
            epoch=[18, 14, 16], batch_size=[2048, 256, 16], 
            test_mode=test_net, thresh=[0.6, 0.6, 0.7], min_face_size=25,
            stride=2, slide_window=False, shuffle=False, vis=False)

    save_hard_example(net, data_dir)





def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='RNet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['../data/MTCNN_model/PNet_No_Landmark/PNet', '../data/MTCNN_model/RNet_No_Landmark/RNet', '../data/MTCNN_model/ONet_No_Landmark/ONet'],
                        type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[18, 14, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.3, 0.1, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=20, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    #parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


