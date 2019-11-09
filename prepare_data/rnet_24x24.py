import os
import cv2
import random
import numpy as np
from numpy import random as npr
from scipy import io

import torch

from config import configer
from utils import iou, show_bbox
from processbar import ProcessBar

import sys
sys.path.append('../mtcnn_py/')

from detector import MtcnnDetector

# 创建文件夹保存截取图片
if not os.path.exists(configer.rImage):
    os.makedirs(configer.rImage)

# 初始化全局变量
DETS_BY_PNET = dict()
SAVE_CNT = 0                            # 图片保存计数
FACE_CNT = 0                            # 人脸计数
FACE_USED_CNT = 0                       # 使用的人脸计数
SAVE_IMAGE_NAME = configer.rImage + '{:d}.jpg'
SAVE_ANNO_FP    = open(configer.rAnno[0], 'w')

# 读取标注文档
with open(configer.annotations, 'r') as f:
    annotations = f.readlines()
n_annotation = len(annotations)
random.shuffle(annotations)

# 利用PNet生成候选框
if not os.path.exists(configer.rDets):  # 若已生成则跳过
    detector = MtcnnDetector()

    bar = ProcessBar(n_annotation)
    for i_annotation, annotation in enumerate(annotations):
        bar.step()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 读取图片
        imname = annotation.split('jpg')[0] + 'jpg'
        image  = cv2.imread(configer.images + imname, cv2.IMREAD_COLOR)
        _, boxpreds, _ = detector._detect_pnet(image)                           # x1, y1, x2, y2

        if boxpreds.shape[0] == 0:
            boxpreds = np.empty((0, 5))

        settype  = annotation.split('/')[0]
        if settype == 'WIDER':        # 来自`WIDER FACE`，无关键点

            boxgts = annotation.split('jpg')[-1].strip().split(' ')
            try:
                boxgts = np.array(list(map(int, boxgts))).reshape(-1, 4)        # x1, y1,  w,  h
            except:
                continue
            DETS_BY_PNET[imname] = [boxpreds, boxgts, np.array([])]

        else:
            annotation = annotation.split('jpg')[1].strip().split(' ')
            boxgts  = np.array(list(map(int, annotation[:4]))).reshape(-1, 4)   # x1, y1, x2, y2
            landgts = np.array(list(map(float, annotation[4:]))).reshape(-1, 10)
            DETS_BY_PNET[imname] = [boxpreds, boxgts, landgts]

    io.savemat(configer.rDets, DETS_BY_PNET)

else:
    DETS_BY_PNET = io.loadmat(configer.rDets)

# 进行采样 
n_image = len(DETS_BY_PNET)
for i_image, (imname, boxpreds_boxgts_langts) in enumerate(DETS_BY_PNET.items()):
    
    if imname[:2] == '__': continue
    boxpreds, boxgts, landgts = boxpreds_boxgts_langts[0]
    # 读取图片
    image  = cv2.imread(configer.images + imname, cv2.IMREAD_COLOR)
    imH, imW = image.shape[:2]

    if landgts.shape[0] == 0:             # 来自`WIDER FACE`，无关键点
        
        n_boxgt = boxgts.shape[0]; FACE_CNT += n_boxgt

        # 滤除真实样本中的小框与异常框
        idx1 = np.bitwise_or(np.max(boxgts[:, 2:], axis=1) < configer.sideMin, 
                                    np.min(boxgts[:, 2:], axis=1) < 0)
        idx2 = np.bitwise_or(boxgts[:, 0] < 0, boxgts[:, 1] < 0)
        idx3 = np.bitwise_or(boxgts[:, 0] + boxgts[:, 2] > imW - 1, boxgts[:, 1] + boxgts[:, 3] > imH - 1)
        index = np.bitwise_not(np.bitwise_or(idx1, idx2, idx3))
        boxgts = boxgts[index]; n_boxgt = boxgts.shape[0]; FACE_USED_CNT += n_boxgt

        if n_boxgt == 0:
            continue
        
        # 转换坐标
        boxgts[:, 2] += boxgts[:, 0]; boxgts[:, 3] += boxgts[:, 1]          # x1, y1, x2, y2

        # 对每个候选框进行判别
        i_pos, i_part, i_neg = 0, 0, 0
        for i_det, boxr in enumerate(boxpreds):
            
            boxr = MtcnnDetector._square(boxr.reshape(1, -1)).reshape(-1)   # 转换为方形
            boxr = np.round(boxr[:4]).astype(np.int)                        # 取整
            x1r, y1r, x2r, y2r = boxr
            wr = x2r - x1r + 1; hr = y2r - y1r + 1
            if wr < configer.sideMin or hr < configer.sideMin or\
                x1r < 0 or y1r < 0 or x2r > imW or y2r > imH:
                continue                                                    # 异常 舍去

            iour = iou(boxr, boxgts)                                        # iou
            if iour.max() < configer.iouThresh[0]:                          # neg

                if i_neg < configer.rNums[0]:
                    # 图像
                    imr = image[y1r: y2r, x1r: x2r]
                    imr = cv2.resize(imr, (24, 24))
                    pathr = SAVE_IMAGE_NAME.format(SAVE_CNT)
                    cv2.imwrite(pathr, imr)
                    # 标注
                    annor = '{} {}\n'.format(pathr, configer.label['neg'])
                    SAVE_ANNO_FP.write(annor)
                    # 计数
                    SAVE_CNT += 1
                    i_neg += 1

                    print('IMAGE: [{}]/[{}] | DET: [{}]/[{}] | NEG: [{}]/[{}]'.\
                            format(i_image, n_image, i_det, boxpreds.shape[0], i_neg, configer.rNums[0]))
                    
            elif iour.max() > configer.iouThresh[1]:                        # part & pos
                
                # 图像
                imr = image[y1r: y2r, x1r: x2r]
                imr = cv2.resize(imr, (24, 24))
                pathr = SAVE_IMAGE_NAME.format(SAVE_CNT)
                cv2.imwrite(pathr, imr)
                # 计算偏移量
                sr  = wr    # or `sr  = hr`
                boxgt = boxgts[iour.argmax()]
                x1f = (boxgt[0] - boxr[0]) / sr
                y1f = (boxgt[1] - boxr[1]) / sr
                x2f = (boxgt[2] - boxr[2]) / sr
                y2f = (boxgt[3] - boxr[3]) / sr
                boxf = np.array([x1f, y1f, x2f, y2f])
                boxf = ' '.join(list(map(str, boxf)))
                # 标注
                if iour.max() < configer.iouThresh[2]:                      # part
                    label = configer.label['part']
                    i_part += 1
                    print('IMAGE: [{}]/[{}] | DET: [{}]/[{}] | PART: [{}]'.\
                            format(i_image, n_image, i_det, boxpreds.shape[0], i_part))
                
                else:                                                       # pos
                    label = configer.label['pos']
                    i_pos  += 1

                    print('IMAGE: [{}]/[{}] | DET: [{}]/[{}] | POS: [{}]'.\
                            format(i_image, n_image, i_det, boxpreds.shape[0], i_pos))

                annor = '{} {} {}\n'.format(pathr, label, boxf)
                SAVE_ANNO_FP.write(annor)
                # 计数
                SAVE_CNT += 1
            
    else:                               # 来自`POINT FACE`，含关键点，仅用作关键点回归
        
        x1gt, x2gt, y1gt, y2gt = boxgts[0]
        wgt, hgt   = x2gt - x1gt, y2gt - y1gt
        cxgt, cygt = (x1gt + x2gt) / 2, (y1gt + y2gt) / 2   # 中心
        landmarkgt = landgts.reshape(5, -1)                 # xx1, yy1, ..., xx5, yy5

        FACE_CNT += 1
        i_landmark = 0
        while i_landmark < configer.rNums[-1]:

            imager = image.copy()
            landmarkgtr = landmarkgt.copy()
            
            sl = np.floor(min(wgt, hgt) * 0.80)
            sh = np.ceil (max(wgt, hgt) * 1.25)
            sr = npr.randint(sl, sh)   
            dx = npr.randint(- wgt * 0.2, wgt * 0.2 + 1)# 随机偏移
            dy = npr.randint(- hgt * 0.2, hgt * 0.2 + 1)# 随机偏移
            x1r = int(max(0, cxgt + dx - sr / 2))       # 左上角x
            y1r = int(max(0, cygt + dy - sr / 2))       # 左上角y
            x2r, y2r = x1r + sr, y1r + sr               # 右下角坐标
            cxr, cyr = (x1r + x2r) / 2, (y1r + y2r) / 2 # 中心
            
            if x2r > imW or y2r > imH:
                continue
            
            boxr = np.array([x1r, y1r, x2r, y2r])       # 随机框
            boxgt = np.array([x1gt, y1gt, x2gt, y2gt])  # 真实框
            iour = iou(boxr, boxgt.reshape(1, -1))      # 计算与真实框的IoU

            if iour < configer.iouThresh[2]:            # 非`pos`样本，舍去
                continue

            FACE_USED_CNT += 1
            
            # 水平翻转
            if configer.augment and npr.rand() > 0.5:
                imager = imager[:, ::-1]
                boxr [[0, 2]] = imW - boxr [[2, 0]]
                boxgt[[0, 2]] = imW - boxgt[[2, 0]]
                landmarkgtr[:, 0] = imW - landmarkgtr[:, 0]
                landmarkgtr[[0, 1]] = landmarkgtr[[1, 0]]
                landmarkgtr[[3, 4]] = landmarkgtr[[4, 3]]

            # 旋转
            alphar = npr.randint(-15, 15)
            M = cv2.getRotationMatrix2D((cxr, cyr), alphar, 1)
            imager = cv2.warpAffine(imager, M, imager.shape[:2][::-1])
            landmarkgtr = np.c_[landmarkgtr, np.ones(5)].dot(M.T)

            # 保存图片
            imr = imager[boxr[0]: boxr[2], boxr[1]: boxr[3]]
            imr = cv2.resize(imr, (24, 24))
            pathr = SAVE_IMAGE_NAME.format(SAVE_CNT)
            cv2.imwrite(pathr, imr)
            SAVE_CNT += 1
            
            # 保存标注
            x1f = (boxgt[0] - boxr[0]) / sr
            y1f = (boxgt[1] - boxr[1]) / sr
            x2f = (boxgt[2] - boxr[2]) / sr
            y2f = (boxgt[3] - boxr[3]) / sr
            boxf = np.array([x1f, y1f, x2f, y2f])
            boxf = ' '.join(list(map(str, boxf)))

            landmarkf = np.c_[(landmarkgtr[:, 0] - boxr[0]) / sr, (landmarkgtr[:, 1] - boxr[1]) / sr].reshape(-1)
            landmarkf = ' '.join(list(map(str, landmarkf)))
            
            annor = '{} {} {}\n'.\
                    format(pathr, configer.label['landmark'], landmarkf)
            SAVE_ANNO_FP.write(annor)

            i_landmark += 1

            print("IMAGE: [{}]/[{}] | LANDMARK: [{}]/[{}]".\
                format(i_image, n_image, i_landmark, configer.rNums[-1]))

            # show_bbox(imager, np.c_[np.r_[np.c_[boxgt, boxr]].T, np.array([1, iour])], landmarkgtr.reshape((1, 10)), show_score=True)

print("Number of faces: {}, Number of faces used: {}".format(FACE_CNT, FACE_USED_CNT))

SAVE_ANNO_FP.close()

# 划分数据集
TRAIN = []; VALID = []; TEST = []

with open(configer.rAnno[0], 'r') as f:
    lines = np.array(f.readlines())
    
n_samples = lines.shape[0] 
sampletypes = np.array(list(map(lambda x: int(x.split(' ')[1]), lines)))
for st in ['pos', 'part', 'neg', 'landmark']:
    lines_st   = lines[sampletypes == configer.label[st]]
    idx        = np.arange(lines_st.shape[0])
    _n_samples = idx.shape[0] 
    _n_train   = int(_n_samples*configer.splitratio[0])
    _n_valid   = int(_n_samples*configer.splitratio[1])
    idx_train = npr.choice(idx, _n_train)
    idx_valid_test = list(filter(lambda x: x not in idx_train, idx))
    idx_valid = npr.choice(idx_valid_test, _n_valid)
    idx_test  = list(filter(lambda x: x not in idx_valid, idx_valid_test))

    TRAIN += [lines_st[idx_train]]
    VALID += [lines_st[idx_valid]]
    TEST  += [lines_st[idx_test ]]

TRAIN = np.concatenate(TRAIN)
VALID = np.concatenate(VALID)
TEST  = np.concatenate(TEST )

n_train   = TRAIN.shape[0]
n_valid   = VALID.shape[0]
n_test    = TEST.shape[0]

print("Train: {} | Valid: {} | Test: {} || {}: {}: {}".\
            format(n_train, n_valid, n_test, 
                    n_train / n_samples, n_valid / n_samples, n_test / n_samples))

with open(configer.rAnno[1], 'w') as f:
    f.writelines(TRAIN)
with open(configer.rAnno[2], 'w') as f:
    f.writelines(VALID)
with open(configer.rAnno[3], 'w') as f:
    f.writelines(TEST )