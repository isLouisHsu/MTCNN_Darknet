# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-25 12:25:16
@LastEditTime: 2019-10-25 20:34:23
@Update: 
'''
import os
import cv2
import numpy as np
from numpy import random as npr

from config import configer
from utils import iou, show_bbox

# 创建文件夹保存截取图片
if not os.path.exists(configer.pImage):
    os.makedirs(configer.pImage)

# 初始化全局变量
SAVE_CNT = 0                            # 图片保存计数
SAVE_IMAGE_NAME = configer.pImage + '{:d}.jpg'
SAVE_ANNO_FP    = open(configer.pAnno, 'w')

# 读取标注文档
with open(configer.annotations, 'r') as f:
    annotations = f.readlines()

# 进行采样
for annotation in annotations:  # 每张图片进行采样
    
    # 读取图片
    impath = configer.images + annotation.split('jpg')[0] + 'jpg'
    image  = cv2.imread(impath, cv2.IMREAD_COLOR)
    imH, imW = image.shape[:2]
    
    settype  = annotation.split('/')[0]
    if settype == 'WIDER':        # 来自`WIDER FACE`，无关键点

        boxgts = annotation.split('jpg')[-1].strip().split(' ')
        boxgts = np.array(list(map(int, boxgts))).reshape(-1, 4)        # x1, y1,  w,  h
        boxgts[:, 2] += boxgts[:, 0]; boxgts[:, 3] += boxgts[:, 1]      # x1, y1, x2, y2
        n_boxgt = boxgts.shape[0]

        # ----------------------- 依据图片中框的个数进行随机采样 -----------------------
        n_neg, i_neg = configer.pNums[0] * n_boxgt, 0
        while i_neg < n_neg:
            
            sr = npr.randint(12, min(imH, imW) / 2)                         # 随机尺寸[12, (w + h) / 2)
            x1r, y1r = npr.randint(0, imW - sr), npr.randint(0, imH - sr)   # 左上角坐标
            x2r, y2r = x1r + sr, y1r + sr                                   # 右下角坐标
            boxr = np.array([x1r, y1r, x2r, y2r])                           # 随机框
            iour = iou(boxr, boxgts)                                         # 计算与所有真实框的IoU

            if iour.max() < configer.iouThresh[0]:                          # 小于时，记作`neg`样本
                # 保存图片
                imr = image[y1r: y2r, x1r: x2r]
                imr = cv2.resize(imr, (12, 12))
                pathr = SAVE_IMAGE_NAME.format(SAVE_CNT)
                cv2.imwrite(pathr, imr)
                # 保存标注
                annor = '{} {}\n'.format(pathr, configer.label['neg'])
                SAVE_ANNO_FP.write(annor)
                # 计数
                i_neg += 1
                SAVE_CNT += 1
        
        # ----------------------- 对于每个框，在其附近进行采样 -----------------------
        for x1gt, y1gt, x2gt, y2gt in boxgts:
            
            wgt, hgt = x2gt - x1gt, y2gt - y1gt                             # 长宽
            if max(wgt, hgt) < configer.sideMin or x1gt < 0 or y1gt < 0:    # 忽略过小的框
                continue
            cxgt, cygt = (x1gt + x2gt) / 2, (y1gt + y2gt) / 2               # 中心
            
            # -------------- 附近采样：neg样本  --------------
            i_neg = 0
            while i_neg  < configer.pNums[1]:
                
                sr = npr.randint(12, min(imH, imW) / 2)     # 随机尺寸[12, (w + h) / 2)
                dx = npr.randint(max(-sr, -x1gt), wgt)      # 随机偏移
                dy = npr.randint(max(-sr, -y1gt), hgt)      # 随机偏移
                x1r, y1r = x1gt + dx, y1gt + dy             # 左上角坐标
                x2r, y2r = x1r + sr, y1r + sr               # 右下角坐标
                if x2r > imW or y2r > imH:
                    continue

                boxr = np.array([x1r, y1r, x2r, y2r])       # 随机框
                iour = iou(boxr, boxgts)                    # 计算与所有真实框的IoU

                if iour.max() < configer.iouThresh[0]:      # 小于时，记作`neg`样本
                    # 保存图片
                    imr = image[y1r: y2r, x1r: x2r]
                    imr = cv2.resize(imr, (12, 12))
                    pathr = SAVE_IMAGE_NAME.format(SAVE_CNT)
                    cv2.imwrite(pathr, imr)
                    SAVE_CNT += 1
                    # 保存标注
                    annor = '{} {}\n'.format(pathr, configer.label['neg'])
                    SAVE_ANNO_FP.write(annor)
                    # 计数
                    i_neg += 1

            # -------------- 附近采样：part,pos 样本 --------------
            i_part, i_pos = 0, 0
            while i_part < configer.pNums[2] or i_pos  < configer.pNums[3]:

                sl = np.floor(min(wgt, hgt) * 0.8)
                sh = np.ceil (max(wgt, hgt) * 1.25)
                sr = npr.randint(sl, sh)      
                dx = npr.randint(- wgt * 0.2, wgt * 0.2)# 随机偏移
                dy = npr.randint(- hgt * 0.2, hgt * 0.2)# 随机偏移

                x1r = int(max(0, cxgt + dx - sr / 2))       # 左上角x
                y1r = int(max(0, cygt + dy - sr / 2))       # 左上角y
                x2r, y2r = x1r + sr, y1r + sr               # 右下角坐标
                if x2r > imW or y2r > imH:
                    continue

                boxr = np.array([x1r, y1r, x2r, y2r])       # 随机框
                boxgt = np.array([x1gt, y1gt, x2gt, y2gt])  # 真实框
                iour = iou(boxr, boxgt.reshape(1, -1))      # 计算与真实框的IoU

                if iour < configer.iouThresh[1]:            # `neg`样本，舍去
                    continue
            
                # 保存图片
                imr = image[y1r: y2r, x1r: x2r]

                # 图像扩增：水平镜像
                if configer.augment and npr.rand() > 0.5:
                    imr = imr[:, ::-1]
                    boxr [[0, 2]] = imW - boxr [[2, 0]]
                    boxgt[[0, 2]] = imW - boxgt[[2, 0]]

                imr = cv2.resize(imr, (12, 12))
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

                if iour < configer.iouThresh[2]:            # `part`样本
                    label = configer.label['part']
                    if i_part > configer.pNums[2]:
                        continue
                    i_part += 1
                elif iour > configer.iouThresh[2]:          # `pos`样本
                    label = configer.label['pos']
                    if i_pos > configer.pNums[3]:
                        continue
                    i_pos += 1
                    
                annor = '{} {} {}\n'.format(pathr, label, boxf)
                SAVE_ANNO_FP.write(annor)
    
                # show_bbox(image, np.c_[np.r_[np.c_[boxgt, boxr]].T, np.array([1, iour])], show_score=True)
    
    elif settype == 'Align':        # 来自`POINT FACE`，含关键点，仅用作关键点回归
        
        box_landmark_gt = annotation.split('jpg')[-1].strip().split(' ')
        boxgt = np.array(list(map(int, box_landmark_gt[:4])))           # x1, x2, y1, y2
        x1gt, x2gt, y1gt, y2gt = boxgt
        wgt, hgt = x2gt - x1gt, y2gt - y1gt
        cxgt, cygt = (x1gt + x2gt) / 2, (y1gt + y2gt) / 2                           # 中心
        landmarkgt = np.array(list(map(float, box_landmark_gt[4:]))).reshape(5, -1) # xx1, yy1, ..., xx5, yy5
        
        i_landmark = 0
        while i_landmark < configer.pNums[4]:

            imager = image.copy()
            landmarkgtr = landmarkgt.copy()
            
            sl = np.floor(min(wgt, hgt) * 0.80)
            sh = np.ceil (max(wgt, hgt) * 1.25)
            sr = npr.randint(sl, sh)   
            dx = npr.randint(- wgt * 0.2, wgt * 0.2)    # 随机偏移
            dy = npr.randint(- hgt * 0.2, hgt * 0.2)    # 随机偏移
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
            
            # 水平翻转
            if configer.augment and npr.rand() > 0.5:
                imager = imager[:, ::-1]
                boxr [[0, 2]] = imW - boxr [[2, 0]]
                boxgt[[0, 2]] = imW - boxgt[[2, 0]]
                landmarkgtr[:, 0] = imW - landmarkgtr[:, 0]
                landmarkgtr[[0, 1]] = landmarkgtr[[1, 0]]
                landmarkgtr[[3, 4]] = landmarkgtr[[4, 3]]

            # 旋转
            alphar = npr.randint(-10, 10)
            M = cv2.getRotationMatrix2D((cxr, cyr), alphar, 1)
            imager = cv2.warpAffine(imager, M, imager.shape[:2][::-1])
            landmarkgtr = np.c_[landmarkgtr, np.ones(5)].dot(M.T)

            # 保存图片
            imr = imager[boxr[0]: boxr[2], boxr[1]: boxr[3]]
            imr = cv2.resize(imr, (12, 12))
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
            
            annor = '{} {} {} {}\n'.\
                    format(pathr, -2, boxf, landmarkf)
            SAVE_ANNO_FP.write(annor)

            i_landmark += 1
            
            # show_bbox(imager, np.c_[np.r_[np.c_[boxgt, boxr]].T, np.array([1, iour])], landmarkgtr.reshape((1, 10)), show_score=True)

SAVE_ANNO_FP.close()