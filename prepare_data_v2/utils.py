# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-25 12:50:26
@LastEditTime: 2019-10-25 18:49:03
@Update: 
'''
import os
import cv2
import numpy as np

def iou(box, bboxes):
    """
    Params:
        box: {ndarray(4,)}
        bboxes: {ndarray(n, 4)}
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    xx1 = np.maximum(box[0], bboxes[:, 0])
    yy1 = np.maximum(box[1], bboxes[:, 1])
    xx2 = np.minimum(box[2], bboxes[:, 2])
    yy2 = np.minimum(box[3], bboxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    over = inter / (box_area + areas - inter)

    return over

def show_bbox(image, bbox, landmark=None, show_score=False):
    """
    Params: 
        image:  {ndarray(H, W, C)}
        bbox:   {ndarray(n_box, 5)} x1, y1, x2, y2, score
        landmark: {ndarray(n_landmark, 10)} xx1, yy1, ..., xx5, yy5
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