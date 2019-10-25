import cv2
import time
import numpy as np

getTime = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def finetune(state_pre, state_to):

    dict_update = {k: v for k, v in state_pre.items() 
            if (k in state_to.keys()) and (state_to[k].shape==state_pre[k].shape)}
    state_to.update(dict_update)
    
    return state_to

def py_nms(dets, thresh, mode="Union"):
    """
    Params:
        dets:   {ndarray(n_boxes, 5)} x1, y1, x2, y2 score
        thresh: {float} retain overlap <= thresh
        mode:   {str} 'Union' or 'Minimum'
    Returns:
        keep:   {list[int]} indexes to keep
    Notes:
        greedily select boxes with high confidence
        keep boxes overlap <= thresh
        rule out overlap > thresh
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

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

    return keep
