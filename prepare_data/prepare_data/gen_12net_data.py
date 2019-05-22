#coding:utf-8
import os
import cv2
import numpy as np
import numpy.random as npr

from prepare_data.utils import IoU

def save_12net_data(data_dir):
    anno_file = "./prepare_data/wider_face_train.txt"
    im_dir        = data_dir + "/WIDER_train/images"
    save_dir      = data_dir + "/12"
    pos_save_dir  = data_dir + "/12/positive"
    part_save_dir = data_dir + "/12/part"
    neg_save_dir  = data_dir + '/12/negative'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
    f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
    f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # don't care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        #image path
        im_path = annotation[0]
        #boxed change to float type
        bbox = list(map(float, annotation[1:]))
        #gt
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        #load image
        img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
        height, width, channel = img.shape
        idx += 1
        #if idx % 100 == 0:
            #print(idx, "images done")




        neg_num = 0
        # crop image randomly, 1---->50
        # keep crop random parts, until have 50 negative examples
        # get 50 negative sample from every image
        while neg_num < 50:

            # ============================================================== #
            #neg_num's size [40,min(width, height) / 2],min_size:40
            # size is a random number between 12 and min(width,height)
            size = npr.randint(12, min(width, height) / 2)
            #top_left coordinate
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            #random crop
            crop_box = np.array([nx, ny, nx + size, ny + size])
            #calculate iou
            Iou = IoU(crop_box, boxes)

            #crop a part from inital image
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            #resize the cropped image to size 12*12
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                # ../data/12/negative/%s.jpg
                f2.write(save_dir + "/negative/%s.jpg"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            # ============================================================== #



        #for every bounding boxes
        for box in boxes:
            
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            #gt's width and height
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces and those faces has left-top corner out of the image
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue


            # crop another 5 images near the bounding box if IoU less than 0.5, save as negative samples
            for i in range(5):
                #size of the image to be cropped
                size = npr.randint(12, min(width, height) / 2)
                
                # ============================================================== #
                # delta_x and delta_y are offsets of (x1, y1)
                # max can make sure if the delta is a negative number , x1+delta_x >0
                # parameter high of randint make sure there will be intersection between bbox and cropped_box
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                # max here not really necessary
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # if the right bottom point is out of image then skip
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)
        
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                #rexize cropped image to be 12 * 12
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        
                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_dir + "/negative/%s.jpg" % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                # ============================================================== #


            #generate positive examples and part faces
            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                if w<5:
                    print (w)
                    continue

                # ============================================================== #
                # delta here is the offset of box center
                # - x1 + w/2 is the central point
                # - add offset to move the bbox
                # - deduct size/2 to compute x1, and also to make sure 
                #   that the right bottom corner will be out of bbox
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)
                # show this way: nx1 = max(x1+w/2-size/2+delta_x)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                # show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                if nx2 > width or ny2 > height:
                    continue 
                crop_box = np.array([nx1, ny1, nx2, ny2])
                
                # yu gt de offset
                # - will not be effected by resize option
                # - x, y is the location of rectangle box
                # - nx, ny is the location of square box
                # - offset_x = (x - nx) / w
                #       => x = offset_x*w + nx
                # - offset_y = (y - ny) / h
                #       => y = offset_y*h + ny
                # - [offset_x1, offset_y1, offset_x2, offset_y2] is the regression target
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                #crop
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                #resize
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                iou = IoU(crop_box, box_)
                if iou  >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    f1.write(save_dir + "/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif iou >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    f3.write(save_dir + "/part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
                # ============================================================== #

            box_idx += 1
            if idx % 100 == 0:
                print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
    f1.close()
    f2.close()
    f3.close()

