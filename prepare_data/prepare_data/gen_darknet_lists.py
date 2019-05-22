import numpy as np
import numpy.random as npr
import os, shutil
import random

def save_imglist_darknet(data_dir, net, train=0.7):
    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        return

    dir_path = os.path.join(data_dir, 'imglists', "%s" %(net))
    if not os.path.exists(dir_path):
        print("no such directory, run other steps first")
        return

    with open(os.path.join(dir_path, "train_%s_landmark.txt" % (net)), "r") as f:
        datalists = f.readlines()
    n_item = len(datalists)
    print("net: %s, total: %d" % (net, n_item))

    datalist_pos = []
    datalist_neg = []
    datalist_part = []
    datalist_landmark = []

    for i_item in range(n_item):
        datalist = datalists[i_item]
        if i_item%100000 == 0: print("net: %s, item %d" % (net, i_item))

        savedict = dict()
        savedict['filepath'] = ""
        savedict['label'] = 0.
        savedict['x1'] = 0.
        savedict['y1'] = 0.
        savedict['x2'] = 0.
        savedict['y2'] = 0.
        savedict['lmx1'] = 0.
        savedict['lmy1'] = 0.
        savedict['lmx2'] = 0.
        savedict['lmy2'] = 0.
        savedict['lmx3'] = 0.
        savedict['lmy3'] = 0.
        savedict['lmx4'] = 0.
        savedict['lmy4'] = 0.
        savedict['lmx5'] = 0.
        savedict['lmy5'] = 0.

        datalist = datalist.strip().split(' ')
        savedict['filepath'] = datalist[0]
        datalist = [float(i) for i in datalist[1:]]
        savedict['label']    = datalist[0]
        
        if savedict['label'] == 1:
            savedict['x1'] = datalist[1]
            savedict['y1'] = datalist[2]
            savedict['x2'] = datalist[3]
            savedict['y2'] = datalist[4]
            datalist_pos += [savedict.values()]
        elif savedict['label'] == 0:
            pass
            datalist_neg += [savedict.values()]
        elif savedict['label'] == -1:
            savedict['x1'] = datalist[1]
            savedict['y1'] = datalist[2]
            savedict['x2'] = datalist[3]
            savedict['y2'] = datalist[4]
            datalist_part += [savedict.values()]
        elif savedict['label'] == -2:
            savedict['lmx1'] = datalist[1]
            savedict['lmy1'] = datalist[2]
            savedict['lmx2'] = datalist[3]
            savedict['lmy2'] = datalist[4]
            savedict['lmx3'] = datalist[5]
            savedict['lmy3'] = datalist[6]
            savedict['lmx4'] = datalist[7]
            savedict['lmy4'] = datalist[8]
            savedict['lmx5'] = datalist[9]
            savedict['lmy5'] = datalist[10]
            datalist_landmark += [savedict.values()]
    print("net: %s, item %d" % (net, i_item))

    print("sample from pos")
    pos_train  = random.sample(datalist_pos, int(train*len(datalist_pos)))
    pos_valid   = [i for i in datalist_pos if i not in pos_train]
    print("sample from neg")
    neg_train  = random.sample(datalist_neg, int(train*len(datalist_neg)))
    neg_valid   = [i for i in datalist_neg if i not in neg_train]
    print("sample from part")
    part_train = random.sample(datalist_part, int(train*len(datalist_part)))
    part_valid  = [i for i in datalist_part if i not in part_train]
    print("sample from landmark")
    landmark_train = random.sample(datalist_landmark, int(train*len(datalist_landmark)))
    landmark_valid  = [i for i in datalist_landmark if i not in landmark_train]
    print("split done! ")

    trainset = pos_train + neg_train + part_train + landmark_train
    validset = pos_valid + neg_valid + part_valid + landmark_valid
    n_pos = len(pos_train) + len(pos_valid)
    n_neg = len(neg_train) + len(neg_valid)
    n_part = len(part_train) + len(part_valid)
    n_landmark = len(landmark_train) + len(landmark_valid)
    n_train = len(trainset)
    n_valid = len(validset)
    print("net: %s, pos: %d, neg: %d, part: %d, landmark: %d, ratio: %f: %f: %f: %f" % 
            (net, n_pos, n_neg, n_part, n_landmark, n_pos/n_item, n_neg/n_item, n_part/n_item, n_landmark/n_item))
    print("net: %s, train: %d, valid: %d, ratio: %f: %f" % 
            (net, n_train, n_valid, n_train/n_item, n_valid/n_item))

    
    f_train = open(os.path.join(dir_path + "/%s_train.txt" % (net.lower())), "w")
    f_valid = open(os.path.join(dir_path, "%s_valid.txt" % (net.lower())), "w")

    print("shuffle and write trainset ")
    random.shuffle(trainset)
    for item in trainset:
        item = [str(it) for it in item]
        item = ' '.join(item) + '\n'
        f_train.write(item)
    
    print("write validset ")
    for item in validset:
        item = [str(it) for it in item]
        item = ' '.join(item) + '\n'
        f_valid.write(item)

    f_train.close(); f_valid.close()
