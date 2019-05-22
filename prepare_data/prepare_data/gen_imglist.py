import numpy as np
import numpy.random as npr
import os

def save_imglist(data_dir, net):
    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        return

    with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
        part = f.readlines()
    with open(os.path.join(data_dir,'%s/landmark_%s_aug.txt' %(size,size)), 'r') as f:
        landmark = f.readlines()
        
    dir_path = os.path.join(data_dir, 'imglists', "%s" %(net))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(os.path.join(dir_path, "train_%s_landmark.txt" % (net)), "w") as f:
        if net == "PNet":
            nums = [len(neg), len(pos), len(part)]
            ratio = [3, 1, 1]
            # base_num = min(nums)
            base_num = 250000
            print(len(neg), len(pos), len(part), base_num)

            # shuffle the order of the initial data
            # if negative examples are more than 750k then only choose 750k
            if len(neg) > base_num * 3:
                neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
            else:
                neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
            pos_keep = npr.choice(len(pos), size=base_num, replace=True)
            part_keep = npr.choice(len(part), size=base_num, replace=True)
            print(len(neg_keep), len(pos_keep), len(part_keep))

            # write the data according to the shuffled order
            for i in pos_keep:
                f.write(pos[i])
            for i in neg_keep:
                f.write(neg[i])
            for i in part_keep:
                f.write(part[i])
            for item in landmark:
                f.write(item)
        else:
            print(len(neg))
            print(len(pos))
            print(len(part))
            print(len(landmark))
            for i in np.arange(len(pos)):
                f.write(pos[i])
            for i in np.arange(len(neg)):
                f.write(neg[i])
            for i in np.arange(len(part)):
                f.write(part[i])
            for i in np.arange(len(landmark)):
                f.write(landmark[i])