from prepare_data.gen_12net_data        import save_12net_data
from prepare_data.gen_hard_example      import gen_hard_example
from prepare_data.gen_landmark_aug      import generateData_aug
from prepare_data.gen_imglist           import save_imglist
from prepare_data.gen_darknet_lists     import save_imglist_darknet
"""
\-- data_dir
                                            # ------- origin data -------
    \-- WIDER_train
        \-- images
            \-- Occasion
                |-- Occasion_*.jpg
    \-- Align
        \-- lfw_5590
        \-- net_7876



                                            # --------- for PNet --------
    \-- 12
                                            # afer run `save_12net_data`
        \-- negative
            |-- {image_id}.jpg
        \-- part
            |-- {image_id}.jpg
        \-- positive
            |-- {image_id}.jpg
        |-- neg_12.txt                      : filename  0
        |-- part_12.txt                     : filename -1 offset_x1 offset_y1 offset_x2 offset_y2
        |-- pos_12.txt                      : filename  1 offset_x1 offset_y1 offset_x2 offset_y2

                                            # after run `generateData_aug`
        \-- train_PNet_landmark_aug
            |-- {image_id}.jpg
        |-- landmark_12_aug.txt             : filename -2 offset_x1 offset_y1 ... offset_x5 offset_y5



                                            # --------- for RNet --------
                                            # afer run `gen_hard_example/t_net`
    \-- RNet
        |-- detections.pkl
                                            # afer run `gen_hard_example/save_hard_example`
    \-- 24
        \-- negative 
            |-- {image_id}.jpg
        \-- positive
            |-- {image_id}.jpg
        \-- part
            |-- {image_id}.jpg
        |-- neg_24.txt                      : filename  0
        |-- pos_24.txt                      : filename -1 offset_x1 offset_y1 offset_x2 offset_y2
        |-- part_24.txt                     : filename  1 offset_x1 offset_y1 offset_x2 offset_y2

                                            # after run `generateData_aug`
        \-- train_RNet_landmark_aug
            |-- {image_id}.jpg
        |-- landmark_24_aug.txt             : filename -2 offset_x1 offset_y1 ... offset_x5 offset_y5


                                            # --------- for ONet --------
                                            # afer run `gen_hard_example/t_net`
    \-- ONet
        |-- detections.pkl
                                            # afer run `gen_hard_example/save_hard_example`
    \-- 48
        \-- negative 
            |-- {image_id}.jpg
        \-- positive
            |-- {image_id}.jpg
        \-- part
            |-- {image_id}.jpg
        |-- neg_48.txt                      : filename  0
        |-- pos_48.txt                      : filename -1 offset_x1 offset_y1 offset_x2 offset_y2
        |-- part_48.txt                     : filename  1 offset_x1 offset_y1 offset_x2 offset_y2

                                            # after run `generateData_aug`
        \-- train_ONet_landmark_aug
            |-- {image_id}.jpg
        |-- landmark_48_aug.txt             : filename -2 offset_x1 offset_y1 ... offset_x5 offset_y5


                                            # ----- merge size/*.txt -----
                                            # afer run `save_imglist`
    \-- imglists                            
        \-- PNet
            \-- train_PNet_landmark.txt
        \-- RNet
            \-- train_RNet_landmark.txt
        \-- ONet
            \-- train_ONet_landmark.txt
    
"""

def gen_data(data_dir, net):

    if net == "PNet":
        save_12net_data(data_dir)

    elif net in ["RNet", "ONet"]:
        gen_hard_example(data_dir, net)
    
    else:
        print("Net type error! ")
        return 

    generateData_aug(data_dir, net, argument=True)
    save_imglist(data_dir, net)
    save_imglist_darknet(data_dir, net, train=0.8)


def main():
    data_dir = "/home/louishsu/Work/Codes/MTCNN_Darknet/data"
    
    ### For PNet
    gen_data(data_dir, "PNet")

    ### For RNet
    gen_data(data_dir, "RNet")

    ### For Onet
    gen_data(data_dir, "ONet")



if __name__ == "__main__":
    main()