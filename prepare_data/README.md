# Prepare Data

## Notes

1. Generating training data using pretrained MTCNN, which can be found at [](https://github.com/AITTSMD/MTCNN-Tensorflow).

## Download Data Set

The origin data set can be downloaded from [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [CNN for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm). [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) is used for face detection and [CNN for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) is for landmark regression.


As files listed below, the files with * is needed.

1. WIDER Face
    - WIDER_train.zip (*)
    - WIDER_val.zip
    - Face annotations.zip
2. CNN for Facial Point Detection
    - train.zip (*)
    - test.zip

Extract the files to the directory `data/`
```
./data
├── WIDER_train
│    └── images
│        └── {Occasion}
│            └── {Occasion}_*.jpg
└── Align
    ├── lfw_5590
    │    └── *.jpg
    └── net_7876
        └── *.jpg
```

## Create Training Data

Run `main.py` to create training data.

1. For Detection

    - Run `prepare_data/gen_12net_data.py/save_12net_data` to generate training data for PNet;
    - Run `prepare_data/gen_hard_example.py/gen_hard_example` to generate training data for RNet and ONet;

2. For Landmark Regression

    Run `prepare_data/gen_landmark_aug.py/generateData_aug` to generate training data for landmark regression;

3. Sort Image Lists

    Run `prepare_data/gen_imglist.py/save_imglist` to sort image list files;

```
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
```