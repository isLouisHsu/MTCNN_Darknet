## Description
Reproduce MTCNN using Darknet.

## Requirements
- python(>=3.6)
- tensorflow(>=1.13)  
- torch(>=1.0.0)  
- [darkernet](https://github.com/isLouisHsu/DarkerNet)
- cmake(>=2.8)
- OpenCV(>=3.4.0)
- [OpenBLAS](http://www.openblas.net/)

## Usage
### Install
``` shell
bash make_install.sh
```

### Python
``` shell
cd train_mtcnn
python detector <image file>
```

### C
``` shell
cd mtcnn
./mtcnn --help
```

## Prepare Data

See more about [prepare_data](prepare_data/README.md).

1. Origin data
    Download dataset from [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [CNN for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
    1. WIDER Face
        - WIDER_train.zip (*)
        - WIDER_val.zip
        - Face annotations.zip
    2. CNN for Facial Point Detection
        - train.zip (*)
        - test.zip
2. Sort
    ```
    ├── ./data
    │   ├── WIDER_train
    │   │    └── images
    │   │        └── {Occasion}
    │   │            └── {Occasion}_*.jpg
    │   └── Align
    │        ├── lfw_5590
    │        │    └── *.jpg
    │        └── net_7876
    │            └── *.jpg
    ```
3. Generate data
    Run `prepare_data/main.py`

## Model
![mtcnn](/images/mtcnn.png)

## Details
1. Make training data using pretrained MTCNN;
1. Train MTCNN using PyTorch;
1. Convert PyTorch module's weights to fit darknet;
1. Use `logistic` in classification task instead of `softmax`;

## Reference
1. Kaipeng Zhang Zhanpeng Zhang Zhifeng Li Yu Qiao  " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. AITTSMD/MTCNN-Tensorflow: Reproduce MTCNN using Tensorflow https://github.com/AITTSMD/MTCNN-Tensorflow
