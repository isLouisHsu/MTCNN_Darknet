## Description

训练详细过程查看[]()

## Requirements

- python(>=3.6)
- PyTorch(>=1.0.0)  
- [DarkerNet](https://github.com/isLouisHsu/DarkerNet)
- Cmake(>=2.8)
- OpenCV(>=3.4.0)
- [OpenBLAS](http://www.openblas.net/)

## Usage

### Install

``` shell
bash install.sh
```

### Python

``` shell
cd mtcnn_py
python detector <image file>
```

### C

``` shell
cd mtcnn_c
./mtcnn --help
```

## Reference
1. Kaipeng Zhang Zhanpeng Zhang Zhifeng Li Yu Qiao  " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. AITTSMD/MTCNN-Tensorflow: Reproduce MTCNN using Tensorflow https://github.com/AITTSMD/MTCNN-Tensorflow
