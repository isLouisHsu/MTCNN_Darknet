## Description

训练详细过程查看[详细！MTCNN训练全过程！](https://louishsu.xyz/2019/05/06/%E8%AF%A6%E7%BB%86%EF%BC%81MTCNN%E8%AE%AD%E7%BB%83%E5%85%A8%E8%BF%87%E7%A8%8B%EF%BC%81/).

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
