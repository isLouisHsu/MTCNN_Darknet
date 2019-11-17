#! bin/bash
### 
# @Description: 
 # @Version: 1.0.0
 # @Author: louishsu
 # @E-mail: is.louishsu@foxmail.com
 # @Date: 2019-11-10 11:58:54
 # @LastEditTime: 2019-11-10 17:36:19
 # @Update: 
 ###
#

# Download weights

wget -O mtcnn_py/ckptdir/PNet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/PNet.pkl
wget -O mtcnn_py/ckptdir/RNet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/RNet.pkl
wget -O mtcnn_py/ckptdir/ONet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/ONet.pkl

wget -O mtcnn_c/weights/PNet.weights https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/PNet.weights
wget -O mtcnn_c/weights/RNet.weights https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/RNet.weights
wget -O mtcnn_c/weights/ONet.weights https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/ONet.weights

# Compile

cd mtcnn_c
mkdir build
cd build
cmake .. && make
cd ..

echo "============================================="
echo "MTCNN compiled!  You can use MTCNN like: "
echo "	./mtcnn --help"