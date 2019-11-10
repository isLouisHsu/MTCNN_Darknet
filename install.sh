#! bin/bash
### 
# @Description: 
 # @Version: 1.0.0
 # @Author: louishsu
 # @E-mail: is.louishsu@foxmail.com
 # @Date: 2019-11-10 11:58:54
 # @LastEditTime: 2019-11-10 12:04:37
 # @Update: 
 ###
#

# Download weights

wget -O mtcnn_py/ckptdir/PNet.pkl TODO:
wget -O mtcnn_py/ckptdir/RNet.pkl TODO:
wget -O mtcnn_py/ckptdir/ONet.pkl TODO:

wget -O mtcnn_c/weights/PNet.weights TODO:
wget -O mtcnn_c/weights/RNet.weights TODO:
wget -O mtcnn_c/weights/ONet.weights TODO:

# Compile

cd mtcnn_c
mkdir build
cd build
cmake .. && make
cd ..

echo "============================================="
echo "MTCNN compiled!  You can use MTCNN like: "
echo "	./mtcnn --help"