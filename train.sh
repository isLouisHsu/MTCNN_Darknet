#! bin/bash
### 
# @Description: 
 # @Version: 1.0.0
 # @Author: louishsu
 # @E-mail: is.louishsu@foxmail.com
 # @Date: 2019-11-10 12:23:11
 # @LastEditTime: 2019-11-10 12:26:02
 # @Update: 
 ###

cd prepare_data
python merge_annotations.py

# -----------------------------
python pnet_12x12.py
python statistic.py p

cd ../mtcnn_py
python main_pnet.py

# -----------------------------
cd ../prepare_data
python rnet_24x24.py
python statistic.py r

cd ../mtcnn_py
python main_rnet.py

# -----------------------------
cd ../prepare_data
python onet_48x48.py
python statistic.py o

cd ../mtcnn_py
python main_onet.py