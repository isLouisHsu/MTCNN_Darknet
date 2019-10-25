
# generate mtcnn weights
cd mtcnn_c && mkdir weights
cd ..
cd mtcnn_py
python extract_weights.py
cd ..
echo "Weights generated! "
echo "============================================="

# compile mtcnn
cd mtcnn_c
mkdir build
cd build
cmake .. && make
cd ..
echo "============================================="

echo "MTCNN compiled!  You can use MTCNN like: "
echo "	./mtcnn --help"




