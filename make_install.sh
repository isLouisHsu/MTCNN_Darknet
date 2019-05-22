
# generate mtcnn weights
cd mtcnn && mkdir weights
cd ..
cd torch_mtcnn
python extract_weights.py
cd ..
echo "Weights generated! "
echo "============================================="

# compile mtcnn
cd mtcnn
mkdir build
cd build
cmake .. && make
cd ..
echo "============================================="

echo "MTCNN compiled!  You can use MTCNN like: "
echo "	./mtcnn --help"




