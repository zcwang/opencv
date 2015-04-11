OPENCV_SRC=${1:-../opencv}
shift
cmake -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF \
-DBUILD_opencv_ocl=OFF -DBUILD_opencv_nonfree=OFF -DBUILD_opencv_gpu=OFF \
-DBUILD_opencv_stitching=OFF -DBUILD_opencv_photo=OFF \
-DWITH_ANDROIDGSTREAMER=ON \
-DCMAKE_TOOLCHAIN_FILE=${OPENCV_SRC}/modules/androidgstreamer/toolchain.cmake $@ ${OPENCV_SRC}
