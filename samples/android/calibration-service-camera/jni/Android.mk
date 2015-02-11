LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#OPENCV_CAMERA_MODULES:=off
#OPENCV_INSTALL_MODULES:=off
#OPENCV_LIB_TYPE:=SHARED
ifdef OPENCV_ANDROID_SDK
  include ${OPENCV_ANDROID_SDK}/sdk/native/jni/OpenCV.mk
else
  include ../../sdk/native/jni/OpenCV.mk
endif
