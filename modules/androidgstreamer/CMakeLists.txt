# We actually don't declare any OpenCV modules here, we only hacking "highgui" module

if(NOT ANDROID OR NOT WITH_ANDROIDGSTREAMER)
  return()
endif()

set(HAVE_GSTREAMER ON CACHE INTERNAL "" FORCE)

if (NOT COMMAND gstreamer_android_configure)
  include(${CMAKE_CURRENT_LIST_DIR}/gstreamer.cmake)
  gstreamer_android_configure()
endif()

list(APPEND OPENCV_MODULE_opencv_highgui_PRIVATE_REQ_DEPS "-Wl,-no-whole-archive" ${GSTREAMER_ANDROID_LIBS})
set(OPENCV_MODULE_opencv_highgui_PRIVATE_REQ_DEPS "${OPENCV_MODULE_opencv_highgui_PRIVATE_REQ_DEPS}" CACHE INTERNAL "" FORCE)

set(GSTREAMER_ANDROID_INIT_SOURCE_FILE "${CMAKE_CURRENT_LIST_DIR}/gstreamer_init.cpp" CACHE INTERNAL "" FORCE)
macro(ocv_get_module_external_sources)
  if(the_module STREQUAL opencv_highgui)
    message(STATUS "Add GStreamer initialization code into highgui: ${GSTREAMER_ANDROID_INIT_SOURCE_FILE}")
    list(APPEND OPENCV_MODULE_${the_module}_SOURCES "${GSTREAMER_ANDROID_INIT_SOURCE_FILE}")
    include_directories(${GSTREAMER_ANDROID_INCLUDES})
    link_directories(${GSTREAMER_ANDROID_LINK_DIRS})
    add_definitions(-Dgst_init=opencv_gst_init)
  endif()
endmacro()
