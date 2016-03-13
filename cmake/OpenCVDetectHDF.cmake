if(WIN32)
  # Detection is broken, but you can specify settings via CMake parameters
else()
  find_package(HDF5)
endif()

if(HDF5_FOUND)
  include(CheckCXXSourceRuns)
  function(hdf5_validate HDF5_CHECK_VAR)
    set(CMAKE_REQUIRED_LIBRARIES "${HDF5_LIBRARIES}")
    set(CMAKE_REQUIRED_INCLUDES "${HDF5_INCLUDE_DIRS}")
    CHECK_CXX_SOURCE_RUNS(
"
#include <hdf5.h>\n
int main()\n
{\n
    H5open();\n
    H5close();\n
    return 0;\n
}
"
      ${HDF5_CHECK_VAR}
    )
    set(${HDF5_CHECK_VAR} "${${HDF5_CHECK_VAR}}" PARENT_SCOPE)
  endfunction()

  hdf5_validate(OPENCV_HDF5_CHECK)
  if(NOT OPENCV_HDF5_CHECK)
    message(MESSAGE "HDF5 is found, but can't be used")
    return()
  endif()

  set(HAVE_HDF5 ON CACHE BOOL "HDF5 library is found")
  set(OPENCV_HDF5_INCLUDE_DIRS "${HDF5_INCLUDE_DIRS}" CACHE INTERNAL "")
  set(OPENCV_HDF5_LIBRARIES "${HDF5_LIBRARIES}" CACHE INTERNAL "")
endif()
