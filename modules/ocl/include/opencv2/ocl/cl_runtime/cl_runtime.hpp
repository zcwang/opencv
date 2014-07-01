#ifndef __OPENCV_OCL_CL_RUNTIME_HPP__
#define __OPENCV_OCL_CL_RUNTIME_HPP__

#ifdef HAVE_OPENCL

#if defined(HAVE_OPENCL_STATIC)

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if defined __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#else // HAVE_OPENCL_STATIC

#include "cl_runtime_opencl.hpp"

#endif // HAVE_OPENCL_STATIC

#endif // HAVE_OPENCL

#endif // __OPENCV_OCL_CL_RUNTIME_HPP__
