#include "precomp.hpp"

#if defined(HAVE_OPENCL) && !defined(HAVE_OPENCL_STATIC)

#include "opencv2/core.hpp" // CV_Error
#include <sstream> // std::ostringstream

#include "opencv2/core/ocl_runtime/ocl_runtime.hpp"

static const char* funcToCheckOpenCL1_1 = "clEnqueueReadBufferRect";
#define ERROR_MSG_CANT_LOAD "Failed to load OpenCL runtime\n"
#define ERROR_MSG_INVALID_VERSION "Failed to load OpenCL runtime (expected version 1.1+)\n"

#if defined(__APPLE__)
#include <dlfcn.h>

static void* AppleCLGetProcAddress(const char* name)
{
    static bool initialized = false;
    static void* handle = NULL;
    if (!handle)
    {
        if(!initialized)
        {
            initialized = true;
            const char* path = "/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL";
            const char* envPath = getenv("OPENCV_OPENCL_RUNTIME");
            if (envPath)
                path = envPath;
            handle = dlopen(oclpath, RTLD_LAZY | RTLD_GLOBAL);
            if (handle == NULL)
            {
                fprintf(stderr, ERROR_MSG_CANT_LOAD);
            }
            else if (dlsym(handle, funcToCheckOpenCL1_1) == NULL)
            {
                fprintf(stderr, ERROR_MSG_INVALID_VERSION);
                handle = NULL;
            }
        }
        if (!handle)
            return NULL;
    }
    return dlsym(handle, name);
}
#define CV_CL_GET_PROC_ADDRESS(name) AppleCLGetProcAddress(name)
#endif // __APPLE__

#if defined(_WIN32)
static void* WinGetProcAddress(const char* name)
{
    static bool initialized = false;
    static HMODULE handle = NULL;
    if (!handle)
    {
        if(!initialized)
        {
            initialized = true;
            handle = GetModuleHandleA("OpenCL.dll");
            if (!handle)
            {
                const char* path = "OpenCL.dll";
                const char* envPath = getenv("OPENCV_OPENCL_RUNTIME");
                if (envPath)
                    path = envPath;
                handle = LoadLibraryA(path);
                if (!handle)
                {
                    fprintf(stderr, ERROR_MSG_CANT_LOAD);
                }
                else if (GetProcAddress(handle, funcToCheckOpenCL1_1) == NULL)
                {
                    fprintf(stderr, ERROR_MSG_INVALID_VERSION);
                    handle = NULL;
                }
            }
        }
        if (!handle)
            return NULL;
    }
    return (void*)GetProcAddress(handle, name);
}
#define CV_CL_GET_PROC_ADDRESS(name) WinGetProcAddress(name)
#endif // _WIN32

#if defined(linux)
#include <dlfcn.h>
#include <stdio.h>

static void* GetProcAddress (const char* name)
{
    static bool initialized = false;
    static void* handle = NULL;
    if (!handle)
    {
        if(!initialized)
        {
            initialized = true;
            const char* path = "libOpenCL.so";
            const char* envPath = getenv("OPENCV_OPENCL_RUNTIME");
            if (envPath)
                path = envPath;
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
            if (handle == NULL)
            {
                fprintf(stderr, ERROR_MSG_CANT_LOAD);
            }
            else if (dlsym(handle, funcToCheckOpenCL1_1) == NULL)
            {
                fprintf(stderr, ERROR_MSG_INVALID_VERSION);
                handle = NULL;
            }
        }
        if (!handle)
            return NULL;
    }
    return dlsym(handle, name);
}
#define CV_CL_GET_PROC_ADDRESS(name) GetProcAddress(name)
#endif

#ifndef CV_CL_GET_PROC_ADDRESS
#define CV_CL_GET_PROC_ADDRESS(name) NULL
#endif

static void* opencl_check_fn(int ID);

#include "ocl_runtime_opencl_impl.hpp"

static void* opencl_check_fn(int ID)
{
    void* func = CV_CL_GET_PROC_ADDRESS(opencl_fn_names[ID]);
    if (!func)
    {
        std::ostringstream msg;
        msg << "OpenCL function is not available: [" << opencl_fn_names[ID] << "]";
        CV_Error(CV_StsBadFunc, msg.str().c_str());
    }
    *(void**)(opencl_fn_ptrs[ID]) = func;
    return func;
}

#endif
