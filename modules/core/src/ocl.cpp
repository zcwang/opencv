/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <map>

#ifdef HAVE_OPENCL

#include "opencv2/core/ocl_runtime/ocl_runtime.hpp"

namespace cv { namespace ocl {

struct UMat2D
{
    UMat2D(const UMat& m)
    {
        offset = m.offset;
        step = m.step;
        rows = m.rows;
        cols = m.cols;
    }
    size_t offset;
    size_t step;
    int rows;
    int cols;
};

struct UMat3D
{
    UMat3D(const UMat& m)
    {
        offset = m.offset;
        step = m.step.p[1];
        slicestep = m.step.p[0];
        slices = m.size.p[0];
        rows = m.size.p[1];
        cols = m.size.p[2];
    }
    size_t offset;
    size_t slicestep;
    size_t step;
    int slices;
    int rows;
    int cols;
};

// Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
static uint64 crc64( const uchar* data, size_t size, uint64 crc0=0 )
{
    static uint64 table[256];
    static bool initialized = false;

    if( !initialized )
    {
        for( int i = 0; i < 256; i++ )
        {
            uint64 c = i;
            for( int j = 0; j < 8; j++ )
                c = ((c & 1) ? CV_BIG_UINT(0xc96c5795d7870f42) : 0) ^ (c >> 1);
            table[i] = c;
        }
        initialized = true;
    }

    uint64 crc = ~crc0;
    for( size_t idx = 0; idx < size; idx++ )
        crc = table[(uchar)crc ^ data[idx]] ^ (crc >> 8);

    return ~crc;
}

struct HashKey
{
    typedef uint64 part;
    HashKey(part _a, part _b) : a(_a), b(_b) {}
    part a, b;
};

inline bool operator == (const HashKey& h1, const HashKey& h2)
{
    return h1.a == h2.a && h1.b == h2.b;
}

inline bool operator < (const HashKey& h1, const HashKey& h2)
{
    return h1.a < h2.a || (h1.a == h2.a && h1.b < h2.b);
}

static cv::Mutex g_initMutex;
static bool g_isInitialized = false;
static bool g_isOpenCLAvailable = false;
bool haveOpenCL()
{
    if (!g_isInitialized)
    {
        cv::AutoLock lock(g_initMutex);
        if (!g_isInitialized)
        {
            try
            {
                cl_uint n = 0;
                cl_int err = ::clGetPlatformIDs(0, NULL, &n);
                if (err != CL_SUCCESS)
                    g_isOpenCLAvailable = false;
                else
                    g_isOpenCLAvailable = true; // TODO !!! We must check default device
            }
            catch (...)
            {
                g_isOpenCLAvailable = false;
            }
            g_isInitialized = true;
        }
    }
    return g_isOpenCLAvailable;
}

bool useOpenCL()
{
    TLSData* data = TLSData::get();
    if( data->useOpenCL < 0 )
        data->useOpenCL = (int)haveOpenCL();
    return data->useOpenCL > 0;
}

void setUseOpenCL(bool flag)
{
    if( haveOpenCL() )
    {
        TLSData* data = TLSData::get();
        data->useOpenCL = flag ? 1 : 0;
    }
}

void finish2()
{
    Queue::getDefault().finish();
}

#define IMPLEMENT_REFCOUNTABLE() \
    void addref() { CV_XADD(&refcount, 1); } \
    void release() { if( CV_XADD(&refcount, -1) == 1 ) delete this; } \
    int refcount

class Platform
{
public:
    Platform();
    ~Platform();
    Platform(const Platform& p);
    Platform& operator = (const Platform& p);

    void* ptr() const;
    static Platform& getDefault();
protected:
    struct Impl;
    Impl* p;
};

struct Platform::Impl
{
    Impl()
    {
        refcount = 1;
        handle = 0;
        initialized = false;
    }

    ~Impl() {}

    void init()
    {
        if( !initialized )
        {
            //cl_uint num_entries
            cl_uint n = 0;
            if( clGetPlatformIDs(1, &handle, &n) < 0 || n == 0 )
                handle = 0;
            if( handle != 0 )
            {
                char buf[1000];
                size_t len = 0;
                clGetPlatformInfo(handle, CL_PLATFORM_VENDOR, sizeof(buf), buf, &len);
                buf[len] = '\0';
                vendor = String(buf);
            }

            initialized = true;
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_platform_id handle;
    String vendor;
    bool initialized;
};

Platform::Platform()
{
    p = 0;
}

Platform::~Platform()
{
    if(p)
        p->release();
}

Platform::Platform(const Platform& pl)
{
    p = (Impl*)pl.p;
    if(p)
        p->addref();
}

Platform& Platform::operator = (const Platform& pl)
{
    Impl* newp = (Impl*)pl.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

void* Platform::ptr() const
{
    return p ? p->handle : 0;
}

Platform& Platform::getDefault()
{
    static Platform p;
    if( !p.p )
    {
        p.p = new Impl;
        p.p->init();
    }
    return p;
}

///////////////////////////////////////////////////////////////////////////////////

struct Device::Impl
{
    Impl(void* d)
    {
        handle = (cl_device_id)d;
    }

    template<typename _TpCL, typename _TpOut>
    _TpOut getProp(cl_device_info prop) const
    {
        _TpCL temp=_TpCL();
        size_t sz = 0;

        return clGetDeviceInfo(handle, prop, sizeof(temp), &temp, &sz) >= 0 &&
            sz == sizeof(temp) ? _TpOut(temp) : _TpOut();
    }

    bool getBoolProp(cl_device_info prop) const
    {
        cl_bool temp = CL_FALSE;
        size_t sz = 0;

        return clGetDeviceInfo(handle, prop, sizeof(temp), &temp, &sz) >= 0 &&
            sz == sizeof(temp) ? temp != 0 : false;
    }

    String getStrProp(cl_device_info prop) const
    {
        char buf[1024];
        size_t sz=0;
        return clGetDeviceInfo(handle, prop, sizeof(buf)-16, buf, &sz) >= 0 &&
            sz < sizeof(buf) ? String(buf) : String();
    }

    IMPLEMENT_REFCOUNTABLE();
    cl_device_id handle;
};


Device::Device()
{
    p = 0;
}

Device::Device(void* d)
{
    p = 0;
    set(d);
}

Device::Device(const Device& d)
{
    p = d.p;
    if(p)
        p->addref();
}

Device& Device::operator = (const Device& d)
{
    Impl* newp = (Impl*)d.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Device::~Device()
{
    if(p)
        p->release();
}

void Device::set(void* d)
{
    if(p)
        p->release();
    p = new Impl(d);
}

void* Device::ptr() const
{
    return p ? p->handle : 0;
}

String Device::name() const
{ return p ? p->getStrProp(CL_DEVICE_NAME) : String(); }

String Device::extensions() const
{ return p ? p->getStrProp(CL_DEVICE_EXTENSIONS) : String(); }

String Device::vendor() const
{ return p ? p->getStrProp(CL_DEVICE_VENDOR) : String(); }

String Device::OpenCL_C_Version() const
{ return p ? p->getStrProp(CL_DEVICE_OPENCL_C_VERSION) : String(); }

String Device::OpenCLVersion() const
{ return p ? p->getStrProp(CL_DEVICE_EXTENSIONS) : String(); }

String Device::driverVersion() const
{ return p ? p->getStrProp(CL_DRIVER_VERSION) : String(); }

int Device::type() const
{ return p ? p->getProp<cl_device_type, int>(CL_DEVICE_TYPE) : 0; }

int Device::addressBits() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_ADDRESS_BITS) : 0; }

bool Device::available() const
{ return p ? p->getBoolProp(CL_DEVICE_AVAILABLE) : false; }

bool Device::compilerAvailable() const
{ return p ? p->getBoolProp(CL_DEVICE_COMPILER_AVAILABLE) : false; }

bool Device::linkerAvailable() const
{ return p ? p->getBoolProp(CL_DEVICE_LINKER_AVAILABLE) : false; }

int Device::doubleFPConfig() const
{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_DOUBLE_FP_CONFIG) : 0; }

int Device::singleFPConfig() const
{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_SINGLE_FP_CONFIG) : 0; }

// TODO cl_ext.h
//int Device::halfFPConfig() const
//{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_HALF_FP_CONFIG) : 0; }

bool Device::endianLittle() const
{ return p ? p->getBoolProp(CL_DEVICE_ENDIAN_LITTLE) : false; }

bool Device::errorCorrectionSupport() const
{ return p ? p->getBoolProp(CL_DEVICE_ERROR_CORRECTION_SUPPORT) : false; }

int Device::executionCapabilities() const
{ return p ? p->getProp<cl_device_exec_capabilities, int>(CL_DEVICE_EXECUTION_CAPABILITIES) : 0; }

size_t Device::globalMemCacheSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) : 0; }

int Device::globalMemCacheType() const
{ return p ? p->getProp<cl_device_mem_cache_type, int>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE) : 0; }

int Device::globalMemCacheLineSize() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) : 0; }

size_t Device::globalMemSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_GLOBAL_MEM_SIZE) : 0; }

size_t Device::localMemSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_LOCAL_MEM_SIZE) : 0; }

int Device::localMemType() const
{ return p ? p->getProp<cl_device_local_mem_type, int>(CL_DEVICE_LOCAL_MEM_TYPE) : 0; }

bool Device::hostUnifiedMemory() const
{ return p ? p->getBoolProp(CL_DEVICE_HOST_UNIFIED_MEMORY) : false; }

bool Device::imageSupport() const
{ return p ? p->getBoolProp(CL_DEVICE_IMAGE_SUPPORT) : false; }

size_t Device::image2DMaxWidth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE2D_MAX_WIDTH) : 0; }

size_t Device::image2DMaxHeight() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE2D_MAX_HEIGHT) : 0; }

size_t Device::image3DMaxWidth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_WIDTH) : 0; }

size_t Device::image3DMaxHeight() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_HEIGHT) : 0; }

size_t Device::image3DMaxDepth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_DEPTH) : 0; }

size_t Device::imageMaxBufferSize() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE) : 0; }

size_t Device::imageMaxArraySize() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE) : 0; }

int Device::maxClockFrequency() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_CLOCK_FREQUENCY) : 0; }

int Device::maxComputeUnits() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_COMPUTE_UNITS) : 0; }

int Device::maxConstantArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_CONSTANT_ARGS) : 0; }

size_t Device::maxConstantBufferSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) : 0; }

size_t Device::maxMemAllocSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_MEM_ALLOC_SIZE) : 0; }

size_t Device::maxParameterSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_PARAMETER_SIZE) : 0; }

int Device::maxReadImageArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_READ_IMAGE_ARGS) : 0; }

int Device::maxWriteImageArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_WRITE_IMAGE_ARGS) : 0; }

int Device::maxSamplers() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_SAMPLERS) : 0; }

size_t Device::maxWorkGroupSize() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE) : 0; }

int Device::maxWorkItemDims() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS) : 0; }

void Device::maxWorkItemSizes(size_t* sizes) const
{
    if(p)
    {
        const int MAX_DIMS = 32;
        size_t retsz = 0;
        clGetDeviceInfo(p->handle, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                MAX_DIMS*sizeof(sizes[0]), &sizes[0], &retsz);
    }
}

int Device::memBaseAddrAlign() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) : 0; }

int Device::nativeVectorWidthChar() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR) : 0; }

int Device::nativeVectorWidthShort() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT) : 0; }

int Device::nativeVectorWidthInt() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT) : 0; }

int Device::nativeVectorWidthLong() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG) : 0; }

int Device::nativeVectorWidthFloat() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT) : 0; }

int Device::nativeVectorWidthDouble() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE) : 0; }

int Device::nativeVectorWidthHalf() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF) : 0; }

int Device::preferredVectorWidthChar() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) : 0; }

int Device::preferredVectorWidthShort() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) : 0; }

int Device::preferredVectorWidthInt() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) : 0; }

int Device::preferredVectorWidthLong() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) : 0; }

int Device::preferredVectorWidthFloat() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) : 0; }

int Device::preferredVectorWidthDouble() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) : 0; }

int Device::preferredVectorWidthHalf() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) : 0; }

size_t Device::printfBufferSize() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_PRINTF_BUFFER_SIZE) : 0; }

size_t Device::profilingTimerResolution() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION) : 0; }

const Device& Device::getDefault()
{
    const Context2& ctx = Context2::getDefault();
    int idx = TLSData::get()->device;
    return ctx.device(idx);
}

/////////////////////////////////////////////////////////////////////////////////////////

struct Context2::Impl
{
    Impl(int dtype0)
    {
        refcount = 1;
        handle = 0;

        cl_int retval = 0;
        cl_platform_id pl = (cl_platform_id)Platform::getDefault().ptr();
        cl_context_properties prop[] =
        {
            CL_CONTEXT_PLATFORM, (cl_context_properties)pl,
            0
        };

        cl_uint i, nd0 = 0, nd = 0;
        int dtype = dtype0 & 15;
        clGetDeviceIDs( pl, dtype, 0, 0, &nd0 );
        if(retval < 0)
            return;
        AutoBuffer<void*> dlistbuf(nd0*2+1);
        cl_device_id* dlist = (cl_device_id*)(void**)dlistbuf;
        cl_device_id* dlist_new = dlist + nd0;
        clGetDeviceIDs(	pl, dtype, nd0, dlist, &nd0 );
        String name0;

        for(i = 0; i < nd0; i++)
        {
            Device d(dlist[i]);
            if( !d.available() || !d.compilerAvailable() )
                continue;
            if( dtype0 == Device::TYPE_DGPU && d.hostUnifiedMemory() )
                continue;
            if( dtype0 == Device::TYPE_IGPU && !d.hostUnifiedMemory() )
                continue;
            String name = d.name();
            if( nd != 0 && name != name0 )
                continue;
            name0 = name;
            dlist_new[nd++] = dlist[i];
        }

        if(nd == 0)
            return;

        // !!! in the current implementation force the number of devices to 1 !!!
        nd = 1;

        handle = clCreateContext(prop, nd, dlist_new, 0, 0, &retval);
        bool ok = handle != 0 && retval >= 0;
        if( ok )
        {
            devices.resize(nd);
            for( i = 0; i < nd; i++ )
                devices[i].set(dlist_new[i]);
        }
    }

    ~Impl()
    {
        if(handle)
            clReleaseContext(handle);
        devices.clear();
    }

    Program getProg(const ProgramSource2& src,
                    const String& buildflags, String& errmsg)
    {
        String prefix = Program::getPrefix(buildflags);
        HashKey k(src.hash(), crc64((const uchar*)prefix.c_str(), prefix.size()));
        phash_t::iterator it = phash.find(k);
        if( it != phash.end() )
            return it->second;
        //String filename = format("%08x%08x_%08x%08x.clb2",
        Program prog(src, buildflags, errmsg);
        if(prog.ptr())
            phash.insert(std::pair<HashKey,Program>(k, prog));
        return prog;
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_context handle;
    std::vector<Device> devices;
    bool initialized;

    typedef ProgramSource2::hash_t hash_t;

    struct HashKey
    {
        HashKey(hash_t _a, hash_t _b) : a(_a), b(_b) {}
        bool operator < (const HashKey& k) const { return a < k.a || (a == k.a && b < k.b); }
        bool operator == (const HashKey& k) const { return a == k.a && b == k.b; }
        bool operator != (const HashKey& k) const { return a != k.a || b != k.b; }
        hash_t a, b;
    };
    typedef std::map<HashKey, Program> phash_t;
    phash_t phash;
};


Context2::Context2()
{
    p = 0;
}

Context2::Context2(int dtype)
{
    p = 0;
    create(dtype);
}

bool Context2::create(int dtype0)
{
    if( !haveOpenCL() )
        return false;
    if(p)
        p->release();
    p = new Impl(dtype0);
    if(!p->handle)
    {
        delete p;
        p = 0;
    }
    return p != 0;
}

Context2::~Context2()
{
    p->release();
}

Context2::Context2(const Context2& c)
{
    p = (Impl*)c.p;
    if(p)
        p->addref();
}

Context2& Context2::operator = (const Context2& c)
{
    Impl* newp = (Impl*)c.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

void* Context2::ptr() const
{
    return p->handle;
}

size_t Context2::ndevices() const
{
    return p ? p->devices.size() : 0;
}

const Device& Context2::device(size_t idx) const
{
    static Device dummy;
    return !p || idx >= p->devices.size() ? dummy : p->devices[idx];
}

Context2& Context2::getDefault()
{
    static Context2 ctx;
    if( !ctx.p && haveOpenCL() )
    {
        // do not create new Context2 right away.
        // First, try to retrieve existing context of the same type.
        // In its turn, Platform::getContext() may call Context2::create()
        // if there is no such context.
        ctx.create(Device::TYPE_ACCELERATOR);
        if(!ctx.p)
            ctx.create(Device::TYPE_DGPU);
        if(!ctx.p)
            ctx.create(Device::TYPE_IGPU);
        if(!ctx.p)
            ctx.create(Device::TYPE_CPU);
    }

    return ctx;
}

Program Context2::getProg(const ProgramSource2& prog,
                         const String& buildopts, String& errmsg)
{
    return p ? p->getProg(prog, buildopts, errmsg) : Program();
}

struct Queue::Impl
{
    Impl(const Context2& c, const Device& d)
    {
        refcount = 1;
        const Context2* pc = &c;
        cl_context ch = (cl_context)pc->ptr();
        if( !ch )
        {
            pc = &Context2::getDefault();
            ch = (cl_context)pc->ptr();
        }
        cl_device_id dh = (cl_device_id)d.ptr();
        if( !dh )
            dh = (cl_device_id)pc->device(0).ptr();
        cl_int retval = 0;
        handle = clCreateCommandQueue(ch, dh, 0, &retval);
    }

    ~Impl()
    {
        if(handle)
        {
            clFinish(handle);
            clReleaseCommandQueue(handle);
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_command_queue handle;
    bool initialized;
};

Queue::Queue()
{
    p = 0;
}

Queue::Queue(const Context2& c, const Device& d)
{
    p = 0;
    create(c, d);
}

Queue::Queue(const Queue& q)
{
    p = q.p;
    if(p)
        p->addref();
}

Queue& Queue::operator = (const Queue& q)
{
    Impl* newp = (Impl*)q.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Queue::~Queue()
{
    if(p)
        p->release();
}

bool Queue::create(const Context2& c, const Device& d)
{
    if(p)
        p->release();
    p = new Impl(c, d);
    return p->handle != 0;
}

void Queue::finish()
{
    if(p && p->handle)
        clFinish(p->handle);
}

void* Queue::ptr() const
{
    return p ? p->handle : 0;
}

Queue& Queue::getDefault()
{
    Queue& q = TLSData::get()->oclQueue;
    if( !q.p )
        q.create(Context2::getDefault());
    return q;
}

static cl_command_queue getQueue(const Queue& q)
{
    cl_command_queue qq = (cl_command_queue)q.ptr();
    if(!qq)
        qq = (cl_command_queue)Queue::getDefault().ptr();
    return qq;
}

KernelArg::KernelArg()
    : flags(0), m(0), obj(0), sz(0), wscale(1)
{
}

KernelArg::KernelArg(int _flags, UMat* _m, int _wscale, const void* _obj, size_t _sz)
    : flags(_flags), m(_m), obj(_obj), sz(_sz), wscale(_wscale)
{
}

KernelArg KernelArg::Constant(const Mat& m)
{
    CV_Assert(m.isContinuous());
    return KernelArg(CONSTANT, 0, 1, m.data, m.total()*m.elemSize());
}


struct Kernel::Impl
{
    Impl(const char* kname, const Program& prog)
    {
        e = 0; refcount = 1;
        cl_program ph = (cl_program)prog.ptr();
        cl_int retval = 0;
        handle = ph != 0 ?
            clCreateKernel(ph, kname, &retval) : 0;
        for( int i = 0; i < MAX_ARRS; i++ )
            u[i] = 0;
    }

    void cleanupUMats()
    {
        for( int i = 0; i < MAX_ARRS; i++ )
            if( u[i] )
            {
                if( CV_XADD(&u[i]->urefcount, -1) == 1 )
                    u[i]->currAllocator->deallocate(u[i]);
                u[i] = 0;
            }
        nu = 0;
    }

    void addUMat(const UMat& m)
    {
        CV_Assert(nu < MAX_ARRS && m.u && m.u->urefcount > 0);
        u[nu] = m.u;
        CV_XADD(&m.u->urefcount, 1);
        nu++;
    }

    void finit()
    {
        cleanupUMats();
        if(e) { clReleaseEvent(e); e = 0; }
        release();
    }

    ~Impl()
    {
        if(handle)
            clReleaseKernel(handle);
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_kernel handle;
    cl_event e;
    enum { MAX_ARRS = 16 };
    UMatData* u[MAX_ARRS];
    int nu;
};

}}

extern "C"
{
static void CL_CALLBACK oclCleanupCallback(cl_event, cl_int, void *p)
{
    ((cv::ocl::Kernel::Impl*)p)->finit();
}

}

namespace cv { namespace ocl {

Kernel::Kernel()
{
    p = 0;
}

Kernel::Kernel(const char* kname, const Program& prog)
{
    p = 0;
    create(kname, prog);
}

Kernel::Kernel(const char* kname, const ProgramSource2& src,
               const String& buildopts, String* errmsg)
{
    p = 0;
    create(kname, src, buildopts, errmsg);
}

Kernel::Kernel(const Kernel& k)
{
    p = k.p;
    if(p)
        p->addref();
}

Kernel& Kernel::operator = (const Kernel& k)
{
    Impl* newp = (Impl*)k.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Kernel::~Kernel()
{
    if(p)
        p->release();
}

bool Kernel::create(const char* kname, const Program& prog)
{
    if(p)
        p->release();
    p = new Impl(kname, prog);
    if(p->handle == 0)
    {
        p->release();
        p = 0;
    }
    return p != 0;
}

bool Kernel::create(const char* kname, const ProgramSource2& src,
                    const String& buildopts, String* errmsg)
{
    if(p)
    {
        p->release();
        p = 0;
    }
    String tempmsg;
    if( !errmsg ) errmsg = &tempmsg;
    const Program& prog = Context2::getDefault().getProg(src, buildopts, *errmsg);
    return create(kname, prog);
}

void* Kernel::ptr() const
{
    return p ? p->handle : 0;
}

bool Kernel::empty() const
{
    return ptr() == 0;
}

int Kernel::set(int i, const void* value, size_t sz)
{
    CV_Assert(i >= 0);
    if( i == 0 )
        p->cleanupUMats();
    if( !p || !p->handle || clSetKernelArg(p->handle, (cl_uint)i, sz, value) < 0 )
        return -1;
    return i+1;
}

int Kernel::set(int i, const UMat& m)
{
    return set(i, KernelArg(KernelArg::READ_WRITE, (UMat*)&m, 0, 0));
}

int Kernel::set(int i, const KernelArg& arg)
{
    CV_Assert( i >= 0 );
    if( i == 0 )
        p->cleanupUMats();
    if( !p || !p->handle )
        return -1;
    if( arg.m )
    {
        int accessFlags = ((arg.flags & KernelArg::READ_ONLY) ? ACCESS_READ : 0) +
                          ((arg.flags & KernelArg::WRITE_ONLY) ? ACCESS_WRITE : 0);
        cl_mem h = (cl_mem)arg.m->handle(accessFlags);

        if( arg.m->dims <= 2 )
        {
            UMat2D u2d(*arg.m);
            clSetKernelArg(p->handle, (cl_uint)i, sizeof(h), &h);
            clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u2d.step), &u2d.step);
            clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u2d.offset), &u2d.offset);
            i += 3;

            if( !(arg.flags & KernelArg::NO_SIZE) )
            {
                int cols = u2d.cols*arg.wscale;
                clSetKernelArg(p->handle, (cl_uint)i, sizeof(u2d.rows), &u2d.rows);
                clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u2d.cols), &cols);
                i += 2;
            }
        }
        else
        {
            UMat3D u3d(*arg.m);
            clSetKernelArg(p->handle, (cl_uint)i, sizeof(h), &h);
            clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u3d.slicestep), &u3d.slicestep);
            clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u3d.step), &u3d.step);
            clSetKernelArg(p->handle, (cl_uint)(i+3), sizeof(u3d.offset), &u3d.offset);
            i += 4;
            if( !(arg.flags & KernelArg::NO_SIZE) )
            {
                int cols = u3d.cols*arg.wscale;
                clSetKernelArg(p->handle, (cl_uint)i, sizeof(u3d.slices), &u3d.rows);
                clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u3d.rows), &u3d.rows);
                clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u3d.cols), &cols);
                i += 3;
            }
        }
        p->addUMat(*arg.m);
        return i;
    }
    clSetKernelArg(p->handle, (cl_uint)i, arg.sz, arg.obj);
    return i+1;
}


bool Kernel::run(int dims, size_t globalsize[], size_t localsize[],
                 bool sync, const Queue& q)
{
    if(!p || !p->handle || p->e != 0)
        return false;
    cl_command_queue qq = getQueue(q);
    size_t offset[CV_MAX_DIM] = {0};
    cl_int retval = clEnqueueNDRangeKernel(qq, p->handle, (cl_uint)dims,
                                           offset, globalsize, localsize, 0, 0,
                                           sync ? 0 : &p->e);
    if( sync || retval < 0 )
    {
        clFinish(qq);
        p->cleanupUMats();
    }
    else
    {
        p->addref();
        clSetEventCallback(p->e, CL_COMPLETE, oclCleanupCallback, p);
    }
    return retval >= 0;
}

bool Kernel::runTask(bool sync, const Queue& q)
{
    if(!p || !p->handle || p->e != 0)
        return false;

    cl_command_queue qq = getQueue(q);
    cl_int retval = clEnqueueTask(qq, p->handle, 0, 0, sync ? 0 : &p->e);
    if( sync || retval < 0 )
    {
        clFinish(qq);
        p->cleanupUMats();
    }
    else
    {
        p->addref();
        clSetEventCallback(p->e, CL_COMPLETE, oclCleanupCallback, p);
    }
    return retval >= 0;
}


size_t Kernel::workGroupSize() const
{
    if(!p)
        return 0;
    size_t val = 0, retsz = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(val), &val, &retsz) >= 0 ? val : 0;
}

bool Kernel::compileWorkGroupSize(size_t wsz[]) const
{
    if(!p || !wsz)
        return 0;
    size_t retsz = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                    sizeof(wsz[0]*3), wsz, &retsz) >= 0;
}

size_t Kernel::localMemSize() const
{
    if(!p)
        return 0;
    size_t retsz = 0;
    cl_ulong val = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(val), &val, &retsz) >= 0 ? (size_t)val : 0;
}

////////////////////////////////////////////////////////////////////////////////////////

struct Program::Impl
{
    Impl(const ProgramSource2& _src,
         const String& _buildflags, String& errmsg)
    {
        refcount = 1;
        const Context2& ctx = Context2::getDefault();
        src = _src;
        buildflags = _buildflags;
        const String& srcstr = src.source();
        const char* srcptr = srcstr.c_str();
        size_t srclen = srcstr.size();
        cl_int retval = 0;

        handle = clCreateProgramWithSource((cl_context)ctx.ptr(), 1, &srcptr, &srclen, &retval);
        if( handle && retval >= 0 )
        {
            int i, n = ctx.ndevices();
            AutoBuffer<void*> deviceListBuf(n+1);
            void** deviceList = deviceListBuf;
            for( i = 0; i < n; i++ )
                deviceList[i] = ctx.device(i).ptr();
            printf("Building the OpenCL program ...\n");
            retval = clBuildProgram(handle, n,
                                    (const cl_device_id*)deviceList,
                                    buildflags.c_str(), 0, 0);
            if( retval == CL_BUILD_PROGRAM_FAILURE )
            {
                char buf[1<<16];
                size_t retsz = 0;
                clGetProgramBuildInfo(handle, (cl_device_id)deviceList[0], CL_PROGRAM_BUILD_LOG,
                                      sizeof(buf)-16, buf, &retsz);
                errmsg = String(buf);
                CV_Error_(Error::StsAssert, ("OpenCL program can not be built: %s", errmsg.c_str()));
            }
            CV_Assert(retval >= 0);
        }
    }

    Impl(const String& _buf, const String& _buildflags)
    {
        refcount = 1;
        handle = 0;
        buildflags = _buildflags;
        if(_buf.empty())
            return;
        String prefix0 = Program::getPrefix(buildflags);
        const Context2& ctx = Context2::getDefault();
        const Device& dev = Device::getDefault();
        const char* pos0 = _buf.c_str();
        const char* pos1 = strchr(pos0, '\n');
        if(!pos1)
            return;
        const char* pos2 = strchr(pos1+1, '\n');
        if(!pos2)
            return;
        const char* pos3 = strchr(pos2+1, '\n');
        if(!pos3)
            return;
        size_t prefixlen = (pos3 - pos0)+1;
        String prefix(pos0, prefixlen);
        if( prefix != prefix0 )
            return;
        const uchar* bin = (uchar*)(pos3+1);
        void* devid = dev.ptr();
        size_t codelen = _buf.length() - prefixlen;
        cl_int binstatus = 0, retval = 0;
        handle = clCreateProgramWithBinary((cl_context)ctx.ptr(), 1, (cl_device_id*)&devid,
                                           &codelen, &bin, &binstatus, &retval);
    }

    String store()
    {
        if(!handle)
            return String();
        size_t progsz = 0, retsz = 0;
        String prefix = Program::getPrefix(buildflags);
        size_t prefixlen = prefix.length();
        if(clGetProgramInfo(handle, CL_PROGRAM_BINARY_SIZES, sizeof(progsz), &progsz, &retsz) < 0)
            return String();
        AutoBuffer<uchar> bufbuf(prefixlen + progsz + 16);
        uchar* buf = bufbuf;
        memcpy(buf, prefix.c_str(), prefixlen);
        buf += prefixlen;
        if(clGetProgramInfo(handle, CL_PROGRAM_BINARIES, sizeof(buf), &buf, &retsz) < 0)
            return String();
        buf[progsz] = (uchar)'\0';
        return String((const char*)(uchar*)bufbuf, prefixlen + progsz);
    }

    ~Impl()
    {
        if( handle )
            clReleaseProgram(handle);
    }

    IMPLEMENT_REFCOUNTABLE();

    ProgramSource2 src;
    String buildflags;
    cl_program handle;
};


Program::Program() { p = 0; }

Program::Program(const ProgramSource2& src,
        const String& buildflags, String& errmsg)
{
    p = 0;
    create(src, buildflags, errmsg);
}

Program::Program(const Program& prog)
{
    p = prog.p;
    if(p)
        p->addref();
}

Program& Program::operator = (const Program& prog)
{
    Impl* newp = (Impl*)prog.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Program::~Program()
{
    if(p)
        p->release();
}

bool Program::create(const ProgramSource2& src,
            const String& buildflags, String& errmsg)
{
    if(p)
        p->release();
    p = new Impl(src, buildflags, errmsg);
    if(!p->handle)
    {
        p->release();
        p = 0;
    }
    return p != 0;
}

const ProgramSource2& Program::source() const
{
    static ProgramSource2 dummy;
    return p ? p->src : dummy;
}

void* Program::ptr() const
{
    return p ? p->handle : 0;
}

bool Program::read(const String& bin, const String& buildflags)
{
    if(p)
        p->release();
    p = new Impl(bin, buildflags);
    return p->handle != 0;
}

bool Program::write(String& bin) const
{
    if(!p)
        return false;
    bin = p->store();
    return !bin.empty();
}

String Program::getPrefix() const
{
    if(!p)
        return String();
    return getPrefix(p->buildflags);
}

String Program::getPrefix(const String& buildflags)
{
    const Context2& ctx = Context2::getDefault();
    const Device& dev = ctx.device(0);
    return format("name=%s\ndriver=%s\nbuildflags=%s\n",
                  dev.name().c_str(), dev.driverVersion().c_str(), buildflags.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////

struct ProgramSource2::Impl
{
    Impl(const char* _src)
    {
        init(String(_src));
    }
    Impl(const String& _src)
    {
        init(_src);
    }
    void init(const String& _src)
    {
        refcount = 1;
        src = _src;
        h = crc64((uchar*)src.c_str(), src.size());
    }

    IMPLEMENT_REFCOUNTABLE();
    String src;
    ProgramSource2::hash_t h;
};


ProgramSource2::ProgramSource2()
{
    p = 0;
}

ProgramSource2::ProgramSource2(const char* prog)
{
    p = new Impl(prog);
}

ProgramSource2::ProgramSource2(const String& prog)
{
    p = new Impl(prog);
}

ProgramSource2::~ProgramSource2()
{
    if(p)
        p->release();
}

ProgramSource2::ProgramSource2(const ProgramSource2& prog)
{
    p = prog.p;
    if(p)
        p->addref();
}

ProgramSource2& ProgramSource2::operator = (const ProgramSource2& prog)
{
    Impl* newp = (Impl*)prog.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

const String& ProgramSource2::source() const
{
    static String dummy;
    return p ? p->src : dummy;
}

ProgramSource2::hash_t ProgramSource2::hash() const
{
    return p ? p->h : 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

class OpenCLAllocator : public MatAllocator
{
public:
    OpenCLAllocator() {}

    UMatData* defaultAllocate(int dims, const int* sizes, int type, size_t* step) const
    {
        UMatData* u = Mat::getStdAllocator()->allocate(dims, sizes, type, step);
        u->urefcount = 1;
        u->refcount = 0;
        return u;
    }

    void getBestFlags(const Context2& ctx, int& createFlags, int& flags0) const
    {
        const Device& dev = ctx.device(0);
        createFlags = CL_MEM_READ_WRITE;

        if( dev.hostUnifiedMemory() )
            flags0 = 0;
        else
            flags0 = UMatData::COPY_ON_MAP;
    }

    UMatData* allocate(int dims, const int* sizes, int type, size_t* step) const
    {
        if(!useOpenCL())
            return defaultAllocate(dims, sizes, type, step);
        size_t total = CV_ELEM_SIZE(type);
        for( int i = dims-1; i >= 0; i-- )
        {
            if( step )
                step[i] = total;
            total *= sizes[i];
        }

        Context2& ctx = Context2::getDefault();
        int createFlags = 0, flags0 = 0;
        getBestFlags(ctx, createFlags, flags0);

        cl_int retval = 0;
        void* handle = clCreateBuffer((cl_context)ctx.ptr(),
                                      createFlags, total, 0, &retval);
        if( !handle || retval < 0 )
            return defaultAllocate(dims, sizes, type, step);
        UMatData* u = new UMatData(this);
        u->data = 0;
        u->size = total;
        u->handle = handle;
        u->urefcount = 1;
        u->flags = flags0;

        return u;
    }

    bool allocate(UMatData* u, int accessFlags) const
    {
        if(!u)
            return false;

        UMatDataAutoLock lock(u);

        if(u->handle == 0)
        {
            CV_Assert(u->origdata != 0);
            Context2& ctx = Context2::getDefault();
            int createFlags = 0, flags0 = 0;
            getBestFlags(ctx, createFlags, flags0);

            cl_context ctx_handle = (cl_context)ctx.ptr();
            cl_int retval = 0;
            int tempUMatFlags = UMatData::TEMP_UMAT;
            u->handle = clCreateBuffer(ctx_handle, CL_MEM_USE_HOST_PTR|createFlags,
                                       u->size, u->origdata, &retval);
            if((!u->handle || retval < 0) && !(accessFlags & ACCESS_FAST))
            {
                u->handle = clCreateBuffer(ctx_handle, CL_MEM_COPY_HOST_PTR|createFlags,
                                           u->size, u->origdata, &retval);
                tempUMatFlags = UMatData::TEMP_COPIED_UMAT;
            }
            if(!u->handle || retval < 0)
                return false;
            u->prevAllocator = u->currAllocator;
            u->currAllocator = this;
            u->flags |= tempUMatFlags;
        }
        if(accessFlags & ACCESS_WRITE)
            u->markHostCopyObsolete(true);
        CV_XADD(&u->urefcount, 1);
        return true;
    }

    void deallocate(UMatData* u) const
    {
        if(!u)
            return;

        // TODO: !!! when we add Shared Virtual Memory Support,
        // this function (as well as the others should be corrected)
        CV_Assert(u->handle != 0 && u->urefcount == 0);
        if(u->tempUMat())
        {
            if( u->hostCopyObsolete() && u->refcount > 0 && u->tempCopiedUMat() )
            {
                clEnqueueWriteBuffer((cl_command_queue)Queue::getDefault().ptr(),
                                     (cl_mem)u->handle, CL_TRUE, 0,
                                     u->size, u->origdata, 0, 0, 0);
            }
            u->markHostCopyObsolete(false);
            clReleaseMemObject((cl_mem)u->handle);
            u->currAllocator = u->prevAllocator;
            if(u->data && u->copyOnMap())
                fastFree(u->data);
            u->data = u->origdata;
            if(u->refcount == 0)
                u->currAllocator->deallocate(u);
        }
        else
        {
            if(u->data && u->copyOnMap())
                fastFree(u->data);
            clReleaseMemObject((cl_mem)u->handle);
            delete u;
        }
    }

    void map(UMatData* u, int accessFlags) const
    {
        if(!u)
            return;

        CV_Assert( u->handle != 0 );

        UMatDataAutoLock autolock(u);

        if(accessFlags & ACCESS_WRITE)
            u->markDeviceCopyObsolete(true);

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        if( u->refcount == 0 )
        {
            if( !u->copyOnMap() )
            {
                CV_Assert(u->data == 0);
                // because there can be other map requests for the same UMat with different access flags,
                // we use the universal (read-write) access mode.
                cl_int retval = 0;
                u->data = (uchar*)clEnqueueMapBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                                     (CL_MAP_READ | CL_MAP_WRITE),
                                                     0, u->size, 0, 0, 0, &retval);
                if(u->data && retval >= 0)
                {
                    u->markHostCopyObsolete(false);
                    return;
                }

                // if map failed, switch to copy-on-map mode for the particular buffer
                u->flags |= UMatData::COPY_ON_MAP;
            }

            if(!u->data)
            {
                u->data = (uchar*)fastMalloc(u->size);
                u->markHostCopyObsolete(true);
            }
        }

        if( (accessFlags & ACCESS_READ) != 0 && u->hostCopyObsolete() )
        {
            CV_Assert( clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                           u->size, u->data, 0, 0, 0) >= 0 );
            u->markHostCopyObsolete(false);
        }
    }

    void unmap(UMatData* u) const
    {
        if(!u)
            return;

        CV_Assert(u->handle != 0);

        UMatDataAutoLock autolock(u);

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
        if( !u->copyOnMap() && u->data )
        {
            CV_Assert( clEnqueueUnmapMemObject(q, (cl_mem)u->handle, u->data, 0, 0, 0) >= 0 );
            u->data = 0;
        }
        else if( u->copyOnMap() && u->deviceCopyObsolete() )
        {
            CV_Assert( clEnqueueWriteBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                            u->size, u->data, 0, 0, 0) >= 0 );
        }
        u->markDeviceCopyObsolete(false);
        u->markHostCopyObsolete(false);
    }

    bool checkContinuous(int dims, const size_t sz[],
                         const size_t srcofs[], const size_t srcstep[],
                         const size_t dstofs[], const size_t dststep[],
                         size_t& total, size_t new_sz[],
                         size_t& srcrawofs, size_t new_srcofs[], size_t new_srcstep[],
                         size_t& dstrawofs, size_t new_dstofs[], size_t new_dststep[]) const
    {
        bool iscontinuous = true;
        srcrawofs = srcofs ? srcofs[dims-1] : 0;
        dstrawofs = dstofs ? dstofs[dims-1] : 0;
        total = sz[dims-1];
        for( int i = dims-2; i >= 0; i-- )
        {
            if( i >= 0 && (total != srcstep[i] || total != dststep[i]) )
                iscontinuous = false;
            total *= sz[i];
            if( srcofs )
                srcrawofs += srcofs[i]*srcstep[i];
            if( dstofs )
                dstrawofs += dstofs[i]*dststep[i];
        }

        if( !iscontinuous )
        {
            // OpenCL uses {x, y, z} order while OpenCV uses {z, y, x} order.
            if( dims == 2 )
            {
                new_sz[0] = sz[1]; new_sz[1] = sz[0]; new_sz[2] = 1;
                // we assume that new_... arrays are initialized by caller
                // with 0's, so there is no else branch
                if( srcofs )
                {
                    new_srcofs[0] = srcofs[1];
                    new_srcofs[1] = srcofs[0];
                    new_srcofs[2] = 0;
                }

                if( dstofs )
                {
                    new_dstofs[0] = dstofs[1];
                    new_dstofs[1] = dstofs[0];
                    new_dstofs[2] = 0;
                }

                new_srcstep[0] = srcstep[0]; new_srcstep[1] = 0;
                new_dststep[0] = dststep[0]; new_dststep[1] = 0;
            }
            else
            {
                // we could check for dims == 3 here,
                // but from user perspective this one is more informative
                CV_Assert(dims <= 3);
                new_sz[0] = sz[2]; new_sz[1] = sz[1]; new_sz[2] = sz[0];
                if( srcofs )
                {
                    new_srcofs[0] = srcofs[2];
                    new_srcofs[1] = srcofs[1];
                    new_srcofs[2] = srcofs[0];
                }

                if( dstofs )
                {
                    new_dstofs[0] = dstofs[2];
                    new_dstofs[1] = dstofs[1];
                    new_dstofs[2] = dstofs[0];
                }

                new_srcstep[0] = srcstep[1]; new_srcstep[1] = srcstep[0];
                new_dststep[0] = dststep[1]; new_dststep[1] = dststep[0];
            }
        }
        return iscontinuous;
    }

    void download(UMatData* u, void* dstptr, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const
    {
        if(!u)
            return;
        UMatDataAutoLock autolock(u);

        if( u->data && !u->hostCopyObsolete() )
        {
            Mat::getStdAllocator()->download(u, dstptr, dims, sz, srcofs, srcstep, dststep);
            return;
        }
        CV_Assert( u->handle != 0 );

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, srcofs, srcstep, 0, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);
        if( iscontinuous )
        {
            CV_Assert( clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                           srcrawofs, total, dstptr, 0, 0, 0) >= 0 );
        }
        else
        {
            CV_Assert( clEnqueueReadBufferRect(q, (cl_mem)u->handle, CL_TRUE,
                            new_srcofs, new_dstofs, new_sz, new_srcstep[0], new_srcstep[1],
                            new_dststep[0], new_dststep[1], dstptr, 0, 0, 0) >= 0 );
        }
    }

    void upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const
    {
        if(!u)
            return;

        // there should be no user-visible CPU copies of the UMat which we are going to copy to
        CV_Assert(u->refcount == 0);

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, 0, srcstep, dstofs, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);

        UMatDataAutoLock autolock(u);

        // if there is cached CPU copy of the GPU matrix,
        // we could use it as a destination.
        // we can do it in 2 cases:
        //    1. we overwrite the whole content
        //    2. we overwrite part of the matrix, but the GPU copy is out-of-date
        if( u->data && (u->hostCopyObsolete() <= u->deviceCopyObsolete() || total == u->size))
        {
            Mat::getStdAllocator()->upload(u, srcptr, dims, sz, dstofs, dststep, srcstep);
            u->markHostCopyObsolete(false);
            u->markDeviceCopyObsolete(true);
            return;
        }

        CV_Assert( u->handle != 0 );
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        if( iscontinuous )
        {
            int crc = 0;
            for( size_t i = 0; i < total; i++ )
                crc ^= ((uchar*)srcptr)[i];
            CV_Assert( clEnqueueWriteBuffer(q, (cl_mem)u->handle,
                CL_TRUE, dstrawofs, total, srcptr, 0, 0, 0) >= 0 );
        }
        else
        {
            CV_Assert( clEnqueueWriteBufferRect(q, (cl_mem)u->handle, CL_TRUE,
                new_dstofs, new_srcofs, new_sz, new_dststep[0], new_dststep[1],
                new_srcstep[0], new_srcstep[1], srcptr, 0, 0, 0) >= 0 );
        }

        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);

        clFinish(q);
    }

    void copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool sync) const
    {
        if(!src || !dst)
            return;

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, srcofs, srcstep, dstofs, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);

        UMatDataAutoLock src_autolock(src);
        UMatDataAutoLock dst_autolock(dst);

        if( !src->handle || (src->data && src->hostCopyObsolete() <= src->deviceCopyObsolete()) )
        {
            upload(dst, src->data + srcrawofs, dims, sz, dstofs, dststep, srcstep);
            return;
        }
        if( !dst->handle || (dst->data && dst->hostCopyObsolete() <= dst->deviceCopyObsolete()) )
        {
            download(src, dst->data + dstrawofs, dims, sz, srcofs, srcstep, dststep);
            dst->markHostCopyObsolete(false);
            dst->markDeviceCopyObsolete(true);
            return;
        }

        // there should be no user-visible CPU copies of the UMat which we are going to copy to
        CV_Assert(dst->refcount == 0);
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        if( iscontinuous )
        {
            CV_Assert( clEnqueueCopyBuffer(q, (cl_mem)src->handle, (cl_mem)dst->handle,
                                           srcrawofs, dstrawofs, total, 0, 0, 0) >= 0 );
        }
        else
        {
            cl_int retval;
            CV_Assert( (retval = clEnqueueCopyBufferRect(q, (cl_mem)src->handle, (cl_mem)dst->handle,
                                               new_srcofs, new_dstofs, new_sz,
                                               new_srcstep[0], new_srcstep[1], new_dststep[0], new_dststep[1],
                                               0, 0, 0)) >= 0 );
        }

        dst->markHostCopyObsolete(true);
        dst->markDeviceCopyObsolete(false);

        if( sync )
            clFinish(q);
    }
};

MatAllocator* getOpenCLAllocator()
{
    static OpenCLAllocator allocator;
    return &allocator;
}

const char* typeToStr(int t)
{
    static const char* tab[]=
    {
        "uchar", "uchar2", "uchar3", "uchar4",
        "char", "char2", "char3", "char4",
        "ushort", "ushort2", "ushort3", "ushort4",
        "short", "short2", "short3", "short4",
        "int", "int2", "int3", "int4",
        "float", "float2", "float3", "float4",
        "double", "double2", "double3", "double4",
        "?", "?", "?", "?"
    };
    int cn = CV_MAT_CN(t);
    return cn > 4 ? "?" : tab[CV_MAT_DEPTH(t)*4 + cn-1];
}

const char* memopTypeToStr(int t)
{
    static const char* tab[]=
    {
        "uchar", "uchar2", "uchar3", "uchar4",
        "uchar", "uchar2", "uchar3", "uchar4",
        "ushort", "ushort2", "ushort3", "ushort4",
        "ushort", "ushort2", "ushort3", "ushort4",
        "int", "int2", "int3", "int4",
        "int", "int2", "int3", "int4",
        "long", "long2", "long3", "long4",
        "?", "?", "?", "?"
    };
    int cn = CV_MAT_CN(t);
    return cn > 4 ? "?" : tab[CV_MAT_DEPTH(t)*4 + cn-1];
}

const char* convertTypeStr(int sdepth, int ddepth, int cn, char* buf)
{
    if( sdepth == ddepth )
        return "noconvert";
    const char *typestr = typeToStr(CV_MAKETYPE(ddepth, cn));
    if( ddepth >= CV_32F ||
        (ddepth == CV_32S && sdepth < CV_32S) ||
        (ddepth == CV_16S && sdepth <= CV_8S) ||
        (ddepth == CV_16U && sdepth == CV_8U))
    {
        sprintf(buf, "convert_%s", typestr);
    }
    else if( sdepth >= CV_32F )
    {
        sprintf(buf, "convert_%s%s_rte", typestr, (ddepth < CV_32S ? "_sat" : ""));
    }
    else
    {
        sprintf(buf, "convert_%s_sat", typestr);
    }
    return buf;
}

}}

#endif
