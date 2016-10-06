#include "precomp.hpp"

#if defined CV_ENABLE_INTRINSICS && !CV_NEON
#error "Compiler configuration error"
#endif

#include "test.opt.hpp"

namespace cv {

void test_cpu_dispatch_neon(float* a, float* b, float* c, size_t len)
{
    printf("Run NEON code\n");
    size_t i = 0;
    for (; i < len; i++)
    {
        c[i] = std::max(a[i], b[i]);
    }
}

}
