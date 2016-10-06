#include "precomp.hpp"

#if defined CV_ENABLE_INTRINSICS && !CV_CPU_COMPILE_FP16
#error "Compiler configuration error"
#endif

#include "test.opt.hpp"

namespace cv {

void test_cpu_dispatch_fp16(float* a, float* b, float* c, size_t len)
{
    printf("Run FP16+ code\n");
    size_t i = 0;
    for (; i < len; i++)
    {
        c[i] = std::max(a[i], b[i]); // compiler should use vmaxss here (x86 FP16 forces AVX)
    }
#if CV_AVX && !defined CV_CPU_BASELINE_COMPILE_AVX
    _mm256_zeroupper();
#endif
}

}
