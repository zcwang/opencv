#include "precomp.hpp"

#include "test.opt.hpp"

namespace cv {

void test_cpu_dispatch(float* a, float* b, float* c, size_t len)
{
    CV_CPU_CALL_AVX2(test_cpu_dispatch_avx2(a, b, c, len));
    CV_CPU_CALL_FP16(test_cpu_dispatch_fp16(a, b, c, len));
    CV_CPU_CALL_AVX(test_cpu_dispatch_avx(a, b, c, len));
    CV_CPU_CALL_SSE4_2(test_cpu_dispatch_sse4_2(a, b, c, len));
    CV_CPU_CALL_POPCNT(test_cpu_dispatch_popcnt(a, b, c, len));
    CV_CPU_CALL_NEON(test_cpu_dispatch_neon(a, b, c, len));

    printf("Run generic code\n");
    for (size_t i = 0; i < len; i++)
    {
        c[i] = std::max(a[i], b[i]);
    }
}

}
