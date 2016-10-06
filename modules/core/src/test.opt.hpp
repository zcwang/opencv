namespace cv {

CV_EXPORTS void test_cpu_dispatch(float* a, float* b, float* c, size_t num);
void test_cpu_dispatch_avx2(float* a, float* b, float* c, size_t num);
void test_cpu_dispatch_fp16(float* a, float* b, float* c, size_t num);
void test_cpu_dispatch_avx(float* a, float* b, float* c, size_t num);
void test_cpu_dispatch_sse4_2(float* a, float* b, float* c, size_t num);
void test_cpu_dispatch_popcnt(float* a, float* b, float* c, size_t num);
void test_cpu_dispatch_neon(float* a, float* b, float* c, size_t num);

}
