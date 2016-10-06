#ifdef _MSC_VER
# if _MSC_VER >= 1700
#  pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
# endif
#endif


#include "test_precomp.hpp"

CV_TEST_MAIN("cv")



namespace cv {
void test_cpu_dispatch(float* a, float* b, float* c, size_t len);
}

TEST(cpu_dispatch, simple)
{
    float a[22] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    float b[22] = { 1, -2, -3, 4, 5, -6, 7, -8, 9, 10, -11, 12, 13, -14, -15, -16, -17, 18, 19, 20, 21, -22};
    float c[22] = { 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0};
    cv::test_cpu_dispatch(a, b, c, 22);
    for (size_t i = 0; i < 22; i++)
    {
        printf("%3g %3g %3g\n", a[i], b[i], c[i]);
    }
}
