// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace {

using namespace cv;
using namespace std;

TEST(Imgproc_putText, no_garbage)
{
  Size sz(640, 480);
  Mat mat = Mat::zeros(sz, CV_8UC1);

  mat = Scalar::all(0);
  putText(mat, "029", Point(10, 350), 0, 10, Scalar(128), 15);

  EXPECT_EQ(0, cv::countNonZero(mat(Rect(0, 0,           10, sz.height))));
  EXPECT_EQ(0, cv::countNonZero(mat(Rect(sz.width-10, 0, 10, sz.height))));
  EXPECT_EQ(0, cv::countNonZero(mat(Rect(205, 0,         10, sz.height))));
  EXPECT_EQ(0, cv::countNonZero(mat(Rect(405, 0,         10, sz.height))));
}

} // namespace
