#ifdef HAVE_OPENCV_CORE

#include "opencv2/core.hpp"

#ifdef __cv_strict_types
CV_PY_FROM_ENUM(ElemType);
CV_PY_TO_ENUM(ElemType);

CV_PY_FROM_ENUM(ElemDepth);
CV_PY_TO_ENUM(ElemDepth);
#endif

#endif
