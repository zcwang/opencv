// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "persistence_hdf5.hpp"

#include <memory>

namespace cv {

class HDF5Loader {
public:
    HDF5Loader() { H5open(); }
    ~HDF5Loader() { H5close(); }
};

void hdf5_activate()
{
    static cv::Ptr<HDF5Loader> instance; \
    if (!instance) \
    { \
        cv::AutoLock lock(cv::getInitializationMutex()); \
        if (!instance) \
            instance.reset(new HDF5Loader()); \
    } \
}

} // namespace
