// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "opencv_samples_config.hpp"

#include <vector>
#include <fstream>

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/filesystem.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef small
#undef min
#undef max
#undef abs
#endif

namespace cv { namespace samples {

static cv::Ptr< std::vector<cv::String> > g_data_search_path;
static cv::Ptr< std::vector<cv::String> > g_data_search_subdir;

static std::vector<cv::String>& _getDataSearchPath()
{
    if (g_data_search_path.empty())
        g_data_search_path.reset(new std::vector<cv::String>());
    return *(g_data_search_path.get());
}

static std::vector<cv::String>& _getDataSearchSubDirectory()
{
    if (g_data_search_subdir.empty())
    {
        g_data_search_subdir.reset(new std::vector<cv::String>());
        g_data_search_subdir->push_back("samples/data");
        g_data_search_subdir->push_back("data");
        g_data_search_subdir->push_back("");
    }
    return *(g_data_search_subdir.get());
}


CV_EXPORTS void addDataSearchPath(const cv::String& path)
{
    if (utils::fs::isDirectory(path))
        _getDataSearchPath().push_back(path);
}
CV_EXPORTS void addDataSearchSubDirectory(const cv::String& subdir)
{
    _getDataSearchSubDirectory().push_back(subdir);
}

static bool isPathSep(char c)
{
    return c == '/' || c == '\\';
}
static bool isSubDirectory(const cv::String& base_path, const cv::String& path)
{
    CV_LOG_VERBOSE(NULL, 0, "isSubDirectory(): base: " << base_path << "  path: " << path);
    size_t N = base_path.size();
    if (N == 0)
        return false;
    if (isPathSep(base_path[N - 1]))
        N--;
    if (path.size() < N)
        return false;
    for (size_t i = 0; i < N; i++)
    {
        if (path[i] == base_path[i])
            continue;
        if (isPathSep(path[i]) && isPathSep(base_path[i]))
            continue;
        return false;
    }
    return true;
}

static cv::String getModuleLocation(void* addr)
{
#ifdef _WIN32
    HMODULE m = 0;
#if _WIN32_WINNT >= 0x0501
    ::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCTSTR>(addr),
        &m);
#endif
    if (m)
    {
        char path[MAX_PATH];
        const size_t path_size = sizeof(path)/sizeof(*path);
        size_t sz = GetModuleFileNameA(m, path, path_size); // no unicode support
        if (sz > 0 && sz < path_size)
        {
            path[sz] = '\0';
            return cv::String(path);
        }
    }
#elif defined(__linux__)
    std::ifstream fs("/proc/self/maps");
    std::string line;
    while (std::getline(fs, line, '\n'))
    {
        long long int addr_begin = 0, addr_end = 0;
        if (2 == sscanf(line.c_str(), "%llx-%llx", &addr_begin, &addr_end))
        {
            if ((intptr_t)addr >= (intptr_t)addr_begin && (intptr_t)addr < (intptr_t)addr_end)
            {
                size_t pos = line.rfind("  "); // 2 spaces
                if (pos == cv::String::npos)
                    pos = line.rfind(' '); // 1 spaces
                else
                    pos++;
                if (pos == cv::String::npos)
                {
                    CV_LOG_DEBUG(NULL, "Can't parse module path: '" << line << '\'');
                }
                return line.substr(pos + 1);
            }
        }
    }
#else // MacOSX ?
    CV_UNUSED(addr);
#endif
    return cv::String();
}

cv::String findFile(const cv::String& relative_path, bool required)
{
    CV_LOG_DEBUG(NULL, cv::format("cv::samples::findFile('%s', %s)", relative_path.c_str(), required ? "true" : "false"));
#define TRY_FILE_WITH_PREFIX(prefix) \
{ \
    cv::String path = utils::fs::join(prefix, relative_path); \
    CV_LOG_DEBUG(NULL, cv::format("... Line %d Trying %s", __LINE__, path.c_str())); \
    FILE* f = fopen(path.c_str(), "rb"); \
    if(f) { \
        fclose(f); \
        return path; \
    } \
}

    TRY_FILE_WITH_PREFIX("");  // check current directory or absolute path at first

    const std::vector<cv::String>& search_path = _getDataSearchPath();
    for(size_t i = search_path.size(); i > 0; i--)
    {
        const cv::String& prefix = search_path[i - 1];
        TRY_FILE_WITH_PREFIX(prefix);
    }

    const std::vector<cv::String>& search_subdir = _getDataSearchSubDirectory();

#ifndef WINRT
    char* datapath_dir = getenv("OPENCV_SAMPLES_DATA_PATH");
#else
    char* datapath_dir = NULL;
#endif

    cv::String datapath;
    if (datapath_dir)
    {
        datapath = datapath_dir;
        if (utils::fs::isDirectory(datapath))
        {
            CV_LOG_DEBUG(NULL, "samples::findFile(): trying OPENCV_SAMPLES_DATA_PATH=" << datapath);
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const cv::String& subdir = search_subdir[i - 1];
                cv::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
        else
        {
            CV_LOG_WARNING(NULL, "OPENCV_SAMPLES_DATA_PATH is specified but it is not a directory: " << datapath);
        }
    }

    cv::String cwd = utils::fs::getcwd();
    cv::String build_dir(OPENCV_BUILD_DIR);
    bool has_tested_build_directory = false;
    if (isSubDirectory(build_dir, cwd) || isSubDirectory(utils::fs::canonical(build_dir), utils::fs::canonical(cwd)))
    {
        CV_LOG_DEBUG(NULL, "samples::findFile(): the current directory is build sub-directory: " << cwd);
        const char* build_subdirs[] = { OPENCV_BUILD_DIR_SEARCH_PATHS };
        for (size_t k = 0; k < sizeof(build_subdirs)/sizeof(build_subdirs[0]); k++)
        {
            CV_LOG_DEBUG(NULL, "samples::findFile(): <build>/" << build_subdirs[k]);
            datapath = utils::fs::join(build_dir, build_subdirs[k]);
            if (utils::fs::isDirectory(datapath))
            {
                for(size_t i = search_subdir.size(); i > 0; i--)
                {
                    const cv::String& subdir = search_subdir[i - 1];
                    cv::String prefix = utils::fs::join(datapath, subdir);
                    TRY_FILE_WITH_PREFIX(prefix);
                }
            }
        }
        has_tested_build_directory = true;
    }

    cv::String source_dir;
    cv::String try_source_dir = cwd;
    for (int levels = 0; levels < 3; ++levels)
    {
        if (utils::fs::exists(utils::fs::join(try_source_dir, "modules/core/include/opencv2/core/version.hpp")))
        {
            source_dir = try_source_dir;
            break;
        }
        try_source_dir = utils::fs::join(try_source_dir, "/..");
    }
    if (!source_dir.empty())
    {
        CV_LOG_DEBUG(NULL, "samples::findFile(): the current directory is source sub-directory: " << source_dir);
        CV_LOG_DEBUG(NULL, "samples::findFile(): <source>" << source_dir);
        datapath = source_dir;
        if (utils::fs::isDirectory(datapath))
        {
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const cv::String& subdir = search_subdir[i - 1];
                cv::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }

    cv::String module_path = getModuleLocation((void*)findFile);  // use code addr, doesn't work with static linkage!
    CV_LOG_DEBUG(NULL, "Detected module path: '" << module_path << '\'');

    if (!has_tested_build_directory &&
        (isSubDirectory(build_dir, module_path) || isSubDirectory(utils::fs::canonical(build_dir), utils::fs::canonical(module_path)))
    )
    {
        CV_LOG_DEBUG(NULL, "samples::findFile(): the binary module directory is build sub-directory: " << module_path);
        const char* build_subdirs[] = { OPENCV_BUILD_DIR_SEARCH_PATHS };
        for (size_t k = 0; k < sizeof(build_subdirs)/sizeof(build_subdirs[0]); k++)
        {
            CV_LOG_DEBUG(NULL, "samples::findFile(): <build>/" << build_subdirs[k]);
            datapath = utils::fs::join(build_dir, build_subdirs[k]);
            if (utils::fs::isDirectory(datapath))
            {
                for(size_t i = search_subdir.size(); i > 0; i--)
                {
                    const cv::String& subdir = search_subdir[i - 1];
                    cv::String prefix = utils::fs::join(datapath, subdir);
                    TRY_FILE_WITH_PREFIX(prefix);
                }
            }
        }
    }

#if defined OPENCV_INSTALL_SAMPLES_DIR_RELATIVE
    if (!module_path.empty())  // require module path
    {
        size_t pos = module_path.rfind('/');
        if (pos == cv::String::npos)
            pos = module_path.rfind('\\');
        cv::String module_dir = (pos == cv::String::npos) ? module_path : module_path.substr(0, pos);
        datapath = utils::fs::join(module_dir, OPENCV_INSTALL_SAMPLES_DIR_RELATIVE);
        if (utils::fs::isDirectory(datapath))
        {
            CV_LOG_DEBUG(NULL, "samples::findFile(): trying install path (from binary path): " << datapath);
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const cv::String& subdir = search_subdir[i - 1];
                cv::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }
#endif

#if defined OPENCV_INSTALL_PREFIX && defined OPENCV_SAMPLES_DATA_INSTALL_PATH
    cv::String install_dir(OPENCV_INSTALL_PREFIX);
    // use core/world module path and verify that library is running from installation directory
    // It is neccessary to avoid touching of unrelated common /usr/local path
    if (module_path.empty()) // can't determine
        module_path = install_dir;
    if (isSubDirectory(install_dir, module_path) || isSubDirectory(utils::fs::canonical(install_dir), utils::fs::canonical(module_path)))
    {
        datapath = utils::fs::join(install_dir, OPENCV_SAMPLES_DATA_INSTALL_PATH);
        if (utils::fs::isDirectory(datapath))
        {
            CV_LOG_DEBUG(NULL, "samples::findFile(): trying install path: " << datapath);
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const cv::String& subdir = search_subdir[i - 1];
                cv::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }
#endif

    if (required)
        CV_Error(cv::Error::StsError, cv::format("OpenCV samples: Can't find required data file: %s", relative_path.c_str()));
    return cv::String();
}


}} // namespace
