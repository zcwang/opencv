function(gstreamer_android_configure)
  message(STATUS "Run ndk-build to configure GStreamer...")
  if (NOT DEFINED GSTREAMER_ROOT)
    if (NOT DEFINED ENV{GSTREAMER_ROOT})
      message(FATAL_ERROR "Specify GSTREAMER_ROOT")
    endif()
    set(GSTREAMER_ROOT "$ENV{GSTREAMER_ROOT}")
  endif()
  execute_process(COMMAND ${ANDROID_NDK}/ndk-build GSTREAMER_ROOT=${GSTREAMER_ROOT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/configure
    RESULT_VARIABLE result
    OUTPUT_VARIABLE out_var
    ERROR_VARIABLE err_var
  )
  if (result)
    message(WARNING "${err_var} ")
    message(FATAL_ERROR "GStreamer configuration failed!")
  endif()
  foreach(v GSTREAMER_PLUGINS G_IO_MODULES GSTREAMER_ANDROID_CFLAGS GSTREAMER_ANDROID_LIBS)
    unset(${v} CACHE)
    string(REGEX MATCH "\n${v}=[^\n]*\n" value "${out_var}")
    string(REGEX REPLACE "(\n${v}=)(.*)(\n)" "\\2" value "${value}")
    set(${v} "${value}")
    message(STATUS "${v}=${${v}}")
  endforeach()
  set(GST_PLUGINS_DECLARATION_CODE "")
  set(GST_PLUGINS_REGISTRATION_CODE "")
  string(REPLACE " " ";" GSTREAMER_PLUGINS "${GSTREAMER_PLUGINS}")
  foreach(p ${GSTREAMER_PLUGINS})
    set(GST_PLUGINS_DECLARATION_CODE "${GST_PLUGINS_DECLARATION_CODE}GST_PLUGIN_STATIC_DECLARE(${p});\n")
    set(GST_PLUGINS_REGISTRATION_CODE "${GST_PLUGINS_REGISTRATION_CODE}GST_PLUGIN_STATIC_REGISTER(${p});\n")
  endforeach()
  set(GST_IO_MODULES_DECLARE_CODE "")
  set(GST_IO_MODULES_LOAD_CODE "")
  string(REPLACE " " ";" G_IO_MODULES "${G_IO_MODULES}")
  foreach(m ${G_IO_MODULES})
    set(GST_IO_MODULES_DECLARE_CODE "${GST_IO_MODULES_DECLARE_CODE}GST_G_IO_MODULE_DECLARE(${m});\n")
    set(GST_IO_MODULES_LOAD_CODE "${GST_IO_MODULES_LOAD_CODE}GST_G_IO_MODULE_LOAD(${m});\n")
  endforeach()

  configure_file(${CMAKE_CURRENT_LIST_DIR}/gstreamer_init.cpp.in ${CMAKE_CURRENT_LIST_DIR}/gstreamer_init.cpp)

  string(REGEX REPLACE " *-l| +" ";" GSTREAMER_ANDROID_LIBS "${GSTREAMER_ANDROID_LIBS}")
  string(REPLACE ";stdc++" "" GSTREAMER_ANDROID_LIBS "${GSTREAMER_ANDROID_LIBS}") # OpenCV dependency with '+' brokes CMake
  string(REPLACE ";supc++" "" GSTREAMER_ANDROID_LIBS "${GSTREAMER_ANDROID_LIBS}") # OpenCV dependency with '+' brokes CMake
  string(REPLACE ";vpx" ";${GSTREAMER_ROOT}/lib/libvpx.a" GSTREAMER_ANDROID_LIBS "${GSTREAMER_ANDROID_LIBS}") # force absolute path for this lib also
  set(GSTREAMER_ANDROID_LIBS "-Wl,--whole-archive;-Wl,--allow-multiple-definition;${GSTREAMER_ANDROID_LIBS};-Wl,--no-whole-archive")

  string(REGEX REPLACE " *-I" ";" includes "${GSTREAMER_ANDROID_CFLAGS}")

  # Non needed, we use absolute paths
  #set(GSTREAMER_ANDROID_LINK_DIRS ${GSTREAMER_ROOT}/lib;${GSTREAMER_ROOT}/lib/gstreamer-1.0/static;${GSTREAMER_ROOT}/lib/gio/modules/static)
  set(GSTREAMER_ANDROID_LINK_DIRS "")

  set(GSTREAMER_ANDROID_LIBS "${GSTREAMER_ANDROID_LIBS}" CACHE INTERNAL "" FORCE)
  set(GSTREAMER_ANDROID_LINK_DIRS "${GSTREAMER_ANDROID_LINK_DIRS}" CACHE INTERNAL "" FORCE)
  set(GSTREAMER_ANDROID_INCLUDES "${includes}" CACHE INTERNAL "" FORCE)
endfunction()
