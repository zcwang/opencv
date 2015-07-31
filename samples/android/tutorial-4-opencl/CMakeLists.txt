set(sample example-tutorial-4-opencl)

if(BUILD_FAT_JAVA_LIB)
  set(native_deps opencv_java)
else()
  set(native_deps opencv_imgproc)
endif()

# add_android_project_2(target_name ${CMAKE_CURRENT_SOURCE_DIR} LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET 11)
function(add_android_project_2 target path)

  set(ndkbuild_name "ndk-build")
  if(CMAKE_SYSTEM MATCHES Windows)
    set(ndkbuild_name "ndk-build.cmd ndk-build.bat")
  endif()
  find_host_program(NDKBUILD_EXECUTABLE
    NAMES ${ndkbuild_name}
    PATHS
      ENV ANDROID_NDK
      ${ANDROID_NDK}
    NO_DEFAULT_PATH
  )

  if(NOT (ANDROID_EXECUTABLE AND ANT_EXECUTABLE))
    message(FATAL_ERROR "Can't build")
    return()
  endif()

  if(EXISTS ${path}/jni/Android.mk AND NOT NDKBUILD_EXECUTABLE)
    message(FATAL_ERROR "Can't build native")
    return()
  endif()

  cmake_parse_arguments(android_proj "COPY_LIBS" "SDK_TARGET" "NATIVE_DEPS;LIBRARY_DEPS" ${ARGN})

  ocv_check_dependencies(${android_proj_NATIVE_DEPS})

  if(OCV_DEPENDENCIES_FOUND)

    project(${target})
    set(android_proj_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")

    # get project sources
    file(GLOB_RECURSE android_proj_files RELATIVE "${path}" "${path}/*")
    ocv_list_filterout(android_proj_files "CMakeLists.txt")
    if(EXISTS ${path}/jni/Application.mk.in)
      ocv_list_filterout(android_proj_files jni/Application.mk.in)
      ocv_list_filterout(android_proj_files jni/Application.mk)
      configure_file(${path}/jni/Application.mk.in ${android_proj_bin_dir}/jni/Application.mk)
    endif()

    # copy sources out from the build tree
    set(android_proj_file_deps "")
    foreach(f ${android_proj_files})
      add_custom_command(
        OUTPUT "${android_proj_bin_dir}/${f}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${path}/${f}" "${android_proj_bin_dir}/${f}"
        MAIN_DEPENDENCY "${path}/${f}"
        COMMENT "Copying ${f}")
      list(APPEND android_proj_file_deps "${path}/${f}" "${android_proj_bin_dir}/${f}")
    endforeach()

    set(android_proj_lib_deps_commands "")
    set(android_proj_target_files ${ANDROID_PROJECT_FILES})
    ocv_list_add_prefix(android_proj_target_files "${android_proj_bin_dir}/")

    # process Android library dependencies
    foreach(dep ${android_proj_LIBRARY_DEPS})
      file(RELATIVE_PATH __dep "${android_proj_bin_dir}" "${dep}")
      list(APPEND android_proj_lib_deps_commands
        COMMAND ${ANDROID_EXECUTABLE} --silent update project --path "${android_proj_bin_dir}" --library "${__dep}")
    endforeach()

    # fix Android project
    add_custom_target(
        ${target}_fix_android
        COMMAND ${ANDROID_EXECUTABLE} --silent update project --path "${android_proj_bin_dir}" --target "${android_proj_SDK_TARGET}" --name "${target}"
        ${android_proj_lib_deps_commands}
        DEPENDS ${android_proj_file_deps}
        COMMENT "Updating Android project at ${path}. SDK target: ${android_proj_SDK_TARGET}"
        )

    set(build_deps "${target}_fix_android")

    if(EXISTS ${path}/jni/Android.mk)
        add_custom_target(${target}_native
            COMMAND ${NDKBUILD_EXECUTABLE} "OPENCV_BUILD_DIR=${android_proj_LIBRARY_DEPS}"
            DEPENDS ${target}_fix_android
            WORKING_DIRECTORY "${android_proj_bin_dir}"
            COMMENT "Build native part of Android project at ${path}"
        )
        list(APPEND build_deps "${target}_native")
    endif()

    if (android_proj_COPY_LIBS OR ANDROID_EXAMPLES_WITH_LIBS)
      message(STATUS "Android project with libs: " ${target})
      add_custom_target(
        ${target}_copy_libs
        COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${OpenCV_BINARY_DIR}/lib -DDST_DIR=${android_proj_bin_dir}/libs -P ${OpenCV_SOURCE_DIR}/cmake/copyAndroidLibs.cmake
        WORKING_DIRECTORY ${OpenCV_BINARY_DIR}/lib
      )
      if (ANDROID_EXAMPLES_WITH_LIBS)
        add_dependencies(${target}_copy_libs "${OpenCV_BINARY_DIR}/bin/classes.jar.dephelper" opencv_java)
      endif()
      list(APPEND build_deps "${target}_copy_libs")
    endif()

    string(TOLOWER ${CMAKE_BUILD_TYPE} android_build_target)
    add_custom_target(${target} ALL
        COMMAND ${ANT_EXECUTABLE} ${android_build_target}
        DEPENDS ${build_deps}
        WORKING_DIRECTORY "${android_proj_bin_dir}"
        COMMENT "Build Android project at ${path}"
    )

# TODO !!!!!!!!!!!!!!!!
    # put the final .apk to the OpenCV's bin folder
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${android_proj_bin_dir}/bin/${target}-debug.apk" "${OpenCV_BINARY_DIR}/bin/${target}.apk")
    if(INSTALL_ANDROID_EXAMPLES AND "${target}" MATCHES "^example-")
      #apk
      install(FILES "${OpenCV_BINARY_DIR}/bin/${target}.apk" DESTINATION "samples" COMPONENT samples)
      get_filename_component(sample_dir "${path}" NAME)
      #java part
      list(REMOVE_ITEM android_proj_files ${ANDROID_MANIFEST_FILE})
      foreach(f ${android_proj_files} ${ANDROID_MANIFEST_FILE})
        get_filename_component(install_subdir "${f}" PATH)
        install(FILES "${android_proj_bin_dir}/${f}" DESTINATION "samples/${sample_dir}/${install_subdir}" COMPONENT samples)
      endforeach()
      #jni part + eclipse files
      file(GLOB_RECURSE jni_files RELATIVE "${path}" "${path}/jni/*" "${path}/.cproject")
      ocv_list_filterout(jni_files "\\\\.svn")
      foreach(f ${jni_files} ".classpath" ".project" ".settings/org.eclipse.jdt.core.prefs")
        get_filename_component(install_subdir "${f}" PATH)
        install(FILES "${path}/${f}" DESTINATION "samples/${sample_dir}/${install_subdir}" COMPONENT samples)
      endforeach()
      #update proj
      if(android_proj_lib_deps_commands)
        set(inst_lib_opt " --library ../../sdk/java")
      endif()
      install(CODE "EXECUTE_PROCESS(COMMAND ${ANDROID_EXECUTABLE} --silent update project --path . --target \"${android_proj_sdk_target}\" --name \"${target}\" ${inst_lib_opt}
                                    WORKING_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/samples/${sample_dir}\"
                                   )"  COMPONENT samples)
      #empty 'gen'
      install(CODE "MAKE_DIRECTORY(\"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/samples/${sample_dir}/gen\")" COMPONENT samples)
    endif()
  endif()
endfunction()

add_android_project_2(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" NATIVE_DEPS ${native_deps} LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET android-21)
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
