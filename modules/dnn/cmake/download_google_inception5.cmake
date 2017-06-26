function(download_google_inception5 dst_dir status_var)
    set(${status_var} TRUE PARENT_SCOPE)
    set(model_graph_filename "tensorflow_inception_graph.pb")
    set(model_graph_dest "${dst_dir}/google/inception5/")
    if(EXISTS ${model_graph_dest}${model_graph_filename})
        set(res TRUE)
    else()
        ocv_download(FILENAME ${model_graph_filename}
                HASH "be71c0d3ba9c5952b11656133588c75c"
                URL "https://raw.githubusercontent.com/opencv/opencv_extra/add.google.inception5.test/testdata/dnn/google/inception5/"
                DESTINATION_DIR ${model_graph_dest}
                ID "dnn/google/inception5"
                RELATIVE_URL
                STATUS res)
    endif()
    set(dnn_samples_dest "${dst_dir}/samples/")
    set(dnn_samples_image1 "space_shuttle.jpg")
    if(NOT EXISTS ${dnn_samples_dest}${dnn_samples_image1})
        ocv_download(FILENAME ${dnn_samples_image1}
                HASH "c6bf45f56551707841620f245e70a252"
                URL "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn"
                DESTINATION_DIR ${dnn_samples_dest}
                ID "dnn/samples"
                RELATIVE_URL
                STATUS res)
    endif()
    if(NOT res)
      set(${status_var} FALSE PARENT_SCOPE)
    endif()
endfunction()