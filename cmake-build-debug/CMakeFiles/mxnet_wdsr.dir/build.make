# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/zhiyu/softWares/intelliJ/clion-2019.2.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/zhiyu/softWares/intelliJ/clion-2019.2.1/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhiyu/CLionProjects/mxnet-wdsr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/mxnet_wdsr.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mxnet_wdsr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mxnet_wdsr.dir/flags.make

CMakeFiles/mxnet_wdsr.dir/src/main.cpp.o: CMakeFiles/mxnet_wdsr.dir/flags.make
CMakeFiles/mxnet_wdsr.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mxnet_wdsr.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mxnet_wdsr.dir/src/main.cpp.o -c /home/zhiyu/CLionProjects/mxnet-wdsr/src/main.cpp

CMakeFiles/mxnet_wdsr.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mxnet_wdsr.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiyu/CLionProjects/mxnet-wdsr/src/main.cpp > CMakeFiles/mxnet_wdsr.dir/src/main.cpp.i

CMakeFiles/mxnet_wdsr.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mxnet_wdsr.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiyu/CLionProjects/mxnet-wdsr/src/main.cpp -o CMakeFiles/mxnet_wdsr.dir/src/main.cpp.s

CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.o: CMakeFiles/mxnet_wdsr.dir/flags.make
CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.o: ../src/predict.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.o -c /home/zhiyu/CLionProjects/mxnet-wdsr/src/predict.cpp

CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiyu/CLionProjects/mxnet-wdsr/src/predict.cpp > CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.i

CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiyu/CLionProjects/mxnet-wdsr/src/predict.cpp -o CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.s

# Object files for target mxnet_wdsr
mxnet_wdsr_OBJECTS = \
"CMakeFiles/mxnet_wdsr.dir/src/main.cpp.o" \
"CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.o"

# External object files for target mxnet_wdsr
mxnet_wdsr_EXTERNAL_OBJECTS =

mxnet_wdsr: CMakeFiles/mxnet_wdsr.dir/src/main.cpp.o
mxnet_wdsr: CMakeFiles/mxnet_wdsr.dir/src/predict.cpp.o
mxnet_wdsr: CMakeFiles/mxnet_wdsr.dir/build.make
mxnet_wdsr: /usr/local/lib/libopencv_gapi.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_stitching.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_aruco.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_bgsegm.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_bioinspired.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_ccalib.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudabgsegm.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudafeatures2d.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudaobjdetect.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudastereo.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_dnn_objdetect.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_dnn_superres.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_dpm.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_face.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_freetype.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_fuzzy.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_hdf.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_hfs.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_img_hash.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_quality.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_reg.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_rgbd.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_saliency.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_sfm.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_stereo.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_structured_light.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_superres.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_surface_matching.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_tracking.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_videostab.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_xphoto.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_highgui.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_shape.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_datasets.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_plot.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_text.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_dnn.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_ml.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudacodec.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_videoio.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudaoptflow.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudalegacy.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudawarping.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_optflow.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_ximgproc.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_video.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_objdetect.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_calib3d.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_features2d.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_flann.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_photo.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudaimgproc.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudafilters.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_imgproc.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudaarithm.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_core.so.4.2.0
mxnet_wdsr: /usr/local/lib/libopencv_cudev.so.4.2.0
mxnet_wdsr: CMakeFiles/mxnet_wdsr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable mxnet_wdsr"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mxnet_wdsr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mxnet_wdsr.dir/build: mxnet_wdsr

.PHONY : CMakeFiles/mxnet_wdsr.dir/build

CMakeFiles/mxnet_wdsr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mxnet_wdsr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mxnet_wdsr.dir/clean

CMakeFiles/mxnet_wdsr.dir/depend:
	cd /home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhiyu/CLionProjects/mxnet-wdsr /home/zhiyu/CLionProjects/mxnet-wdsr /home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug /home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug /home/zhiyu/CLionProjects/mxnet-wdsr/cmake-build-debug/CMakeFiles/mxnet_wdsr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mxnet_wdsr.dir/depend
