# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /snap/clion/129/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/129/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kostya/test2/gym_interf/gym_interf/envs/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/interf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/interf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/interf.dir/flags.make

CMakeFiles/interf.dir/src/utils.cpp.o: CMakeFiles/interf.dir/flags.make
CMakeFiles/interf.dir/src/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/interf.dir/src/utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/interf.dir/src/utils.cpp.o -c /home/kostya/test2/gym_interf/gym_interf/envs/cpp/src/utils.cpp

CMakeFiles/interf.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/interf.dir/src/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kostya/test2/gym_interf/gym_interf/envs/cpp/src/utils.cpp > CMakeFiles/interf.dir/src/utils.cpp.i

CMakeFiles/interf.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/interf.dir/src/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kostya/test2/gym_interf/gym_interf/envs/cpp/src/utils.cpp -o CMakeFiles/interf.dir/src/utils.cpp.s

CMakeFiles/interf.dir/src/interflib.cpp.o: CMakeFiles/interf.dir/flags.make
CMakeFiles/interf.dir/src/interflib.cpp.o: ../src/interflib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/interf.dir/src/interflib.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/interf.dir/src/interflib.cpp.o -c /home/kostya/test2/gym_interf/gym_interf/envs/cpp/src/interflib.cpp

CMakeFiles/interf.dir/src/interflib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/interf.dir/src/interflib.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kostya/test2/gym_interf/gym_interf/envs/cpp/src/interflib.cpp > CMakeFiles/interf.dir/src/interflib.cpp.i

CMakeFiles/interf.dir/src/interflib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/interf.dir/src/interflib.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kostya/test2/gym_interf/gym_interf/envs/cpp/src/interflib.cpp -o CMakeFiles/interf.dir/src/interflib.cpp.s

# Object files for target interf
interf_OBJECTS = \
"CMakeFiles/interf.dir/src/utils.cpp.o" \
"CMakeFiles/interf.dir/src/interflib.cpp.o"

# External object files for target interf
interf_EXTERNAL_OBJECTS =

libinterf.so: CMakeFiles/interf.dir/src/utils.cpp.o
libinterf.so: CMakeFiles/interf.dir/src/interflib.cpp.o
libinterf.so: CMakeFiles/interf.dir/build.make
libinterf.so: CMakeFiles/interf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libinterf.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/interf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/interf.dir/build: libinterf.so

.PHONY : CMakeFiles/interf.dir/build

CMakeFiles/interf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/interf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/interf.dir/clean

CMakeFiles/interf.dir/depend:
	cd /home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kostya/test2/gym_interf/gym_interf/envs/cpp /home/kostya/test2/gym_interf/gym_interf/envs/cpp /home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug /home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug /home/kostya/test2/gym_interf/gym_interf/envs/cpp/cmake-build-debug/CMakeFiles/interf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/interf.dir/depend

