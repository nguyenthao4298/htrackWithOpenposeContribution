# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dante/Documents/handtrack/htrack

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dante/Documents/handtrack/htrack

# Utility rule file for util_headers.

# Include the progress variables for this target.
include util/CMakeFiles/util_headers.dir/progress.make

util_headers: util/CMakeFiles/util_headers.dir/build.make

.PHONY : util_headers

# Rule to build all files generated by this target.
util/CMakeFiles/util_headers.dir/build: util_headers

.PHONY : util/CMakeFiles/util_headers.dir/build

util/CMakeFiles/util_headers.dir/clean:
	cd /home/dante/Documents/handtrack/htrack/util && $(CMAKE_COMMAND) -P CMakeFiles/util_headers.dir/cmake_clean.cmake
.PHONY : util/CMakeFiles/util_headers.dir/clean

util/CMakeFiles/util_headers.dir/depend:
	cd /home/dante/Documents/handtrack/htrack && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dante/Documents/handtrack/htrack /home/dante/Documents/handtrack/htrack/util /home/dante/Documents/handtrack/htrack /home/dante/Documents/handtrack/htrack/util /home/dante/Documents/handtrack/htrack/util/CMakeFiles/util_headers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : util/CMakeFiles/util_headers.dir/depend

