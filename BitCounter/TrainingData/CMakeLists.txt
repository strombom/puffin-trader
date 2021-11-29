# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-EHsc")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project ("TrainingData")

# Include sub-projects.
add_subdirectory ("TrainingData")
