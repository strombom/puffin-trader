# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.19)

set(CMAKE_TOOLCHAIN_FILE "/home/strombom/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)

project ("BitBackend")
add_subdirectory ("BitBackend")

find_package(cppzmq CONFIG REQUIRED)
find_package(Boost)
find_package(OpenSSL REQUIRED)
target_link_libraries(BitBackend PRIVATE OpenSSL::SSL)
target_link_libraries(BitBackend PRIVATE ${Boost_LIBRARIES})
target_include_directories(BitBackend PRIVATE ${Boost_INCLUDE_DIRS})
