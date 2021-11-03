# CMakeList.txt : CMake project for BitBackend, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

add_executable (BitBackend "BitBackend.cpp" "ByBitAuthentication.cpp" "ByBitWebsocket.cpp" "BitLib/json11/json11.cpp" "BitLib/DateTime.cpp" "BitLib/Logger.cpp" "Portfolio.cpp" "Portfolio.h" "BitLib/Uuid.cpp" "BitLib/Uuid.h" "Symbols.cpp")

target_precompile_headers(BitBackend PUBLIC precompiled_headers.h)

add_definitions(-DUUID_SYSTEM_GENERATOR)

set(CMAKE_TOOLCHAIN_FILE "/home/strombom/vcpkg/scripts/buildsystems/vcpkg.cmake")
include("/home/strombom/vcpkg/scripts/buildsystems/vcpkg.cmake")
find_package(unofficial-libuuid CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(Boost)
find_package(OpenSSL REQUIRED)
target_link_libraries(BitBackend PRIVATE unofficial::UUID::uuid)
target_link_libraries(BitBackend PRIVATE cppzmq cppzmq-static)
target_link_libraries(BitBackend PRIVATE OpenSSL::SSL)
target_link_libraries(BitBackend PRIVATE ${Boost_LIBRARIES})
target_include_directories(BitBackend PRIVATE ${Boost_INCLUDE_DIRS})
