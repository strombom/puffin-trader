# CMakeList.txt : CMake project for BitBackend, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

add_executable (BitBackend 
    "BitBackend.cpp" "ByBitAuthentication.cpp" "ByBitWebsocket.cpp" "ByBitRest.cpp"
    "BitLib/json11/json11.cpp" "BitLib/DateTime.cpp" "BitLib/Logger.cpp" "BitLib/Uuid.cpp" 
    "Symbols.cpp" "OrderManager.cpp" "Portfolio.cpp" "OrderBook.cpp" "Common.h" "Common.cpp")

target_precompile_headers(BitBackend PUBLIC precompiled_headers.h)
#add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

add_definitions(-DUUID_SYSTEM_GENERATOR)

set(CMAKE_TOOLCHAIN_FILE "/home/strombom/vcpkg/scripts/buildsystems/vcpkg.cmake")
#include("/home/strombom/vcpkg/scripts/buildsystems/vcpkg.cmake")

find_package(Boost 1.74.0 EXACT REQUIRED)
find_package(unofficial-libuuid CONFIG REQUIRED)
#find_package(cppzmq CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(simdjson CONFIG REQUIRED)

target_include_directories(BitBackend PRIVATE "/usr/local/include")
target_include_directories(BitBackend PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(BitBackend PRIVATE unofficial::UUID::uuid)
#target_link_libraries(BitBackend PRIVATE cppzmq cppzmq-static)
target_link_libraries(BitBackend PRIVATE OpenSSL::SSL)
target_link_libraries(BitBackend PRIVATE ${Boost_LIBRARIES})
target_link_libraries(BitBackend PRIVATE simdjson::simdjson)

#target_link_libraries(BitBackend PRIVATE "/usr/local/lib/libnghttp2.so")
#target_link_libraries(BitBackend PRIVATE "/usr/local/lib/libnghttp2.so")
#target_link_libraries(BitBackend PRIVATE "/usr/local/lib/libnghttp2.a")
target_link_libraries(BitBackend PRIVATE "/usr/local/lib/libnghttp2_asio.a")
