# CMakeList.txt : CMake project for BitBackend, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_executable (BitBackend "BitBackend.cpp"  "ByBitWebsocket.h" "ByBitWebsocket.cpp" "ByBitAuthentication.cpp" "ByBitAuthentication.h" "BitLib/Logger.cpp" "BitLib/date/date.h" "ByBitConfig.h" "BitLib/json11/json11.cpp" "BitLib/DateTime.cpp")

# TODO: Add tests and install targets if needed.
