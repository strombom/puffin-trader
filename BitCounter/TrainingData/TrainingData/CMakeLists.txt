# CMakeList.txt : CMake project for TrainingData, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_executable (TrainingData "TrainingData.cpp" "BitLib/DateTime.cpp" "BitLib/Logger.cpp" "TickData.cpp" "Symbols.cpp" "BitLib/csv.h" "Config.h")

target_precompile_headers(TrainingData PUBLIC precompiled_headers.h)

