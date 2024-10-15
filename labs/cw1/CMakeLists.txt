project(set10108-cw1)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

include_directories(../contrib)

# Create executable for main.cpp
add_executable(cw1-original main.cpp)

# Create executable for main2.cpp
add_executable(cw1-serial-chrono main-serial-chrono.cpp)
