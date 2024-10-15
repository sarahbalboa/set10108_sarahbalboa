project(set10108-cw1)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

include_directories(../contrib)

add_executable(cw1-original main.cpp)

add_executable(cw1-serial-chrono main-serial-chrono.cpp)


# Copy the dataset directory to the build directory
file(COPY ${CMAKE_SOURCE_DIR}/dataset DESTINATION ${CMAKE_BINARY_DIR})
