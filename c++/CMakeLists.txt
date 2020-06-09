cmake_minimum_required(VERSION 3.15)
project(run_simpleicp)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(simpleicp
        src/simpleicp-cli.cpp
        src/simpleicp.cpp src/simpleicp.h
        src/pointcloud.cpp src/pointcloud.h
        src/corrpts.cpp src/corrpts.h)

target_link_libraries(simpleicp Eigen3::Eigen)
