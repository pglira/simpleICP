cmake_minimum_required(VERSION 3.15)
project(simpleicp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(cxxopts REQUIRED)
find_package(nanoflann REQUIRED)

add_library(simpleicp_lib
    src/simpleicp.cpp
    src/pointcloud.cpp
    src/corrpts.cpp)
target_link_libraries(simpleicp_lib PUBLIC Eigen3::Eigen cxxopts::cxxopts nanoflann::nanoflann)
target_include_directories(simpleicp_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_executable(simpleicp src/simpleicp-cli.cpp)
target_link_libraries(simpleicp PRIVATE simpleicp_lib)


