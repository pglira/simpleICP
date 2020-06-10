#!/bin/bash

mkdir -p build
[ ! -d "build" ] && exit 1
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
