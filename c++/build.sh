#!/usr/bin/env bash

set -eux
set -o pipefail

mkdir -p build
[ ! -d "build" ] && exit 1
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake
make -j 4
