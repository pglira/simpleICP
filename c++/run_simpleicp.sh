#!/usr/bin/env bash

set -e

simpleicp=build/simpleicp

dataset=all

if [[ "${dataset}" == "Dragon" || "${dataset}" == "all" ]]; then
    echo "Processing dataset \"Dragon\""
    ${simpleicp} \
    --fixed ../data/dragon1.xyz \
    --movable ../data/dragon2.xyz
fi

if [[ "${dataset}" == "Airborne Lidar" || "${dataset}" == "all" ]]; then
    echo "Processing dataset \"Airborne Lidar\""
    ${simpleicp} \
    --fixed ../data/airborne_lidar1.xyz \
    --movable ../data/airborne_lidar2.xyz
fi

if [[ "${dataset}" == "Terrestrial Lidar" || "${dataset}" == "all" ]]; then
    echo "Processing dataset \"Airborne Lidar\""
    ${simpleicp} \
    --fixed ../data/terrestrial_lidar1.xyz \
    --movable ../data/terrestrial_lidar2.xyz
fi

if [[ "${dataset}" == "Bunny" || "${dataset}" == "all" ]]; then
    echo "Processing dataset \"Bunny\""
    ${simpleicp} \
    --fixed ../data/bunny_part1.xyz \
    --movable ../data/bunny_part2.xyz \
    --max_overlap_distance 0.01
fi