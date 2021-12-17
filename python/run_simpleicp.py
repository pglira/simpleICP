"""
Read two point clouds from xyz files and run simpleICP.
"""

from pathlib import Path

import numpy as np

import simpleicp

dataset = "Bunny"
export_results = True

if dataset == "Dragon":
    X_fix = np.genfromtxt(Path("../data/dragon1.xyz"))
    X_mov = np.genfromtxt(Path("../data/dragon2.xyz"))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov)

elif dataset == "Airborne Lidar":
    X_fix = np.genfromtxt(Path("../data/airborne_lidar1.xyz"))
    X_mov = np.genfromtxt(Path("../data/airborne_lidar2.xyz"))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov)

elif dataset == "Terrestrial Lidar":
    X_fix = np.genfromtxt(Path("../data/terrestrial_lidar1.xyz"))
    X_mov = np.genfromtxt(Path("../data/terrestrial_lidar2.xyz"))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov)

elif dataset == "Bunny":
    X_fix = np.genfromtxt(Path("../data/bunny_part1.xyz"))
    X_mov = np.genfromtxt(Path("../data/bunny_part2.xyz"))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov, max_overlap_distance=0.01)

elif dataset == "Multisensor":
    X_fix = np.genfromtxt(Path("../data/multisensor_lidar.xyz"))
    X_mov = np.genfromtxt(Path("../data/multisensor_radar.xyz"))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov, max_overlap_distance=2.0)

# Export original and adjusted point clouds to xyz files to check the result
if export_results:
    target_dir = Path("check")
    target_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(target_dir.joinpath(Path("X_fix.xyz")), X_fix)
    np.savetxt(target_dir.joinpath(Path("X_mov.xyz")), X_mov)
    np.savetxt(target_dir.joinpath(Path("X_mov_transformed.xyz")), X_mov_transformed)
