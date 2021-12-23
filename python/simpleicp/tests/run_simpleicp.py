"""
Read two point clouds from xyz files and run simpleICP.
"""

from pathlib import Path

import numpy as np

from simpleicp import simpleicp

dataset = "all"
export_results = False
plot_results = False

tests_dirpath = Path(__file__).parent

if dataset == "Dragon" or dataset == "all":
    print('Processing dataset "Dragon"')
    X_fix = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/dragon1.xyz")))
    X_mov = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/dragon2.xyz")))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov)

if dataset == "Airborne Lidar" or dataset == "all":
    print('Processing dataset "Airborne Lidar"')
    X_fix = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/airborne_lidar1.xyz")))
    X_mov = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/airborne_lidar2.xyz")))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov)

if dataset == "Terrestrial Lidar" or dataset == "all":
    print('Processing dataset "Terrestrial Lidar"')
    X_fix = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/terrestrial_lidar1.xyz")))
    X_mov = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/terrestrial_lidar2.xyz")))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov)

if dataset == "Bunny" or dataset == "all":
    print('Processing dataset "Bunny"')
    X_fix = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/bunny_part1.xyz")))
    X_mov = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/bunny_part2.xyz")))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov, max_overlap_distance=1)

if dataset == "Multisensor" or dataset == "all":
    print('Processing dataset "Multisensor"')
    X_fix = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/multisensor_lidar.xyz")))
    X_mov = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/multisensor_radar.xyz")))
    H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov, max_overlap_distance=2.0)

# Export original and adjusted point clouds to xyz files to check the result
if export_results:
    target_dir = Path("check")
    target_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(target_dir.joinpath(Path("X_fix.xyz")), X_fix)
    np.savetxt(target_dir.joinpath(Path("X_mov.xyz")), X_mov)
    np.savetxt(target_dir.joinpath(Path("X_mov_transformed.xyz")), X_mov_transformed)

# Plot original and adjusted point clouds with open3d viewer
if plot_results:
    import open3d as o3d

    pcd_fix = o3d.geometry.PointCloud()
    pcd_fix.points = o3d.utility.Vector3dVector(X_fix)
    colors = [[1, 0, 0] for i in range(len(pcd_fix.points))]  # red
    pcd_fix.colors = o3d.utility.Vector3dVector(colors)

    pcd_mov = o3d.geometry.PointCloud()
    pcd_mov.points = o3d.utility.Vector3dVector(X_mov)
    colors = [[0, 1, 0] for i in range(len(pcd_mov.points))]  # green
    pcd_mov.colors = o3d.utility.Vector3dVector(colors)

    pcd_mov_transformed = o3d.geometry.PointCloud()
    pcd_mov_transformed.points = o3d.utility.Vector3dVector(X_mov_transformed)
    colors = [[0, 0, 1] for i in range(len(pcd_mov_transformed.points))]  # blue
    pcd_mov_transformed.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd_fix, pcd_mov, pcd_mov_transformed])
