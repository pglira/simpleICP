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

# Plot original and adjusted point clouds with matplotlib
if plot_results:
    import matplotlib.pyplot as plt

    # We need to select a small subset of points for the plot as scatter plots are very slow in mpl
    # https://stackoverflow.com/questions/18179928/speeding-up-matplotlib-scatter-plots
    no_points_to_plot = 10000
    idx_points_fix = np.random.permutation(np.shape(X_fix)[0])[0:no_points_to_plot]
    idx_points_mov = np.random.permutation(np.shape(X_mov)[0])[0:no_points_to_plot]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        X_fix[idx_points_fix, 0],
        X_fix[idx_points_fix, 1],
        X_fix[idx_points_fix, 2],
        c="r",
        marker=",",
    )
    ax.scatter(
        X_mov[idx_points_mov, 0],
        X_mov[idx_points_mov, 1],
        X_mov[idx_points_mov, 2],
        c="g",
        marker=",",
    )
    ax.scatter(
        X_mov_transformed[idx_points_mov, 0],
        X_mov_transformed[idx_points_mov, 1],
        X_mov_transformed[idx_points_mov, 2],
        c="b",
        marker=",",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
