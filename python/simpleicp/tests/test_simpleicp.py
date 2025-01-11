"""
Read two point clouds from xyz files and run simpleICP.
"""

import time
from pathlib import Path

import numpy as np
import pytest
from simpleicp import PointCloud, SimpleICP

export_results = False
plot_results = False

tests_dirpath = Path(__file__).parent


def run_simpleicp(X_fix, X_mov, kwargs):
    """Small helper function to run simpleICP for each dataset."""

    # Create point cloud for fixed point cloud
    pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])

    # Create point cloud for movable point cloud (copy=True to keep original X_mov for plot)
    pc_mov = PointCloud(X_mov, columns=["x", "y", "z"], copy=True)

    # Create simpleICP object, add point clouds, and run algorithm!
    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    _, X_mov_transformed, _, _ = icp.run(**kwargs)

    return X_mov_transformed


@pytest.mark.parametrize(
    "dataset, file1, file2, kwargs",
    [
        (
            "Dragon",
            "dragon1.xyz",
            "dragon2.xyz",
            {"debug_dirpath": str(Path("debug").joinpath(f"Dragon_{time.time()}"))},
        ),
        (
            "Airborne Lidar",
            "airborne_lidar1.xyz",
            "airborne_lidar2.xyz",
            {
                "debug_dirpath": str(
                    Path("debug").joinpath(f"Airborne_Lidar_{time.time()}")
                )
            },
        ),
        (
            "Terrestrial Lidar",
            "terrestrial_lidar1.xyz",
            "terrestrial_lidar2.xyz",
            {
                "debug_dirpath": str(
                    Path("debug").joinpath(f"Terrestrial_Lidar_{time.time()}")
                )
            },
        ),
        (
            "Bunny",
            "bunny_part1.xyz",
            "bunny_part2.xyz",
            {
                "max_overlap_distance": 1,
                "debug_dirpath": str(Path("debug").joinpath(f"Bunny_{time.time()}")),
            },
        ),
        (
            "Multisensor",
            "multisensor_lidar.xyz",
            "multisensor_radar.xyz",
            {
                "max_overlap_distance": 1,
                "rbp_observed_values": (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
                "rbp_observation_weights": (np.inf, np.inf, 0.0, 0.0, 0.0, 0.0),
                "debug_dirpath": str(
                    Path("debug").joinpath(f"Multisensor_{time.time()}")
                ),
            },
        ),
        (
            "Webots",
            "webots1.xyz",
            "webots2.xyz",
            {
                "neighbors": 40,
                "max_overlap_distance": 0.5,
                "rbp_observed_values": (0.0, 0.0, -60.0, -0.05, -0.09, 0.0),
                "rbp_observation_weights": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "debug_dirpath": str(Path("debug").joinpath(f"Webots_{time.time()}")),
            },
        ),
    ],
)
def test_simpleicp(dataset, file1, file2, kwargs):
    print(f'Processing dataset "{dataset}"')
    X_fix = np.genfromtxt(tests_dirpath.joinpath(Path(f"../../../data/{file1}")))
    X_mov = np.genfromtxt(tests_dirpath.joinpath(Path(f"../../../data/{file2}")))
    X_mov_transformed = run_simpleicp(X_fix, X_mov, kwargs)

    # Export original and adjusted point clouds to xyz files to check the result
    if export_results:
        target_dir = Path("check")
        target_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(target_dir.joinpath(Path("X_fix.xyz")), X_fix)
        np.savetxt(target_dir.joinpath(Path("X_mov.xyz")), X_mov)
        np.savetxt(
            target_dir.joinpath(Path("X_mov_transformed.xyz")), X_mov_transformed
        )

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
            marker=".",
        )
        ax.scatter(
            X_mov[idx_points_mov, 0],
            X_mov[idx_points_mov, 1],
            X_mov[idx_points_mov, 2],
            c="g",
            marker=".",
        )
        ax.scatter(
            X_mov_transformed[idx_points_mov, 0],
            X_mov_transformed[idx_points_mov, 1],
            X_mov_transformed[idx_points_mov, 2],
            c="b",
            marker=".",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
