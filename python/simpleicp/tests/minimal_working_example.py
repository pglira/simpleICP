"""
Read two point clouds from xyz files and run simpleICP.
"""

from pathlib import Path

import numpy as np
from simpleicp import PointCloud, SimpleICP

# Read point clouds from xyz files into n-by-3 numpy arrays
tests_dirpath = Path(__file__).parent
X_fix = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/bunny_part1.xyz")))
X_mov = np.genfromtxt(tests_dirpath.joinpath(Path("../../../data/bunny_part2.xyz")))

# Create point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rbp = icp.run(max_overlap_distance=1)
