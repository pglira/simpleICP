"""
Read two point clouds from xyz files and run simpleICP.
"""

import csv
from pathlib import Path

import numpy as np

import simpleicp


def read_xyz(path_to_pc: Path) -> np.array:
    """Generate numpy array from xyz file."""
    X = []
    with open(path_to_pc, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            X.append(list(map(float, row)))
    return X


path_to_pc1 = Path("../data/dragon1.xyz")
path_to_pc2 = Path("../data/dragon2.xyz")

# path_to_pc1 = Path('../data/airborne_lidar1.xyz')
# path_to_pc2 = Path('../data/airborne_lidar2.xyz')

# path_to_pc1 = Path('../data/terrestrial_lidar1.xyz')
# path_to_pc2 = Path('../data/terrestrial_lidar2.xyz')

X_fix = np.array(read_xyz(path_to_pc1))
X_mov = np.array(read_xyz(path_to_pc2))

H = simpleicp.simpleicp(X_fix, X_mov)
