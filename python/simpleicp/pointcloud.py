"""
PointCloud class.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import spatial

from . import utils


@dataclass
class PointCloud:
    """Data class for working with point cloud data."""

    _X_orig: np.array  # original points as n-by-3 array
    _H: np.array = np.eye(4)  # homogeneous transformation matrix

    X: np.array = field(init=False)  # original points transformed by H
    no_points: int = field(init=False)
    sel: np.array = field(init=False)  # indices of currently "selected" points
    N: np.array = field(init=False)  # normal vectors as n-by-3 array
    planarity: np.array = field(init=False)  # planarity value for each point

    def __post_init__(self):
        self.X = self._X_orig
        self.no_points = self.X.shape[0]
        self.sel = np.arange(0, self.no_points, 1)
        self.N = np.full_like(self.X, np.nan)
        self.planarity = np.full((self.no_points,), np.nan)

    @property
    def no_selected_points(self) -> int:
        """Returns number of selected points."""
        return len(self.sel)

    def set_H(self, H: np.array) -> None:
        """Set new homogeneous transformation matrix H and transform points with it."""

        assert np.shape(H)[0] == 4, "H must have 4 rows!"
        assert np.shape(H)[1] == 4, "H must have 4 columns!"

        self._H = H

        # Update X by applying new transformation to original points
        X_orig_h = utils.euler_coord_to_homogeneous_coord(self._X_orig)
        X_h = np.transpose(self._H @ X_orig_h.T)
        self.X = utils.homogeneous_coord_to_euler_coord(X_h)

    def select_in_range(self, X: np.array, max_range: float):
        """Select points within range of points X."""

        assert np.shape(X)[1] == 3, "X must have 3 columns!"

        kdtree = spatial.cKDTree(X)

        distances, _ = kdtree.query(
            self.X[self.sel, :], k=1, p=2, distance_upper_bound=max_range, workers=-1
        )

        keep = np.isfinite(distances)  # distances > max_range are infinite

        self.sel = self.sel[keep]

    def select_n_points(self, n: int):
        """Select n points equidistantly."""

        if self.no_selected_points > n:
            idx = np.round(np.linspace(0, self.no_selected_points - 1, n)).astype(int)
            self.sel = self.sel[idx]

    def estimate_normals(self, neighbors: int):
        """Estimate normal vectors for selected points from its neighborhood."""

        kdtree = spatial.cKDTree(self.X)
        _, idxNN_all_qp = kdtree.query(
            self.X[self.sel, :], k=neighbors, p=2, workers=-1
        )

        for (i, idxNN) in enumerate(idxNN_all_qp):
            selected_points = self.X[idxNN, :]
            C = np.cov(selected_points.T, bias=False)
            eig_vals, eig_vecs = np.linalg.eig(C)
            idx_sort = eig_vals.argsort()[::-1]  # sort from large to small
            eig_vals = eig_vals[idx_sort]
            eig_vecs = eig_vecs[:, idx_sort]
            self.N[self.sel[i], 0] = eig_vecs[0, 2]
            self.N[self.sel[i], 1] = eig_vecs[1, 2]
            self.N[self.sel[i], 2] = eig_vecs[2, 2]
            self.planarity[self.sel[i]] = (eig_vals[1] - eig_vals[2]) / eig_vals[0]
