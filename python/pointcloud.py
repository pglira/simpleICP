"""
PointCloud class.
"""

import numpy as np
from scipy import spatial


class PointCloud:
    """Class for working with point cloud data."""

    def __init__(self, x: np.array, y: np.array, z: np.array):
        self.x = x
        self.y = y
        self.z = z

        self.nx = None
        self.ny = None
        self.nz = None
        self.planarity = None

        self.no_points = len(x)
        self.sel = np.arange(0, len(x), 1)

    def select_in_range(self, X: np.array, max_range: float):
        """Select points within range of points X."""

        assert np.shape(X)[1] == 3, "X must have 3 columns!"

        kdtree = spatial.cKDTree(X)

        distances, _ = kdtree.query(
            self.X_sel, k=1, p=2, distance_upper_bound=max_range, workers=-1
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

        self.nx = np.full(self.no_points, np.nan)
        self.ny = np.full(self.no_points, np.nan)
        self.nz = np.full(self.no_points, np.nan)
        self.planarity = np.full(self.no_points, np.nan)

        kdtree = spatial.cKDTree(self.X)
        _, idxNN_all_qp = kdtree.query(self.X_sel, k=neighbors, p=2, workers=-1)

        for (i, idxNN) in enumerate(idxNN_all_qp):
            selected_points = np.column_stack(
                (self.x[idxNN], self.y[idxNN], self.z[idxNN])
            )
            C = np.cov(selected_points.T, bias=False)
            eig_vals, eig_vecs = np.linalg.eig(C)
            idx_sort = eig_vals.argsort()[::-1]  # sort from large to small
            eig_vals = eig_vals[idx_sort]
            eig_vecs = eig_vecs[:, idx_sort]
            self.nx[self.sel[i]] = eig_vecs[0, 2]
            self.ny[self.sel[i]] = eig_vecs[1, 2]
            self.nz[self.sel[i]] = eig_vecs[2, 2]
            self.planarity[self.sel[i]] = (eig_vals[1] - eig_vals[2]) / eig_vals[0]

    def transform(self, H: np.array):
        """Transform point cloud by given homogeneous transformation matrix H."""

        XInH = PointCloud.euler_coord_to_homogeneous_coord(self.X)
        XOutH = np.transpose(H @ XInH.T)
        XOut = PointCloud.homogeneous_coord_to_euler_coord(XOutH)

        self.x = XOut[:, 0]
        self.y = XOut[:, 1]
        self.z = XOut[:, 2]

    @property
    def x_sel(self) -> np.array:
        """Returns x coordinates of selected points."""
        return self.x[self.sel]

    @property
    def y_sel(self) -> np.array:
        """Returns y coordinates of selected points."""
        return self.y[self.sel]

    @property
    def z_sel(self) -> np.array:
        """Returns z coordinates of selected points."""
        return self.z[self.sel]

    @property
    def X(self) -> np.array:
        """Returns n-by-3 matrix of all points."""
        return np.column_stack((self.x, self.y, self.z))

    @property
    def X_sel(self) -> np.array:
        """Returns n-by-3 matrix of selected points."""
        return np.column_stack((self.x[self.sel], self.y[self.sel], self.z[self.sel]))

    @property
    def no_selected_points(self) -> int:
        """Returns number of selected points."""
        return len(self.sel)

    @staticmethod
    def euler_coord_to_homogeneous_coord(XE: np.array):
        """Convert Euler coordinates to homogeneous coordinates."""

        no_points = np.shape(XE)[0]
        XH = np.column_stack((XE, np.ones(no_points)))

        return XH

    @staticmethod
    def homogeneous_coord_to_euler_coord(XH: np.array):
        """Convert homogeneous coordinates to Euler coordinates."""

        XE = np.column_stack(
            (XH[:, 0] / XH[:, 3], XH[:, 1] / XH[:, 3], XH[:, 2] / XH[:, 3])
        )

        return XE
