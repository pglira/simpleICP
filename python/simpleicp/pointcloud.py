"""
PointCloud class.
"""

from typing import List

import numpy as np
import pandas as pd
from scipy import spatial

from . import mathutils


class PointCloud(pd.DataFrame):
    """Class for working with point cloud data.

    Dev notes:
        - The PointCloud class is a child class of pandas.DataFrame.
        - All attributes which should not be changed outside this class are declared as private,
        e.g. self._num_points. To access these attributes outside this class, getter methods are
        defined, e.g. self.num_points.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Constructor method.

        The PointCloud class is a child class of pandas.DataFrame. Thus, when creating an object
        all arguments are passes to the pandas.DataFrame constructor:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

        Note that the created DataFrame must contain the columns "x", "y", and "z" for the
        coordinates of the points.

        An additional column "selected" of dtype bool is automatically added if missing. It is used
        to mark a selected subset of points.
        """
        super().__init__(*args, **kwargs)

        for coordinate in ("x", "y", "z"):
            if coordinate not in self:
                raise PointCloudException(
                    f'Column "{coordinate}" is missing in DataFrame.'
                )

        self._num_points = len(self)

        if "selected" not in self:
            self["selected"] = [True] * self._num_points

    @property
    def x(self) -> np.array:
        """Returns x coordinates of all points as 1-dim numpy array."""
        return self["x"].to_numpy()

    @property
    def x_selected(self) -> np.array:
        """Returns x coordinates of selected points as 1-dim numpy array."""
        return self.loc[self["selected"], "x"].to_numpy()

    @property
    def y(self) -> np.array:
        """Returns y coordinates of all points as 1-dim numpy array."""
        return self["y"].to_numpy()

    @property
    def y_selected(self) -> np.array:
        """Returns y coordinates of selected points as 1-dim numpy array."""
        return self.loc[self["selected"], "y"].to_numpy()

    @property
    def z(self) -> np.array:
        """Returns z coordinates of all points as 1-dim numpy array."""
        return self["z"].to_numpy()

    @property
    def z_selected(self) -> np.array:
        """Returns z coordinates of selected points as 1-dim numpy array."""
        return self.loc[self["selected"], "z"].to_numpy()

    @property
    def X(self) -> np.array:
        """Returns x,y,z coordinates of all points as numpy array of shape (n,3)."""
        return self[["x", "y", "z"]].to_numpy()

    @property
    def X_selected(self) -> np.array:
        """Returns x,y,z coordinates of selected points as numpy array of shape (n,3)."""
        return self.loc[self["selected"], ["x", "y", "z"]].to_numpy()

    @property
    def idx_selected(self) -> np.array:
        """Returns indices of selected points as 1-dim numpy array."""
        return np.where(self["selected"])[0]

    @idx_selected.setter
    def idx_selected(self, idx_selected: List[int]) -> None:
        """Set indices of selected points."""
        self.unselect_all_points()
        self.loc[idx_selected, "selected"] = True

    @property
    def num_points(self) -> int:
        """Returns total number of points."""
        return self._num_points

    @property
    def num_selected_points(self) -> int:
        """Returns number of selected points."""
        return sum(self["selected"])

    def select_all_points(self) -> None:
        """Select all points."""
        self["selected"].values[:] = True

    def unselect_all_points(self) -> None:
        """Unselect all points."""
        self["selected"].values[:] = False

    def select_by_indices(self, indices: List[int]) -> None:
        """Select points by a list of indices.

        Note that this method selects a subset of the currently selected points. I.e., if a point
        is unselected but its index is contained in idx_selected, it remains unselected. To avoid
        this, call select_all_points() beforehand.

        Args:
            indices (List[int]): List of indices to select.
        """
        self.idx_selected = np.intersect1d(self.idx_selected, indices)

    def select_n_points(self, n: int) -> None:
        """Select n points.

        Note that this method selects a subset of the currently selected points. Points are selected
        equidistantly accross the indices of the currently selected points.

        Args:
            n (int): Number of points to be selected.
        """
        if self.num_selected_points > n:
            idx_subset_n_points = np.round(
                np.linspace(0, self.num_selected_points - 1, n)
            ).astype(int)
            idx_selected_new = self.idx_selected[idx_subset_n_points]
            self.unselect_all_points()
            self.loc[idx_selected_new, "selected"] = True

    def select_in_range(self, X: np.array, max_range: float) -> None:
        """Select points within range of points X.

        Note that this method selects a subset of the currently selected points.

        Args:
            X (np.array): Points of shape (n,3).
            max_range (float): Maximum range from points X.
        """
        if np.shape(X)[1] != 3:
            raise PointCloudException("X must have 3 columns!")

        kdtree = spatial.cKDTree(X)

        distances, _ = kdtree.query(
            self.X_selected, k=1, p=2, distance_upper_bound=max_range, workers=-1
        )

        keep = np.isfinite(distances)  # distances > max_range are infinite

        idx_selected_new = self.idx_selected[keep]
        self.unselect_all_points()
        self.loc[idx_selected_new, "selected"] = True

    def estimate_normals(self, neighbors: int) -> None:
        """Estimate normal vectors for selected points from its neighborhood.

        Args:
            neighbors (int): Number of nearest neighbors to use for normal vector estimation.
        """
        # Initialize point attributes
        nx = np.full((self._num_points,), np.nan, dtype=np.float32)
        ny = np.full((self._num_points,), np.nan, dtype=np.float32)
        nz = np.full((self._num_points,), np.nan, dtype=np.float32)
        planarity = np.full((self._num_points,), np.nan, dtype=np.float32)

        kdtree = spatial.cKDTree(self.X)
        _, idxNN_all_qp = kdtree.query(self.X_selected, k=neighbors, p=2, workers=-1)

        for (i, idxNN) in enumerate(idxNN_all_qp):
            selected_points = self.X[idxNN, :]
            C = np.cov(selected_points.T, bias=False)
            eig_vals, eig_vecs = np.linalg.eig(C)
            idx_sort = eig_vals.argsort()[::-1]  # sort from large to small
            eig_vals = eig_vals[idx_sort]
            eig_vecs = eig_vecs[:, idx_sort]
            nx[self.idx_selected[i]] = eig_vecs[0, 2]
            ny[self.idx_selected[i]] = eig_vecs[1, 2]
            nz[self.idx_selected[i]] = eig_vecs[2, 2]
            planarity[self.idx_selected[i]] = (eig_vals[1] - eig_vals[2]) / eig_vals[0]

        self["nx"] = pd.arrays.SparseArray(nx)
        self["ny"] = pd.arrays.SparseArray(ny)
        self["nz"] = pd.arrays.SparseArray(nz)
        self["planarity"] = pd.arrays.SparseArray(planarity)

    def transform_by_H(self, H: np.array) -> None:
        """Transform points by applying a homogeneous transformation matrix H.

        Args:
            H (np.array): Homogenous transformation matrix H of shape (4,4).
        """
        Xh = mathutils.euler_coord_to_homogeneous_coord(self.X)
        Xh = np.transpose(H @ Xh.T)  # transform in-place to save memory
        Xe = mathutils.homogeneous_coord_to_euler_coord(Xh)

        self["x"] = Xe[:, 0]
        self["y"] = Xe[:, 1]
        self["z"] = Xe[:, 2]


class PointCloudException(Exception):
    """The PointCloud class raises this when the class is misused."""
