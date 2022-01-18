"""
Class for corresponding points, i.e. correspondences.
"""

import numpy as np
import pandas as pd
from scipy import spatial, stats

from . import pointcloud


class CorrPts:
    """Class for managing corresponding points between two overlapping point clouds."""

    def __init__(self, pc1: pointcloud.PointCloud, pc2: pointcloud.PointCloud) -> None:
        """Constructor method.

        Args:
            pc1 (pointcloud.PointCloud): Point cloud 1.
            pc2 (pointcloud.PointCloud): Point cloud 2.
        """
        self._pc1 = pc1
        self._pc2 = pc2

        # DataFrame containing the data about the correspondences
        self._df = pd.DataFrame()

    @property
    def pc1_x(self) -> np.array:
        """Returns x coordinates of point cloud 1 of the correspondences."""
        if "pc1_idx" in self._df:
            return self._pc1.iloc[self._df["pc1_idx"]]["x"].to_numpy()
        return None

    @property
    def pc1_y(self) -> np.array:
        """Returns y coordinates of point cloud 1 of the correspondences."""
        if "pc1_idx" in self._df:
            return self._pc1.iloc[self._df["pc1_idx"]]["y"].to_numpy()
        return None

    @property
    def pc1_z(self) -> np.array:
        """Returns z coordinates of point cloud 1 of the correspondences."""
        if "pc1_idx" in self._df:
            return self._pc1.iloc[self._df["pc1_idx"]]["z"].to_numpy()
        return None

    @property
    def pc1_nx(self) -> np.array:
        """Returns x component of the normal vectors of point cloud 1 of the correspondences."""
        if "pc1_idx" in self._df:
            return self._pc1.iloc[self._df["pc1_idx"]]["nx"].to_numpy()
        return None

    @property
    def pc1_ny(self) -> np.array:
        """Returns y component of the normal vectors of point cloud 1 of the correspondences."""
        if "pc1_idx" in self._df:
            return self._pc1.iloc[self._df["pc1_idx"]]["ny"].to_numpy()
        return None

    @property
    def pc1_nz(self) -> np.array:
        """Returns z component of the normal vectors of point cloud 1 of the correspondences."""
        if "pc1_idx" in self._df:
            return self._pc1.iloc[self._df["pc1_idx"]]["nz"].to_numpy()
        return None

    @property
    def pc2_x(self) -> np.array:
        """Returns x coordinates of point cloud 2 of the correspondences."""
        if "pc2_idx" in self._df:
            return self._pc2.iloc[self._df["pc2_idx"]]["x"].to_numpy()
        return None

    @property
    def pc2_y(self) -> np.array:
        """Returns y coordinates of point cloud 2 of the correspondences."""
        if "pc2_idx" in self._df:
            return self._pc2.iloc[self._df["pc2_idx"]]["y"].to_numpy()
        return None

    @property
    def pc2_z(self) -> np.array:
        """Returns z coordinates of point cloud 2 of the correspondences."""
        if "pc2_idx" in self._df:
            return self._pc2.iloc[self._df["pc2_idx"]]["z"].to_numpy()
        return None

    @property
    def pc2_nx(self) -> np.array:
        """Returns x component of the normal vectors of point cloud 2 of the correspondences."""
        if "pc2_idx" in self._df:
            return self._pc2.iloc[self._df["pc2_idx"]]["nx"].to_numpy()
        return None

    @property
    def pc2_ny(self) -> np.array:
        """Returns y component of the normal vectors of point cloud 2 of the correspondences."""
        if "pc2_idx" in self._df:
            return self._pc2.iloc[self._df["pc2_idx"]]["ny"].to_numpy()
        return None

    @property
    def pc2_nz(self) -> np.array:
        """Returns z component of the normal vectors of point cloud 2 of the correspondences."""
        if "pc2_idx" in self._df:
            return self._pc2.iloc[self._df["pc2_idx"]]["nz"].to_numpy()
        return None

    @property
    def point_to_plane_distances(self) -> np.array:
        """Returns point-to-plane distances of correspondences."""
        return self._df["point_to_plane_distances"].to_numpy()

    @property
    def num_corr_pts(self) -> int:
        """Returns number of corresponding points."""
        return len(self._df)

    def match(self) -> None:
        """Matching of point clouds.

        For each *selected* point in self._pc1 the nearest neighbor in all *selected* points of
        self._pc2 is searched.
        """

        kdtree = spatial.cKDTree(self._pc2.X_selected)
        _, idx_nn = kdtree.query(self._pc1.X_selected, k=1, p=2, n_jobs=-1)

        self._df["pc1_idx"] = self._pc1.idx_selected
        self._df["pc2_idx"] = self._pc2.idx_selected[idx_nn]

        self.__compute_point_to_plane_distances()

    def reject_wrt_planarity(self, min_planarity: float) -> None:
        """Rejects corresponding points w.r.t. the planarity of the points.

        Notes:
            - Point attribute "planarity" is estimated in PointCloud.estimate_normals() and is
              defined between 0 and 1.
            - The "planarity" attribute of both point clouds is checked.

        Args:
            min_planarity (float): Minimum planarity value needed to keep correspondence alive.
        """

        if "planarity" in self._pc1:
            keep = (
                self._pc1.iloc[self._df["pc1_idx"]]["planarity"].to_numpy()
                >= min_planarity
            )
            self._df = self._df.loc[keep][:]

        if "planarity" in self._pc2:
            keep = (
                self._pc2.iloc[self._df["pc2_idx"]]["planarity"].to_numpy()
                >= min_planarity
            )
            self._df = self._df.loc[keep][:]

    def reject_wrt_point_to_plane_distances(self) -> None:
        """Reject corresponding points w.r.t. to their point-to-plane distance.

        This method is used to remove correspondence outliers. All correspondences with a
        point-to-plane distance (dp) outside the range

        [median-3*sigma_mad(dp) median+3*sigma_mad(dp)]

        are rejected.

        Note: sigma_mad is a robust estimator for the standard deviation of a data set under the
        assumption that the set has a Gaussian distribution:

        sigma_mad = 1.4812 * mad

        where mad is the median of the absolute differences (with respect to the median) of the data
        set.
        """

        distances = self._df["point_to_plane_distances"]
        median = np.median(distances)
        sigma_mad = stats.median_absolute_deviation(distances)
        keep = [abs(d - median) <= 3 * sigma_mad for d in distances]
        self._df = self._df.loc[keep][:]

    def reject_wrt_to_angle_between_normals(self) -> None:
        """Reject corresponding points w.r.t. to the angle between their normal vectors."""

        raise NotImplementedError

    def __compute_point_to_plane_distances(self) -> None:
        """Compute point-to-plane distances between corresponding points."""

        dx = self.pc2_x - self.pc1_x
        dy = self.pc2_y - self.pc1_y
        dz = self.pc2_z - self.pc1_z

        nx = self.pc1_nx
        ny = self.pc1_ny
        nz = self.pc1_nz

        point_to_plane_distances = np.empty(self.num_corr_pts)

        for i in range(0, self.num_corr_pts):
            point_to_plane_distances[i] = dx[i] * nx[i] + dy[i] * ny[i] + dz[i] * nz[i]

        self._df["point_to_plane_distances"] = point_to_plane_distances
