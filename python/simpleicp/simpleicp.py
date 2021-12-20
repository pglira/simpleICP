"""
Implementation of a rather simple version of the Iterative Closest Point (ICP) algorithm.
"""

import time
from datetime import datetime
from typing import Tuple

import numpy as np
from scipy import spatial, stats

from .pointcloud import PointCloud


def log(text: str):
    """Print text with time stamp."""
    logtime = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{logtime}] {text}")


def matching(pcfix: PointCloud, pcmov: PointCloud) -> np.array:
    """Matching of point clouds."""
    kdtree = spatial.cKDTree(pcmov.X)
    _, pcmov.sel = kdtree.query(pcfix.X_sel, k=1, p=2, n_jobs=-1)

    dx = pcmov.x_sel - pcfix.x_sel
    dy = pcmov.y_sel - pcfix.y_sel
    dz = pcmov.z_sel - pcfix.z_sel

    nx = pcfix.nx[pcfix.sel]
    ny = pcfix.ny[pcfix.sel]
    nz = pcfix.nz[pcfix.sel]

    no_correspondences = pcfix.no_selected_points
    distances = np.empty(no_correspondences)
    for i in range(0, no_correspondences):
        distances[i] = dx[i] * nx[i] + dy[i] * ny[i] + dz[i] * nz[i]

    return distances


def reject(
    pcfix: PointCloud, pcmov: PointCloud, min_planarity: float, distances: np.array
) -> np.array:
    """Rejection of correspondences on the basis of multiple criterias."""

    planarity = pcfix.planarity[pcfix.sel]

    med = np.median(distances)
    sigmad = stats.median_absolute_deviation(distances)

    keep_distance = np.array([abs(d - med) <= 3 * sigmad for d in distances])
    keep_planarity = np.array([p >= min_planarity for p in planarity])

    keep = keep_distance & keep_planarity

    pcfix.sel = pcfix.sel[keep]
    pcmov.sel = pcmov.sel[keep]
    distances = distances[keep]

    return distances


def estimate_rigid_body_transformation(
    x_fix: np.array,
    y_fix: np.array,
    z_fix: np.array,
    nx_fix: np.array,
    ny_fix: np.array,
    nz_fix: np.array,
    x_mov: np.array,
    y_mov: np.array,
    z_mov: np.array,
) -> Tuple[np.array, np.array]:
    """Estimate rigid body transformation for a given set of correspondences."""

    A = np.column_stack(
        (
            -z_mov * ny_fix + y_mov * nz_fix,
            z_mov * nx_fix - x_mov * nz_fix,
            -y_mov * nx_fix + x_mov * ny_fix,
            nx_fix,
            ny_fix,
            nz_fix,
        )
    )

    l = nx_fix * (x_fix - x_mov) + ny_fix * (y_fix - y_mov) + nz_fix * (z_fix - z_mov)

    x, _, _, _ = np.linalg.lstsq(A, l, rcond=None)

    residuals = A @ x - l

    R = euler_angles_to_linearized_rotation_matrix(x[0], x[1], x[2])

    t = x[3:6]

    H = create_homogeneous_transformation_matrix(R, t)

    return H, residuals


def euler_angles_to_linearized_rotation_matrix(
    alpha1: float, alpha2: float, alpha3: float
) -> np.array:
    """Compute linearized rotation matrix from three Euler angles."""

    dR = np.array([[1, -alpha3, alpha2], [alpha3, 1, -alpha1], [-alpha2, alpha1, 1]])

    return dR


def create_homogeneous_transformation_matrix(R: np.array, t: np.array) -> np.array:
    """Create homogeneous transformation matrix from rotation matrix R and translation vector t."""

    H = np.array(
        [
            [R[0, 0], R[0, 1], R[0, 2], t[0]],
            [R[1, 0], R[1, 1], R[1, 2], t[1]],
            [R[2, 0], R[2, 1], R[2, 2], t[2]],
            [0, 0, 0, 1],
        ]
    )

    return H


def check_convergence_criteria(
    distances_new: np.array, distances_old: np.array, min_change: float
) -> bool:
    """Check if the convergence criteria is met."""

    def change(new, old):
        return np.abs((new - old) / old * 100)

    change_of_mean = change(np.mean(distances_new), np.mean(distances_old))
    change_of_std = change(np.std(distances_new), np.std(distances_old))

    return True if change_of_mean < min_change and change_of_std < min_change else False


def simpleicp(
    X_fix: PointCloud,
    X_mov: PointCloud,
    correspondences: int = 1000,
    neighbors: int = 10,
    min_planarity: float = 0.3,
    max_overlap_distance: float = np.inf,
    min_change: float = 1.0,
    max_iterations: int = 100,
) -> Tuple[np.array, np.array]:
    """Implementation of a rather simple version of the Iterative Closest Point (ICP) algorithm."""

    start_time = time.time()
    log("Create point cloud objects ...")
    pcfix = PointCloud(X_fix[:, 0], X_fix[:, 1], X_fix[:, 2])
    pcmov = PointCloud(X_mov[:, 0], X_mov[:, 1], X_mov[:, 2])

    if np.isfinite(max_overlap_distance):
        log("Consider partial overlap of point clouds ...")
        pcfix.select_in_range(pcmov.X, max_range=max_overlap_distance)
        assert pcfix.no_selected_points > 0, (
            "Point clouds do not overlap within max_overlap_distance = ",
            f"{max_overlap_distance:.3f}! Consider increasing the value of max_overlap_distance.",
        )

    log(
        "Select points for correspondences within overlap area of fixed point cloud ..."
    )
    pcfix.select_n_points(correspondences)
    sel_orig = pcfix.sel

    log("Estimate normals of selected points ...")
    pcfix.estimate_normals(neighbors)

    H = np.eye(4)
    residual_distances = []

    log("Start iterations ...")
    for i in range(0, max_iterations):

        initial_distances = matching(pcfix, pcmov)

        initial_distances = reject(pcfix, pcmov, min_planarity, initial_distances)

        dH, residuals = estimate_rigid_body_transformation(
            pcfix.x_sel,
            pcfix.y_sel,
            pcfix.z_sel,
            pcfix.nx[pcfix.sel],
            pcfix.ny[pcfix.sel],
            pcfix.nz[pcfix.sel],
            pcmov.x_sel,
            pcmov.y_sel,
            pcmov.z_sel,
        )

        residual_distances.append(residuals)

        pcmov.transform(dH)

        H = dH @ H
        pcfix.sel = sel_orig

        if i > 0:
            if check_convergence_criteria(
                residual_distances[i], residual_distances[i - 1], min_change
            ):
                log("Convergence criteria fulfilled -> stop iteration!")
                break

        if i == 0:
            log(
                f"{'Iteration':9s} | "
                f"{'correspondences':15s} | "
                f"{'mean(residuals)':15s} | "
                f"{'std(residuals)':15s}"
            )
            log(
                f"{0:9d} | "
                f"{len(initial_distances):15d} | "
                f"{np.mean(initial_distances):15.4f} | "
                f"{np.std(initial_distances):15.4f}"
            )
        log(
            f"{i+1:9d} | "
            f"{len(residual_distances[i]):15d} | "
            f"{np.mean(residual_distances[i]):15.4f} | "
            f"{np.std(residual_distances[i]):15.4f}"
        )

    log("Estimated transformation matrix H:")
    log(f"H = [{H[0, 0]:12.6f} {H[0, 1]:12.6f} {H[0, 2]:12.6f} {H[0, 3]:12.6f}]")
    log(f"H = [{H[1, 0]:12.6f} {H[1, 1]:12.6f} {H[1, 2]:12.6f} {H[1, 3]:12.6f}]")
    log(f"H = [{H[2, 0]:12.6f} {H[2, 1]:12.6f} {H[2, 2]:12.6f} {H[2, 3]:12.6f}]")
    log(f"H = [{H[3, 0]:12.6f} {H[3, 1]:12.6f} {H[3, 2]:12.6f} {H[3, 3]:12.6f}]")
    log(f"Finished in {time.time() - start_time:.3f} seconds!")

    return H, pcmov.X
