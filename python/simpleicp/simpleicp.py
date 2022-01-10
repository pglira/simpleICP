"""
Implementation of a rather simple version of the Iterative Closest Point (ICP) algorithm.
"""

import time
from dataclasses import dataclass
from typing import Tuple

import lmfit
import numpy as np
from lmfit.parameter import Parameters
from scipy import spatial, stats

from . import utils
from .pointcloud import PointCloud


@dataclass
class Parameter:
    """Data class for a single optimization parameter."""

    estimated_value: float = np.nan
    estimated_uncertainty: float = np.nan
    observed_value: float = np.nan
    observation_weight: float = np.nan
    scale_for_report: float = 1


@dataclass
class RigidBodyParameters:

    alpha1: Parameter = Parameter()
    alpha2: Parameter = Parameter()
    alpha3: Parameter = Parameter()
    tx: Parameter = Parameter()
    ty: Parameter = Parameter()
    tz: Parameter = Parameter()


def matching(pcfix: PointCloud, pcmov: PointCloud) -> np.array:
    """Matching of point clouds."""
    kdtree = spatial.cKDTree(pcmov.X)
    _, pcmov.sel = kdtree.query(pcfix.X[pcfix.sel, :], k=1, p=2, n_jobs=-1)

    dx = pcmov.X[pcmov.sel, 0] - pcfix.X[pcfix.sel, 0]
    dy = pcmov.X[pcmov.sel, 1] - pcfix.X[pcfix.sel, 1]
    dz = pcmov.X[pcmov.sel, 2] - pcfix.X[pcfix.sel, 2]

    nx = pcfix.N[pcfix.sel, 0]
    ny = pcfix.N[pcfix.sel, 1]
    nz = pcfix.N[pcfix.sel, 2]

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


def compute_residuals(
    parameter: lmfit.Parameters,
    X_fix: np.array,
    N_fix: np.array,
    X_mov: np.array,
    weights_distance_residuals: float,
    tf_obs: Tuple[float],
    weights_tf_obs_residuals: Tuple[float],
) -> np.array:
    """Compute residuals of optimization problem."""

    (
        x_mov_transformed,
        y_mov_transformed,
        z_mov_transformed,
    ) = transform_point_cloud(parameter, X_mov)

    weighted_distance_residuals = compute_weighted_distance_residuals(
        x_fix,
        y_fix,
        z_fix,
        nx_fix,
        ny_fix,
        nz_fix,
        x_mov_transformed,
        y_mov_transformed,
        z_mov_transformed,
        weights_distance_residuals,
    )

    weighted_tf_obs_residuals = compute_weighted_tf_obs_residuals(
        parameter, tf_obs, weights_tf_obs_residuals
    )

    # Combine all residuals
    weighted_residuals = np.concatenate(
        (weighted_distance_residuals, weighted_tf_obs_residuals)
    )

    return weighted_residuals


def transform_point_cloud(
    parameter: lmfit.Parameters, x: np.array, y: np.array, z: np.array
) -> Tuple[np.array, np.array, np.array]:
    """Transform point cloud by rigid body transformation."""

    H = params_to_homogeneous_transformation_matrix(parameter)

    X = np.column_stack((x, y, z))
    X_h = utils.euler_coord_to_homogeneous_coord(X)
    X_transformed_h = np.transpose(H @ X_h.T)
    X_transformed = utils.homogeneous_coord_to_euler_coord(X_transformed_h)

    x_transformed = X_transformed[:, 0]
    y_transformed = X_transformed[:, 1]
    z_transformed = X_transformed[:, 2]

    return x_transformed, y_transformed, z_transformed


def compute_weighted_distance_residuals(
    x_fix: np.array,
    y_fix: np.array,
    z_fix: np.array,
    nx_fix: np.array,
    ny_fix: np.array,
    nz_fix: np.array,
    x_mov: np.array,
    y_mov: np.array,
    z_mov: np.array,
    weights_distance_residuals: float,
) -> np.array:
    """Compute weighted residuals for point-to-plane distances between correspondences."""

    no_correspondences = len(x_fix)

    dx = x_mov - x_fix
    dy = y_mov - y_fix
    dz = z_mov - z_fix

    distance_residuals = np.empty(no_correspondences)
    for i in range(0, no_correspondences):
        distance_residuals[i] = (
            dx[i] * nx_fix[i] + dy[i] * ny_fix[i] + dz[i] * nz_fix[i]
        )

    weighted_distance_residuals = weights_distance_residuals * distance_residuals

    return weighted_distance_residuals


def compute_weighted_tf_obs_residuals(
    parameter: lmfit.Parameters,
    tf_obs: Tuple[float],
    weights_tf_obs_residuals: Tuple[float],
) -> np.array:
    """Compute weighted residuals for direct observation of transform parameters."""

    tf_obs_residuals = np.empty(6)

    tf_obs_residuals[0] = parameter["alpha1"].value
    tf_obs_residuals[1] = parameter["alpha2"].value
    tf_obs_residuals[2] = parameter["alpha3"].value
    tf_obs_residuals[3] = parameter["tx"].value
    tf_obs_residuals[4] = parameter["ty"].value
    tf_obs_residuals[5] = parameter["tz"].value

    weighted_tf_obs_residuals = weights_tf_obs_residuals * tf_obs_residuals

    keep = np.isfinite(weights_tf_obs_residuals) & (
        np.array(weights_tf_obs_residuals) > 0
    )

    # Remove residuals where weight is infinite
    weighted_tf_obs_residuals = weighted_tf_obs_residuals[keep]

    return weighted_tf_obs_residuals


def estimate_rigid_body_transformation(
    X_fix: np.array,
    N_fix: np.array,
    X_mov: np.array,
    weights_distance_residuals: float,
    tf_obs: Tuple[float],
    weights_tf_obs_residuals: Tuple[float],
) -> Tuple[np.array, np.array, lmfit.minimizer.MinimizerResult]:
    """Estimate rigid body transformation for a given set of correspondences."""

    # Define rigid body parameters
    params = lmfit.Parameters()
    params.add("alpha1", value=0, vary=np.isfinite(weights_tf_obs_residuals[0]))
    params.add("alpha2", value=0, vary=np.isfinite(weights_tf_obs_residuals[1]))
    params.add("alpha3", value=0, vary=np.isfinite(weights_tf_obs_residuals[2]))
    params.add("tx", value=0, vary=np.isfinite(weights_tf_obs_residuals[3]))
    params.add("ty", value=0, vary=np.isfinite(weights_tf_obs_residuals[4]))
    params.add("tz", value=0, vary=np.isfinite(weights_tf_obs_residuals[5]))

    optim_result = lmfit.minimize(
        compute_residuals,
        params,
        args=(
            X_fix,
            N_fix,
            X_mov,
            weights_distance_residuals,
            tf_obs,
            weights_tf_obs_residuals,
        ),
    )

    H = params_to_homogeneous_transformation_matrix(optim_result.params)

    # Compute unweighted distance residuals
    no_correspondences = X_fix.shape[0]
    weighted_distance_residuals = optim_result.residual[0 : no_correspondences - 1]
    unweighted_distance_residuals = (
        weighted_distance_residuals / weights_distance_residuals
    )

    return H, unweighted_distance_residuals, optim_result


def params_to_homogeneous_transformation_matrix(
    params: lmfit.Parameters,
) -> np.array:
    """Compute homogeneous transformation matrix from lmfit parameters."""

    R = utils.euler_angles_to_linearized_rotation_matrix(
        params["alpha1"].value,
        params["alpha2"].value,
        params["alpha3"].value,
    )

    t = np.array(
        [
            params["tx"].value,
            params["ty"].value,
            params["tz"].value,
        ]
    )

    H = utils.create_homogeneous_transformation_matrix(R, t)

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
    X_fix: np.array,
    X_mov: np.array,
    correspondences: int = 1000,
    neighbors: int = 10,
    min_planarity: float = 0.3,
    max_overlap_distance: float = np.inf,
    min_change: float = 1.0,
    max_iterations: int = 100,
    weights_distance_residuals: float = 1.0,
    tf_obs: Tuple[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    weights_tf_obs_residuals: Tuple[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
) -> Tuple[np.array, np.array]:
    """Implementation of a rather simple version of the Iterative Closest Point (ICP) algorithm."""

    # Some input validation
    # TODO Input of tf_obs in degree for first 3 elements?
    assert weights_distance_residuals > 0, "Weights must be > 0!"
    assert len(tf_obs) == 6, "Tuple must have exactly 6 elements!"
    assert len(weights_tf_obs_residuals) == 6, "Tuple must have exactly 6 elements!"
    assert all([w >= 0 for w in weights_tf_obs_residuals]), "Weights must be >= 0!"
    assert any(
        np.isfinite(weights_tf_obs_residuals)
    ), "At least one weight must be finite!"

    start_time = time.time()
    print("Create point cloud objects ...")
    pcfix = PointCloud(X_fix)
    pcmov = PointCloud(X_mov)

    # Set initial transform
    R = utils.euler_angles_to_rotation_matrix(tf_obs[0], tf_obs[1], tf_obs[2])
    t = tf_obs[3:]
    H = utils.create_homogeneous_transformation_matrix(R, t)
    pcmov.set_H(H)

    if np.isfinite(max_overlap_distance):
        print("Consider partial overlap of point clouds ...")
        pcfix.select_in_range(pcmov.X, max_range=max_overlap_distance)
        assert pcfix.no_selected_points > 0, (
            "Point clouds do not overlap within max_overlap_distance = ",
            f"{max_overlap_distance:.5f}! Consider increasing the value of max_overlap_distance.",
        )

    print(
        "Select points for correspondences within overlap area of fixed point cloud ..."
    )
    pcfix.select_n_points(correspondences)
    sel_orig = pcfix.sel

    print("Estimate normals of selected points ...")
    pcfix.estimate_normals(neighbors)

    # Initialize some variables
    rigid_body_parameters = RigidBodyParameters
    rigid_body_parameters.alpha1.observed_value = tf_obs[0]
    rigid_body_parameters.alpha2.observed_value = tf_obs[1]
    rigid_body_parameters.alpha3.observed_value = tf_obs[2]
    rigid_body_parameters.tx.observed_value = tf_obs[3]
    rigid_body_parameters.ty.observed_value = tf_obs[4]
    rigid_body_parameters.tz.observed_value = tf_obs[5]
    rigid_body_parameters.alpha1.observation_weight = weights_tf_obs_residuals[0]
    rigid_body_parameters.alpha2.observation_weight = weights_tf_obs_residuals[1]
    rigid_body_parameters.alpha3.observation_weight = weights_tf_obs_residuals[2]
    rigid_body_parameters.tx.observation_weight = weights_tf_obs_residuals[3]
    rigid_body_parameters.ty.observation_weight = weights_tf_obs_residuals[4]
    rigid_body_parameters.tz.observation_weight = weights_tf_obs_residuals[5]
    rigid_body_parameters.alpha1.scale_for_report = 180 / np.pi
    rigid_body_parameters.alpha2.scale_for_report = 180 / np.pi
    rigid_body_parameters.alpha3.scale_for_report = 180 / np.pi

    residual_distances = []

    print("Start iterations ...")
    for i in range(0, max_iterations):

        initial_distances = matching(pcfix, pcmov)

        initial_distances = reject(pcfix, pcmov, min_planarity, initial_distances)

        # Estimate weight of distances if value is < 0
        if weights_distance_residuals < 0:
            weights_distance_residuals = 1 / (np.std(initial_distances) ** 2)

        dH, residuals, optim_result = estimate_rigid_body_transformation(
            pcfix.X[pcfix.sel, :],
            pcfix.N[pcfix.sel, :],
            pcmov.X[pcmov.sel, :],
            weights_distance_residuals,
            tf_obs,
            weights_tf_obs_residuals,
        )

        residual_distances.append(residuals)

        pcmov.transform(dH)

        H = dH @ H

        update_rigid_body_parameters(rigid_body_parameters, H, optim_result)

        pcfix.sel = sel_orig

        if i > 0:
            if check_convergence_criteria(
                residual_distances[i], residual_distances[i - 1], min_change
            ):
                print("Convergence criteria fulfilled -> stop iteration!")
                break

        if i == 0:
            print(
                f"{'iteration':>9s} | "
                f"{'correspondences':>15s} | "
                f"{'mean(residuals)':>15s} | "
                f"{'std(residuals)':>15s}"
            )
            print(
                f"{0:9d} | "
                f"{len(initial_distances):15d} | "
                f"{np.mean(initial_distances):15.4f} | "
                f"{np.std(initial_distances):15.4f}"
            )
        print(
            f"{i+1:9d} | "
            f"{len(residual_distances[i]):15d} | "
            f"{np.mean(residual_distances[i]):15.4f} | "
            f"{np.std(residual_distances[i]):15.4f}"
        )

    print("Estimated transformation matrix H:")
    print(f"[{H[0, 0]:12.6f} {H[0, 1]:12.6f} {H[0, 2]:12.6f} {H[0, 3]:12.6f}]")
    print(f"[{H[1, 0]:12.6f} {H[1, 1]:12.6f} {H[1, 2]:12.6f} {H[1, 3]:12.6f}]")
    print(f"[{H[2, 0]:12.6f} {H[2, 1]:12.6f} {H[2, 2]:12.6f} {H[2, 3]:12.6f}]")
    print(f"[{H[3, 0]:12.6f} {H[3, 1]:12.6f} {H[3, 2]:12.6f} {H[3, 3]:12.6f}]")

    # TODO tf_obs does not make too much sense as we estimate always dH only
    # TODO report final global transformation parameters (instead of params from last iteration)

    print(
        "... which corresponds to the following rigid body transformation parameters:"
    )
    print(
        f"{'parameter':>9s} | "
        f"{'est. value':>16s} | "
        f"{'est. uncertainty':>16s} | "
        f"{'obs. value':>16s} | "
        f"{'obs. weight':>16s}"
    )
    print(
        f"{'alpha1':>9s} | "
        f"{rigid_body_parameters.alpha1.estimated_value*rigid_body_parameters.alpha1.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha1.estimated_uncertainty*rigid_body_parameters.alpha1.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha1.observed_value*rigid_body_parameters.alpha1.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha1.observation_weight:16.6f}"
    )
    print(
        f"{'alpha2':>9s} | "
        f"{rigid_body_parameters.alpha2.estimated_value*rigid_body_parameters.alpha2.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha2.estimated_uncertainty*rigid_body_parameters.alpha2.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha2.observed_value*rigid_body_parameters.alpha2.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha2.observation_weight:16.6f}"
    )
    print(
        f"{'alpha3':>9s} | "
        f"{rigid_body_parameters.alpha3.estimated_value*rigid_body_parameters.alpha3.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha3.estimated_uncertainty*rigid_body_parameters.alpha3.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha3.observed_value*rigid_body_parameters.alpha3.scale_for_report:16.6f} | "
        f"{rigid_body_parameters.alpha3.observation_weight:16.6f}"
    )
    print(
        f"{'tx':>9s} | "
        f"{rigid_body_parameters.tx.estimated_value:16.6f} | "
        f"{rigid_body_parameters.tx.estimated_uncertainty:16.6f} | "
        f"{rigid_body_parameters.tx.observed_value:16.6f} | "
        f"{rigid_body_parameters.tx.observation_weight:16.6f}"
    )
    print(
        f"{'ty':>9s} | "
        f"{rigid_body_parameters.ty.estimated_value:16.6f} | "
        f"{rigid_body_parameters.ty.estimated_uncertainty:16.6f} | "
        f"{rigid_body_parameters.ty.observed_value:16.6f} | "
        f"{rigid_body_parameters.ty.observation_weight:16.6f}"
    )
    print(
        f"{'tz':>9s} | "
        f"{rigid_body_parameters.tz.estimated_value:16.6f} | "
        f"{rigid_body_parameters.tz.estimated_uncertainty:16.6f} | "
        f"{rigid_body_parameters.tz.observed_value:16.6f} | "
        f"{rigid_body_parameters.tz.observation_weight:16.6f}"
    )

    print(
        "(Unit of est. value, est. uncertainty, and obs.value for alpha1/2/3 is degree)"
    )

    print(f"Finished in {time.time() - start_time:.3f} seconds!")

    return H, pcmov.X


def update_rigid_body_parameters(rigid_body_parameters, H, optim_result):
    (
        rigid_body_parameters.alpha1.estimated_value,
        rigid_body_parameters.alpha2.estimated_value,
        rigid_body_parameters.alpha3.estimated_value,
    ) = utils.rotation_matrix_to_euler_angles(H[0:3, 0:3])

    rigid_body_parameters.tx.estimated_value = H[0, 3]
    rigid_body_parameters.ty.estimated_value = H[1, 3]
    rigid_body_parameters.tz.estimated_value = H[2, 3]

    # rigid_body_parameters.alpha1.estimated_uncertainty = optim_result.params[
    #     "alpha1"
    # ].stderr
    # rigid_body_parameters.alpha2.estimated_uncertainty = optim_result.params[
    #     "alpha2"
    # ].stderr
    # rigid_body_parameters.alpha3.estimated_uncertainty = optim_result.params[
    #     "alpha3"
    # ].stderr
    # rigid_body_parameters.tx.estimated_uncertainty = optim_result.params["tx"].stderr
    # rigid_body_parameters.ty.estimated_uncertainty = optim_result.params["ty"].stderr
    # rigid_body_parameters.tz.estimated_uncertainty = optim_result.params["tz"].stderr
