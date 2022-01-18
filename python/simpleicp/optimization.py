"""
SimpleICPOptimization class.
"""

from __future__ import (
    annotations,
)  # needed for type hinting of RigidBodyParameters before it is defined

from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import lmfit
import numpy as np

from . import corrpts, mathutils


class SimpleICPOptimization:
    """Class for setting up and solve least squares optimization for simpleICP.

    Dev notes:
        - The basic structure of the class does not depend on the solver specifically used to
        solve the LS optimization, e.g. lmfit. Consequently, it should be fairly easy to substitute
        the solver in the future.
    """

    def __init__(
        self,
        corr_pts: corrpts.CorrPts,
        distance_weights: Optional[float],  # can also be None
        rbp_initial_values: Tuple[float],
        rbp_observed_values: Tuple[float],
        rbp_observation_weights: Tuple[float],
    ) -> None:
        """[summary]

        Args:
            corr_pts (corrpts.CorrPts): Corresponding points between pc1 and pc2.
            distance_weights (Optional[float]): See docstring of SimpleICP.run.
            rbp_observed_values (Tuple[float]): See docstring of SimpleICP.run.
            rbp_observation_weights (Tuple[float]): See docstring of SimpleICP.run.
        """
        self._cp = corr_pts

        self._rbp = RigidBodyParameters()
        self._rbp.set_parameter_attributes_from_list(
            "initial_value", rbp_initial_values
        )
        self._rbp.set_parameter_attributes_from_list(
            "observed_value", rbp_observed_values
        )
        self._rbp.set_parameter_attributes_from_list(
            "observation_weight", rbp_observation_weights
        )

        self._distance_weights = distance_weights

        self._optim_results = None

    @property
    def rbp(self) -> RigidBodyParameters:
        """Returns rigid body parameters."""
        return self._rbp

    def estimate_parameters(self) -> np.array:
        """Estimate values of unknown parameters by setting up and solving a LS optimization.

        Returns:
            np.array: Distance residuals. The distances are signed point-to-plane distances.
        """
        # Define rigid body parameters for lmfit
        params = lmfit.Parameters()
        for parameter in fields(self._rbp):

            initial_value = getattr(self._rbp, parameter.name).initial_value
            observed_value = getattr(self._rbp, parameter.name).observed_value
            observation_weight = getattr(self._rbp, parameter.name).observation_weight
            vary = np.isfinite(observation_weight)

            params[parameter.name] = lmfit.Parameter(
                name=parameter.name,
                value=initial_value,
                vary=vary,
                user_data={
                    "observed_value": observed_value,
                    "observation_weight": observation_weight,
                    "is_observed": observation_weight > 0
                    and np.isfinite(observation_weight),
                },
            )

        # Optimize with lmfit
        self._optim_results = lmfit.minimize(
            SimpleICPOptimization.__residuals,
            params,
            method="least_squares",
            args=(
                self._cp,
                self._distance_weights,
            ),
        )

        # Save estimated values
        estimated_values = [
            self._optim_results.params["alpha1"].value,
            self._optim_results.params["alpha2"].value,
            self._optim_results.params["alpha3"].value,
            self._optim_results.params["tx"].value,
            self._optim_results.params["ty"].value,
            self._optim_results.params["tz"].value,
        ]
        self._rbp.set_parameter_attributes_from_list(
            "estimated_value", estimated_values
        )

        # Compute unweighted distance residuals
        weighted_distance_residuals = self._optim_results.residual[
            0 : self._cp.num_corr_pts - 1
        ]
        unweighted_distance_residuals = (
            weighted_distance_residuals / self._distance_weights
        )

        return unweighted_distance_residuals

    def estimate_parameter_uncertainties(self):
        """Estimate the uncertainties of the unknown parameters.

        The uncertainty of a parameter is defined as the a posteriori standard deviation of the
        estimated value.
        """
        # Attention: Unfortunately we can not simply use the stderr values which lmfit provides
        # (e.g. out.params["alpha1"].stderr) for each parameter if the residuals are weighted.
        # Instead we need to "unweight" the Jacobian A and the residuals and calculate the
        # uncertainties as usual from them.

        # Create array with weights of all observations
        weights = np.full((self._cp.num_corr_pts,), self._distance_weights)
        for param_name in self._optim_results.params:
            if self._optim_results.params[param_name].user_data["is_observed"]:
                observation_weight = self._optim_results.params[param_name].user_data[
                    "observation_weight"
                ]
                weights = np.append(weights, observation_weight)

        # Compute unweighted Jakobian A
        A_unweighted = np.empty_like(self._optim_results.jac)
        for col in range(0, A_unweighted.shape[1]):
            A_unweighted[:, col] = self._optim_results.jac[:, col] / weights

        # Compute unweighted residuals
        residuals_unweighted = self._optim_results.residual / weights

        # Compute covariance matrix
        P = np.diag(weights)
        N = A_unweighted.T @ P @ A_unweighted
        Qxx = np.linalg.inv(N)
        vPv = np.sum(weights * residuals_unweighted ** 2)
        [num_obs, num_prm] = np.shape(self._optim_results.jac)
        s0 = np.sqrt(vPv / (num_obs - num_prm))
        Cxx = s0 ** 2 * Qxx

        # Get uncertainties from covariance matrix
        idx_prm = 0
        for param_name in self._optim_results.params:
            if self._optim_results.params[param_name].vary:
                getattr(self._rbp, param_name).estimated_uncertainty = np.sqrt(
                    Cxx[idx_prm, idx_prm]
                )
                idx_prm += 1

    @staticmethod
    def __residuals(
        params: lmfit.Parameters,
        cp: corrpts.CorrPts,
        distance_weights: float,
    ) -> np.array:
        """Returns weighted residuals as numpy array for lmfit."""
        (
            pc2_x_transformed,
            pc2_y_transformed,
            pc2_z_transformed,
        ) = SimpleICPOptimization.__transform_point_cloud(
            params, cp.pc2_x, cp.pc2_y, cp.pc2_z
        )

        weighted_distance_residuals = (
            SimpleICPOptimization.__compute_weighted_distance_residuals(
                cp.pc1_x,
                cp.pc1_y,
                cp.pc1_z,
                cp.pc1_nx,
                cp.pc1_ny,
                cp.pc1_nz,
                pc2_x_transformed,
                pc2_y_transformed,
                pc2_z_transformed,
                distance_weights,
            )
        )

        weighted_rbp_observation_residuals = (
            SimpleICPOptimization.__compute_weighted_rbp_observation_residuals(params)
        )

        # Combine all residuals
        weighted_residuals = np.concatenate(
            (weighted_distance_residuals, weighted_rbp_observation_residuals)
        )

        return weighted_residuals

    @staticmethod
    def __transform_point_cloud(
        params: lmfit.Parameters, x: np.array, y: np.array, z: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """Transform point cloud by rigid body transformation."""
        R = mathutils.euler_angles_to_rotation_matrix(
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

        H = mathutils.create_homogeneous_transformation_matrix(R, t)

        Xe = np.column_stack((x, y, z))
        Xh = mathutils.euler_coord_to_homogeneous_coord(Xe)
        Xh_transformed = np.transpose(H @ Xh.T)
        Xe_transformed = mathutils.homogeneous_coord_to_euler_coord(Xh_transformed)

        x_transformed = Xe_transformed[:, 0]
        y_transformed = Xe_transformed[:, 1]
        z_transformed = Xe_transformed[:, 2]

        return x_transformed, y_transformed, z_transformed

    @staticmethod
    def __compute_weighted_distance_residuals(
        pc1_x: np.array,
        pc1_y: np.array,
        pc1_z: np.array,
        pc1_nx: np.array,
        pc1_ny: np.array,
        pc1_nz: np.array,
        pc2_x: np.array,
        pc2_y: np.array,
        pc2_z: np.array,
        distance_weights: float,
    ) -> np.array:
        """Compute weighted residuals for point-to-plane distances between correspondences."""
        no_correspondences = len(pc1_x)

        dx = pc2_x - pc1_x
        dy = pc2_y - pc1_y
        dz = pc2_z - pc1_z

        distance_residuals = np.empty(no_correspondences)
        for i in range(0, no_correspondences):
            distance_residuals[i] = (
                dx[i] * pc1_nx[i] + dy[i] * pc1_ny[i] + dz[i] * pc1_nz[i]
            )

        weighted_distance_residuals = distance_weights * distance_residuals

        return weighted_distance_residuals

    @staticmethod
    def __compute_weighted_rbp_observation_residuals(
        params: lmfit.Parameters,
    ) -> np.array:
        """Compute weighted residuals for direct observation of rbp."""

        weighted_residual_list = []

        for param_name in params:
            if params[param_name].user_data["is_observed"]:
                residual = (
                    params[param_name].value
                    - params[param_name].user_data["observed_value"]
                )
                weighted_residual = (
                    params[param_name].user_data["observation_weight"] * residual
                )
                weighted_residual_list.append(weighted_residual)

        return np.array(weighted_residual_list)


@dataclass
class Parameter:
    """Data class for a single optimization parameter."""

    initial_value: float = np.nan
    observed_value: float = np.nan
    observation_weight: float = np.nan
    estimated_value: float = np.nan
    estimated_uncertainty: float = np.nan
    scale_for_logging: float = 1

    @property
    def initial_value_scaled(self):
        """Returns scaled initial value for logging."""
        return self.initial_value * self.scale_for_logging

    @property
    def observed_value_scaled(self):
        """Returns observed value for logging."""
        return self.observed_value * self.scale_for_logging

    @property
    def estimated_value_scaled(self):
        """Returns estimated value for logging."""
        return self.estimated_value * self.scale_for_logging

    @property
    def estimated_uncertainty_scaled(self):
        """Returns uncertainty for logging."""
        return self.estimated_uncertainty * self.scale_for_logging


@dataclass
class RigidBodyParameters:
    """Data class for rigid body parameters."""

    alpha1: Parameter = Parameter(scale_for_logging=180 / np.pi)
    alpha2: Parameter = Parameter(scale_for_logging=180 / np.pi)
    alpha3: Parameter = Parameter(scale_for_logging=180 / np.pi)
    tx: Parameter = Parameter()
    ty: Parameter = Parameter()
    tz: Parameter = Parameter()

    @property
    def H(self) -> np.array:
        """Returns homogeneous transformation matrix from estimated values."""

        R = mathutils.euler_angles_to_rotation_matrix(
            self.alpha1.estimated_value,
            self.alpha2.estimated_value,
            self.alpha3.estimated_value,
        )

        t = np.array(
            [self.tx.estimated_value, self.ty.estimated_value, self.tz.estimated_value]
        )

        H = mathutils.create_homogeneous_transformation_matrix(R, t)

        return H

    def set_parameter_attributes_from_list(self, attribute_name: str, array: List):
        """Set an attribute for all six rbp parameters from a given array.

        Args:
            attribute_name (str): Name of attribute.
            array (List): Array with 6 elements with attribute values.
        """
        setattr(self.alpha1, attribute_name, array[0])
        setattr(self.alpha2, attribute_name, array[1])
        setattr(self.alpha3, attribute_name, array[2])
        setattr(self.tx, attribute_name, array[3])
        setattr(self.ty, attribute_name, array[4])
        setattr(self.tz, attribute_name, array[5])

    def get_parameter_attributes_as_list(self, attribute_name: str) -> List:
        """Returns an attribute of all six rbp parameters as array.

        Args:
            attribute_name (str): Name of attribute.

        Returns:
            List: Attribute values as array.
        """
        return [
            getattr(self.alpha1, attribute_name),
            getattr(self.alpha2, attribute_name),
            getattr(self.alpha3, attribute_name),
            getattr(self.tx, attribute_name),
            getattr(self.ty, attribute_name),
            getattr(self.tz, attribute_name),
        ]
