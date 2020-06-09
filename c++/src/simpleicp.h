#ifndef RUN_SIMPLEICP_SIMPLEICP_H
#define RUN_SIMPLEICP_SIMPLEICP_H

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

Eigen::Matrix<double, 4, 4> SimpleICP(const Eigen::MatrixXd& X_fix,
                                      const Eigen::MatrixXd& X_mov,
                                      const int& correspondences = 1000,
                                      const int& neighbors = 10,
                                      const double& min_planarity = 0.3,
                                      const double& min_change = 1,
                                      const int& max_iterations = 100);

const char* Timestamp();

Eigen::MatrixXi KnnSearch(const Eigen::MatrixXd& X,
                          const Eigen::MatrixXd& X_query,
                          const int& k = 1);

double Median(const Eigen::VectorXd& v);

// Median of absolute differences (mad) with respect to the median
double MAD(const Eigen::VectorXd& v);

double Std(const Eigen::VectorXd& v);

double Change(const double& new_val, const double& old_val);

bool CheckConvergenceCriteria(const std::vector<double>& residual_dists_mean,
                              const std::vector<double>& residual_dists_std,
                              const double& min_change);

#endif  // RUN_SIMPLEICP_SIMPLEICP_H
