#ifndef RUN_SIMPLEICP_SIMPLEICP_H
#define RUN_SIMPLEICP_SIMPLEICP_H

#include <fstream>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>

using namespace Eigen;

Matrix<double, 4, 4> SimpleICP(const MatrixXd& X_fix, const MatrixXd& X_mov,
                               const int& correspondences = 1000, const int& neighbors = 10,
                               const double& min_planarity = 0.3, const int& min_change = 1,
                               const int& max_iterations = 100);

MatrixXd ImportXYZFileToMatrix(const std::string& path_to_pc);

const char* Timestamp();

MatrixXi KnnSearch(const MatrixXd& X, const MatrixXd& X_query, const int& k=1);

double Median(const VectorXd& v);

// Median of absolute differences (mad) with respect to the median
double MAD(const VectorXd& v);

double Std(const VectorXd& v);

double Change(const double& new_val, const double& old_val);

bool CheckConvergenceCriteria(const std::vector<double>& residual_dists_mean,
                              const std::vector<double>& residual_dists_std,
                              const double& min_change);

#endif //RUN_SIMPLEICP_SIMPLEICP_H
