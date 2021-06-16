#include "pointcloud.h"
#include "simpleicp.h"

PointCloud::PointCloud(Eigen::MatrixXd X) : X_{X}, sel_{std::vector<bool>(X.rows(), true)} {}

Eigen::MatrixXd PointCloud::GetXOfSelectedPts() {
  auto sel_idx = GetIdxOfSelectedPts();
  Eigen::MatrixXd X_sel(sel_idx.size(), 3);
  for (int i = 0; i < sel_idx.size(); i++) {
    X_sel(i, 0) = X_(sel_idx[i], 0);
    X_sel(i, 1) = X_(sel_idx[i], 1);
    X_sel(i, 2) = X_(sel_idx[i], 2);
  }
  return X_sel;
}

std::vector<int> PointCloud::GetIdxOfSelectedPts() {
  std::vector<int> idx;
  for (int i = 0; i < NoPts(); i++) {
    if (sel_[i]) {
      idx.push_back(i);
    }
  }
  return idx;
}

void PointCloud::SelectNPts(const int& n) {
  auto sel_idx{GetIdxOfSelectedPts()};

  if (n < sel_idx.size()) {
    // Deactivate all points first
    for (int i = 1; i < NoPts(); i++) {
      sel_[i] = false;
    }

    // Re-activate n points
    auto idx_not_rounded{Eigen::VectorXd::LinSpaced(n, 0, sel_idx.size() - 1)};
    for (int i = 0; i < n; i++) {
      int idx_rounded{static_cast<int>(round(idx_not_rounded(i)))};
      sel_[sel_idx[idx_rounded]] = true;
    }
  }
}

void PointCloud::EstimateNormals(const int& neighbors) {
  // Initialize vectors with NANs
  nx_ = Eigen::VectorXd(NoPts());
  nx_.fill(NAN);
  ny_ = Eigen::VectorXd(NoPts());
  ny_.fill(NAN);
  nz_ = Eigen::VectorXd(NoPts());
  nz_.fill(NAN);
  planarity_ = Eigen::VectorXd(NoPts());
  planarity_.fill(NAN);

  Eigen::MatrixXi mat_idx_nn(X_.rows(), neighbors);
  mat_idx_nn = KnnSearch(X_, GetXOfSelectedPts(), neighbors);

  auto sel_idx = GetIdxOfSelectedPts();
  for (int i = 0; i < sel_idx.size(); i++) {
    // Build matrix with nn
    Eigen::MatrixXd X_nn(neighbors, 3);
    for (int j = 0; j < neighbors; j++) {
      X_nn(j, 0) = X_(mat_idx_nn(i, j), 0);
      X_nn(j, 1) = X_(mat_idx_nn(i, j), 1);
      X_nn(j, 2) = X_(mat_idx_nn(i, j), 2);
    }

    // Covariance matrix
    Eigen::MatrixXd centered = X_nn.rowwise() - X_nn.colwise().mean();
    Eigen::MatrixXd C = (centered.adjoint() * centered) / double(X_nn.rows() - 1);

    // Normal vector as eigenvector corresponding to smallest eigenvalue
    // Note that SelfAdjointEigenSolver is faster than EigenSolver for symmetric matrices. Moreover,
    // the eigenvalues are sorted in increasing order.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    auto eigenvectors{es.eigenvectors()};
    auto eigenvalues{es.eigenvalues()};
    nx_[sel_idx[i]] = eigenvectors(0, 0);
    ny_[sel_idx[i]] = eigenvectors(1, 0);
    nz_[sel_idx[i]] = eigenvectors(2, 0);
    planarity_[sel_idx[i]] = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2];
  }
}

void PointCloud::Transform(Eigen::Matrix<double, 4, 4>& H) {
  Eigen::MatrixXd X_in_H(NoPts(), 4);
  Eigen::MatrixXd X_out_H(NoPts(), 4);
  X_in_H << X_, Eigen::VectorXd::Ones(NoPts());
  X_out_H = H * X_in_H.transpose();
  X_ << X_out_H.row(0).transpose(), X_out_H.row(1).transpose(), X_out_H.row(2).transpose();
}

int PointCloud::NoPts() { return X_.rows(); }

// Getters
const Eigen::MatrixXd& PointCloud::X() { return X_; }
const Eigen::VectorXd& PointCloud::nx() { return nx_; }
const Eigen::VectorXd& PointCloud::ny() { return ny_; }
const Eigen::VectorXd& PointCloud::nz() { return nz_; }
const Eigen::VectorXd& PointCloud::planarity() { return planarity_; }
const std::vector<bool>& PointCloud::sel() { return sel_; }
