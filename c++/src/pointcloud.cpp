#include "pointcloud.h"
#include "simpleicp.h"

PointCloud::PointCloud(MatrixXd X): X_{X}, sel_ {std::vector<bool> (X.rows(), true)} {}

MatrixXd PointCloud::GetXOfSelectedPts() {
  auto sel_idx = GetIdxOfSelectedPts();
  MatrixXd X_sel(sel_idx.size(),3);
  for (int i = 0; i < sel_idx.size(); i++) {
    X_sel(i,0) = X_(sel_idx[i],0);
    X_sel(i,1) = X_(sel_idx[i],1);
    X_sel(i,2) = X_(sel_idx[i],2);
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

  auto sel_idx {GetIdxOfSelectedPts()};

  if (n < sel_idx.size()) {

    // Deactivate all points first
    for (int i = 1; i < NoPts(); i++) {
      sel_[i] = false;
    }

    // Re-activate n points
    auto idx_not_rounded {VectorXd::LinSpaced(n, 0, sel_idx.size()-1)};
    for (int i = 0; i < n; i++) {
      int idx_rounded {static_cast<int>(round(idx_not_rounded(i)))};
      sel_[sel_idx[idx_rounded]] = true;
    }

  }
}

void PointCloud::EstimateNormals(const int& neighbors) {

  // Initialize vectors with NANs
  nx_ = VectorXd(NoPts()); nx_.fill(NAN);
  ny_ = VectorXd(NoPts()); ny_.fill(NAN);
  nz_ = VectorXd(NoPts()); nz_.fill(NAN);

  MatrixXi mat_idx_nn(X_.rows(), neighbors);
  mat_idx_nn = KnnSearch(X_, GetXOfSelectedPts(), neighbors);

  auto sel_idx = GetIdxOfSelectedPts();
  for (int i = 0; i < sel_idx.size(); i++) {

    // Build matrix with nn
    MatrixXd X_nn(neighbors, 3);
    for (int j = 0; j < neighbors; j++) {
      X_nn(j, 0) = X_(mat_idx_nn(i, j), 0);
      X_nn(j, 1) = X_(mat_idx_nn(i, j), 1);
      X_nn(j, 2) = X_(mat_idx_nn(i, j), 2);
    }

    // Covariance matrix
    MatrixXd centered = X_nn.rowwise() - X_nn.colwise().mean();
    MatrixXd C = (centered.adjoint() * centered) / double(X_nn.rows() - 1);

    // Normal vector as eigenvector corresponding to smallest eigenvalue
    EigenSolver<MatrixXd> es(C);
    auto eigenvectors = es.eigenvectors();
    int min_index;
    es.eigenvalues().real().minCoeff(&min_index);
    nx_[sel_idx[i]] = eigenvectors(0, min_index).real();
    ny_[sel_idx[i]] = eigenvectors(1, min_index).real();
    nz_[sel_idx[i]] = eigenvectors(2, min_index).real();
  }

}

void PointCloud::Transform(Matrix<double, 4, 4>& H) {
  MatrixXd X_in_H(NoPts(), 4);
  MatrixXd X_out_H(NoPts(), 4);
  X_in_H << X_, VectorXd::Ones(NoPts());
  X_out_H = H*X_in_H.transpose();
  X_ << X_out_H.row(0).transpose(), X_out_H.row(1).transpose(), X_out_H.row(2).transpose();
}

int PointCloud::NoPts() {
  return X_.rows();
}

// Getters
const MatrixXd& PointCloud::X() {return X_;}
const VectorXd& PointCloud::nx() {return nx_;}
const VectorXd& PointCloud::ny() {return ny_;}
const VectorXd& PointCloud::nz() {return nz_;}
const std::vector<bool>& PointCloud::sel() {return sel_;}
