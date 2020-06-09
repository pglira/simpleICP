#include "corrpts.h"
#include "simpleicp.h"

CorrPts::CorrPts(PointCloud& pc1, PointCloud& pc2) : pc1_{pc1}, pc2_{pc2} {}

void CorrPts::Match() {
  auto X_sel_pc1 = pc1_.GetXOfSelectedPts();
  auto X_sel_pc2 = pc2_.GetXOfSelectedPts();

  idx_pc1_ = pc1_.GetIdxOfSelectedPts();
  idx_pc2_ = std::vector<int>(idx_pc1_.size());

  Eigen::MatrixXi mat_idx_nn(idx_pc1_.size(), 1);
  mat_idx_nn = KnnSearch(X_sel_pc2, X_sel_pc1, 1);
  for (int i = 0; i < mat_idx_nn.rows(); i++) {
    idx_pc2_[i] = mat_idx_nn(i, 0);
  }

  GetPlanarityFromPc1();
  ComputeDists();
}

void CorrPts::GetPlanarityFromPc1() {
  planarity_ = Eigen::VectorXd(idx_pc1_.size());
  for (int i = 0; i < idx_pc1_.size(); i++) {
    planarity_[i] = pc1_.planarity()[idx_pc1_[i]];
  }
}

void CorrPts::ComputeDists() {
  dists_ = Eigen::VectorXd(idx_pc1_.size());
  dists_.fill(NAN);

  for (int i = 0; i < idx_pc1_.size(); i++) {
    double x_pc1 = pc1_.X()(idx_pc1_[i], 0);
    double y_pc1 = pc1_.X()(idx_pc1_[i], 1);
    double z_pc1 = pc1_.X()(idx_pc1_[i], 2);

    double x_pc2 = pc2_.X()(idx_pc2_[i], 0);
    double y_pc2 = pc2_.X()(idx_pc2_[i], 1);
    double z_pc2 = pc2_.X()(idx_pc2_[i], 2);

    double nx_pc1 = pc1_.nx()(idx_pc1_[i]);
    double ny_pc1 = pc1_.ny()(idx_pc1_[i]);
    double nz_pc1 = pc1_.nz()(idx_pc1_[i]);

    double dist{(x_pc2 - x_pc1) * nx_pc1 + (y_pc2 - y_pc1) * ny_pc1 + (z_pc2 - z_pc1) * nz_pc1};

    dists_(i) = dist;
  }
}

void CorrPts::Reject(const double& min_planarity) {
  auto med{Median(dists_)};
  auto sigmad{1.4826 * MAD(dists_)};
  std::vector<bool> keep(dists_.size(), true);
  for (int i = 0; i < dists_.size(); i++) {
    if ((abs(dists_[i] - med) > 3 * sigmad) | (planarity_[i] < min_planarity)) {
      keep[i] = false;
    }
  }
  size_t no_remaining_pts = count(keep.begin(), keep.end(), true);
  std::vector<int> idx_pc1_new(no_remaining_pts);
  std::vector<int> idx_pc2_new(no_remaining_pts);
  Eigen::VectorXd dists_new(no_remaining_pts);
  int j{0};
  for (int i = 0; i < dists_.size(); i++) {
    if (keep[i]) {
      idx_pc1_new[j] = idx_pc1_[i];
      idx_pc2_new[j] = idx_pc2_[i];
      dists_new[j] = dists_[i];
      j++;
    }
  }
  idx_pc1_ = idx_pc1_new;
  idx_pc2_ = idx_pc2_new;
  dists_ = dists_new;
}

void CorrPts::EstimateRigidBodyTransformation(Eigen::Matrix<double, 4, 4>& H,
                                              Eigen::VectorXd& residuals) {
  auto no_corr_pts{idx_pc1_.size()};

  Eigen::MatrixXd A(no_corr_pts, 6);
  Eigen::VectorXd l(no_corr_pts);

  for (int i = 0; i < no_corr_pts; i++) {
    double x_pc1 = pc1_.X()(idx_pc1_[i], 0);
    double y_pc1 = pc1_.X()(idx_pc1_[i], 1);
    double z_pc1 = pc1_.X()(idx_pc1_[i], 2);

    double x_pc2 = pc2_.X()(idx_pc2_[i], 0);
    double y_pc2 = pc2_.X()(idx_pc2_[i], 1);
    double z_pc2 = pc2_.X()(idx_pc2_[i], 2);

    double nx_pc1 = pc1_.nx()(idx_pc1_[i]);
    double ny_pc1 = pc1_.ny()(idx_pc1_[i]);
    double nz_pc1 = pc1_.nz()(idx_pc1_[i]);

    A(i, 0) = -z_pc2 * ny_pc1 + y_pc2 * nz_pc1;
    A(i, 1) = z_pc2 * nx_pc1 - x_pc2 * nz_pc1;
    A(i, 2) = -y_pc2 * nx_pc1 + x_pc2 * ny_pc1;
    A(i, 3) = nx_pc1;
    A(i, 4) = ny_pc1;
    A(i, 5) = nz_pc1;

    l(i) = nx_pc1 * (x_pc1 - x_pc2) + ny_pc1 * (y_pc1 - y_pc2) + nz_pc1 * (z_pc1 - z_pc2);
  }

  Eigen::Matrix<double, 6, 1> x{A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(l)};

  double alpha1{x(0)};
  double alpha2{x(1)};
  double alpha3{x(2)};
  double tx{x(3)};
  double ty{x(4)};
  double tz{x(5)};

  H(0, 0) = 1;
  H(0, 1) = -alpha3;
  H(0, 2) = alpha2;
  H(0, 3) = tx;

  H(1, 0) = alpha3;
  H(1, 1) = 1;
  H(1, 2) = -alpha1;
  H(1, 3) = ty;

  H(2, 0) = -alpha2;
  H(2, 1) = alpha1;
  H(2, 2) = 1;
  H(2, 3) = tz;

  H(3, 0) = 0;
  H(3, 1) = 0;
  H(3, 2) = 0;
  H(3, 3) = 1;

  residuals = A * x - l;
}

// Getters
const PointCloud& CorrPts::pc1() { return pc1_; }
const PointCloud& CorrPts::pc2() { return pc2_; }
const std::vector<int>& CorrPts::idx_pc1() { return idx_pc1_; }
const std::vector<int>& CorrPts::idx_pc2() { return idx_pc2_; }
const Eigen::VectorXd& CorrPts::dists() { return dists_; }
