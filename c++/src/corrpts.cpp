#include "corrpts.h"
#include "simpleicp.h"

#ifdef USE_CERES
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#endif

CorrPts::CorrPts(PointCloud &pc1, PointCloud &pc2) : pc1_{pc1}, pc2_{pc2} {}

void CorrPts::Match()
{
  auto X_sel_pc1 = pc1_.GetXOfSelectedPts();
  auto X_sel_pc2 = pc2_.GetXOfSelectedPts();

  idx_pc1_ = pc1_.GetIdxOfSelectedPts();
  idx_pc2_ = std::vector<int>(idx_pc1_.size());

  Eigen::MatrixXi mat_idx_nn(idx_pc1_.size(), 1);
  mat_idx_nn = KnnSearch(X_sel_pc2, X_sel_pc1, 1);
  for (int i = 0; i < mat_idx_nn.rows(); i++)
  {
    idx_pc2_[i] = mat_idx_nn(i, 0);
  }

  GetPlanarityFromPc1();
  ComputeDists();
}

void CorrPts::GetPlanarityFromPc1()
{
  planarity_ = Eigen::VectorXd(idx_pc1_.size());
  for (uint i = 0; i < idx_pc1_.size(); i++)
  {
    planarity_[i] = pc1_.planarity()[idx_pc1_[i]];
  }
}

void CorrPts::ComputeDists()
{
  dists_ = Eigen::VectorXd(idx_pc1_.size());
  dists_.fill(NAN);

  for (uint i = 0; i < idx_pc1_.size(); i++)
  {
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

void CorrPts::Reject(const double &min_planarity)
{
  auto med{Median(dists_)};
  auto sigmad{1.4826 * MAD(dists_)};
  std::vector<bool> keep(dists_.size(), true);
  for (int i = 0; i < dists_.size(); i++)
  {
    if ((abs(dists_[i] - med) > 3 * sigmad) | (planarity_[i] < min_planarity))
    {
      keep[i] = false;
    }
  }
  size_t no_remaining_pts = count(keep.begin(), keep.end(), true);
  std::vector<int> idx_pc1_new(no_remaining_pts);
  std::vector<int> idx_pc2_new(no_remaining_pts);
  Eigen::VectorXd dists_new(no_remaining_pts);
  int j{0};
  for (int i = 0; i < dists_.size(); i++)
  {
    if (keep[i])
    {
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

#ifdef USE_CERES
struct PointToPointFunctor {
  PointToPointFunctor(PointCloud *pcf, PointCloud *pcm, size_t i, size_t j) : pcf_(pcf), pcm_(pcm), i_(i) , j_(j) {}

  template <typename T>
  bool operator()(const T* const rbp, T* residual) const;

  PointCloud *pcf_, *pcm_;
  size_t i_, j_;
};

template <typename T>
bool PointToPointFunctor::operator()(const T* const rbp, T* residual) const
{
  const Eigen::Vector3d &Xf = pcf_->X().row(i_);
  const Eigen::Vector3d &Xm = pcm_->X().row(j_);

  T X[3];
  X[0] = T(Xm.x());
  X[1] = T(Xm.y());
  X[2] = T(Xm.z());

  T p[3];
  ceres::AngleAxisRotatePoint(rbp, X, p);
  p[0] += rbp[3]; p[1] += rbp[4]; p[2] += rbp[5];

  residual[0] = T(Xf.x()) - p[0];
  residual[1] = T(Xf.y()) - p[1];
  residual[2] = T(Xf.z()) - p[2];

  return true;
}


struct PointToPlaneFunctor {
  PointToPlaneFunctor(PointCloud *pcf, PointCloud *pcm, size_t i, size_t j) : pcf_(pcf), pcm_(pcm), i_(i) , j_(j) {}

  template <typename T>
  bool operator()(const T* const rbp, T* residual) const;

  PointCloud *pcf_, *pcm_;
  size_t i_, j_;
};

template <typename T>
bool PointToPlaneFunctor::operator()(const T* const rbp, T* residual) const
{
  const Eigen::Vector3d &Xf = pcf_->X().row(i_);
  const Eigen::Vector3d &Xm = pcm_->X().row(j_);

  T X[3];
  X[0] = T(Xm.x());
  X[1] = T(Xm.y());
  X[2] = T(Xm.z());

  T p[3];
  ceres::AngleAxisRotatePoint(rbp, X, p);
  p[0] += rbp[3]; p[1] += rbp[4]; p[2] += rbp[5];

  p[0] = T(pcf_->nx()(i_)) * (T(Xf.x()) - p[0]);
  p[1] = T(pcf_->ny()(i_)) * (T(Xf.y()) - p[1]);
  p[2] = T(pcf_->nz()(i_)) * (T(Xf.z()) - p[2]);

  residual[0] = p[0] + p[1] + p[2];

  return true;
}

void CorrPts::EstimateRigidBodyTransformation(Eigen::Matrix<double, 4, 4> &H,
                                              Eigen::VectorXd &residuals)
{
  auto no_corr_pts{idx_pc1_.size()};

  ceres::Problem problem;

  std::array<double, 6> parameters = { 0, 0, 0, 0, 0, 0 };
  problem.AddParameterBlock(parameters.data(), parameters.size());

  for (size_t i = 0; i < no_corr_pts; ++i)
  {
    using Functor = PointToPlaneFunctor;
    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<Functor, 1, 6>(new Functor(&pc1_, &pc2_, idx_pc1_[i], idx_pc2_[i]));

    problem.AddResidualBlock(cost_function, nullptr, parameters.data());
  }

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  ceres::AngleAxisToRotationMatrix(parameters.data(), ceres::ColumnMajorAdapter3x3(R.data()));
  t << parameters[3], parameters[4], parameters[5];

  H = Eigen::Matrix4d::Identity();
  H.topLeftCorner<3, 3>() = R;
  H.topRightCorner<3, 1>() << t;

  {
    std::vector<double> residuals_vec;
    residuals_vec.reserve(no_corr_pts);
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.apply_loss_function = false;

    problem.Evaluate(eval_options, nullptr, &residuals_vec, nullptr, nullptr);
    residuals = Eigen::Map<const Eigen::VectorXd>(residuals_vec.data(), residuals_vec.size());
  }
}
#else
Eigen::Matrix3d EulerAnglesToRotationMatrix(float alpha1, float alpha2, float alpha3)
{
  Eigen::Matrix3d R;
  R <<
      std::cos(alpha2) * std::cos(alpha3),
      -std::cos(alpha2) * std::sin(alpha3),
      std::sin(alpha2),

      std::cos(alpha1) * std::sin(alpha3) + std::sin(alpha1) * std::sin(alpha2) * std::cos(alpha3),
      std::cos(alpha1) * std::cos(alpha3) - std::sin(alpha1) * std::sin(alpha2) * std::sin(alpha3),
      -std::sin(alpha1) * std::cos(alpha2),

      std::sin(alpha1) * std::sin(alpha3) - std::cos(alpha1) * std::sin(alpha2) * std::cos(alpha3),
      std::sin(alpha1) * std::cos(alpha3) + std::cos(alpha1) * std::sin(alpha2) * std::sin(alpha3),
      std::cos(alpha1) * std::cos(alpha2);

  return R;
}

void CorrPts::EstimateRigidBodyTransformation(Eigen::Matrix<double, 4, 4> &H,
                                              Eigen::VectorXd &residuals)
{
  auto no_corr_pts{idx_pc1_.size()};

  Eigen::MatrixXd A(no_corr_pts, 6);
  Eigen::VectorXd l(no_corr_pts);

  for (uint i = 0; i < no_corr_pts; i++)
  {
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

  H = Eigen::Matrix4d::Identity();
  H.topLeftCorner<3, 3>() = EulerAnglesToRotationMatrix(alpha1, alpha2, alpha3);
  H.topRightCorner<3, 1>() << tx, ty, tz;

  residuals = A * x - l;
}
#endif

// Getters
const PointCloud &CorrPts::pc1() { return pc1_; }
const PointCloud &CorrPts::pc2() { return pc2_; }
const std::vector<int> &CorrPts::idx_pc1() { return idx_pc1_; }
const std::vector<int> &CorrPts::idx_pc2() { return idx_pc2_; }
const Eigen::VectorXd &CorrPts::dists() { return dists_; }
