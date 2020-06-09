#include "simpleicp.h"
#include "corrpts.h"
#include "nanoflann.hpp"
#include "pointcloud.h"

const int LEAF_SIZE{200};

Eigen::Matrix<double, 4, 4> SimpleICP(const Eigen::MatrixXd& X_fix,
                                      const Eigen::MatrixXd& X_mov,
                                      const int& correspondences,
                                      const int& neighbors,
                                      const double& min_planarity,
                                      const double& min_change,
                                      const int& max_iterations) {
  auto start = std::chrono::system_clock::now();

  printf("[%s] Create point cloud objects ...\n", Timestamp());
  PointCloud pc_fix{X_fix};
  PointCloud pc_mov{X_mov};

  printf("[%s] Select points for correspondences in fixed point cloud ...\n", Timestamp());
  pc_fix.SelectNPts(correspondences);

  printf("[%s] Estimate normals of selected points ...\n", Timestamp());
  pc_fix.EstimateNormals(neighbors);

  // Initialization
  Eigen::Matrix<double, 4, 4> H_old{Eigen::Matrix<double, 4, 4>::Identity()};
  Eigen::Matrix<double, 4, 4> H_new;
  Eigen::Matrix<double, 4, 4> dH;
  Eigen::VectorXd residual_dists;
  std::vector<double> residual_dists_mean;
  std::vector<double> residual_dists_std;

  printf("[%s] Start iterations ...\n", Timestamp());
  for (int i = 0; i < max_iterations; i++) {
    CorrPts cp = CorrPts(pc_fix, pc_mov);

    cp.Match();
    cp.Reject(min_planarity);
    auto initial_dists{cp.dists()};

    cp.EstimateRigidBodyTransformation(dH, residual_dists);

    pc_mov.Transform(dH);

    H_new = H_old * dH;
    H_old = H_new;

    residual_dists_mean.push_back(residual_dists.mean());
    residual_dists_std.push_back(Std(residual_dists));

    if (i > 0) {
      if (CheckConvergenceCriteria(residual_dists_mean, residual_dists_std, min_change)) {
        printf("[%s] Convergence criteria fulfilled -> stop iteration!\n", Timestamp());
        break;
      }
    }

    if (i == 0) {
      printf("[%s] %9s | %15s | %15s | %15s\n",
             Timestamp(),
             "Iteration",
             "correspondences",
             "mean(residuals)",
             "std(residuals)");
      printf("[%s] %9d | %15d | %15.4f | %15.4f\n",
             Timestamp(),
             i,
             int(initial_dists.size()),
             initial_dists.mean(),
             Std(initial_dists));
    }
    printf("[%s] %9d | %15d | %15.4f | %15.4f\n",
           Timestamp(),
           i + 1,
           int(residual_dists.size()),
           residual_dists_mean.back(),
           residual_dists_std.back());
  }

  printf("[%s] Estimated transformation matrix H:\n", Timestamp());
  printf("[%s] [%12.6f %12.6f %12.6f %12.6f]\n",
         Timestamp(),
         H_new(0, 0),
         H_new(0, 1),
         H_new(0, 2),
         H_new(0, 3));
  printf("[%s] [%12.6f %12.6f %12.6f %12.6f]\n",
         Timestamp(),
         H_new(1, 0),
         H_new(1, 1),
         H_new(1, 2),
         H_new(1, 3));
  printf("[%s] [%12.6f %12.6f %12.6f %12.6f]\n",
         Timestamp(),
         H_new(2, 0),
         H_new(2, 1),
         H_new(2, 2),
         H_new(2, 3));
  printf("[%s] [%12.6f %12.6f %12.6f %12.6f]\n",
         Timestamp(),
         H_new(3, 0),
         H_new(3, 1),
         H_new(3, 2),
         H_new(3, 3));

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  printf("[%s] Finished in %.3f seconds!\n", Timestamp(), elapsed_seconds.count());

  return H_new;
}

// https://stackoverflow.com/a/35157784
const char* Timestamp() {
  using namespace std::chrono;

  auto now = system_clock::now();  // current time

  // Get number of milliseconds for the current second
  // (remainder after division into seconds)
  auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

  // Convert to std::time_t in order to convert to std::tm (broken time)
  auto timer = system_clock::to_time_t(now);

  // convert to broken time
  std::tm bt = *std::localtime(&timer);

  std::ostringstream oss;

  oss << std::put_time(&bt, "%H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

  return oss.str().c_str();
}

Eigen::MatrixXi KnnSearch(const Eigen::MatrixXd& X, const Eigen::MatrixXd& X_query, const int& k) {
  // Create kd tree
  typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      kd_tree;
  kd_tree mat_index(3, std::cref(X), LEAF_SIZE);

  // Iterate over all query points
  Eigen::MatrixXi mat_idx_nn(X_query.rows(), k);
  for (size_t i = 0; i < X_query.rows(); i++) {
    // Query point
    std::vector<double> qp{X_query(i, 0), X_query(i, 1), X_query(i, 2)};

    // Search for nn of query point
    std::vector<size_t> idx_nn(k);
    std::vector<double> dists_nn(k);  // not used
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(&idx_nn[0], &dists_nn[0]);
    mat_index.index->findNeighbors(resultSet, &qp[0], nanoflann::SearchParams(10));

    // Save indices of nn to matrix
    for (int j = 0; j < k; j++) {
      mat_idx_nn(i, j) = idx_nn[j];
    }
  }
  return mat_idx_nn;
}

double Median(const Eigen::VectorXd& v) {
  // VectorXd --> vector<double>
  std::vector<double> vv(v.size());
  for (int i = 1; i < v.size(); i++) {
    vv[i] = v[i];
  }

  // Median
  const auto median_it = vv.begin() + vv.size() / 2;
  nth_element(vv.begin(), median_it, vv.end());
  auto median = *median_it;

  return median;
}

double MAD(const Eigen::VectorXd& v) {
  auto med{Median(v)};
  Eigen::VectorXd dmed(v.size());
  for (int i = 1; i < v.size(); i++) {
    dmed[i] = abs(v[i] - med);
  }
  auto mad{Median(dmed)};
  return mad;
}

double Std(const Eigen::VectorXd& v) {
  double std{sqrt((v.array() - v.mean()).square().sum() / (v.size() - 1))};
  return std;
}

double Change(const double& new_val, const double& old_val) {
  return abs((new_val - old_val) / old_val * 100);
}

bool CheckConvergenceCriteria(const std::vector<double>& residual_dists_mean,
                              const std::vector<double>& residual_dists_std,
                              const double& min_change) {
  if (Change(residual_dists_mean.end()[-1], residual_dists_mean.end()[-2]) < min_change) {
    if (Change(residual_dists_std.end()[-1], residual_dists_std.end()[-2]) < min_change) {
      return true;
    }
  }
  return false;
}
