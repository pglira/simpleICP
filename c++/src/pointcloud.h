#ifndef RUN_SIMPLEICP_POINTCLOUD_H
#define RUN_SIMPLEICP_POINTCLOUD_H

#include <Eigen/Dense>
#include <vector>

class PointCloud {
 public:
  PointCloud(Eigen::MatrixXd X);

  Eigen::MatrixXd GetXOfSelectedPts();

  std::vector<int> GetIdxOfSelectedPts();

  void SelectNPts(const int& n);

  void EstimateNormals(const int& neighbors);

  void Transform(Eigen::Matrix<double, 4, 4>& H);

  int NoPts();

  // Getters
  const Eigen::MatrixXd& X();
  const Eigen::VectorXd& nx();
  const Eigen::VectorXd& ny();
  const Eigen::VectorXd& nz();
  const Eigen::VectorXd& planarity();
  const std::vector<bool>& sel();

 private:
  Eigen::MatrixXd X_;
  Eigen::VectorXd nx_;
  Eigen::VectorXd ny_;
  Eigen::VectorXd nz_;
  Eigen::VectorXd planarity_;
  std::vector<bool> sel_;
};

#endif  // RUN_SIMPLEICP_POINTCLOUD_H
