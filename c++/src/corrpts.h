#ifndef RUN_SIMPLEICP_CORRPTS_H
#define RUN_SIMPLEICP_CORRPTS_H

#include "pointcloud.h"

class CorrPts {
 public:
  CorrPts(PointCloud& pc1, PointCloud& pc2);

  // Matching of each selected point of pc1 --> nn of selected points of pc2
  void Match();

  void GetPlanarityFromPc1();

  void ComputeDists();

  void Reject(const double& min_planarity);

  void EstimateRigidBodyTransformation(Matrix<double, 4, 4>& H, VectorXd& residuals);

  // Getters
  const PointCloud& pc1();
  const PointCloud& pc2();
  const std::vector<int>& idx_pc1();
  const std::vector<int>& idx_pc2();
  const VectorXd& dists();
  const VectorXd& planarity();

 private:
  PointCloud pc1_;
  PointCloud pc2_;
  std::vector<int> idx_pc1_;
  std::vector<int> idx_pc2_;
  VectorXd dists_;
  VectorXd planarity_;
};

#endif  // RUN_SIMPLEICP_CORRPTS_H
