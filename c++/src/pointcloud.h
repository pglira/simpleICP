#ifndef RUN_SIMPLEICP_POINTCLOUD_H
#define RUN_SIMPLEICP_POINTCLOUD_H

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

class PointCloud {
  public:

    PointCloud(MatrixXd X);

    MatrixXd GetXOfSelectedPts();

    std::vector<int> GetIdxOfSelectedPts();

    void SelectNPts(const int& n);

    void EstimateNormals(const int& neighbors);

    void Transform(Matrix<double, 4, 4>& H);

    int NoPts();

    // Getters
    const MatrixXd& X();
    const VectorXd& nx();
    const VectorXd& ny();
    const VectorXd& nz();
    const std::vector<bool>& sel();

  private:
    MatrixXd X_;
    VectorXd nx_;
    VectorXd ny_;
    VectorXd nz_;
    std::vector<bool> sel_;
};

#endif //RUN_SIMPLEICP_POINTCLOUD_H
