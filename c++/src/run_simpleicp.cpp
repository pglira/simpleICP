#include "simpleicp.h"

int main()
{
  // auto X_fix = ImportXYZFileToMatrix(std::string("../../data/dragon1.xyz"));
  // auto X_mov = ImportXYZFileToMatrix(std::string("../../data/dragon2.xyz"));

  // auto X_fix = ImportXYZFileToMatrix(std::string("../../data/airborne_lidar1.xyz"));
  // auto X_mov = ImportXYZFileToMatrix(std::string("../../data/airborne_lidar2.xyz"));

  auto X_fix = ImportXYZFileToMatrix(std::string("../../data/terrestrial_lidar1.xyz"));
  auto X_mov = ImportXYZFileToMatrix(std::string("../../data/terrestrial_lidar2.xyz"));

  auto H = SimpleICP(X_fix, X_mov);

  return 0;
}
