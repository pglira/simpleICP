#include <Eigen/Dense>
#include "cxxopts.hpp"
#include "simpleicp.h"

Eigen::MatrixXd ImportXYZFileToMatrix(const std::string &path_to_pc);

int main(int argc, char **argv)
{
  try
  {

    cxxopts::Options options("simpleicp", "A simple version of the ICP algorithm.");

    // clang-format off
    options.add_options()
      ("f,fixed", "Path to fixed point cloud",
        cxxopts::value<std::string>())
      ("m,movable", "Path to movable point cloud",
        cxxopts::value<std::string>())
      ("c,correspondences", "Number of initially selected correspondences",
        cxxopts::value<int>()->default_value("1000"))
      ("n,neighbors", "Number of neighbors used for plane estimation",
        cxxopts::value<int>()->default_value("10"))
      ("p,min_planarity", "Minimal planarity value of planes used as correspondence",
        cxxopts::value<double>()->default_value("0.3"))
      ("o,max_overlap_distance", "Maximum initial overlap distance. Set to negative value if point "
      "clouds are fully overlapping.",
        cxxopts::value<double>()->default_value("-1"))
      ("i,min_change", "Minimal change of mean and standard deviation of distances (in percent) "
                       "needed to proceed to next iteration",
        cxxopts::value<double>()->default_value("1"))
      ("x,max_iterations", "Maximum number of iterations",
        cxxopts::value<int>()->default_value("100"))
      ("h,help", "Print usage")
      ;
    // clang-format on

    auto result = options.parse(argc, argv);

    if (result.count("help") || argc == 1)
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    auto X_fix = ImportXYZFileToMatrix(std::string(result["fixed"].as<std::string>()));
    auto X_mov = ImportXYZFileToMatrix(std::string(result["movable"].as<std::string>()));

    Eigen::Matrix<double, 4, 4> H = SimpleICP(X_fix,
                                              X_mov,
                                              result["correspondences"].as<int>(),
                                              result["neighbors"].as<int>(),
                                              result["min_planarity"].as<double>(),
                                              result["max_overlap_distance"].as<double>(),
                                              result["min_change"].as<double>(),
                                              result["max_iterations"].as<int>());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception." << std::endl;
    return 1;
  }

  return 0;
}

Eigen::MatrixXd ImportXYZFileToMatrix(const std::string &path_to_pc)
{
  std::ifstream data(path_to_pc);
  if (data.is_open())
  {
    // Read data from file
    std::vector<std::vector<std::string>> parsedData;
    std::string line;
    while (getline(data, line))
    {
      std::stringstream lineStream(line);
      std::string cell; // single value
      std::vector<std::string> parsedRow;
      while (getline(lineStream, cell, ' '))
      {
        parsedRow.push_back(cell);
      }
      parsedData.push_back(parsedRow);
    }

    // Check if each line contains exactly 3 values
    for (uint i = 0; i < parsedData.size(); i++)
    {
      if (parsedData[i].size() != 3)
      {
        std::cerr << "Line " << i + 1 << " does not contain exactly 3 values!" << std::endl;
        exit(-1);
      }
    }

    // Create eigen array
    Eigen::MatrixXd X(parsedData.size(), 3);
    for (uint i = 0; i < parsedData.size(); i++)
    {
      for (uint j = 0; j < parsedData[i].size(); j++)
      {
        try
        {
          X(i, j) = stod(parsedData[i][j]);
        }
        catch (std::exception &e)
        {
          std::cerr << "Conversion of " << parsedData[i][j] << " on row/column=" << i << "/" << j
                    << " is not possible!" << std::endl;
          exit(-1);
        }
      }
    }

    return X;
  }
  else
  {
    std::cerr << "Error opening file!" << std::endl;
    exit(-1);
  }
}
