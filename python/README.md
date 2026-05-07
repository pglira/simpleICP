# simpleICP

This package contains an implementation of a rather simple version of the [Iterative Closest Point (ICP) algorithm](https://en.wikipedia.org/wiki/Iterative_closest_point).

## Documentation

This python implementation is just one of several (almost identical) implementations of the ICP algorithm in various programming languages. They all share a common documentation here: https://github.com/pglira/simpleICP

## Installation

You can install the simpleicp package from [PyPI](https://pypi.org/project/simpleicp/):

```
pip install simpleicp
```

## How to use

```python
from simpleicp import PointCloud, SimpleICP
import numpy as np

# Read point clouds from xyz files into n-by-3 numpy arrays
X_fix = np.genfromtxt("bunny_part1.xyz")
X_mov = np.genfromtxt("bunny_part2.xyz")

# Create point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)
```

This should give this output:

```
Consider partial overlap of point clouds ...
Select points for correspondences in fixed point cloud ...
Estimate normals of selected points ...
Start iterations ...
iteration | correspondences | mean(residuals) |  std(residuals)
   orig:0 |             863 |          0.0403 |          0.1825
        1 |             862 |          0.0096 |          0.1113
        2 |             775 |          0.0050 |          0.0553
        3 |             807 |          0.0022 |          0.0407
        4 |             825 |          0.0016 |          0.0346
        5 |             825 |          0.0010 |          0.0253
        6 |             816 |          0.0008 |          0.0198
        7 |             785 |         -0.0000 |          0.0142
        8 |             764 |          0.0008 |          0.0091
        9 |             753 |          0.0003 |          0.0061
       10 |             735 |          0.0002 |          0.0040
       11 |             742 |         -0.0001 |          0.0032
       12 |             747 |         -0.0000 |          0.0030
       13 |             752 |         -0.0000 |          0.0030
       14 |             752 |         -0.0000 |          0.0029
Convergence criteria fulfilled -> stop iteration!
Estimated transformation matrix H:
[    0.984798    -0.173702    -0.000053     0.000676]
[    0.173702     0.984798     0.000084    -0.001150]
[    0.000038    -0.000092     1.000000     0.000113]
[    0.000000     0.000000     0.000000     1.000000]
... which corresponds to the following rigid-body transformation parameters:
parameter |       est.value | est.uncertainty |       obs.value |      obs.weight
   alpha1 |       -0.004804 |        0.004491 |        0.000000 |       0.000e+00
   alpha2 |       -0.003061 |        0.002104 |        0.000000 |       0.000e+00
   alpha3 |       10.003124 |        0.005680 |        0.000000 |       0.000e+00
       tx |        0.000676 |        0.000418 |        0.000000 |       0.000e+00
       ty |       -0.001150 |        0.000885 |        0.000000 |       0.000e+00
       tz |        0.000113 |        0.000189 |        0.000000 |       0.000e+00
(Unit of est.value, est.uncertainty, and obs.value for alpha1/2/3 is degree)
Finished in 4.737 seconds!
```

Note that ``bunny_part1.xyz`` and ``bunny_part2.xyz`` are not included in this package. They can be downloaded (among other example files) [here](https://github.com/pglira/simpleICP/tree/master/data).