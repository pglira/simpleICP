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
H, X_mov_transformed = icp.run(max_overlap_distance=1)
```

This should give this output:

```
Consider partial overlap of point clouds ...
Select points for correspondences in fixed point cloud ...
Estimate normals of selected points ...
Start iterations ...
iteration | correspondences | mean(residuals) |  std(residuals)
   orig:0 |             951 |          0.0401 |          0.2397
        1 |             950 |          0.0027 |          0.1356
        2 |             889 |          0.0026 |          0.0586
        3 |             897 |          0.0020 |          0.0407
        4 |             873 |          0.0004 |          0.0303
        5 |             854 |          0.0004 |          0.0245
        6 |             847 |          0.0003 |          0.0208
        7 |             826 |         -0.0006 |          0.0154
        8 |             799 |          0.0005 |          0.0099
        9 |             787 |          0.0002 |          0.0068
       10 |             783 |         -0.0001 |          0.0047
       11 |             779 |         -0.0001 |          0.0037
       12 |             776 |         -0.0000 |          0.0033
       13 |             776 |         -0.0000 |          0.0033
Convergence criteria fulfilled -> stop iteration!
Estimated transformation matrix H:
[    0.984804    -0.173671    -0.000041     0.000420]
[    0.173671     0.984804     0.000051    -0.000750]
[    0.000032    -0.000057     1.000000     0.000054]
[    0.000000     0.000000     0.000000     1.000000]
... which corresponds to the following rigid body transformation parameters:
parameter |       est.value | est.uncertainty |       obs.value |      obs.weight
   alpha1 |       -0.002906 |        0.004963 |        0.000000 |        0.000000
   alpha2 |       -0.002353 |        0.002339 |        0.000000 |        0.000000
   alpha3 |       10.001317 |        0.006276 |        0.000000 |        0.000000
       tx |        0.000420 |        0.000459 |        0.000000 |        0.000000
       ty |       -0.000750 |        0.000974 |        0.000000 |        0.000000
       tz |        0.000054 |        0.000209 |        0.000000 |        0.000000
(Unit of est.value, est.uncertainty, and obs.value for alpha1/2/3 is degree)
Finished in 4.320 seconds!
```

Note that ``bunny_part1.xyz`` and ``bunny_part2.xyz`` are not included in this package. They can be downloaded (among other example files) [here](https://github.com/pglira/simpleICP/tree/master/data).