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
from simpleicp import simpleicp
import numpy as np

# Read fixed and movable point cloud from xyz files into n-by-3 numpy arrays
X_fix = np.genfromtxt("dragon1.xyz")
X_mov = np.genfromtxt("dragon2.xyz")

# Run simpleICP!
H, X_mov_transformed = simpleicp.simpleicp(X_fix, X_mov)
```

This should give (except for timestamps and timings) this output:

```log
[11:18:30.890] Create point cloud objects ...
[11:18:30.891] Select points for correspondences within overlap area of fixed point cloud ...
[11:18:30.891] Estimate normals of selected points ...
[11:18:31.048] Start iterations ...
[11:18:31.084] Iteration | correspondences | mean(residuals) | std(residuals)
[11:18:31.084]         0 |             768 |          0.0464 |          0.3179
[11:18:31.084]         1 |             768 |          0.0124 |          0.2534
[11:18:31.155]         2 |             773 |          0.0100 |          0.1655
[11:18:31.212]         3 |             772 |          0.0040 |          0.0830
[11:18:31.306]         4 |             749 |          0.0010 |          0.0191
[11:18:31.415]         5 |             758 |         -0.0000 |          0.0024
[11:18:31.487]         6 |             774 |          0.0000 |          0.0022
[11:18:31.556] Convergence criteria fulfilled -> stop iteration!
[11:18:31.556] Estimated transformation matrix H:
[11:18:31.557] H = [    0.998693     0.052911    -0.034201    -0.197402]
[11:18:31.557] H = [   -0.052402     0.999035     0.019119    -0.400079]
[11:18:31.557] H = [    0.034816    -0.017661     0.999446    -0.602222]
[11:18:31.558] H = [    0.000000     0.000000     0.000000     1.000000]
[11:18:31.558] Finished in 0.669 seconds!
```

Note that ``dragon1.xyz`` and ``dragon2.xyz`` are not included in this package. They can be downloaded (among other example files) [here](https://github.com/pglira/simpleICP/tree/master/data).