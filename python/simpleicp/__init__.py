__version__ = "2.0.14"

import logging as _logging

# Library best practice: attach a NullHandler so importing simpleicp without
# configuring logging never raises "No handlers could be found" warnings.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Import statements in order to simplify API
# Reference: https://stackoverflow.com/a/35733111
# Import here only objects which are directly used by the user
from .simpleicp import SimpleICP
from .pointcloud import PointCloud
from .optimization import RigidBodyParameters
