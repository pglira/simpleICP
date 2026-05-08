//! Point-to-plane ICP for 3D point clouds.
//!
//! Public API:
//! - [`PointCloud`]: load points from `.xyz` files or a `Vec<Point3<f64>>`.
//! - [`simple_icp`]: align a movable cloud to a fixed cloud, returning the
//!   homogeneous transform plus per-iteration residual stats.

mod icp;
mod io;
mod point_cloud;

pub use icp::{simple_icp, IcpParams, IcpResult};
pub use point_cloud::PointCloud;
