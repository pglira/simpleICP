use std::path::Path;

use anyhow::Result;
use nalgebra::{Matrix4, Point3, Vector4};

use crate::io::read_xyz;

/// A 3D point cloud — a thin wrapper around `Vec<Point3<f64>>`.
#[derive(Debug, Clone)]
pub struct PointCloud {
    points: Vec<Point3<f64>>,
}

impl PointCloud {
    pub fn from_points(points: Vec<Point3<f64>>) -> Self {
        Self { points }
    }

    pub fn from_xyz(path: &Path) -> Result<Self> {
        Ok(Self {
            points: read_xyz(path)?,
        })
    }

    pub fn points(&self) -> &[Point3<f64>] {
        &self.points
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Apply a 4×4 homogeneous transform in place.
    pub fn transform(&mut self, h: &Matrix4<f64>) {
        for p in &mut self.points {
            let v = Vector4::new(p.x, p.y, p.z, 1.0);
            let r = h * v;
            *p = Point3::new(r.x / r.w, r.y / r.w, r.z / r.w);
        }
    }
}
