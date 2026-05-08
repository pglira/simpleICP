use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use nalgebra::Point3;

/// Read a whitespace-separated XYZ file. Each non-empty line: `x y z`.
pub(crate) fn read_xyz(path: &Path) -> Result<Vec<Point3<f64>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut points = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", lineno + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let coords: Vec<f64> = trimmed
            .split_whitespace()
            .map(|s| {
                s.parse::<f64>()
                    .with_context(|| format!("parsing {:?} on line {}", s, lineno + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        if coords.len() != 3 {
            return Err(anyhow!(
                "line {} does not have 3 columns: {:?}",
                lineno + 1,
                coords
            ));
        }
        points.push(Point3::new(coords[0], coords[1], coords[2]));
    }
    Ok(points)
}
