use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use simpleicp::{simple_icp, IcpParams, PointCloud};

// Long arg names use underscores (e.g. --max_overlap_distance) to match the
// C++ CLI, so scripts/benchmark.sh can invoke either binary identically.
#[derive(Parser, Debug)]
#[command(name = "simpleicp", about = "A simple version of the ICP algorithm.")]
struct Cli {
    /// Path to fixed point cloud (.xyz)
    #[arg(short = 'f', long = "fixed")]
    fixed: PathBuf,

    /// Path to movable point cloud (.xyz)
    #[arg(short = 'm', long = "movable")]
    movable: PathBuf,

    /// Number of initially selected correspondences
    #[arg(short = 'c', long = "correspondences", default_value_t = 1000)]
    correspondences: usize,

    /// Number of neighbors used for plane estimation
    #[arg(short = 'n', long = "neighbors", default_value_t = 10)]
    neighbors: usize,

    /// Minimal planarity value of planes used as correspondence
    #[arg(short = 'p', long = "min_planarity", default_value_t = 0.3)]
    min_planarity: f64,

    /// Maximum initial overlap distance. Set to a negative value if point
    /// clouds are fully overlapping.
    #[arg(short = 'o', long = "max_overlap_distance", default_value_t = -1.0)]
    max_overlap_distance: f64,

    /// Minimal change of mean and standard deviation of distances (in
    /// percent) needed to proceed to the next iteration
    #[arg(short = 'i', long = "min_change", default_value_t = 1.0)]
    min_change: f64,

    /// Maximum number of iterations
    #[arg(short = 'x', long = "max_iterations", default_value_t = 100)]
    max_iterations: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let fixed = PointCloud::from_xyz(&cli.fixed)?;
    let movable = PointCloud::from_xyz(&cli.movable)?;

    let params = IcpParams {
        correspondences: cli.correspondences,
        neighbors: cli.neighbors,
        min_planarity: cli.min_planarity,
        max_overlap_distance: if cli.max_overlap_distance > 0.0 {
            Some(cli.max_overlap_distance)
        } else {
            None
        },
        min_change: cli.min_change,
        max_iterations: cli.max_iterations,
    };

    simple_icp(&fixed, &movable, &params)?;
    Ok(())
}
