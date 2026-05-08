use std::path::PathBuf;

use simpleicp::{simple_icp, IcpParams, PointCloud};

fn data_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    p.push("data");
    p.push(name);
    p
}

fn assert_converged(result: &simpleicp::IcpResult, max_iterations: usize) {
    assert!(
        result.iterations < max_iterations,
        "did not converge within {} iterations",
        max_iterations
    );
    let m = &result.mean_residuals;
    assert!(
        m.last().unwrap().abs() < m.first().unwrap().abs(),
        "mean residual did not improve: first={:.4}, last={:.4}",
        m.first().unwrap(),
        m.last().unwrap()
    );
}

#[test]
fn bunny_converges() {
    let fixed = PointCloud::from_xyz(&data_path("bunny_part1.xyz")).unwrap();
    let movable = PointCloud::from_xyz(&data_path("bunny_part2.xyz")).unwrap();
    let params = IcpParams {
        max_overlap_distance: Some(1.0),
        ..IcpParams::default()
    };
    let result = simple_icp(&fixed, &movable, &params).unwrap();
    assert_converged(&result, params.max_iterations);
}

#[test]
fn dragon_converges() {
    let fixed = PointCloud::from_xyz(&data_path("dragon1.xyz")).unwrap();
    let movable = PointCloud::from_xyz(&data_path("dragon2.xyz")).unwrap();
    let params = IcpParams::default();
    let result = simple_icp(&fixed, &movable, &params).unwrap();
    assert_converged(&result, params.max_iterations);
}

// The two lidar datasets are 1.3M points each and take meaningfully longer.
// Run with `cargo test --release -- --ignored` when refreshing benchmarks.

#[test]
#[ignore]
fn airborne_converges() {
    let fixed = PointCloud::from_xyz(&data_path("airborne_lidar1.xyz")).unwrap();
    let movable = PointCloud::from_xyz(&data_path("airborne_lidar2.xyz")).unwrap();
    let params = IcpParams::default();
    let result = simple_icp(&fixed, &movable, &params).unwrap();
    assert_converged(&result, params.max_iterations);
}

#[test]
#[ignore]
fn terrestrial_converges() {
    let fixed = PointCloud::from_xyz(&data_path("terrestrial_lidar1.xyz")).unwrap();
    let movable = PointCloud::from_xyz(&data_path("terrestrial_lidar2.xyz")).unwrap();
    let params = IcpParams::default();
    let result = simple_icp(&fixed, &movable, &params).unwrap();
    assert_converged(&result, params.max_iterations);
}
