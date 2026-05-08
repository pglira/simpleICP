use std::num::NonZeroUsize;
use std::time::Instant;

use anyhow::{anyhow, Result};
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Point3, Vector3, Vector4};

use crate::point_cloud::PointCloud;

/// ICP parameters. Defaults match the values used by the C++/Julia/Matlab/Python
/// implementations in this repo so timings and results are comparable.
#[derive(Debug, Clone)]
pub struct IcpParams {
    pub correspondences: usize,
    pub neighbors: usize,
    pub min_planarity: f64,
    /// `Some(d)` filters fixed points further than `d` from any movable point
    /// before subsampling correspondences. `None` keeps all fixed points.
    pub max_overlap_distance: Option<f64>,
    /// Convergence threshold: percent change in mean and std of residuals
    /// below which iteration stops.
    pub min_change: f64,
    pub max_iterations: usize,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self {
            correspondences: 1000,
            neighbors: 10,
            min_planarity: 0.3,
            max_overlap_distance: None,
            min_change: 1.0,
            max_iterations: 100,
        }
    }
}

/// Result of a [`simple_icp`] run.
#[derive(Debug, Clone)]
pub struct IcpResult {
    /// 4×4 homogeneous transform aligning the movable cloud to fixed.
    pub transform: Matrix4<f64>,
    pub iterations: usize,
    pub mean_residuals: Vec<f64>,
    pub std_residuals: Vec<f64>,
}

const KD_BUCKET: usize = 32;
// ImmutableKdTree is built bulk from a slice, so its items are indices into
// that slice (item type defaults to u64). It also handles points coplanar on
// an axis cleanly — the mutable KdTree panics on those past bucket_size.
type Kd = ImmutableKdTree<f64, u64, 3, KD_BUCKET>;

/// Align `movable` to `fixed` via point-to-plane ICP.
///
/// Prints per-iteration progress and a final `Finished in N.NNN seconds!`
/// line to stdout, matching the format used by the other simpleICP
/// language implementations.
pub fn simple_icp(
    fixed: &PointCloud,
    movable: &PointCloud,
    params: &IcpParams,
) -> Result<IcpResult> {
    let start = Instant::now();

    let mut movable_pts = movable.points().to_vec();

    // Optional partial-overlap filter on the fixed cloud.
    let fixed_indices: Vec<usize> = match params.max_overlap_distance {
        Some(d) if d > 0.0 => {
            println!("Consider partial overlap of point clouds ...");
            filter_overlap(fixed.points(), &movable_pts, d)
        }
        _ => (0..fixed.len()).collect(),
    };
    if fixed_indices.is_empty() {
        return Err(anyhow!(
            "Point clouds do not overlap within max_overlap_distance"
        ));
    }

    println!("Select points for correspondences in fixed point cloud ...");
    let selected = subsample_indices(&fixed_indices, params.correspondences);

    println!("Estimate normals of selected points ...");
    let (normals, planarity) = estimate_normals(fixed.points(), &selected, params.neighbors)?;

    println!("Start iterations ...");
    let mut total_h = Matrix4::<f64>::identity();
    let mut means: Vec<f64> = Vec::new();
    let mut stds: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut iter_count = 0usize;

    for i in 0..params.max_iterations {
        iter_count = i + 1;

        // 1. Match: NN in current movable for each selected fixed point.
        let movable_tree = build_tree(&movable_pts);
        let corr_movable: Vec<usize> = selected
            .iter()
            .map(|&j| {
                let p = fixed.points()[j];
                movable_tree
                    .nearest_one::<SquaredEuclidean>(&[p.x, p.y, p.z])
                    .item as usize
            })
            .collect();

        // 2. Point-to-plane signed distance using the fixed-side normal.
        let dists: Vec<f64> = (0..selected.len())
            .map(|k| {
                let p1 = fixed.points()[selected[k]];
                let p2 = movable_pts[corr_movable[k]];
                normals[k].dot(&(p2 - p1))
            })
            .collect();

        if i == 0 {
            println!(
                "{:>9} | {:>15} | {:>15} | {:>15}",
                "Iteration", "correspondences", "mean(residuals)", "std(residuals)"
            );
            println!(
                "{:>9} | {:>15} | {:>15.4} | {:>15.4}",
                "orig:0",
                dists.len(),
                mean(&dists),
                std_dev(&dists),
            );
        }

        // 3. Reject outliers (median ± 3·1.4826·MAD) and low-planarity samples.
        let med = median(&dists);
        let sigma = 1.4826 * mad(&dists, med);
        let kept: Vec<usize> = (0..selected.len())
            .filter(|&k| {
                let d = dists[k];
                d >= med - 3.0 * sigma
                    && d <= med + 3.0 * sigma
                    && planarity[k] >= params.min_planarity
            })
            .collect();

        if kept.len() < 6 {
            return Err(anyhow!(
                "Too few correspondences ({}) after rejection",
                kept.len()
            ));
        }

        // 4. Solve linearised least-squares system for incremental transform.
        let kept_fixed: Vec<Point3<f64>> = kept.iter().map(|&k| fixed.points()[selected[k]]).collect();
        let kept_normals: Vec<Vector3<f64>> = kept.iter().map(|&k| normals[k]).collect();
        let kept_movable: Vec<Point3<f64>> =
            kept.iter().map(|&k| movable_pts[corr_movable[k]]).collect();

        let (dh, residuals) = estimate_rigid_body(&kept_fixed, &kept_normals, &kept_movable)?;

        // 5. Apply increment to movable cloud and accumulate into total transform.
        apply_to_points(&mut movable_pts, &dh);
        total_h = dh * total_h;

        // 6. Convergence check on residual mean and std.
        let mean_r = mean(&residuals);
        let std_r = std_dev(&residuals);
        means.push(mean_r);
        stds.push(std_r);

        println!(
            "{:>9} | {:>15} | {:>15.4} | {:>15.4}",
            iter_count,
            residuals.len(),
            mean_r,
            std_r,
        );

        if i >= 1 && check_convergence(&means, &stds, params.min_change) {
            converged = true;
            break;
        }
    }

    if converged {
        println!("Convergence criteria fulfilled -> stop iteration!");
    }

    println!("Estimated transformation matrix H:");
    for row in 0..4 {
        println!(
            "[{:12.6} {:12.6} {:12.6} {:12.6}]",
            total_h[(row, 0)],
            total_h[(row, 1)],
            total_h[(row, 2)],
            total_h[(row, 3)],
        );
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("Finished in {:.3} seconds!", elapsed);

    Ok(IcpResult {
        transform: total_h,
        iterations: iter_count,
        mean_residuals: means,
        std_residuals: stds,
    })
}

fn build_tree(points: &[Point3<f64>]) -> Kd {
    let entries: Vec<[f64; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
    Kd::new_from_slice(&entries)
}

fn filter_overlap(fixed: &[Point3<f64>], movable: &[Point3<f64>], max_dist: f64) -> Vec<usize> {
    let tree = build_tree(movable);
    let max_sq = max_dist * max_dist;
    fixed
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            tree.nearest_one::<SquaredEuclidean>(&[p.x, p.y, p.z])
                .distance
                <= max_sq
        })
        .map(|(i, _)| i)
        .collect()
}

fn subsample_indices(indices: &[usize], n: usize) -> Vec<usize> {
    if indices.len() <= n {
        return indices.to_vec();
    }
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![indices[0]];
    }
    let step = (indices.len() - 1) as f64 / (n - 1) as f64;
    (0..n)
        .map(|i| indices[(i as f64 * step).round() as usize])
        .collect()
}

fn estimate_normals(
    points: &[Point3<f64>],
    selected: &[usize],
    k: usize,
) -> Result<(Vec<Vector3<f64>>, Vec<f64>)> {
    if k < 3 {
        return Err(anyhow!("neighbors must be >= 3, got {}", k));
    }
    if points.len() < k {
        return Err(anyhow!(
            "not enough points ({}) for k={} neighbors",
            points.len(),
            k
        ));
    }

    let tree = build_tree(points);
    let k_nz = NonZeroUsize::new(k).expect("k must be > 0");
    let mut normals = Vec::with_capacity(selected.len());
    let mut planarities = Vec::with_capacity(selected.len());

    for &j in selected {
        let p = points[j];
        let nn = tree.nearest_n::<SquaredEuclidean>(&[p.x, p.y, p.z], k_nz);
        let neighbor_pts: Vec<Point3<f64>> = nn.iter().map(|n| points[n.item as usize]).collect();

        let mut centroid = Vector3::<f64>::zeros();
        for q in &neighbor_pts {
            centroid += q.coords;
        }
        centroid /= k as f64;

        let mut cov = Matrix3::<f64>::zeros();
        for q in &neighbor_pts {
            let d = q.coords - centroid;
            cov += d * d.transpose();
        }
        cov /= (k - 1) as f64;

        let eig = cov.symmetric_eigen();
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| {
            eig.eigenvalues[a]
                .partial_cmp(&eig.eigenvalues[b])
                .unwrap()
        });
        let lambda_min = eig.eigenvalues[idx[0]];
        let lambda_mid = eig.eigenvalues[idx[1]];
        let lambda_max = eig.eigenvalues[idx[2]];
        let col = eig.eigenvectors.column(idx[0]);
        let normal = Vector3::new(col[0], col[1], col[2]);
        let planarity = if lambda_max > 0.0 {
            (lambda_mid - lambda_min) / lambda_max
        } else {
            0.0
        };
        normals.push(normal);
        planarities.push(planarity);
    }
    Ok((normals, planarities))
}

fn estimate_rigid_body(
    fixed: &[Point3<f64>],
    normals: &[Vector3<f64>],
    movable: &[Point3<f64>],
) -> Result<(Matrix4<f64>, Vec<f64>)> {
    let n = fixed.len();
    let mut a = DMatrix::<f64>::zeros(n, 6);
    let mut l = DVector::<f64>::zeros(n);

    for i in 0..n {
        let p1 = &fixed[i];
        let p2 = &movable[i];
        let nm = &normals[i];
        a[(i, 0)] = -p2.z * nm.y + p2.y * nm.z;
        a[(i, 1)] = p2.z * nm.x - p2.x * nm.z;
        a[(i, 2)] = -p2.y * nm.x + p2.x * nm.y;
        a[(i, 3)] = nm.x;
        a[(i, 4)] = nm.y;
        a[(i, 5)] = nm.z;
        l[i] = nm.x * (p1.x - p2.x) + nm.y * (p1.y - p2.y) + nm.z * (p1.z - p2.z);
    }

    let svd = a.clone().svd(true, true);
    let x = svd
        .solve(&l, 1e-12)
        .map_err(|e| anyhow!("least-squares solve failed: {}", e))?;

    let residuals: Vec<f64> = (&a * &x - &l).iter().copied().collect();

    // Linearised rotation: R ≈ I + skew([α1, α2, α3]).
    let h = Matrix4::new(
        1.0, -x[2], x[1], x[3],
        x[2], 1.0, -x[0], x[4],
        -x[1], x[0], 1.0, x[5],
        0.0, 0.0, 0.0, 1.0,
    );
    Ok((h, residuals))
}

fn apply_to_points(points: &mut [Point3<f64>], h: &Matrix4<f64>) {
    for p in points.iter_mut() {
        let r = h * Vector4::new(p.x, p.y, p.z, 1.0);
        *p = Point3::new(r.x / r.w, r.y / r.w, r.z / r.w);
    }
}

fn check_convergence(means: &[f64], stds: &[f64], min_change_pct: f64) -> bool {
    let n = means.len();
    let change_mean = pct_change(means[n - 1], means[n - 2]);
    let change_std = pct_change(stds[n - 1], stds[n - 2]);
    change_mean < min_change_pct && change_std < min_change_pct
}

fn pct_change(new_v: f64, old_v: f64) -> f64 {
    if old_v == 0.0 {
        if new_v == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        ((new_v - old_v) / old_v * 100.0).abs()
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 0 {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    } else {
        s[n / 2]
    }
}

fn mad(v: &[f64], med: f64) -> f64 {
    let abs_dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    median(&abs_dev)
}
