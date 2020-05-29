using Dates
using DelimitedFiles
using LinearAlgebra
using MultivariateStats
using NearestNeighbors # https://github.com/KristofferC/NearestNeighbors.jl
using Printf
using Statistics
using StatsBase

mutable struct PointCloud

    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}

    nx::Vector{Float64}
    ny::Vector{Float64}
    nz::Vector{Float64}
    planarity::Vector{Float64}

    no_points::Int64
    sel::Vector{Int64}

    function PointCloud(x, y, z)
        no_points = length(x)
        new(x,
            y,
            z,
            fill(NaN, no_points),
            fill(NaN, no_points),
            fill(NaN, no_points),
            fill(NaN, no_points),
            no_points,
            Int64[])
    end

end

function select_n_points!(pc::PointCloud, n)

    if pc.no_points > n
        pc.sel = round.(Int, range(1, pc.no_points, length=n))
    else
        pc.sel = collect(1:pc.no_points)
    end

end

function estimate_normals!(pc::PointCloud, neighbors)

    kdtree = KDTree([pc.x'; pc.y'; pc.z'])
    query_points = [pc.x[pc.sel]'; pc.y[pc.sel]'; pc.z[pc.sel]']
    idxNN_all_qp, = knn(kdtree, query_points, neighbors, false)

    for (i, idxNN) in enumerate(idxNN_all_qp)
        selected_points = [pc.x[idxNN]'; pc.y[idxNN]'; pc.z[idxNN]']
        # P = fit(PCA, selected_points, pratio=1.0)
        # pc.nx[pc.sel[i]] = projection(P)[1,3]
        # pc.ny[pc.sel[i]] = projection(P)[2,3]
        # pc.nz[pc.sel[i]] = projection(P)[3,3]
        C = cov(selected_points, dims=2)
        F = eigen(C) # eigenvalues are in ascending order
        pc.nx[pc.sel[i]] = F.vectors[1,1]
        pc.ny[pc.sel[i]] = F.vectors[2,1]
        pc.nz[pc.sel[i]] = F.vectors[3,1]
        pc.planarity[pc.sel[i]] = (F.values[2]-F.values[1])/F.values[3];
    end

end

function transform!(pc, H)

    XInH = euler_coord_to_homogeneous_coord([pc.x pc.y pc.z])
    XOutH = transpose(H*XInH')
    XOut = homogeneous_coord_to_euler_coord(XOutH)

    pc.x = XOut[:,1]
    pc.y = XOut[:,2]
    pc.z = XOut[:,3]

    return pc

end

function matching!(pcmov::PointCloud, pcfix)

    kdtree = KDTree([pcmov.x'; pcmov.y'; pcmov.z'])
    query_points = [pcfix.x[pcfix.sel]'
                    pcfix.y[pcfix.sel]'
                    pcfix.z[pcfix.sel]']
    idxNN, = knn(kdtree, query_points, 1)
    pcmov.sel = vcat(idxNN...)

    dx = pcmov.x[pcmov.sel] - pcfix.x[pcfix.sel]
    dy = pcmov.y[pcmov.sel] - pcfix.y[pcfix.sel]
    dz = pcmov.z[pcmov.sel] - pcfix.z[pcfix.sel]

    nx = pcfix.nx[pcfix.sel]
    ny = pcfix.ny[pcfix.sel]
    nz = pcfix.nz[pcfix.sel]

    distances = [dx[i]*nx[i] + dy[i]*ny[i] + dz[i]*nz[i] for i in 1:length(pcmov.sel)]

    return distances

end

function reject!(pcmov::PointCloud, pcfix::PointCloud, min_planarity, distances)

    planarity = pcfix.planarity[pcfix.sel]

    med = median(distances)
    sigmad = mad(distances, normalize=true)

    keep_distance = [abs(d-med) <= 3*sigmad for d in distances]
    keep_planarity = [p > min_planarity for p in planarity]

    keep = keep_distance .& keep_planarity

    pcmov.sel = pcmov.sel[keep]
    pcfix.sel = pcfix.sel[keep]
    deleteat!(distances, .!keep)

    return nothing

end

function estimate_rigid_body_transformation(x_fix, y_fix, z_fix, nx_fix, ny_fix, nz_fix,
                                            x_mov, y_mov, z_mov)

    A = hcat(-z_mov.*ny_fix + y_mov.*nz_fix,
              z_mov.*nx_fix - x_mov.*nz_fix,
             -y_mov.*nx_fix + x_mov.*ny_fix,
             nx_fix,
             ny_fix,
             nz_fix)

    l = nx_fix.*(x_fix-x_mov) + ny_fix.*(y_fix-y_mov) + nz_fix.*(z_fix-z_mov)

    x = A\l

    residuals = A*x-l

    R = euler_angles_to_linearized_rotation_matrix(x[1], x[2], x[3])

    t = x[4:6]

    H = create_homogeneous_transformation_matrix(R, t)

    return H, residuals

end

function euler_angles_to_linearized_rotation_matrix(α1, α2, α3)

    dR = [  1 -α3  α2
           α3   1 -α1
          -α2  α1   1]

end

function create_homogeneous_transformation_matrix(R, t)

    H = [R          t
         zeros(1,3) 1]

end

function euler_coord_to_homogeneous_coord(XE)

    no_points = size(XE, 1)
    XH = [XE ones(no_points,1)]

end

function homogeneous_coord_to_euler_coord(XH)

    XE = XH[:,1:3]./XH[:,4]

end

function check_convergence_criteria(distances_new, distances_old, min_change)

    change(new, old) = abs((new-old)/old*100)

    change_of_mean = change(mean(distances_new), mean(distances_old))
    change_of_std = change(std(distances_new), std(distances_old))

    return change_of_mean < min_change && change_of_std < min_change ? true : false

end

function simpleicp(X_fix::Array, X_mov::Array;
                   correspondences::Integer=1000,
                   neighbors::Integer=10,
                   min_planarity::Number=0.3,
                   min_change::Number=3,
                   max_iterations::Integer=100)

    size(X_fix)[2] == 3 || error(""""X_fix" must have 3 columns""")
    size(X_mov)[2] == 3 || error(""""X_mov" must have 3 columns""")
    correspondences >= 10 || error(""""correspondences" must be >= 10""")
    min_planarity >= 0 && min_planarity < 1 || error(""""min_planarity" must be >= 0 and < 1""")
    neighbors >= 2 || error(""""neighbors" must be >= 2""")
    min_change > 0 || error(""""min_change" must be > 0""")
    max_iterations > 0 || error(""""max_iterations" must be > 0""")

    dt = @elapsed begin
        @info "Create point cloud objects ..."
        pcfix = PointCloud(X_fix[:,1], X_fix[:,2], X_fix[:,3])
        pcmov = PointCloud(X_mov[:,1], X_mov[:,2], X_mov[:,3])

        @info "Select points for correspondences in fixed point cloud ..."
        select_n_points!(pcfix, correspondences)
        sel_orig = pcfix.sel

        @info "Estimate normals of selected points ..."
        estimate_normals!(pcfix, neighbors)

        H = Matrix{Float64}(I,4,4)
        residual_distances = Any[]

        @info "Start iterations ..."
        for i in 1:max_iterations

            initial_distances = matching!(pcmov, pcfix)

            reject!(pcmov, pcfix, min_planarity, initial_distances)

            dH, residuals = estimate_rigid_body_transformation(
                pcfix.x[pcfix.sel], pcfix.y[pcfix.sel], pcfix.z[pcfix.sel],
                pcfix.nx[pcfix.sel], pcfix.ny[pcfix.sel], pcfix.nz[pcfix.sel],
                pcmov.x[pcmov.sel], pcmov.y[pcmov.sel], pcmov.z[pcmov.sel])

            push!(residual_distances, residuals)

            transform!(pcmov, dH)

            H = dH*H
            pcfix.sel = sel_orig

            if i > 1
                if check_convergence_criteria(residual_distances[i], residual_distances[i-1],
                                              min_change)
                    @info "Convergence criteria fulfilled -> stop iteration!"
                    break
                end
            end

            if i == 1
                @info @sprintf(" %9s | %15s | %15s | %15s", "Iteration", "correspondences",
                               "mean(residuals)", "std(residuals)")
                @info @sprintf(" %9d | %15d | %15.4f | %15.4f", 0, length(initial_distances),
                               mean(initial_distances), std(initial_distances))
            end
            @info @sprintf(" %9d | %15d | %15.4f | %15.4f", i, length(residual_distances[i]),
                           mean(residual_distances[i]), std(residual_distances[i]))

        end

    end
    @info "Estimated transformation matrix H:\n" *
    @sprintf("[%12.6f %12.6f %12.6f %12.6f]\n", H[1,1], H[1,2], H[1,3], H[1,4]) *
    @sprintf("[%12.6f %12.6f %12.6f %12.6f]\n", H[2,1], H[2,2], H[2,3], H[2,4]) *
    @sprintf("[%12.6f %12.6f %12.6f %12.6f]\n", H[3,1], H[3,2], H[3,3], H[3,4]) *
    @sprintf("[%12.6f %12.6f %12.6f %12.6f]\n", H[4,1], H[4,2], H[4,3], H[4,4])
    @info "Finished in " * @sprintf("%.3f", dt) * " seconds!"

    return H

end
