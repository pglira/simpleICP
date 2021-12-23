using Pkg
pkg"add MultivariateStats"
pkg"add NearestNeighbors"
pkg"add StatsBase"

include("simpleicp.jl")

dataset = "all"
write_results = false

if cmp(dataset, "Dragon") == 0 || cmp(dataset, "all") == 0
    println("Processing dataset \"Dragon\"")
    X_fix = readdlm("../data/dragon1.xyz")
    X_mov = readdlm("../data/dragon2.xyz")
    H, X_mov_transformed = simpleicp(X_fix, X_mov)
end

if cmp(dataset, "Airborne Lidar") == 0 || cmp(dataset, "all") == 0
    println("Processing dataset \"Airborne Lidar\"")
    X_fix = readdlm("../data/airborne_lidar1.xyz")
    X_mov = readdlm("../data/airborne_lidar2.xyz")
    H, X_mov_transformed = simpleicp(X_fix, X_mov)
end

if cmp(dataset, "Terrestrial Lidar") == 0 || cmp(dataset, "all") == 0
    println("Processing dataset \"Terrestrial Lidar\"")
    X_fix = readdlm("../data/terrestrial_lidar1.xyz")
    X_mov = readdlm("../data/terrestrial_lidar2.xyz")
    H, X_mov_transformed = simpleicp(X_fix, X_mov)
end

if cmp(dataset, "Bunny") == 0 || cmp(dataset, "all") == 0
    println("Processing dataset \"Bunny\"")
    X_fix = readdlm("../data/bunny_part1.xyz")
    X_mov = readdlm("../data/bunny_part2.xyz")
    H, X_mov_transformed = simpleicp(X_fix, X_mov, max_overlap_distance=1)
end

# Export original and adjusted point clouds to xyz files to check the result
if write_results
    target_dir = "check"
    mkpath(target_dir)
    writedlm(joinpath(target_dir, "X_fix.xyz"), X_fix)
    writedlm(joinpath(target_dir, "X_mov.xyz"), X_mov)
    writedlm(joinpath(target_dir, "X_mov_transformed.xyz"), X_mov_transformed)
end