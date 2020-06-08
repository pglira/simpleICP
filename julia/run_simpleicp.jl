include("simpleicp.jl")

# path_pc_fix = "../data/dragon1.xyz"
# path_pc_mov = "../data/dragon2.xyz"

# path_pc_fix = "../data/airborne_lidar1.xyz"
# path_pc_mov = "../data/airborne_lidar2.xyz"

path_pc_fix = "../data/terrestrial_lidar1.xyz"
path_pc_mov = "../data/terrestrial_lidar2.xyz"

X_fix = readdlm(path_pc_fix)
X_mov = readdlm(path_pc_mov)

H = simpleicp(X_fix, X_mov)
