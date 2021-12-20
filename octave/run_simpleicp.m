clc; clear; close;

pkg load statistics

dataset = "Dragon";
exportResults = true;

if strcmp(dataset, "Dragon") || strcmp(dataset, "all")
    disp('Processing dataset "Dragon"')
    XFix = dlmread("../data/dragon1.xyz");
    XMov = dlmread("../data/dragon2.xyz");
    [H, XMovT] = simpleicp(XFix, XMov);
end

if strcmp(dataset, "Airborne Lidar") || strcmp(dataset, "all")
    disp('Processing dataset "Airborne Lidar"')
    XFix = dlmread("../data/airborne_lidar1.xyz");
    XMov = dlmread("../data/airborne_lidar2.xyz");
    [H, XMovT] = simpleicp(XFix, XMov);
end

if strcmp(dataset, "Terrestrial Lidar") || strcmp(dataset, "all")
    disp('Processing dataset "Terrestrial Lidar"')
    XFix = dlmread("../data/terrestrial_lidar1.xyz");
    XMov = dlmread("../data/terrestrial_lidar2.xyz");
    [H, XMovT] = simpleicp(XFix, XMov);
end

if strcmp(dataset, "Bunny") || strcmp(dataset, "all")
    disp('Processing dataset "Bunny"')
    XFix = dlmread("../data/bunny_part1.xyz");
    XMov = dlmread("../data/bunny_part2.xyz");
    [H, XMovT] = simpleicp(XFix, XMov, 'maxOverlapDistance', 0.01);
end

if exportResults
    targetDir = "check";
    if ~exist(targetDir, "dir")
        mkdir(targetDir);
    end
    dlmwrite(fullfile(targetDir, "X_fix.xyz"), XFix)
    dlmwrite(fullfile(targetDir, "X_mov.xyz"), XMov)
    dlmwrite(fullfile(targetDir, "X_mov_transformed.xyz"), XMovT)
end