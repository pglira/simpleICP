clc; clear; close;

pathToPCFix = "../data/dragon1.xyz";
pathToPCMov = "../data/dragon2.xyz";

% pathToPCFix = "../data/airborne_lidar1.xyz";
% pathToPCMov = "../data/airborne_lidar2.xyz";

% pathToPCFix = "../data/terrestrial_lidar1.xyz";
% pathToPCMov = "../data/terrestrial_lidar2.xyz";

XFix = dlmread(pathToPCFix);
XMov = dlmread(pathToPCMov);

H = simpleicp(XFix, XMov);
