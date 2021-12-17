clc; clear; close;

%% Dataset "Dragon"
XFix = dlmread("../data/dragon1.xyz");
XMov = dlmread("../data/dragon2.xyz");
[H, XMovT] = simpleicp(XFix, XMov);

%% Dataset "Airborne Lidar"
XFix = dlmread("../data/airborne_lidar1.xyz");
XMov = dlmread("../data/airborne_lidar2.xyz");
[H, XMovT] = simpleicp(XFix, XMov);

%% Dataset "Terrestrial Lidar"
XFix = dlmread("../data/terrestrial_lidar1.xyz");
XMov = dlmread("../data/terrestrial_lidar2.xyz");
[H, XMovT] = simpleicp(XFix, XMov);

%% Dataset "Bunny"
XFix = dlmread("../data/bunny_part1.xyz");
XMov = dlmread("../data/bunny_part2.xyz");
[H, XMovT] = simpleicp(XFix, XMov, 'maxOverlapDistance', 0.005);

%% Dataset "Multisensor"
XFix = dlmread("../data/multisensor_lidar.xyz");
XMov = dlmread("../data/multisensor_radar.xyz");
[H, XMovT] = simpleicp(XFix, XMov, 'maxOverlapDistance', 2);

%% Check
addrepo('Point_cloud_tools_for_Matlab'); % https://github.com/pglira/Point_cloud_tools_for_Matlab
pcFix = pointCloud(XFix, 'Label', 'XFix');
pcMov = pointCloud(XMov, 'Label', 'XMov');
pcMovT = pointCloud(XMovT, 'Label', 'XMovT');
pcFix.plot('Color', 'r', 'MarkerSize', 5);
pcMov.plot('Color', 'g', 'MarkerSize', 5);
pcMovT.plot('Color', 'b', 'MarkerSize', 5);