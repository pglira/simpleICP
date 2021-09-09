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

%% Dataset "Lion"
XFix = dlmread("../data/lionscan1.xyz");
XMov = dlmread("../data/lionscan2.xyz");
[H, XMovT] = simpleicp(XFix, XMov, 'maxOverlapDistance', 10);

%% Check
addrepo('Point_cloud_tools_for_Matlab'); % https://github.com/pglira/Point_cloud_tools_for_Matlab
pcFix = pointCloud(XFix, 'Label', 'XFix');
pcMov = pointCloud(XMov, 'Label', 'XMov');
pcMovT = pointCloud(XMovT, 'Label', 'XMovT');
pcFix.plot('Color', 'r');
pcMov.plot('Color', 'b');
pcMovT.plot('Color', 'm');