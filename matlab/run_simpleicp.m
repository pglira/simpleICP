clc; clear; close;

pathToPCFix = "../data/dragon1.xyz";
pathToPCMov = "../data/dragon2.xyz";

XFix = dlmread(pathToPCFix);
XMov = dlmread(pathToPCMov);

H = simpleicp(XFix, XMov);

% pcFix = pointcloud(XFix(:,1), XFix(:,2), XFix(:,3));
% pcMov = pointcloud(XMov(:,1), XMov(:,2), XMov(:,3));
% pcMov.transform(H);

% pcview([pcFix.x pcFix.y pcFix.z]);
% pcview([pcMov.x pcMov.y pcMov.z]);


