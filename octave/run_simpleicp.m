clc; clear; close;

pkg load statistics

pathToPCFix = "../data/dragon1.xyz";
pathToPCMov = "../data/dragon2.xyz";

XFix = dlmread(pathToPCFix);
XMov = dlmread(pathToPCMov);

H = simpleicp(XFix, XMov);
