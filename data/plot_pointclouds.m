clc; clear; close;

pcfile1 = 'bunny_part1.xyz';
pcfile2 = 'bunny_part2.xyz';
pngfile = 'bunny.png';

X1 = dlmread(pcfile1);
X2 = dlmread(pcfile2);

figure('Color', 'w');
hAxes = axes;
plot3(X1(:,1), X1(:,2), X1(:,3), '.');
hold on;
plot3(X2(:,1), X2(:,2), X2(:,3), '.');
axis equal;
box on;
boxColor = 0.7*ones(3,1);
hAxes.XColor = boxColor;
hAxes.YColor = boxColor;
hAxes.ZColor = boxColor;

view(2)

exportgraphics(gca, pngfile)

% Resize this graphic to height = 150px to generate small version
