function [H, XmovT] = simpleicp(XFix, XMov, varargin)

  p = parseParameters(XFix, XMov, varargin{:});

  tic;
  fprintf('Create point cloud objects ...\n');
  pcFix = pointcloud(XFix(:,1), XFix(:,2), XFix(:,3));
  pcMov = pointcloud(XMov(:,1), XMov(:,2), XMov(:,3));

  if ~isinf(p.maxOverlapDistance)
    fprintf('Consider partial overlap of point clouds ...\n');
    pcFix.selectInRange([pcMov.x pcMov.y pcMov.z], p.maxOverlapDistance);
    if numel(pcFix.sel) == 0
      error(['Point clouds do not overlap within maxOverlapDistance = %.5f! ' ...
             'Consider increasing the value of maxOverlapDistance.'], ...
             p.maxOverlapDistance)
    end
  end

  fprintf('Select points for correspondences in fixed point cloud ...\n');
  pcFix.selectNPoints(p.correspondences);
  selOrig = pcFix.sel;

  fprintf('Estimate normals of selected points ...\n');
  pcFix.estimateNormals(p.neighbors);

  H = eye(4);

  fprintf('Start iterations ...\n');
  for i = 1:p.maxIterations

    initialDistances = matching(pcFix, pcMov);

    initialDistances = reject(pcFix, pcMov, p.minPlanarity, initialDistances);

    [dH, residualDistances{i}] = estimateRigidBodyTransformation(...
      pcFix.x(pcFix.sel), pcFix.y(pcFix.sel), pcFix.z(pcFix.sel), ...
      pcFix.nx(pcFix.sel), pcFix.ny(pcFix.sel), pcFix.nz(pcFix.sel), ...
      pcMov.x(pcMov.sel), pcMov.y(pcMov.sel), pcMov.z(pcMov.sel));

    pcMov.transform(dH);

    H = dH*H;
    pcFix.sel = selOrig;

    if i > 1
      if checkConvergenceCriteria(residualDistances{i}, residualDistances{i-1}, p.minChange)
        fprintf('Convergence criteria fulfilled -> stop iteration!\n');
        break;
      end
    end

    if i == 1
      fprintf('%9s | %15s | %15s | %15s\n', 'Iteration', 'correspondences', ...
          'mean(residuals)', 'std(residuals)');
      fprintf('%9s | %15d | %15.4f | %15.4f\n', 'orig:0', numel(initialDistances), ...
          mean(initialDistances), std(initialDistances));
    end
    fprintf('%9d | %15d | %15.4f | %15.4f\n', i, numel(residualDistances{i}), ...
        mean(residualDistances{i}), std(residualDistances{i}));

  end

  XmovT = [pcMov.x pcMov.y pcMov.z];

  fprintf('Estimated transformation matrix H:\n');
  fprintf('[%12.6f %12.6f %12.6f %12.6f]\n', H(1,1:4));
  fprintf('[%12.6f %12.6f %12.6f %12.6f]\n', H(2,1:4));
  fprintf('[%12.6f %12.6f %12.6f %12.6f]\n', H(3,1:4));
  fprintf('[%12.6f %12.6f %12.6f %12.6f]\n', H(4,1:4));
  fprintf('Finished in %.3f seconds!\n', toc);

endfunction

function p = parseParameters(XFix, XMov, varargin)

  p = inputParser();
  p.FunctionName = "simpleicp";
  val_X = @(x) size(x,2)==3 && isreal(x);
  p.addRequired("XFix", val_X);
  p.addRequired("XMov", val_X);
  val_correspondences = @(x) isinteger (x) && x>=10;
  p.addParamValue("correspondences", 1000, val_correspondences);
  val_neighbors = @(x) isinteger (x) && x>=2;
  p.addParamValue("neighbors", 10, val_neighbors);
  val_minPlanarity = @(x) isnumeric (x) && x>=0 && x<1;
  p.addParamValue("minPlanarity", 0.3, val_minPlanarity);
  val_maxOverlapDistance = @(x) isnumeric (x) && x>=0;
  p.addParamValue("maxOverlapDistance", Inf, val_maxOverlapDistance);
  val_minChange = @(x) isnumeric (x) && x>0;
  p.addParamValue("minChange", 1, val_minChange);
  val_maxIterations = @(x) isinteger (x) && x>0;
  p.addParamValue("maxIterations", 100, val_maxIterations);
  p.parse(XFix, XMov, varargin{:})
  p = p.Results;

endfunction

function pc = importPointCloud(pathToPC)

  D = dlmread(pathToPC);
  pc = pointcloud(D(:,1), D(:,2), D(:,3));

endfunction

function distances = matching(pcFix, pcMov)

  pcMov.sel = knnsearch(...
    [pcMov.x pcMov.y pcMov.z], ...
    [pcFix.x(pcFix.sel) pcFix.y(pcFix.sel) pcFix.z(pcFix.sel)], 1);

  distances = dot(...
    [pcMov.x(pcMov.sel) pcMov.y(pcMov.sel) pcMov.z(pcMov.sel)] - ...
    [pcFix.x(pcFix.sel) pcFix.y(pcFix.sel) pcFix.z(pcFix.sel)], ...
    [pcFix.nx(pcFix.sel) pcFix.ny(pcFix.sel) pcFix.nz(pcFix.sel)], 2);

endfunction

function distances = reject(pcFix, pcMov, minPlanarity, distances)

  planarity = pcFix.planarity(pcFix.sel);

  med = median(distances);
  sigmad = 1.4826 * mad(distances,1);

  idxReject = ...
    distances < (-med-3*sigmad) | ...
    distances > (med+3*sigmad) | ...
    planarity < minPlanarity;

  pcFix.sel(idxReject) = [];
  pcMov.sel(idxReject) = [];
  distances(idxReject) = [];

endfunction

function [H, residuals] = estimateRigidBodyTransformation(...
  xFix, yFix, zFix, ...
  nxFix, nyFix, nzFix, ...
  xMov, yMov, zMov)

  A = [-zMov.*nyFix + yMov.*nzFix ...
        zMov.*nxFix - xMov.*nzFix ...
       -yMov.*nxFix + xMov.*nyFix ...
       nxFix ...
       nyFix ...
       nzFix];

  l = nxFix.*(xFix-xMov) + ...
      nyFix.*(yFix-yMov) + ...
      nzFix.*(zFix-zMov);

  x = A\l;

  residuals = A*x-l;

  R = eulerAnglesToLinearizedRotationMatrix(x(1), x(2), x(3));

  t = x(4:6);

  H = createHomogeneousTransformationMatrix(R, t);

endfunction

function dR = eulerAnglesToLinearizedRotationMatrix(alpha1, alpha2, alpha3)

  dR = [      1 -alpha3  alpha2
         alpha3       1 -alpha1
        -alpha2  alpha1       1];

endfunction

function H = createHomogeneousTransformationMatrix(R, t)

  H = [R          t
       zeros(1,3) 1];

endfunction

function stop = checkConvergenceCriteria(distancesNew, distancesOld, minChange)

  changeOfMean = change(mean(distancesNew), mean(distancesOld));
  changeOfStd = change(std(distancesNew), std(distancesOld));

  if changeOfMean < minChange && changeOfStd < minChange
    stop = true;
  else
    stop = false;
  end

  function p = change(new, old)

    p = abs((new-old)/old*100);

  endfunction

endfunction
