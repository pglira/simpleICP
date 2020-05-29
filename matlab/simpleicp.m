function H = simpleicp(XFix, XMov, nva)

    arguments
        XFix(:,3) {mustBeReal}
        XMov(:,3) {mustBeReal}
        nva.correspondences(1,1) {mustBeInteger, ...
            mustBeGreaterThanOrEqual(nva.correspondences, 10)} = 1000;
        nva.neighbors(1,1) {mustBeInteger, ...
            mustBeGreaterThanOrEqual(nva.neighbors, 2)} = 10;
        nva.minPlanarity(1,1) {mustBeGreaterThanOrEqual(nva.minPlanarity, 0), mustBeReal, ...
            mustBeLessThan(nva.minPlanarity, 1)} = 0.3;
        nva.minChange(1,1) {mustBePositive, mustBeReal} = 3;
        nva.maxIterations(1,1) {mustBePositive, mustBeInteger} = 100;
    end

    tic;
    log('Create point cloud objects ...');
    pcFix = pointcloud(XFix(:,1), XFix(:,2), XFix(:,3));
    pcMov = pointcloud(XMov(:,1), XMov(:,2), XMov(:,3));

    log('Select points for correspondences in fixed point cloud ...');
    pcFix.selectNPoints(nva.correspondences);
    selOrig = pcFix.sel;

    log('Estimate normals of selected points ...');
    pcFix.estimateNormals(nva.neighbors);

    H = eye(4);

    log('Start iterations ...');
    for i = 1:nva.maxIterations

        initialDistances = matching(pcFix, pcMov);

        initialDistances = reject(pcFix, pcMov, nva.minPlanarity, initialDistances);

        [dH, residualDistances{i}] = estimateRigidBodyTransformation(...
            pcFix.x(pcFix.sel), pcFix.y(pcFix.sel), pcFix.z(pcFix.sel), ...
            pcFix.nx(pcFix.sel), pcFix.ny(pcFix.sel), pcFix.nz(pcFix.sel), ...
            pcMov.x(pcMov.sel), pcMov.y(pcMov.sel), pcMov.z(pcMov.sel));

        pcMov.transform(dH);

        H = dH*H;
        pcFix.sel = selOrig;

        if i > 1
            if checkConvergenceCriteria(residualDistances{i}, residualDistances{i-1}, nva.minChange)
                log('Convergence criteria fulfilled -> stop iteration!');
                break;
            end
        end

        if i == 1
            log(sprintf('%9s | %15s | %15s | %15s', 'Iteration', 'correspondences', ...
                'mean(residuals)', 'std(residuals)'));
            log(sprintf('%9d | %15d | %15.4f | %15.4f', 0, numel(initialDistances), ...
                mean(initialDistances), std(initialDistances)));
        end
        log(sprintf('%9d | %15d | %15.4f | %15.4f', i, ...
            numel(residualDistances{i}), mean(residualDistances{i}), std(residualDistances{i})));

    end

    log('Estimated transformation matrix H:');
    log(sprintf('[%12.6f %12.6f %12.6f %12.6f]', H(1,1:4)));
    log(sprintf('[%12.6f %12.6f %12.6f %12.6f]', H(2,1:4)));
    log(sprintf('[%12.6f %12.6f %12.6f %12.6f]', H(3,1:4)));
    log(sprintf('[%12.6f %12.6f %12.6f %12.6f]', H(4,1:4)));
    log(sprintf('Finished in %.3f seconds!', toc));

end


function log(text)

    logtime = datestr(now, 'HH:MM:SS.FFF');
    fprintf('[%s] %s\n', logtime, text)

end

function distances = matching(pcFix, pcMov)

    pcMov.sel = knnsearch(...
        [pcMov.x pcMov.y pcMov.z], ...
        [pcFix.x(pcFix.sel) pcFix.y(pcFix.sel) pcFix.z(pcFix.sel)]);

    distances = dot(...
        [pcMov.x(pcMov.sel) pcMov.y(pcMov.sel) pcMov.z(pcMov.sel)] - ...
        [pcFix.x(pcFix.sel) pcFix.y(pcFix.sel) pcFix.z(pcFix.sel)], ...
        [pcFix.nx(pcFix.sel) pcFix.ny(pcFix.sel) pcFix.nz(pcFix.sel)], 2);

end

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

end

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

end

function dR = eulerAnglesToLinearizedRotationMatrix(alpha1, alpha2, alpha3)

    dR = [      1 -alpha3  alpha2
           alpha3       1 -alpha1
          -alpha2  alpha1       1];

end

function H = createHomogeneousTransformationMatrix(R, t)

    H = [R          t
         zeros(1,3) 1];

end

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

    end

end
