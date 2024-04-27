%helperTrackLastKeyFrameCg Estimate the camera pose by tracking the last key frame
%   [currPose, mapPointIdx, featureIdx] = helperTrackLastKeyFrameStereo(mapPoints,
%   views, currFeatures, currPoints, lastKeyFrameId, intrinsics, scaleFactor) estimates
%   the camera pose of the current frame by matching features with the
%   previous key frame.
%
%   This is an example helper function that is subject to change or removal
%   in future releases.
%
%   Inputs
%   ------
%   mapPoints         - A helperMapPoints objects storing map points
%   views             - View attributes of key frames
%   currFeatures      - Features in the current frame
%   currPoints        - Feature points in the current frame
%   lastKeyFrameId    - ViewId of the last key frame
%   intrinsics        - Camera intrinsics
%   scaleFactor       - scale factor of features
%
%   Outputs
%   -------
%   currPose          - Estimated camera pose of the current frame
%   mapPointIdx       - Indices of map points observed in the current frame
%   featureIdx        - Indices of features corresponding to mapPointIdx

%   Copyright 2023 The MathWorks, Inc.
%#codegen

function [currPose, mapPointIdx, featureIdx] = helperTrackLastKeyFrameCg(...
    mapPoints, views, currFeatures, currPoints, lastKeyFrameId, intrinsics, scaleFactor)
% Match features from the previous key frame with known world locations
[index3d, index2d] = findWorldPointsInView(mapPoints, lastKeyFrameId);
index2dVec = index2d{1};
lastKeyFrameFeatures  = views.Features{lastKeyFrameId}(index2dVec,:);
lastKeyFramePoints = views.Points{lastKeyFrameId};
lastKeyFramePointsScale = lastKeyFramePoints.Scale(index2dVec, :);

lastKeyFrameBinaryFeat = binaryFeatures(lastKeyFrameFeatures);
indexPairs  = matchFeatures(currFeatures, lastKeyFrameBinaryFeat,...
    'Unique', true, 'MaxRatio', 0.9, 'MatchThreshold', 40);
% Estimate the camera pose
matchedImagePoints = currPoints.Location(indexPairs(:,1),:);
index3dVal = index3d{1};
matchedWorldPoints = mapPoints.WorldPoints(index3dVal(indexPairs(:,2)), :);

matchedImagePoints = cast(matchedImagePoints, 'like', matchedWorldPoints);
[currPose, inlier, status] = estworldpose(...
    matchedImagePoints, matchedWorldPoints, intrinsics, ...
    'Confidence', 95, 'MaxReprojectionError', 3, 'MaxNumTrials', 1e4);
if status
    pose = rigidtform3d;
    currPose=repmat(pose, 0, 0);
    mapPointIdx = zeros(0, 1);
    featureIdx = zeros(0, 1, class(indexPairs));
    return
end

% Refine camera pose only
currPose = bundleAdjustmentMotion(matchedWorldPoints(inlier,:), ...
    matchedImagePoints(inlier,:), currPose, intrinsics, ...
    'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7,...
    'RelativeTolerance', 1e-15, 'MaxIterations', 20);
% Search for more matches with the map points in the previous key frame
xyzPoints = mapPoints.WorldPoints(index3dVal,:);
[projectedPoints, isInImage] = world2img(xyzPoints, pose2extr(currPose), intrinsics);
projectedPoints = projectedPoints(isInImage, :);

radius            = 4;
lkeyScale    = lastKeyFramePointsScale(isInImage);
lKeyRemain   = bitsll(lkeyScale, 3);
lKey         = (lKeyRemain*10/12);
lKey         = bitsra(lKey, 3);
minScales    = max(1, lKey);
maxScales    = (lastKeyFramePointsScale(isInImage)+ (1e-1*(lastKeyFramePointsScale(isInImage)*2)));
searchRadius = radius*lastKeyFramePointsScale(isInImage);
indexPairs   = matchFeaturesInRadius(binaryFeatures(lastKeyFrameFeatures(isInImage,:)), ...
    binaryFeatures(currFeatures.Features), currPoints, projectedPoints, searchRadius, ...
    'MatchThreshold', 40, 'MaxRatio', 0.8, 'Unique', true);
if size(indexPairs, 1) < 20
    indexPairs   = matchFeaturesInRadius(binaryFeatures(lastKeyFrameFeatures(isInImage,:)), ...
        binaryFeatures(currFeatures.Features), currPoints, projectedPoints, 2*searchRadius, ...
        'MatchThreshold', 40, 'MaxRatio', 1, 'Unique', true);
end
if size(indexPairs, 1) < 10
    pose        = rigidtform3d;
    currPose    = repmat(pose, 0, 0);
    mapPointIdx = zeros(0, 1);
    featureIdx  = zeros(0, 1, class(indexPairs));
    return
end

indexPairs1     = indexPairs(:, 1);
currPointsScale = currPoints.Scale(indexPairs(:, 2));
mini            = minScales(indexPairs1);
maxi            = maxScales(indexPairs1);
miniScale       = zeros(size(mini), 'logical');
maxiScale       = zeros(size(maxi), 'logical');
parfor i = 1:size(mini, 1)
    if (currPointsScale(i, 1) >= (mini(i,1)))
        miniScale(i) = 1;
    else
        miniScale(i) = 0;
    end

    if (currPointsScale(i,1) <= maxi(i,1))
        maxiScale(i) = 1;
    else
        maxiScale(i) = 0;
    end
end
isGoodScale = zeros(size(mini), 'logical');
% Filter by scales
parfor i=1:size(mini, 1)
    isGoodScale(i) = miniScale(i) & maxiScale(i);
end
indexPairs  = indexPairs(isGoodScale, :);
% Obtain the index of matched map points and features
tempIdx            = find(isInImage); % Convert to linear index
coder.varsize('mapPointIdx', [inf, 1], [1, 0]);
mapPointIdx        = index3dVal(tempIdx(indexPairs(:,1)));
coder.varsize('featureIdx', [inf, 1], [1, 0]);
featureIdx         = indexPairs(:,2);

% Refine the camera pose again
matchedWorldPoints = mapPoints.WorldPoints(mapPointIdx, :);
matchedImagePoints = currPoints.Location(featureIdx, :);

currPose = bundleAdjustmentMotion(matchedWorldPoints, matchedImagePoints, ...
    currPose, intrinsics, 'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7,...
    'RelativeTolerance', 1e-15, 'MaxIterations', 20);
end