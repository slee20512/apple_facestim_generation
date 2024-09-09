% Load STL files
fv1 = stlread('SeojinE_.stl');
fv2 = stlread('Neptune_.stl');

vertices1 = fv1.Points;
vertices2 = fv2.Points;

ptCloud1 = pointCloud(vertices1); % Create point cloud from vertices
ptCloud2 = pointCloud(vertices2); % Create point cloud from vertices


% Compare the 3d faces (base + mesh)

% Method 1: Point-to-surface distance calculation
% Measures the distance from each vertex in one mesh to the entire surface
% of the other mesh. 
[indices, distances] = knnsearch(ptCloud2.Location, ptCloud1.Location);

%% Method 2: Hausdorff distance
% Measures the maximum distance from a point on one surface to the closest
% point on another surface (captures the worst-case scenario of how
% different two surfaces are)
[indices1, dist1] = knnsearch(vertices2, vertices1);
[indices2, dist2] = knnsearch(vertices1, vertices2);
hausdorffDist = max(max(dist1), max(dist2));

%% Method 3: Iterative Closest Point (ICP) alignment
% Align two point clouds to minimize the distance between them. After
% alignment, calculate the residual distances.
[tform, ptCloudAligned, rmse] = pcregistericp(ptCloud2, ptCloud1);

%% Method 4: Surface area or volume comparison
% % Compare the surface area or enclosed volume of two meshes.
% area1 = sum(triarea(vertices1, faces1));  % Custom function to compute triangle area
% area2 = sum(triarea(vertices2, faces2));
% 
% volume1 = tetrameshvolume(vertices1, faces1);  % Custom function for volume
% volume2 = tetrameshvolume(vertices2, faces2);
% 
% %% Method 5: Spectral shape analysis
% % Analyze and compare the intrinsic geometry of two shapes by computing the
% % eigenvalues of the Laplace-Beltrami operator of each mesh. 
% % Compute the Laplace-Beltrami eigenvalues for each mesh
% [V1, D1] = eigs(laplacian(vertices1, faces1), 10);
% [V2, D2] = eigs(laplacian(vertices2, faces2), 10);
% 
% % Compare eigenvalues (D1 and D2)
% spectral_diff = norm(diag(D1) - diag(D2));
