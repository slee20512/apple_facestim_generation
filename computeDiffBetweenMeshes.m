%% Method 1. Scale with max radiii
keys = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

scaledCoordsStruct = struct();
scaledRadiiStruct = struct();
partB = [773, 571, 560, 559 ,523, 665, 569, 568, 520, 676, 740, 19, 367, 303, 148, 196, 197, 292, 151, 187, 188, 198, 199, 326, 166, 570, 699, 538, 713, 771, 772, 777, 775, 774 ,765 ,537, 733, 426, 477 ,556, 671 618 602 ,340, 398 ,399, 400, 404 ,402, 401 ,392 ,165 ,360, 54, 105, 184, 298, 245, 229, 220, 593 ,33]; 
max_radii = zeros(length(keys), 1);
centroids = zeros(length(keys), 3);

for i = 1:length(keys)
    key = keys{i}; % Current individual ID
  
    xyz_coords = eval(['partV_orig_before_scale_' key]); % all coords (boundary + inside)
    xyz_coordsB = xyz_coords(partB, :); % boundary coords only
    centroids(i, :) = mean(xyz_coords, 1); % centroid
    xy_centered_boundary = xyz_coordsB(:, 1:2) - centroids(i, 1:2); % distances between xy centroid and boundary xy coords
    max_radii(i) = max(sqrt(sum(xy_centered_boundary.^2, 2))); % max radii

end

% Determine the target maximum radius (maximum among all ids)
target_max_radius = max(max_radii);

% Scale each individual so their radii match the target max radius
for i = 1:length(keys)
    key = keys{i}; % Current individual ID
    xyz_coords = eval(['partV_orig_before_scale_' key]); % original coords
    centroid = centroids(i, :); % original centroids
    xyz_centered = xyz_coords - centroid; % (1) center all xyz coords
    scaling_factor = target_max_radius / max_radii(i); % (2) find the scaling factor for this face (to match the max radii)
    xyz_scaled = xyz_centered * scaling_factor; %  (3) scale the centered coords
    scaled_radii = sqrt(sum(xyz_scaled(partB, :).^2, 2));
    scaledCoordsStruct.(key) = xyz_scaled;
    scaledRadiiStruct.(key) = scaled_radii;
    
end

disp('Scaled Radii for each individual:');
disp(scaledRadiiStruct);

disp('Scaled Coordinates for each individual:');
disp(scaledCoordsStruct);
%% % Initialize the figure window for plotting scaled coordinates
figure;
colors = {'r','b','g','c','m','black','y'};

individualsToPlot = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

individualsToPlot = {'Elias', 'SeojinE'};
% Loop through the selected individuals and plot their scaled data
for i = 1:length(individualsToPlot)
    individual = individualsToPlot{i};
    xyz_scaled = scaledCoordsStructAvg.(individual); % Get the scaled coordinates
    
    % Plot the scaled coordinates
    scatter3(xyz_scaled(:,1), xyz_scaled(:,2), xyz_scaled(:,3), 14, colors{i}, 'filled');
    hold on;
   
    % Plot the centroid with an 'x'
    centroid = mean(xyz_scaled); % Get the centroid for this individual
    plot3(centroid(1), centroid(2), centroid(3), 'x', 'MarkerSize', 14, 'MarkerEdgeColor', 'black', 'LineWidth', 7);
end

% Set labels
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([-0.08, +0.08]);
ylim([-0.08, +0.08]);

% Set the title
title('Scaled Meshes (Scaled on avg radius)');

% Add a legend to distinguish the plotted sets of points
legend(individualsToPlot, 'Location', 'best');

% Hold off to stop adding to this plot
hold off;
%% Method 2. Scale with horizontal distance

keys = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

scaledCoordsStructHz = struct();
scaledRadiiStructHz = struct();
scalingFactorHz = struct();

hz_dist = zeros(length(keys), 1);
centroids = zeros(length(keys), 3);

for i = 1:length(keys)
    key = keys{i}; 
    xyz_coords = eval(['partV_orig_before_scale_' key]); % original coords
    centroids(i, :) = mean(xyz_coords, 1); % original centroids
    xyz_centered = xyz_coords - centroids(i, :);
    
    min_x = min(xyz_centered(:, 1)); % leftmost point
    max_x = max(xyz_centered(:, 1)); % rightmost point
    hz_dist(i) = max_x - min_x; % horizontal distance
    
    scaledCoordsStructHz.(key).xyz_centered = xyz_centered;
end

% Determine the target maximum horizontal distance
max_horizontal_distance = max(hz_dist);

% Scale each individual so their horizontal distances match the target max distance
for i = 1:length(keys)
    key = keys{i}; 
    xyz_centered = scaledCoordsStructHz.(key).xyz_centered; % retrieve centered coordinates
    scaling_factor = max_horizontal_distance / hz_dist(i); % find the scaling factor
    xyz_scaled = xyz_centered * scaling_factor; % apply scaling 
    scaled_radii = sqrt(sum(xyz_scaled(partB, :).^2, 2)); % scaled radius
    
    scaledCoordsStructHz.(key).xyz_scaled = xyz_scaled;
    scaledRadiiStructHz.(key) = scaled_radii;
    scalingFactorHz.(key) = scaling_factor;
end

% Display the scaled radii for each individual
disp('Scaled Radii for each individual:');
disp(scaledRadiiStructHz);

% Display the scaled coordinates for each individual
disp('Scaled Coordinates for each individual:');
disp(scaledCoordsStructHz);
%% %% % Initialize the figure window for plotting scaled coordinates
figure;
individualsToPlot = {'Neptune', 'Elias'};

% Loop through the selected individuals and plot their scaled data
for i = 1:length(individualsToPlot)
    individual = individualsToPlot{i};
    xyz_scaled = scaledCoordsStructAvg.(individual); % Get the scaled coordinates
    
    % Plot the scaled coordinate
    scatter3(xyz_scaled(:,1), xyz_scaled(:,2), xyz_scaled(:,3), 20, colors{i}, 'filled');
    hold on;
end

% Set labels
xlabel('X');
ylabel('Y');
zlabel('Z');

% Set the title
title('Scaled Meshes (Scaled on face width)');

% Add a legend to distinguish the plotted sets of points
legend(individualsToPlot, 'Location', 'best');

% Hold off to stop adding to this plot
hold off;

%% Average radii -- XYZ
keys = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

scaledCoordsMax_1 = struct();
maxDistToCentroid = zeros(length(keys), 1);
centroids = zeros(length(keys), 3);
partB = [773, 571, 560, 559 ,523, 665, 569, 568, 520, 676, 740, 19, 367, 303, 148, 196, 197, 292, 151, 187, 188, 198, 199, 326, 166, 570, 699, 538, 713, 771, 772, 777, 775, 774 ,765 ,537, 733, 426, 477 ,556, 671 618 602 ,340, 398 ,399, 400, 404 ,402, 401 ,392 ,165 ,360, 54, 105, 184, 298, 245, 229, 220, 593 ,33]; 

for i = 1:length(keys)
    key = keys{i}; % Current individual ID
    xyz_coords = eval(['partV_orig_before_scale_' key]); % all coords (boundary + inside)
    xyz_coordsB = xyz_coords(partB, :); % boundary coords only
    centroid = mean(xyz_coords, 1);
    centroids(i, :) = centroid; % centroid
    xy_centered_boundary = xyz_coordsB - centroid; % distances between xy centroid and boundary xy coords
    maxDistToCentroid(i) = max(sqrt(sum(xy_centered_boundary.^2, 2))); % average distance to centroid
end

% Determine the target average distance to centroid (mean among all ids)
target_max_dist = max(maxDistToCentroid);
target_max_dist = 1;

% Scale each individual so their avg distance to centroid matches the target
for i = 1:length(keys)
    key = keys{i}; % Current individual ID
    xyz_coords = eval(['partV_orig_before_scale_' key]); % original coords
    centroid = centroids(i, :); % original centroids
    xyz_centered = xyz_coords - centroid; % (1) center all xyz coords
    scaling_factor = target_max_dist /  maxDistToCentroid(i); % (2) find the scaling factor for this face
    xyz_scaled = xyz_centered * scaling_factor; %  (3) scale the centered coords
    scaledCoordsMax_1.(key) = xyz_scaled;
end


%% Average radii -- XY
keys = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};


scaledCoordsMax_xy_1 = struct();
maxDistToCentroid_xy = zeros(length(keys), 1);
centroids = zeros(length(keys), 3);
partB = [773, 571, 560, 559 ,523, 665, 569, 568, 520, 676, 740, 19, 367, 303, 148, 196, 197, 292, 151, 187, 188, 198, 199, 326, 166, 570, 699, 538, 713, 771, 772, 777, 775, 774 ,765 ,537, 733, 426, 477 ,556, 671 618 602 ,340, 398 ,399, 400, 404 ,402, 401 ,392 ,165 ,360, 54, 105, 184, 298, 245, 229, 220, 593 ,33]; 

for i = 1:length(keys)
    key = keys{i}; % Current individual ID
    xyz_coords = eval(['partV_orig_before_scale_' key]); % all coords (boundary + inside)
    xyz_coordsB = xyz_coords(partB, :); % boundary coords only
    centroid = mean(xyz_coords, 1);
    centroids(i, :) = centroid; % centroid
    xy_centered_boundary = xyz_coordsB(:, 1:2) - centroid(1:2); % distances between xy centroid and boundary xy coords
    maxDistToCentroid_xy(i) = max(sqrt(sum(xy_centered_boundary.^2, 2))); % average distance to centroid
end

% Determine the target average distance to centroid (mean among all ids)
target_max_dist_xy = max(maxDistToCentroid_xy);
target_max_dist_xy = 1;
% Scale each individual so their avg distance to centroid matches the target
for i = 1:length(keys)
    key = keys{i}; % Current individual ID
    xyz_coords = eval(['partV_orig_before_scale_' key]); % original coords
    centroid = centroids(i, :); % original centroids
    xyz_centered = xyz_coords - centroid; % (1) center all xyz coords
    scaling_factor = target_max_dist_xy / maxDistToCentroid_xy(i); % (2) find the scaling factor for this face
    xyz_scaled = xyz_centered * scaling_factor; %  (3) scale the centered coords
    scaledCoordsMax_xy_1.(key) = xyz_scaled;
end
%% Average radii -- YZ
keys = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

scaledCoordsMax_yz = struct();
maxDistToCentroid_yz = zeros(length(keys), 1);
centroids = zeros(length(keys), 3);
partB = [773, 571, 560, 559 ,523, 665, 569, 568, 520, 676, 740, 19, 367, 303, 148, 196, 197, 292, 151, 187, 188, 198, 199, 326, 166, 570, 699, 538, 713, 771, 772, 777, 775, 774 ,765 ,537, 733, 426, 477 ,556, 671 618 602 ,340, 398 ,399, 400, 404 ,402, 401 ,392 ,165 ,360, 54, 105, 184, 298, 245, 229, 220, 593 ,33]; 

for i = 1:length(keys)
    key = keys{i}; % Current individual ID
    xyz_coords = eval(['partV_orig_before_scale_' key]); % all coords (boundary + inside)
    xyz_coordsB = xyz_coords(partB, :); % boundary coords only
    centroid = mean(xyz_coords, 1);
    centroids(i, :) = centroid; % centroid
    xy_centered_boundary = xyz_coordsB(:, 2:3) - centroid(2:3); % distances between xy centroid and boundary xy coords
    maxDistToCentroid_yz(i) = max(sqrt(sum(xy_centered_boundary.^2, 2))); % average distance to centroid

end

% Determine the target average distance to centroid (mean among all ids)
target_max_dist_yz = max(maxDistToCentroid_yz);
% target_max_dist_yz = 1;
% Scale each individual so their avg distance to centroid matches the target
for i = 1:length(keys)
    key = keys{i}; % Current individual ID
    xyz_coords = eval(['partV_orig_before_scale_' key]); % original coords
    centroid = centroids(i, :); % original centroids
    xyz_centered = xyz_coords - centroid; % (1) center all xyz coords
    scaling_factor = target_max_dist_yz / maxDistToCentroid_yz(i); % (2) find the scaling factor for this face
    xyz_scaled = xyz_centered * scaling_factor; %  (3) scale the centered coords
    scaledCoordsMax_yz.(key) = xyz_scaled;
end
%% % Initialize the figure window for plotting scaled coordinates
figure;
colors = {'r','b','g','c','m','black','y'};

individualsToPlot = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
individualsToPlot = {'Elias', 'Neptune'};
% Loop through the selected individuals and plot their scaled data
for i = 1:length(individualsToPlot)
    individual = individualsToPlot{i};
    xyz_scaled = scaledCoordsMax.(individual); % Get the scaled coordinates
    
    % Plot the scaled coordinates
    scatter3(xyz_scaled(:,1), xyz_scaled(:,2), xyz_scaled(:,3), 14, colors{i}, 'filled');
    hold on;
   
    % Plot the centroid with an 'x'
    centroid = mean(xyz_scaled); % Get the centroid for this individual
    plot3(centroid(1), centroid(2), centroid(3), 'x', 'MarkerSize', 14, 'MarkerEdgeColor', 'black', 'LineWidth', 7);
end

% Set labels
xlabel('X');
ylabel('Y');
zlabel('Z');
% xlim([-0.08, +0.08]);
% ylim([-0.08, +0.08]);

% Set the title
title('Scaled Meshes on the yz radii');

% Add a legend to distinguish the plotted sets of points
legend(individualsToPlot, 'Location', 'best');

% Hold off to stop adding to this plot
hold off;
%% Compute Differences Between Scaled Meshes
displacement = sqrt(sum((scaledCoordsMax_yz.("Sreyas") - scaledCoordsMax_yz.("Younah")).^2, 2)); % Euclidean distance between corresponding vertices, calculated per row (vertex)
% normalizedDisplacements = (displacement - min(displacement)) / (max(displacement) - min(displacement));
% normalizedDisplacements = (displacement - 0.00026281) / (0.02943009 - 0.00026281);

% znormalizedDisplacements = abs((displacement - mean(displacement)) / std(displacement)); % z-score

[sortedDisplacements, originalIndices] = sort(displacement, 'ascend');

total_displacement = sum(displacement);
mean_displacement = mean(displacement);
max_displacement = max(displacement);

fprintf('Total Displacement: %.4f\n', total_displacement);
% fprintf('Mean Displacement: %.4f\n', mean_displacement);
% fprintf('Max Displacement: %.4f\n', max_displacement);
%% % Compute differences for each coordinate axis
% Define individuals and number of individuals
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
num_individuals = length(individuals);
num_pairs = num_individuals * (num_individuals - 1) / 2;

% Set color for all plots (blue)
color_blue = [0.1, 0.4, 0.8]; % Blueish color

% Initialize figure for subplots of 1D differences before scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "before scaling"
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the "before scaling" variables using dynamic field names
        mesh1_before = eval(['partV_orig_before_scale_' name1]);
        mesh2_before = eval(['partV_orig_before_scale_' name2]);

        % Compute 1D differences before scaling
        diff_x_before = sum(abs(mesh1_before(:, 1) - mesh2_before(:, 1)));
        diff_y_before = sum(abs(mesh1_before(:, 2) - mesh2_before(:, 2)));
        diff_z_before = sum(abs(mesh1_before(:, 3) - mesh2_before(:, 3)));

        % Plot 1D differences for the current pair (before scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_x_before, diff_y_before, diff_z_before], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
        ylim([0, 8.5]);
        ylabel('Total Displacement');
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('1D Displacement Before Scaling for All Pairs');

% Initialize figure for subplots of 1D differences after scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "after scaling"
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the "after scaling" variables from scaledCoordsMax
        mesh1_after = scaledCoordsMax.(name1);
        mesh2_after = scaledCoordsMax.(name2);

        % Compute 1D differences after scaling
        diff_x_after = sum(abs(mesh1_after(:, 1) - mesh2_after(:, 1)));
        diff_y_after = sum(abs(mesh1_after(:, 2) - mesh2_after(:, 2)));
        diff_z_after = sum(abs(mesh1_after(:, 3) - mesh2_after(:, 3)));

        % Plot 1D differences for the current pair (after scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_x_after, diff_y_after, diff_z_after], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
        ylabel('Total Displacement');
        ylim([0, 8.5]);
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('1D Displacement After Scaling for All Pairs');

% Initialize figure for subplots of 2D differences before scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "before scaling"
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the "before scaling" variables using dynamic field names
        mesh1_before = eval(['partV_orig_before_scale_' name1]);
        mesh2_before = eval(['partV_orig_before_scale_' name2]);

        % Compute 2D differences before scaling
        diff_xy_before = sum(sqrt((mesh1_before(:, 1) - mesh2_before(:, 1)).^2 + (mesh1_before(:, 2) - mesh2_before(:, 2)).^2));
        diff_yz_before = sum(sqrt((mesh1_before(:, 2) - mesh2_before(:, 2)).^2 + (mesh1_before(:, 3) - mesh2_before(:, 3)).^2));
        diff_zx_before = sum(sqrt((mesh1_before(:, 3) - mesh2_before(:, 3)).^2 + (mesh1_before(:, 1) - mesh2_before(:, 1)).^2));

        % Plot 2D differences for the current pair (before scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_xy_before, diff_yz_before, diff_zx_before], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
        ylabel('Total Displacement');
        ylim([0, 12]);
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('2D Displacement Before Scaling for All Pairs');

% Initialize figure for subplots of 2D differences after scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "after scaling"
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the "after scaling" variables from scaledCoordsMax
        mesh1_after = scaledCoordsMax.(name1);
        mesh2_after = scaledCoordsMax.(name2);

        % Compute 2D differences after scaling
        diff_xy_after = sum(sqrt((mesh1_after(:, 1) - mesh2_after(:, 1)).^2 + (mesh1_after(:, 2) - mesh2_after(:, 2)).^2));
        diff_yz_after = sum(sqrt((mesh1_after(:, 2) - mesh2_after(:, 2)).^2 + (mesh1_after(:, 3) - mesh2_after(:, 3)).^2));
        diff_zx_after = sum(sqrt((mesh1_after(:, 3) - mesh2_after(:, 3)).^2 + (mesh1_after(:, 1) - mesh2_after(:, 1)).^2));

        % Plot 2D differences for the current pair (after scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_xy_after, diff_yz_after, diff_zx_after], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
        ylabel('Total Displacement');
        ylim([0, 12]);

        title(sprintf('%s vs %s', name1, name2));
        
        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('2D Displacement After Scaling for All Pairs');
%% % Define individuals and number of individuals
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
num_individuals = length(individuals);
num_pairs = num_individuals * (num_individuals - 1) / 2;

% Initialize variables to accumulate total displacements
total_displacement_1d_before = zeros(1, 3); % [X, Y, Z]
total_displacement_1d_after = zeros(1, 3);  % [X, Y, Z]
total_displacement_2d_before = zeros(1, 3); % [XY, YZ, ZX]
total_displacement_2d_after = zeros(1, 3);  % [XY, YZ, ZX]

% Loop through each pair of individuals to accumulate displacements
for i = 1:num_individuals
    for j = i+1:num_individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the "before scaling" variables using dynamic field names
        mesh1_before = eval(['partV_orig_before_scale_' name1]);
        mesh2_before = eval(['partV_orig_before_scale_' name2]);

        % Access the "after scaling" variables from scaledCoordsMax
        mesh1_after = scaledCoordsMax.(name1);
        mesh2_after = scaledCoordsMax.(name2);

        % Compute 1D differences before and after scaling
        diff_x_before = sum(abs(mesh1_before(:, 1) - mesh2_before(:, 1)));
        diff_y_before = sum(abs(mesh1_before(:, 2) - mesh2_before(:, 2)));
        diff_z_before = sum(abs(mesh1_before(:, 3) - mesh2_before(:, 3)));

        diff_x_after = sum(abs(mesh1_after(:, 1) - mesh2_after(:, 1)));
        diff_y_after = sum(abs(mesh1_after(:, 2) - mesh2_after(:, 2)));
        diff_z_after = sum(abs(mesh1_after(:, 3) - mesh2_after(:, 3)));

        % Accumulate total 1D displacements
        total_displacement_1d_before = total_displacement_1d_before + [diff_x_before, diff_y_before, diff_z_before];
        total_displacement_1d_after = total_displacement_1d_after + [diff_x_after, diff_y_after, diff_z_after];

        % Compute 2D differences before and after scaling
        diff_xy_before = sum(sqrt((mesh1_before(:, 1) - mesh2_before(:, 1)).^2 + (mesh1_before(:, 2) - mesh2_before(:, 2)).^2));
        diff_yz_before = sum(sqrt((mesh1_before(:, 2) - mesh2_before(:, 2)).^2 + (mesh1_before(:, 3) - mesh2_before(:, 3)).^2));
        diff_zx_before = sum(sqrt((mesh1_before(:, 3) - mesh2_before(:, 3)).^2 + (mesh1_before(:, 1) - mesh2_before(:, 1)).^2));

        diff_xy_after = sum(sqrt((mesh1_after(:, 1) - mesh2_after(:, 1)).^2 + (mesh1_after(:, 2) - mesh2_after(:, 2)).^2));
        diff_yz_after = sum(sqrt((mesh1_after(:, 2) - mesh2_after(:, 2)).^2 + (mesh1_after(:, 3) - mesh2_after(:, 3)).^2));
        diff_zx_after = sum(sqrt((mesh1_after(:, 3) - mesh2_after(:, 3)).^2 + (mesh1_after(:, 1) - mesh2_after(:, 1)).^2));

        % Accumulate total 2D displacements
        total_displacement_2d_before = total_displacement_2d_before + [diff_xy_before, diff_yz_before, diff_zx_before];
        total_displacement_2d_after = total_displacement_2d_after + [diff_xy_after, diff_yz_after, diff_zx_after];
    end
end

% Calculate the average displacements
avg_displacement_1d_before = total_displacement_1d_before / num_pairs;
avg_displacement_1d_after = total_displacement_1d_after / num_pairs;
avg_displacement_2d_before = total_displacement_2d_before / num_pairs;
avg_displacement_2d_after = total_displacement_2d_after / num_pairs;

% Plot Averaged 1D Displacement Before Scaling
figure;
bar(avg_displacement_1d_before);
set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
ylabel('Average Displacement');
ylim([0, 4]);
title('Averaged 1D Displacement, Before Scaling');

% Plot Averaged 1D Displacement After Scaling
figure;
bar(avg_displacement_1d_after);
set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
ylabel('Average Displacement');
ylim([0, 4]);
title('Averaged 1D Displacement After Scaling');

% Plot Averaged 2D Displacement Before Scaling
figure;
bar(avg_displacement_2d_before);
set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
ylabel('Average Displacement');
ylim([0, 6]);
title('Averaged 2D Displacement Before Scaling');

% Plot Averaged 2D Displacement After Scaling
figure;
bar(avg_displacement_2d_after);
set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
ylabel('Average Displacement');
ylim([0, 6]);
title('Averaged 2D Displacement After Scaling');
%% 
% Define individuals and number of individuals
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
num_individuals = length(individuals);
num_pairs = num_individuals * (num_individuals - 1) / 2;

% Initialize figure for subplots
figure;
subplot_idx = 1;

% Loop through each pair of individuals to compute and plot z-score standardized histograms
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the vertex data for both individuals (before scaling)
        mesh1 = scaledCoordsMax.(name1);
        mesh2 = scaledCoordsMax.(name2);

        % Compute the Euclidean distances (displacements) between corresponding vertices
        displacements = sqrt(sum((mesh1 - mesh2).^2, 2));

        % Compute the z-score for the displacements
        mean_disp = mean(displacements);
        std_disp = std(displacements);
        z_scores = (displacements - mean_disp) / std_disp;

        % Determine a symmetric x-axis range around 0
        max_abs_z = max(abs(z_scores));
        x_range = [-max_abs_z, max_abs_z];
        % Test for normality using the Kolmogorov-Smirnov test
        [h_kstest, p_kstest] = kstest(z_scores);

        % % Test for normality using the Shapiro-Wilk test
        % if exist('swtest', 'file')
        %     [h_swtest, p_swtest] = swtest(z_scores);
        %     fprintf('%s vs %s: SW test - h = %d, p = %.4f\n', name1, name2, h_swtest, p_swtest);
        % else
        %     fprintf('%s vs %s: SW test not available\n', name1, name2);
        % end

        % Print the results of the Kolmogorov-Smirnov test
        fprintf('%s vs %s: KS test - h = %d, p = %.4f\n', name1, name2, h_kstest, p_kstest);

        % Plot the z-score standardized displacements as a histogram
        subplot(ceil(num_pairs/7), 7, subplot_idx); % 3 columns in subplot grid
        histogram(z_scores, 'Normalization', 'pdf'); % Normalized histogram to pdf (probability density function)
        xlim(x_range); % Set x-axis limits to be symmetric around zero
        xlabel('Z-Score of Displacement');
        ylabel('Probability Density');
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('Z-Score Standardized Displacement Histograms');
%% 
% Define individuals and number of individuals
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
num_individuals = length(individuals);

% Initialize an array to store all the displacements
all_displacements = [];

% Loop through each pair of individuals to compute and collect displacements
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the vertex data for both individuals (before scaling)
        mesh1 = scaledCoordsMax.(name1);
        mesh2 = scaledCoordsMax.(name2);

        % Compute the Euclidean distances (displacements) between corresponding vertices
        displacements = sqrt(sum((mesh1 - mesh2).^2, 2));

        % Compute the z-score for the displacements
        mean_disp = mean(displacements);
        std_disp = std(displacements);
        z_scores = (displacements - mean_disp) / std_disp;

        % Determine a symmetric x-axis range around 0
        max_abs_z = max(abs(z_scores));
        x_range = [-max_abs_z, max_abs_z];

        % Append the displacements to the combined array
        all_displacements = [all_displacements; displacements];
    end
end

% Compute the z-score for the combined displacements
mean_disp = mean(all_displacements);
std_disp = std(all_displacements);
z_scores = (all_displacements - mean_disp) / std_disp;
% Test for normality using the Kolmogorov-Smirnov test
[h_kstest, p_kstest] = kstest(z_scores);
fprintf('%s vs %s: KS test - h = %d, p = %.4f\n', name1, name2, h_kstest, p_kstest);

% Plot the histogram of the averaged z-score standardized displacements
figure;
histogram(z_scores, 'Normalization', 'pdf'); % Normalized histogram to pdf
xlabel('Z-Score of Displacement');
ylabel('Probability Density');
title('Standardized Displacement Histogram Across All Identity Pairs');
%% % Define individuals and number of individuals
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
num_individuals = length(individuals);
num_pairs = num_individuals * (num_individuals - 1) / 2;

% Initialize figure for subplots
figure;
subplot_idx = 1;
all_displacements = [];
% Loop through each pair of individuals to compute and plot normalized histograms
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the vertex data for both individuals (after scaling)
        mesh1 = scaledCoordsMax.(name1);
        mesh2 = scaledCoordsMax.(name2);

        % Compute the Euclidean distances (displacements) between corresponding vertices
        displacements = sqrt(sum((mesh1 - mesh2).^2, 2));

        % Normalize the displacements to the range [-1, 1]
        min_disp = min(displacements);
        max_disp = max(displacements);
        normalized_displacements = 2 * ((displacements - min_disp) / (max_disp - min_disp)) - 1;

        % Determine a symmetric x-axis range around 0
        x_range = [-1, 1];  % Since the data is normalized to [-1, 1], we set the range accordingly

        % Test for normality using the Kolmogorov-Smirnov test
        [h_kstest, p_kstest] = kstest(normalized_displacements);

        % Print the results of the Kolmogorov-Smirnov test
        fprintf('%s vs %s: KS test - h = %d, p = %.4f\n', name1, name2, h_kstest, p_kstest);

        % Plot the normalized displacements as a histogram
        subplot(ceil(num_pairs/7), 7, subplot_idx); % 7 columns in subplot grid
        histogram(normalized_displacements, 'Normalization', 'pdf'); % Normalized histogram to pdf
        xlim(x_range); % Set x-axis limits to [-1, 1]
        xlabel('norm. displacement');
        ylabel('probability density');
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;

        all_displacements = [all_displacements; normalized_displacements];
    end
end
sgtitle('Normalized Displacement Between Pairs (Identity)');

figure;

histogram(all_displacements, 'Normalization', 'pdf'); % Normalized histogram to pdf
xlabel('norm. displacement');
ylabel('probability density');
title('Normalized Displacement All Pairs (Identity)');

%% Emotions
% Define individuals and number of individuals
individuals = {'neutral', 'happiness', 'sadness', 'disgust', 'fear', 'anger', 'surprise'};
num_individuals = length(individuals);
num_pairs = num_individuals * (num_individuals - 1) / 2;

% Initialize figure for subplots
figure;
subplot_idx = 1;
all_displacements = [];
% Loop through each pair of individuals to compute and plot normalized histograms
for i = 1:num_individuals
    for j = i+1:num_individuals
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the vertex data for both individuals (after scaling)
        mesh1 = eval(['partV_orig_before_scale_Elias_' name1]);
        mesh2 = eval(['partV_orig_before_scale_Elias_' name2]);

        % Compute the Euclidean distances (displacements) between corresponding vertices
        displacements = sqrt(sum((mesh1 - mesh2).^2, 2));

        % Normalize the displacements to the range [-1, 1]
        min_disp = min(displacements);
        max_disp = max(displacements);
        normalized_displacements = 2 * ((displacements - min_disp) / (max_disp - min_disp)) - 1;

        % Determine a symmetric x-axis range around 0
        x_range = [-1, 1];  % Since the data is normalized to [-1, 1], we set the range accordingly

        % Test for normality using the Kolmogorov-Smirnov test
        [h_kstest, p_kstest] = kstest(normalized_displacements);

        % Print the results of the Kolmogorov-Smirnov test
        fprintf('%s vs %s: KS test - h = %d, p = %.4f\n', name1, name2, h_kstest, p_kstest);

        % Plot the normalized displacements as a histogram
        subplot(ceil(num_pairs/7), 7, subplot_idx); % 7 columns in subplot grid
        histogram(normalized_displacements, 'Normalization', 'pdf'); % Normalized histogram to pdf
        xlim(x_range); % Set x-axis limits to [-1, 1]
        xlabel('norm. displacement');
        ylabel('probability density');
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;

        all_displacements = [all_displacements; normalized_displacements];
    end
end
sgtitle('Normalized Displacement Between Pairs (Emotion)');

figure;

histogram(all_displacements, 'Normalization', 'pdf'); % Normalized histogram to pdf
xlabel('norm. displacement');
ylabel('probability density');
title('Normalized Displacement All Pairs (Emotion)');
%% 
