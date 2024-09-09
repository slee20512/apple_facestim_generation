%% % Compute differences for each coordinate axis

emotions = {'neutral', 'happiness', 'sadness', 'disgust', 'fear', 'anger', 'surprise'};
num_emotions = length(emotions);
num_pairs = num_emotions * (num_emotions - 1) / 2;

% Set color for all plots (blue)
color_blue = [0.1, 0.4, 0.8]; % Blueish color

% Initialize figure for subplots of 1D differences before scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "before scaling"
for i = 1:num_emotions
    for j = i+1:num_emotions
        % Get the names of the two individuals
        name1 = emotions{i};
        name2 = emotions{j};

        % Access the "before scaling" variables using dynamic field names
        mesh1 = scaledCoordsMax_Elias.(name1);
        mesh2 = scaledCoordsMax_Elias.(name2);
        
        % Compute 1D differences before scaling
        diff_x_before = sum(abs(mesh1(:, 1) - mesh2(:, 1)));
        diff_y_before = sum(abs(mesh1(:, 2) - mesh2(:, 2)));
        diff_z_before = sum(abs(mesh1(:, 3) - mesh2(:, 3)));

        % Plot 1D differences for the current pair (before scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_x_before, diff_y_before, diff_z_before], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
        ylim([0, 7]);
        ylabel('Total Displacement');
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('1D Displacement, All Emotion Pairs');


% Initialize figure for subplots of 2D differences before scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "before scaling"
for i = 1:num_emotions
    for j = i+1:num_emotions
        % Get the names of the two individuals
        name1 = emotions{i};
        name2 = emotions{j};


        % Access the "before scaling" variables using dynamic field names
        mesh1 = scaledCoordsMax_Elias.(name1);
        mesh2 = scaledCoordsMax_Elias.(name2);

        % Compute 2D differences before scaling
        diff_xy_before = sum(sqrt((mesh1(:, 1) - mesh2(:, 1)).^2 + (mesh1(:, 2) - mesh2(:, 2)).^2));
        diff_yz_before = sum(sqrt((mesh1(:, 2) - mesh2(:, 2)).^2 + (mesh1(:, 3) - mesh2(:, 3)).^2));
        diff_zx_before = sum(sqrt((mesh1(:, 3) - mesh2(:, 3)).^2 + (mesh1(:, 1) - mesh2(:, 1)).^2));

        % Plot 2D differences for the current pair (before scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_xy_before, diff_yz_before, diff_zx_before], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
        ylabel('Total Displacement');
        ylim([0, 9]);
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('2D Displacement, All Emotion Pairs');


% Initialize variables to accumulate total displacements
total_displacement_1d_before = zeros(1, 3); % [X, Y, Z]
total_displacement_2d_before = zeros(1, 3); % [XY, YZ, ZX]

% Loop through each pair of individuals to accumulate displacements
for i = 1:num_emotions
    for j = i+1:num_emotions
        name1 = emotions{i};
        name2 = emotions{j};

        % Access the "before scaling" variables using dynamic field names
        % mesh1 = eval(['partV_orig_before_scale_Elias_' name1]);
        % mesh2 = eval(['partV_orig_before_scale_Elias_' name2]);

        mesh1 = scaledCoordsMax_Elias.(name1);
        mesh2 = scaledCoordsMax_Elias.(name2);

        % Compute 1D differences before and after scaling
        diff_x_before = sum(abs(mesh1(:, 1) - mesh2(:, 1)));
        diff_y_before = sum(abs(mesh1(:, 2) - mesh2(:, 2)));
        diff_z_before = sum(abs(mesh1(:, 3) - mesh2(:, 3)));

        % Accumulate total 1D displacements
        total_displacement_1d_before = total_displacement_1d_before + [diff_x_before, diff_y_before, diff_z_before];

        % Compute 2D differences before and after scaling
        diff_xy_before = sum(sqrt((mesh1(:, 1) - mesh2(:, 1)).^2 + (mesh1(:, 2) - mesh2(:, 2)).^2));
        diff_yz_before = sum(sqrt((mesh1(:, 2) - mesh2(:, 2)).^2 + (mesh1(:, 3) - mesh2(:, 3)).^2));
        diff_zx_before = sum(sqrt((mesh1(:, 3) - mesh2(:, 3)).^2 + (mesh1(:, 1) - mesh2(:, 1)).^2));

        % Accumulate total 2D displacements
        total_displacement_2d_before = total_displacement_2d_before + [diff_xy_before, diff_yz_before, diff_zx_before];
    end
end

% Calculate the average displacements
avg_displacement_1d_before = total_displacement_1d_before / num_pairs;
avg_displacement_2d_before = total_displacement_2d_before / num_pairs;

% Plot Averaged 1D Displacement Before Scaling
figure;
bar(avg_displacement_1d_before);
set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
ylabel('Average Displacement');
ylim([0, 3.5]);
title('1D Displacement, Averaged Emotion Pairs');


% Plot Averaged 2D Displacement Before Scaling
figure;
bar(avg_displacement_2d_before);
set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
ylabel('Average Displacement');
ylim([0, 4.5]);
title('2D Displacement, Averaged Emotion Pairs');
%% Same (after scale) for all identiy pairs
% Compute differences for each coordinate axis

individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
num_identity = length(individuals);
num_pairs = num_identity * (num_identity - 1) / 2;

% Set color for all plots (blue)
color_blue = [0.1, 0.4, 0.8]; % Blueish color

% Initialize figure for subplots of 1D differences before scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "before scaling"
for i = 1:num_identity
    for j = i+1:num_identity
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the "before scaling" variables using dynamic field names
        mesh1 = scaledCoordsMax.(name1);
        mesh2 = scaledCoordsMax.(name2);

        % Compute 1D differences before scaling
        diff_x_before = sum(abs(mesh1(:, 1) - mesh2(:, 1)));
        diff_y_before = sum(abs(mesh1(:, 2) - mesh2(:, 2)));
        diff_z_before = sum(abs(mesh1(:, 3) - mesh2(:, 3)));

        % Plot 1D differences for the current pair (before scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_x_before, diff_y_before, diff_z_before], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
        ylim([0, 7]);
        ylabel('Total Displacement');
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('1D Displacement, All Identity Pairs');


% Initialize figure for subplots of 2D differences before scaling
figure;
subplot_idx = 1;

% Loop through each pair of individuals for "before scaling"
for i = 1:num_identity
    for j = i+1:num_identity
        % Get the names of the two individuals
        name1 = individuals{i};
        name2 = individuals{j};


        % Access the "before scaling" variables using dynamic field names
        mesh1 = scaledCoordsMax.(name1);
        mesh2 = scaledCoordsMax.(name2);

        % Compute 2D differences before scaling
        diff_xy_before = sum(sqrt((mesh1(:, 1) - mesh2(:, 1)).^2 + (mesh1(:, 2) - mesh2(:, 2)).^2));
        diff_yz_before = sum(sqrt((mesh1(:, 2) - mesh2(:, 2)).^2 + (mesh1(:, 3) - mesh2(:, 3)).^2));
        diff_zx_before = sum(sqrt((mesh1(:, 3) - mesh2(:, 3)).^2 + (mesh1(:, 1) - mesh2(:, 1)).^2));

        % Plot 2D differences for the current pair (before scaling)
        subplot(ceil(num_pairs/7), 7, subplot_idx);
        bar([diff_xy_before, diff_yz_before, diff_zx_before], 'FaceColor', color_blue, 'EdgeColor', 'k'); % Blueish bars
        set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
        ylabel('Total Displacement');
        ylim([0, 9]);
        title(sprintf('%s vs %s', name1, name2));

        subplot_idx = subplot_idx + 1;
    end
end
sgtitle('2D Displacement, All Identity Pairs');


% Initialize variables to accumulate total displacements
total_displacement_1d_before = zeros(1, 3); % [X, Y, Z]
total_displacement_2d_before = zeros(1, 3); % [XY, YZ, ZX]

% Loop through each pair of individuals to accumulate displacements
for i = 1:num_identity
    for j = i+1:num_identity
        name1 = individuals{i};
        name2 = individuals{j};

        % Access the "before scaling" variables using dynamic field names
        mesh1 = scaledCoordsMax.(name1);
        mesh2 = scaledCoordsMax.(name2);

        % Compute 1D differences before and after scaling
        diff_x_before = sum(abs(mesh1(:, 1) - mesh2(:, 1)));
        diff_y_before = sum(abs(mesh1(:, 2) - mesh2(:, 2)));
        diff_z_before = sum(abs(mesh1(:, 3) - mesh2(:, 3)));

        % Accumulate total 1D displacements
        total_displacement_1d_before = total_displacement_1d_before + [diff_x_before, diff_y_before, diff_z_before];

        % Compute 2D differences before and after scaling
        diff_xy_before = sum(sqrt((mesh1(:, 1) - mesh2(:, 1)).^2 + (mesh1(:, 2) - mesh2(:, 2)).^2));
        diff_yz_before = sum(sqrt((mesh1(:, 2) - mesh2(:, 2)).^2 + (mesh1(:, 3) - mesh2(:, 3)).^2));
        diff_zx_before = sum(sqrt((mesh1(:, 3) - mesh2(:, 3)).^2 + (mesh1(:, 1) - mesh2(:, 1)).^2));

        % Accumulate total 2D displacements
        total_displacement_2d_before = total_displacement_2d_before + [diff_xy_before, diff_yz_before, diff_zx_before];
    end
end

% Calculate the average displacements
avg_displacement_1d_before = total_displacement_1d_before / num_pairs;
avg_displacement_2d_before = total_displacement_2d_before / num_pairs;

% Plot Averaged 1D Displacement Before Scaling
figure;
bar(avg_displacement_1d_before);
set(gca, 'XTickLabel', {'X', 'Y', 'Z'});
ylabel('Average Displacement');
ylim([0, 3.5]);
title('1D Displacement, Averaged Identity Pairs');


% Plot Averaged 2D Displacement Before Scaling
figure;
bar(avg_displacement_2d_before);
set(gca, 'XTickLabel', {'XY', 'YZ', 'ZX'});
ylabel('Average Displacement');
ylim([0, 4.5]);
title('2D Displacement, Averaged Identity Pairs');

