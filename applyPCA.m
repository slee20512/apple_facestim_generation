% Within-identity

% Reshape each face into a row vector
num_faces = 7;
num_vertices = 936;
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

reshaped_faces_identity = zeros(num_faces, num_vertices * 3); % 7 faces, 2808 features

for i = 1:num_faces
    face = scaledCoordsMax.(individuals{i}); % Extract the i-th face
    reshaped_faces_identity(i, :) = reshape(face, 1, []); % Flatten the face into a row
end

% Center the data
mean_face = mean(reshaped_faces_identity, 1);
centered_faces = reshaped_faces_identity - mean_face;
standardized_faces = (reshaped_faces_identity - mean(reshaped_faces_identity)) ./ std(reshaped_faces_identity);

identity_data = standardized_faces;

% Apply PCA
% Perform PCA on the data
% [coeff, score, latent] = pca(centered_faces);
[coeff_identity, score_identity, latent_identity] = pca(identity_data);

% Total variance
total_variance = sum(latent_identity);

% Explained variance ratio for each principal component
explained_variance_identity = latent_identity / sum(latent_identity);

% Cumulative explained variance
cumulative_variance = cumsum(explained_variance_identity);

% Display the results
disp('Explained variance ratio for each PC:');
disp(explained_variance_identity);

disp('Cumulative variance explained:');
disp(cumulative_variance);

% coeff contains the principal components
% score contains the projections of the faces onto the principal components
% latent contains the variance explained by each principal component

% Visualize the results, e.g., plot the first two principal components
figure;
scatter(score_identity(:, 1), score_identity(:, 2));
xlabel('PC 1');
ylabel('PC 2');
title('PCA of Identity Data');
hold on;
for i = 1:num_faces
    text(score_identity(i, 1), score_identity(i, 2), individuals{i}, 'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
hold off;


% 3D Scatter plot of the first three principal components
figure;
scatter3(score_identity(:, 1), score_identity(:, 2), score_identity(:, 3), 'filled');
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
title('3D PCA of Identity Data');
grid on;

% Add labels to the scatter plot
hold on;
for i = 1:num_faces
    text(score_identity(i, 1), score_identity(i, 2), score_identity(i, 3), individuals{i}, 'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
hold off;
%% Within-emotion
num_faces = 7;
num_vertices = 936;
emotions = {'neutral', 'happiness', 'sadness', 'disgust', 'fear', 'anger', 'surprise'};

reshaped_faces_emotion = zeros(num_faces, num_vertices * 3); % 7 faces, 2808 features

for i = 1:num_faces
    face = scaledCoordsMax_Elias.(emotions{i}); % Extract the i-th face
    reshaped_faces_emotion(i, :) = reshape(face, 1, []); % Flatten the face into a row
end

% Center the data
mean_face = mean(reshaped_faces_emotion, 1);
centered_faces = reshaped_faces_emotion - mean_face;
standardized_faces = (reshaped_faces_emotion - mean(reshaped_faces_emotion)) ./ std(reshaped_faces_emotion);

emotion_data = standardized_faces;

% Apply PCA
% Perform PCA on the data
% [coeff, score, latent] = pca(centered_faces);
[coeff_emotion, score_emotion, latent_emotion] = pca(emotion_data);

% Total variance
total_variance = sum(latent_emotion);

% Explained variance ratio for each principal component
explained_variance_emotion = latent_emotion / sum(latent_emotion);

% Cumulative explained variance
cumulative_variance = cumsum(explained_variance_emotion);

% Display the results
disp('Explained variance ratio for each PC:');
disp(explained_variance_emotion);

disp('Cumulative variance explained:');
disp(cumulative_variance);

% coeff contains the principal components
% score contains the projections of the faces onto the principal components
% latent contains the variance explained by each principal component

% Visualize the results, e.g., plot the first two principal components
figure;
scatter(score_emotion(:, 1), score_emotion(:, 2));
xlabel('PC 1');
ylabel('PC 2');
title('PCA of Emotion Data');
hold on;
for i = 1:num_faces
    text(score_emotion(i, 1), score_emotion(i, 2), emotions{i}, 'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
hold off;


% 3D Scatter plot of the first three principal components
figure;
scatter3(score_emotion(:, 1), score_emotion(:, 2), score_emotion(:, 3), 'filled');
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
title('3D PCA of Emotion Data');
grid on;

% Add labels to the scatter plot
hold on;
for i = 1:num_faces
    text(score_emotion(i, 1), score_emotion(i, 2), score_emotion(i, 3), emotions{i}, 'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
hold off;
%% Between set comparisons (Combined PCA on both sets)
% Combine identity and emotion data into one matrix
combined_data = [reshaped_faces_identity; reshaped_faces_emotion];
individuals_emotions = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah', 'neutral', 'happiness', 'sadness', 'disgust', 'fear', 'anger', 'surprise'};

% Apply PCA on the combined dataset
mean_face = mean(combined_data, 1);
centered_faces = combined_data - mean_face;
standardized_faces = (combined_data - mean(combined_data)) ./ std(combined_data);

combined_faces = standardized_faces;

[coeff_combined, score_combined, latent_combined] = pca(combined_faces);

% Explained variance for the combined set
explained_variance_combined = latent_combined / sum(latent_combined);
% Cumulative explained variance
cumulative_variance = cumsum(explained_variance_combined);

% Display the results
disp('Explained variance ratio for each PC:');
disp(explained_variance_combined);

disp('Cumulative variance explained:');
disp(cumulative_variance);

% Number of identity and emotion points
num_identity = size(reshaped_faces_identity, 1);  % Number of identities
num_emotion = size(reshaped_faces_emotion, 1);    % Number of emotions

% 2D Scatter plot of the first two principal components with different colors
figure;
hold on;
% Plot identities (blue)
scatter(score_combined(1:num_identity, 1), score_combined(1:num_identity, 2), 'filled', 'b');
% Plot emotions (red)
scatter(score_combined(num_identity+1:end, 1), score_combined(num_identity+1:end, 2), 'filled', 'r');

% Add labels to the scatter plot
for i = 1:length(individuals_emotions)
    text(score_combined(i, 1), score_combined(i, 2), individuals_emotions{i}, 'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
xlabel('PC 1');
ylabel('PC 2');
title('PCA of Identity and Emotion Combined (2D)');
hold off;


% Create VideoWriter object to save video as an .mp4 file
videoFile = VideoWriter('3D_PCA_Rotation.mp4', 'MPEG-4');
videoFile.FrameRate = 30;  % Set frame rate (frames per second)
open(videoFile);

% 3D Scatter plot of the first three principal components with different colors
figure;
hold on;
% Plot identities (blue)
scatter3(score_combined(1:num_identity, 1), score_combined(1:num_identity, 2), score_combined(1:num_identity, 3), 'filled', 'b');
% Plot emotions (red)
scatter3(score_combined(num_identity+1:end, 1), score_combined(num_identity+1:end, 2), score_combined(num_identity+1:end, 3), 'filled', 'r');

% Add labels to the scatter plot
for i = 1:length(individuals_emotions)
    text(score_combined(i, 1), score_combined(i, 2), score_combined(i, 3), individuals_emotions{i}, 'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
xlim([-70, 100]);
ylim([-80, 60]);
zlim([-60, 60]);

view(3); % Default 3D view

title('3D PCA of Identity and Emotion Combined');
grid on;
hold off;


% Rotate and capture frames
for angle = 0:360  % Rotate from 0 to 360 degrees
    view(angle, 30);  % Adjust the view angle (azimuth = angle, elevation = 30)
    frame = getframe(gcf);  % Capture the current figure as a frame
    writeVideo(videoFile, frame);  % Write the frame to the video file
end

% Close the video file
close(videoFile);
disp('3D plot saved as video.');
%% Plot ALL identity and emotions 

num_faces = 7;
num_vertices = 936;
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};
emotions = {'neutral', 'happiness', 'sadness', 'disgust', 'fear', 'anger', 'surprise'};

all_faces_data = zeros(num_faces * length(emotions), num_vertices * 3);

% Map for assigning distinct colors for identities
identity_colors = lines(num_faces);  % Use distinct colors for each identity

counter = 1;
for i = 1:num_faces
    for j = 1:length(emotions)
        face = eval(['scaledCoordsMax_' individuals{i}]).(emotions{j});  % Extract emotion data
        all_faces_data(counter, :) = reshape(face, 1, []);  % Flatten and store
        counter = counter + 1;
    end
end

% Mean subtraction (centering the data)
mean_face = mean(all_faces_data, 1);
centered_faces = all_faces_data - mean_face;

% Normalize the data (optional but recommended for PCA)
standardized_faces = centered_faces ./ std(centered_faces);

% Apply PCA
[coeff, score, latent] = pca(standardized_faces);

% Explained variance for the combined set
explained_variance = latent / sum(latent);
disp('Explained variance ratio for each PC:');
disp(explained_variance);

% % Create VideoWriter object to save video as an .mp4 file
% videoFile = VideoWriter('3D_PCA_Rotation_ALL2.mp4', 'MPEG-4');
% videoFile.FrameRate = 10;  % Set frame rate (frames per second)
% open(videoFile);

% 3D Scatter plot with varying color intensity for emotions (with legend)
figure;
hold on;
legend_handles = [];  % Initialize an array for legend handles

for i = 1:num_faces
    base_color = identity_colors(i, :);  % Neutral face color for each identity
    
    % Create a plot for the neutral face only for legend purposes
    h = scatter3(NaN, NaN, NaN, 100, 'filled', 'MarkerFaceColor', base_color);  % Dummy plot for the legend
    legend_handles = [legend_handles h];  % Store the handle for the legend
    
    for j = 1:length(emotions)
        % Cap intensity at 0.8 to avoid excessive blending
        emotion_intensity = 0.9 * (j - 1) / (length(emotions) - 1);  % Scale intensity from 0 to 0.8
        current_color = (1 - emotion_intensity) * base_color + emotion_intensity * [0.9, 0.9, 0.9];  % Blend with gray
        
        % Plot each emotion's projection onto the first three PCs
        index = (i - 1) * length(emotions) + j;
        scatter3(score(index, 1), score(index, 2), score(index, 3), 70, 'filled', 'MarkerFaceColor', current_color);
        
        % Label only the neutral face with the identity name
        if strcmp(emotions{j}, 'neutral')
            text(score(index, 1), score(index, 2), score(index, 3), individuals{i}, 'FontSize', 10);
        end
    end
end

xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
xlim([-70, 100]);
ylim([-80, 60]);
zlim([-60, 60]);
view(3); % Default 3D view
title('3D PCA of Identity and Emotions');
grid on;

% Add the legend with identity names
legend(legend_handles, individuals, 'Location', 'bestoutside');  % Place legend outside plot

% % Rotate and capture frames
% for angle = 0:360  % Rotate from 0 to 360 degrees
%     view(angle, 30);  % Adjust the view angle (azimuth = angle, elevation = 30)
%     frame = getframe(gcf);  % Capture the current figure as a frame
%     writeVideo(videoFile, frame);  % Write the frame to the video file
% end
% 
% % Close the video file
% close(videoFile);
% disp('3D plot saved as video.');

%% Add emotion labels to the legend based on Elias' colors
emotion_handles = [];  % Initialize an array for emotion handles
base_color_elias = identity_colors(1, :);  % Base color for Elias

for j = 1:length(emotions)
    emotion_intensity = 0.85 * (j - 1) / (length(emotions) - 1);  % Same intensity scaling as above
    emotion_color = (1 - emotion_intensity) * base_color_elias + emotion_intensity * [0.9, 0.9, 0.9];  % Blend with gray
    % Create a dummy plot for the emotion legend
    h_emotion = scatter3(NaN, NaN, NaN, 100, 'filled', 'MarkerFaceColor', emotion_color);
    emotion_handles = [emotion_handles h_emotion];  % Store the handle for the legend
end

% Add emotion labels below the identity legend
legend([legend_handles, emotion_handles], [individuals, emotions], 'Location', 'bestoutside');

hold off;


%% Plot Colored by Identity with Correct Legend
figure;
hold on;

% Initialize array for legend handles
legend_handles = [];

% Iterate over each identity and plot all emotions in the same color
for i = 1:num_faces
    base_color = identity_colors(i, :);  % Same color for all emotions of this identity
    
    % Dummy plot for legend
    h = scatter3(NaN, NaN, NaN, 100, 'filled', 'MarkerFaceColor', base_color);  % Invisible dummy point
    legend_handles = [legend_handles h];  % Store handle for the legend
    
    for j = 1:length(emotions)
        % Plot all emotions with the same base color for each identity
        index = (i - 1) * length(emotions) + j;
        scatter3(score(index, 1), score(index, 2), score(index, 3), 80, 'filled', 'MarkerFaceColor', base_color);
    end
end
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
view(3); % Default 3D view
title('3D PCA of Identity & Emotion - Colored by Identity');
grid on;

% Add the legend with correct colors for identities
legend(legend_handles, individuals, 'Location', 'bestoutside');
hold off;

% Plot Colored by Emotion with Correct Legend
figure;
hold on;

% Define a colormap for emotions (use distinct colors for each emotion)
emotion_colors = lines(length(emotions));  % Use different colors for each emotion

% Initialize array for legend handles
legend_handles = [];

% Iterate over each emotion and plot it across all identities in the same color
for j = 1:length(emotions)
    emotion_color = emotion_colors(j, :);  % Same color for this emotion across all identities
    
    % Dummy plot for legend
    h = scatter3(NaN, NaN, NaN, 100, 'filled', 'MarkerFaceColor', emotion_color);  % Invisible dummy point
    legend_handles = [legend_handles h];  % Store handle for the legend
    
    for i = 1:num_faces
        % Plot all identities with the same color for each emotion
        index = (i - 1) * length(emotions) + j;
        scatter3(score(index, 1), score(index, 2), score(index, 3), 80, 'filled', 'MarkerFaceColor', emotion_color);
    end
end
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
view(3); % Default 3D view
title('3D PCA of Identity & Emotion - Colored by Emotion');
grid on;

% Add the legend with correct colors for emotions
legend(legend_handles, emotions, 'Location', 'bestoutside');
hold off;
%% Scree Plot with Labeled Values for the First 15 Principal Components
figure;

% Calculate the explained variance percentage for the first 15 PCs
explained_variance_percentage = (latent / sum(latent)) * 100;
explained_variance_percentage = explained_variance_percentage(1:15);  % Only take the first 15 PCs

% Create a bar plot of the explained variance for the first 15 PCs
bar(explained_variance_percentage);

% Label the axes
xlabel('Principal Component', FontSize=12);
ylabel('Variance Explained (%)', FontSize=12);
title('Scree Plot of First 15 PCs', FontSize=15);
grid on;

% Add labels to the bars (variance percentages)
for i = 1:15
    text(i, explained_variance_percentage(i) + 1, sprintf('%.2f', explained_variance_percentage(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', FontSize=12);
end
%% 1. Calculate the angle between the two separation vectors
% Initialize arrays to store the variances for identities and emotions
identity_variance = zeros(1, 3);  % For the first 3 PCs
emotion_variance = zeros(1, 3);   % For the first 3 PCs

% Calculate the variance along each PC for identities
% Initialize variables for between-identity variance
identity_centroids = zeros(num_faces, 3);  % Centroids for the first 3 PCs

for i = 1:num_faces
    identity_indices = (i-1)*length(emotions) + 1 : i*length(emotions);  % Indices for current identity
    identity_centroids(i, :) = mean(score(identity_indices, 1:3));  % Mean of each identity for the first 3 PCs
end

overall_mean = mean(identity_centroids, 1);  % Overall centroid for all identities
between_identity_variance = var(identity_centroids - overall_mean);  % Between-identity variance
identity_variance = between_identity_variance / num_faces;  % Average variance for identities

% Calculate the variance along each PC for emotions
emotion_centroids = zeros(num_faces, 3);  % Centroids for the first 3 PCs

for i = 1:num_faces
    emotion_indices = j:length(emotions):num_faces*length(emotions);  % Indices for current emotion
    emotion_centroids(j, :) = mean(score(emotion_indices, 1:3));  % Mean of each emotion for the first 3 PCs
end

overall_mean = mean(emotion_centroids, 1);  % Overall centroid for all identities
between_emotion_variance = var(emotion_centroids - overall_mean);  % Between-identity variance
emotion_variance = between_emotion_variance / num_faces;  % Average variance for identities

% Normalize the variance to be used as weights for separation vectors
identity_weights = identity_variance / norm(identity_variance);
emotion_weights = emotion_variance / norm(emotion_variance);

% Display the computed weights for the separation vectors
disp('Identity Weights (for separation vector):');
disp(identity_weights);
disp('Emotion Weights (for separation vector):');
disp(emotion_weights);

% Construct the Separation Vectors Based on Computed Weights

% Emotion separation vector: weighted combination of PC 1, PC 2, and PC 3
emotion_separation_vector = emotion_weights;  % Use the computed weights for emotions

% Identity separation vector: weighted combination of PC 1, PC 2, and PC 3
identity_separation_vector = identity_weights;  % Use the computed weights for identities

% Normalize the vectors (optional)
emotion_separation_vector = emotion_separation_vector / norm(emotion_separation_vector);
identity_separation_vector = identity_separation_vector / norm(identity_separation_vector);

% 3D PCA Plot with Computed Separation Vectors for Emotion and Identity
% emotion_separation_vector = cross([0, 0, 1], emotion_separation_vector);
% identity_separation_vector = cross([0, 0, 1], identity_separation_vector);

figure;
hold on;

% Plot data points colored by identity
for j = 1:length(emotions)
    emotion_color = emotion_colors(j, :);  % Same color for this emotion across all identities
    
    % Dummy plot for legend
    h = scatter3(NaN, NaN, NaN, 100, 'filled', 'MarkerFaceColor', emotion_color);  % Invisible dummy point
    legend_handles = [legend_handles h];  % Store handle for the legend
    
    for i = 1:num_faces
        % Plot all identities with the same color for each emotion
        index = (i - 1) * length(emotions) + j;
        scatter3(score(index, 1), score(index, 2), score(index, 3), 80, 'filled', 'MarkerFaceColor', emotion_color);
    end
end

% Plot the separation vectors as arrows

% Calculate the center of the data to start the arrows from the centroid
centroid = mean(score(:, 1:3));  % Mean of all data points in the first 3 PCs

% Plot the emotion separation vector (weighted by PC contributions)
h_emotion_arrow = quiver3(centroid(1), centroid(2), centroid(3), ...
    emotion_separation_vector(1), emotion_separation_vector(2), emotion_separation_vector(3), ...
    50, 'LineWidth', 2.5, 'Color', 'r', 'MaxHeadSize', 0.4);  % Red arrow for emotion separation

% Plot the identity separation vector (weighted by PC contributions)
h_identity_arrow = quiver3(centroid(1), centroid(2), centroid(3), ...
    identity_separation_vector(1), identity_separation_vector(2), identity_separation_vector(3), ...
    50, 'LineWidth', 2.5, 'Color', 'b', 'MaxHeadSize', 0.4);  % Blue arrow for identity separation

% Create dummy scatter points for the legend to capture identity and emotion arrows
h_dummy_emotion = plot(NaN, NaN, 'r-', 'LineWidth', 2);  % Dummy for red arrow
h_dummy_identity = plot(NaN, NaN, 'b-', 'LineWidth', 2);  % Dummy for blue arrow

% Label the plot and the axes
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
xlim([-60, 100]);
ylim([-80, 60]);
zlim([-60, 60]);
view(3);
title('3D PCA Plot with Emotion & Identity Separation Vectors');
legend([h_dummy_emotion, h_dummy_identity], {'Emotion Separation', 'Identity Separation'}, 'Location', 'best');
grid on;
hold off;
% 

% Compute the magnitudes (norms) of the vectors
norm_emotion = norm(emotion_separation_vector);
norm_identity = norm(identity_separation_vector);

% Compute the dot product (already computed earlier)
dot_product = dot(emotion_separation_vector, identity_separation_vector);

% Compute the angle in radians
theta_radians = acos(dot_product / (norm_emotion * norm_identity));

% Convert the angle to degrees
theta_degrees = rad2deg(theta_radians);

% Display the result
disp('Angle between Emotion and Identity Separation Vectors (in degrees):');
disp(theta_degrees);
%% 
% Step 1: Run standard k-means clustering (unconstrained)
num_clusters = 7;  % Number of clusters
[cluster_indices, cluster_centers] = kmeans(score(:, 1:3), num_clusters);  % Run k-means on the first 3 PCs

% Step 2: Count how many points are in each cluster
cluster_counts = histcounts(cluster_indices, num_clusters);

% Step 3: Ensure that each cluster has exactly 7 points
target_cluster_size = 7;  % Desired size for each cluster

% Initialize an array to hold the adjusted cluster indices
adjusted_cluster_indices = cluster_indices;

% Loop through each cluster and adjust sizes
for i = 1:num_clusters
    % Find the indices of the points in this cluster
    cluster_points = find(cluster_indices == i);
    
    % If the cluster has more than 7 points, reassign the extra points
    if length(cluster_points) > target_cluster_size
        extra_points = cluster_points(target_cluster_size+1:end);  % Points to reassign
        
        % Find clusters with fewer than 7 points and reassign the extra points
        for j = 1:num_clusters
            if cluster_counts(j) < target_cluster_size
                free_slots = target_cluster_size - cluster_counts(j);
                num_reassign = min(free_slots, length(extra_points));  % Reassign up to the available free slots
                adjusted_cluster_indices(extra_points(1:num_reassign)) = j;
                cluster_counts(j) = cluster_counts(j) + num_reassign;  % Update the cluster counts
                extra_points(1:num_reassign) = [];  % Remove reassigned points from extra points
            end
            if isempty(extra_points), break; end  % Stop if no extra points remain
        end
    end
end

% Step 4: Check that each cluster now has exactly 7 points
final_cluster_counts = histcounts(adjusted_cluster_indices, num_clusters);
disp('Final cluster sizes:');
disp(final_cluster_counts);

% Step 5: Create a table to compare true identity labels, true emotion labels, and adjusted cluster labels
comparison_table_adjusted = table((1:total_points)', adjusted_cluster_indices, true_identity_labels', true_emotion_labels', ...
                   'VariableNames', {'DataIndex', 'AdjustedClusterLabel', 'TrueIdentity', 'TrueEmotion'});

% Step 6: Print out the comparison table for the adjusted clustering
disp('Comparison between True Identity, True Emotion, and Adjusted Clustered Labels:');
disp(comparison_table_adjusted);

% Step 7: Calculate and print accuracy based on identity for adjusted clustering
identity_accuracy_adjusted = sum(adjusted_cluster_indices == true_identity_labels') / total_points;
disp(['Adjusted clustering accuracy based on Identity: ', num2str(identity_accuracy_adjusted * 100), '%']);

% Step 8: Calculate and print accuracy based on emotion for adjusted clustering
emotion_accuracy_adjusted = sum(adjusted_cluster_indices == true_emotion_labels') / total_points;
disp(['Adjusted clustering accuracy based on Emotion: ', num2str(emotion_accuracy_adjusted * 100), '%']);
%% % Step 1: Run standard k-means clustering (no constraints)
num_clusters = 7;  % Number of clusters (for the 7 identities)
[cluster_indices, cluster_centers] = kmeans(score(:, 1:3), num_clusters);  % Run k-means on the first 3 PCs

% Step 2: Plot the 3D PCA results with standard k-means cluster labels
figure;
scatter3(score(:, 1), score(:, 2), score(:, 3), 80, cluster_indices, 'filled');
title('3D PCA Plot with K-means Cluster Labels');
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
grid on;
% colorbar;  % Show color bar to represent cluster labels
