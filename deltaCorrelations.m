%% Method 1: Compute Delta
% Initialize struct to store deltas for individual pairs
delta_identity = struct();
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

% Iterate through each pair of individuals
for i = 1:length(individuals)
    for j = i+1:length(individuals) 
        % Extract the vertex data for each individual 
        mesh1 = scaledCoordsMax.(individuals{i});
        mesh2 = scaledCoordsMax.(individuals{j});

        % Compute the Euclidean distances (displacements) between corresponding vertices
        displacements = sqrt(sum((mesh1 - mesh2).^2, 2));

        % Store the result in the struct with a field name like 'Elias_Neptune'
        field_name = [individuals{i} '_' individuals{j}];
        delta_identity.(field_name) = displacements;
    end
end

% Initialize struct to store deltas for emotion pairs
delta_emotion = struct();
emotions = {'neutral', 'happiness', 'sadness', 'disgust', 'fear', 'anger', 'surprise'};

% Iterate through each pair of emotions
for i = 1:length(emotions)
    for j = i+1:length(emotions)  % To avoid duplicate pairs, only compare j > i
        % Extract the vertex data for each individual 
        mesh1 = scaledCoordsMax_Elias.(emotions{i});
        mesh2 = scaledCoordsMax_Elias.(emotions{j});

        % Compute the Euclidean distances (displacements) between corresponding vertices
        displacements = sqrt(sum((mesh1 - mesh2).^2, 2));

        % Store the result in the struct with a field name like 'happiness_sadness'
        field_name = [emotions{i} '_' emotions{j}];
        delta_emotion.(field_name) = displacements;
    end
end
%% Method 1: Print delta values 

% Identity
individual_fields = fieldnames(delta_identity);
for i = 1:length(individual_fields)
    field_name = individual_fields{i};
    field_values = delta_identity.(field_name);

    % Calculate the sum of the values in the field
    field_sum = sum(field_values);

    % Print the field name and sum
    fprintf('%s: Sum = %.4f\n', field_name, field_sum);
end

% Emotion 
individual_fields = fieldnames(delta_emotion);
for i = 1:length(individual_fields)
    field_name = individual_fields{i};
    field_values = delta_emotion.(field_name);

    % Calculate the sum of the values in the field
    field_sum = sum(field_values);

    % Print the field name and sum
    fprintf('%s: Sum = %.4f\n', field_name, field_sum);
end

%% Method 1: Scatterplot 

% Define the two field names you want to compare
fieldname1 = 'Elias_Neptune';  % Replace with your actual field name
fieldname2 = 'Elias_Sophie';  % Replace with your actual field name

% Extract the values of the two fields (936 points each)
values1 = delta_identity.(fieldname1);
values2 = delta_identity.(fieldname2);

% Ensure both fields have the same number of points
if length(values1) == length(values2)
    % Create a scatter plot
    figure;
    scatter(values1, values2, 7, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');  % Example of custom dot style
    
    % Add labels and title for clarity
    xlabel(['\Delta ' strrep(fieldname1, '_', '\_')], 'Interpreter', 'tex');  
    ylabel(['\Delta ' strrep(fieldname2, '_', '\_')], 'Interpreter', 'tex');
    title([fieldname1 ' vs ' fieldname2], 'Interpreter', 'none');
    
    correlation_value = corr(values1, values2);
    
    % Display the correlation value
    x_position = min(values1) - 0.01 * range(values1);  % Adjust x position
    y_position = max(values2) - 0.00 * range(values2);  % Adjust y position
    text(x_position, y_position, ['\rho = ' num2str(correlation_value, '%.2f')], ...
        'FontSize', 15, 'Color', 'k');
    
    % Calculate the line of best fit
    coefficients = polyfit(values1, values2, 1);  % Linear fit (degree 1)
    best_fit_line = polyval(coefficients, values1);  % Evaluate the fit line
    
    % Plot the line of best fit
    hold on;  
    plot(values1, best_fit_line, 'black-', 'LineWidth', 2); 
    hold off;
else
    disp('Error: The two fields do not have the same number of points.');
end


%% Method 1: Compute delta correlations for (1) identity-identity, (2) identity-emotion, (3) emotion-emotion
identity_fields = fieldnames(delta_identity);
emotion_fields = fieldnames(delta_emotion);

% Initialize a 21x21 matrix to store the computed values
corr_id_id = zeros(21, 21);
corr_id_em = zeros(21, 21);
corr_em_em = zeros(21, 21);

% (1) identity-identity
for i = 1:length(identity_fields)
    values1 = delta_identity.(identity_fields{i});
    for j = 1:length(identity_fields)
        values2 = delta_identity.(identity_fields{j});
        if length(values1) == length(values2)
            correlation_value = corr(values1, values2);
            corr_id_id(i, j) = correlation_value;
        end
    end
end

% (2) identity-emotion
for i = 1:length(identity_fields)
    values1 = delta_identity.(identity_fields{i});
    for j = 1:length(identity_fields)
        values2 = delta_emotion.(emotion_fields{j});
        if length(values1) == length(values2)
            correlation_value = corr(values1, values2);
            corr_id_em(i, j) = correlation_value;
        end
    end
end

% (3) emotion-emotion
for i = 1:length(emotion_fields)
    values1 = delta_emotion.(emotion_fields{i});
    for j = 1:length(emotion_fields)
        values2 = delta_emotion.(emotion_fields{j});
        if length(values1) == length(values2)
            correlation_value = corr(values1, values2);
            corr_em_em(i, j) = correlation_value;
        end
    end
end

%% Method 1: Correlation plot, display all three correlation matrices
figure;
% (1) identity-identity
subplot(1, 3, 1);  
imagesc(abs(result_matrix_id_id));  % Visualize the correlation matrix
colorbar;  % Add a color bar
clim([0 1]);  % Set color axis to range from 0 to 1
title('Identity vs Identity Correlation', FontSize=13);
xlabel('Identity Fields');
ylabel('Identity Fields');
axis square;  % Make the plot square

% (2) identity-emotion
subplot(1, 3, 2);  
imagesc(abs(result_matrix_em_em));  % Visualize the correlation matrix
colorbar;
clim([0 1]); 
title('Emotion vs Emotion Correlation', FontSize=13);
xlabel('Emotion Fields');
ylabel('Emotion Fields'); % Make the plot square
axis square;  

% (3) emotion-emotion
subplot(1, 3, 3); 
imagesc(abs(result_matrix));  % Visualize the correlation matrix
colorbar; 
clim([0 1]);
title('Identity vs Emotion Correlation', FontSize=13);
xlabel('Emotion Fields');
ylabel('Identity Fields');
axis square;  

set(gcf, 'Position', [100, 100, 1200, 400]);
%% Method 2: Compute delta

delta_identity_mean_centered = struct();
delta_identity_raw = struct();
delta_identity_abs = struct();

% corr_id_id_centroids = zeros(21, 21);

% delta_identity = struct();
individuals = {'Elias', 'Neptune', 'SeojinE', 'Sophie', 'Dan', 'Sreyas', 'Younah'};

% Iterate through each pair of individuals
for i = 1:length(individuals)
    for j = i+1:length(individuals) 
        % Extract the vertex data for each individual 
        mesh1 = scaledCoordsMax.(individuals{i});
        mesh2 = scaledCoordsMax.(individuals{j});

        % Compute the delta between two meshes
        raw_displacements = (mesh1 - mesh2); % 936x3 double ??? should I abs?
        abs_displacements = abs(mesh1 - mesh2);

        delta_x = raw_displacements(:, 1); % 936x1 double indicating x-displacement
        delta_y = raw_displacements(:, 2); % 936x1 double indicating y-displacement
        delta_z = raw_displacements(:, 3); % 936x1 double indicating z-displacement

        delta_xabs = abs_displacements(:, 1); % 936x1 double indicating x-displacement
        delta_yabs = abs_displacements(:, 2); % 936x1 double indicating y-displacement
        delta_zabs = abs_displacements(:, 3); % 936x1 double indicating z-displacement

        centroid_x = mean(delta_x); % x centroid
        centroid_y = mean(delta_y); % y centroid
        centroid_z = mean(delta_z); % z centroid

        centroid_xabs = mean(delta_xabs); % x centroid
        centroid_yabs = mean(delta_yabs); % y centroid
        centroid_zabs = mean(delta_zabs); % z centroid

        % Mean-center x, y, z deltas
        centered_x = delta_x - centroid_x;
        centered_y = delta_y - centroid_y;
        centered_z = delta_z - centroid_z;

        centered_xabs = delta_xabs - centroid_xabs;
        centered_yabs = delta_yabs - centroid_yabs;
        centered_zabs = delta_zabs - centroid_zabs;
        
        % Concatenate the three deltas
        combined_deltas = [delta_x, delta_y, delta_z];
        centered_deltas = [centered_x, centered_y, centered_z]; % 936x3 double 
        centered_deltasabs = [centered_xabs, centered_yabs, centered_zabs]; % 936x3 double 
        % Store mean-centered delta
        field_name = [individuals{i} '_' individuals{j}];
        delta_identity_raw.(field_name) = combined_deltas;
        delta_identity_mean_centered.(field_name) = centered_deltas;
        delta_identity_abs.(field_name) = centered_deltasabs;
        % Correlation between non-mean-centered deltas
        % corr_id_id_centroids(i, j) = mean(corr([delta_x, delta_y, delta_z]), 'all');
    end
end
% 
% % Initialize struct to store deltas for emotion pairs
% delta_emotion = struct();
% emotions = {'neutral', 'happiness', 'sadness', 'disgust', 'fear', 'anger', 'surprise'};
% 
% % Iterate through each pair of emotions
% for i = 1:length(emotions)
%     for j = i+1:length(emotions)  % To avoid duplicate pairs, only compare j > i
%         % Extract the vertex data for each individual 
%         mesh1 = scaledCoordsMax_Elias.(emotions{i});
%         mesh2 = scaledCoordsMax_Elias.(emotions{j});
% 
%         % Compute the Euclidean distances (displacements) between corresponding vertices
%         displacements = sqrt(sum((mesh1 - mesh2).^2, 2));
% 
%         % Store the result in the struct with a field name like 'happiness_sadness'
%         field_name = [emotions{i} '_' emotions{j}];
%         delta_emotion.(field_name) = displacements;
%     end
% end
%% Method 2: Scatterplot 

% Define the two field names you want to compare
fieldname1 = 'Elias_Neptune';  % Replace with your actual field name
fieldname2 = 'Neptune_SeojinE';  % Replace with your actual field name

% Extract the values of the two fields (936 points each)
values1 = delta_identity_mean_centered.(fieldname1)(:);
values2 = delta_identity_mean_centered.(fieldname2)(:);

values1_raw = delta_identity_raw.(fieldname1)(:);
values2_raw = delta_identity_raw.(fieldname2)(:);

values1_abs = delta_identity_abs.(fieldname1)(:);
values2_abs = delta_identity_abs.(fieldname2)(:);
% Ensure both fields have the same number of points
if length(values1) == length(values2)
    % A. Create a scatter plot for mean-centered values
    figure;
    hold on;
    
    scatter(values1(1:936), values2(1:936), 7, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');  
    scatter(values1(937:1872), values2(937:1872), 7, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g');  
    scatter(values1(1873:2808), values2(1873:2808), 7, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');
    
    % Add labels and title for clarity
    xlabel(['\Delta ' strrep(fieldname1, '_', '\_')], 'Interpreter', 'tex');  
    ylabel(['\Delta ' strrep(fieldname2, '_', '\_')], 'Interpreter', 'tex');
    title([fieldname1 ' vs ' fieldname2 ' (Mean-Centered)'], 'Interpreter', 'none');
    % Display correlation value
    correlation_value = corr(values1, values2);
    x_position = min(values1) - 0.05 * range(values1);  % Adjust x position
    y_position = max(values2) - 0.15 * range(values2);  % Adjust y position
    text(x_position, y_position, ['\rho = ' num2str(correlation_value, '%.2f')], ...
        'FontSize', 15, 'Color', 'k');
    % Calculate the line of best fit
    coefficients = polyfit(values1, values2, 1);  % Linear fit (degree 1)
    best_fit_line = polyval(coefficients, values1);  % Evaluate the fit line
    plot(values1, best_fit_line, 'black-', 'LineWidth', 2); 

    hold on;  
    
    % Add a legend to differentiate the x, y, z coordinates
    legend('X Coordinates', 'Y Coordinates', 'Z Coordinates', 'Location', 'best');
    hold off;

    % B. Create a scatter plot for abs mean-centered values
    figure;
    hold on;
    
    scatter(values1_abs(1:936), values2_abs(1:936), 7, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');  
    scatter(values1_abs(937:1872), values2_abs(937:1872), 7, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g');  
    scatter(values1_abs(1873:2808), values2_abs(1873:2808), 7, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');
    
    % Add labels and title for clarity
    xlabel(['\Delta ' strrep(fieldname1, '_', '\_')], 'Interpreter', 'tex');  
    ylabel(['\Delta ' strrep(fieldname2, '_', '\_')], 'Interpreter', 'tex');
    title([fieldname1 ' vs ' fieldname2 ' (Abs. Mean-Centered)'], 'Interpreter', 'none');
    % Display correlation value
    correlation_value = corr(values1_abs, values2_abs);
    x_position = min(values1_abs) - 0.05 * range(values1_abs);  % Adjust x position
    y_position = max(values2_abs) - 0.15 * range(values2_abs);  % Adjust y position
    text(x_position, y_position, ['\rho = ' num2str(correlation_value, '%.2f')], ...
        'FontSize', 15, 'Color', 'k');
    % Calculate the line of best fit
    coefficients = polyfit(values1_abs, values2_abs, 1);  % Linear fit (degree 1)
    best_fit_line = polyval(coefficients, values1_abs);  % Evaluate the fit line
    plot(values1_abs, best_fit_line, 'black-', 'LineWidth', 2); 

    hold on;  
    
    % Add a legend to differentiate the x, y, z coordinates
    legend('X Coordinates', 'Y Coordinates', 'Z Coordinates', 'Location', 'best');
    hold off;
else
    disp('Error: The two fields do not have the same number of points.');
end

%% Method 2: Compute delta correlations for (1) identity-identity, (2) identity-emotion, (3) emotion-emotion
identity_fields = fieldnames(delta_identity);
emotion_fields = fieldnames(delta_emotion);

% Initialize a 21x21 matrix to store the computed values
corr_id_id_mean_centered = zeros(21, 21);
% corr_id_em = zeros(21, 21);
% corr_em_em = zeros(21, 21);

% (1) identity-identity
for i = 1:length(identity_fields)
    values1 = delta_identity_mean_centered.(identity_fields{i});
    for j = 1:length(identity_fields)
        values2 = delta_identity_mean_centered.(identity_fields{j});
        if length(values1) == length(values2)
            correlation_value = mean(corr(values1, values2), 'all');
            corr_id_id_mean_centered(i, j) = correlation_value;
        end
    end
end

% % (2) identity-emotion
% for i = 1:length(identity_fields)
%     values1 = delta_identity.(identity_fields{i});
%     for j = 1:length(identity_fields)
%         values2 = delta_emotion.(emotion_fields{j});
%         if length(values1) == length(values2)
%             correlation_value = corr(values1, values2);
%             corr_id_em(i, j) = correlation_value;
%         end
%     end
% end
% 
% % (3) emotion-emotion
% for i = 1:length(emotion_fields)
%     values1 = delta_emotion.(emotion_fields{i});
%     for j = 1:length(emotion_fields)
%         values2 = delta_emotion.(emotion_fields{j});
%         if length(values1) == length(values2)
%             correlation_value = corr(values1, values2);
%             corr_em_em(i, j) = correlation_value;
%         end
%     end
% end
