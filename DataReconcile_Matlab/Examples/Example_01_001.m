function Example_01_001
% Values for reconciliation
vals_for_reconc = [1; 1]; % [x; y]

% Parameters for the dependency function
params = [1; 0]; % [a; b]

% Covariance matrix
cov_matrix = [0.1, 0.0; 0.0, 0.1]; % Covariance matrix for x and y

% Choose the mode
ChosenMode = 'LS'; % or 'WLS'

% Call the function
[accuracy_increase_ratio, variances_of_DR_result] = ...
    estAccuracyIncreaseByDR(@dep_func, ChosenMode, vals_for_reconc, params, cov_matrix);

% Display the results
disp('Accuracy Increase Ratio:');
disp(accuracy_increase_ratio);

disp('Variances of DR Result:');
disp(variances_of_DR_result);

end

% Define the dependency function
function F = dep_func(vals, params)
    x = vals(1);
    y = vals(2);
    a = params(1);
    b = params(2);
    
    F = y - (a * x + b); % This represents the equation y = a*x + b
end