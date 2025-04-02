function Example_02_002

% Define the dependency function
function F = dep_func(vals, params)
    % vals: [x; y]
    % params: [a1; b1; a2; b2]
    x = vals(1);
    y = vals(2);
    a1 = params(1);
    b1 = params(2);
    a2 = params(3);
    b2 = params(4);
    
    % System of equations
    F = [y - (a1 * x + b1);  % First equation
         y - (a2 * x + b2)]; % Second equation
end

% Values for reconciliation
vals_for_reconc = [1; 1]; % [x; y]

% Parameters for the dependency function
params = [1; 1; 1; 1]; % [a1; b1; a2; b2]

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

