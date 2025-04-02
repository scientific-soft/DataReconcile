function Example_03_003

% Define the dependency function
function F = Func(vals, params)
    % vals: [x; y]
    % params: [a; b]
    x = vals(1);
    y = vals(2);
    a = params(1);
    b = params(2);
    
    F = y - (a * x + b); % Linear dependency: y = a*x + b
end

% Measured data for x and y (each column is a measurement)
msrd_data = [1.0, 1.1, 0.9;  % Measurements for x
             0.9, 1.0, 1.0]; % Measurements for y

% Error parameters (uncertainties in x and y)
error_params = [0.1; 0.1]; % [error_x; error_y]

% Parameters for the dependency function
params_ = [1; 0]; % [a; b] for y = a*x + b


% Call the DRparamEq function
reconciled = DRparamEq(@Func, msrd_data, error_params, params_);

% Display the reconciled values
disp('Reconciled Values:');
disp(reconciled);

end

