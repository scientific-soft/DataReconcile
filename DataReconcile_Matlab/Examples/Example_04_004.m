function Example_04_004
% Measured data for x and y (each column is a measurement)
msrd_data = [1.0, 1.1, 0.9;  % Measurements for x
             0.9, 1.0, 1.1]; % Measurements for y

% Error parameters (uncertainties in x and y)
error_params = [0.1; 0.1]; % [error_x; error_y]

% Parameters for the dependency function
params_ = [1; 0; 1; 0]; % [a1; b1; a2; b2]

% Define the dependency function with two linear equations
function F = Func(vals, params)
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

% Call the DRparamEq function
reconciled = DRparamEq(@Func, msrd_data, error_params, params_);

% Display the reconciled values
disp('Reconciled Values:');
disp(reconciled);



end

