function Example_15_015

% Measured data for x and y (each column is a measurement)
msrd_data = [1.0, 1.1, 0.9;  % Measurements for x
             2.0, 2.2, 1.8]; % Measurements for y

% Model parameters for the dependency function
model_params = [2; 0; 1; 1]; % [a1; b1; a2; b2]

% Prior variances (uncertainties in x and y)
prior_vars = [0.1; 0.2]; % [variance_x; variance_y]

% Bandwidths for kernel-based non-parametric reconciliation
bandwidths = 0.5; % Single bandwidth for both variables

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

% Call the DRnonparamEqRobust function
reconciled_with_Kernels = DRnonparamEqRobust(@Func, msrd_data, model_params, prior_vars, bandwidths);

% Display the reconciled values
disp('Reconciled Values with Kernels (Robust):');
disp(reconciled_with_Kernels);


end

