function Example_26_026

% Linear equation
function F = eqFunc(vals, params)
    % vals: [x; y]
    % params: [a; b]
    x = vals(1);
    y = vals(2);
    a = params(1);
    b = params(2);
    
    F = y - (a * x + b); % Linear equation: y = a*x + b
end

% Linear inequality
function G = ineqFunc(vals, params)
    % vals: [x; y]
    % params: [c; d]
    x = vals(1);
    y = vals(2);
    c = params(1);
    d = params(2);
    
    G = (c * x + d) - y; % Inequality: y <= c*x + d
end

% Measured data for x and y (each column is a measurement)
msrd_data = [1.0, 1.1, 0.9;  % Measurements for x
             2.0, 2.2, 1.8]; % Measurements for y

% Model parameters for the dependency functions
model_params = [2; 0; 1; 1]; % [a; b; c; d]

% Prior variances (uncertainties in x and y)
prior_vars = [0.1; 0.2]; % [variance_x; variance_y]

% Bandwidths for kernel-based non-parametric reconciliation
bandwidths = 0.5; % Single bandwidth for both variables

% Call the DRnonparamEqIneqRobust function
reconciled_with_Kernels = DRnonparamEqIneqRobust(@eqFunc, @ineqFunc, msrd_data, model_params, prior_vars, bandwidths);

% Display the reconciled values
disp('Reconciled Values with Kernels (Robust):');
disp(reconciled_with_Kernels);

end

