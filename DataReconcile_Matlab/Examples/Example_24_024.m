function Example_24_024

% Measured data for x and y (each column is a measurement)
msrd_data = [1.0, 1.1, 0.9;  % Measurements for x
             2.0, 2.2, 1.8]; % Measurements for y

% Error parameters (uncertainties in x and y)
error_params = [0.1; 0.2]; % [error_x; error_y]

% Alpha parameters for semi-parametric reconciliation
alpha_params = [0.5; 0.1; 0.4; 0.2]; % [As1; Ex1; As2; Ex2] for calculMeanAlpha function

% Model parameters for the dependency functions
model_params = [2; 0; 1; 1]; % [a; b; c; d]

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

% Call the DRsemiparamEqIneqRobust function
reconciled_with_AlphaGramCh = DRsemiparamEqIneqRobust(@eqFunc, @ineqFunc, msrd_data, error_params, alpha_params, model_params);

% Display the reconciled values
disp('Reconciled Values with Alpha Gram-Charlier (Robust):');
disp(reconciled_with_AlphaGramCh);

end

