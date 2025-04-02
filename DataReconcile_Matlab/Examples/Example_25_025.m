function Example_25_025

% System of equations
function F = eqFunc(vals, params)
    % vals: [x; y]
    % params: [a1; b1; a2; b2]
    x = vals(1);
    y = vals(2);
    a1 = params(1);
    b1 = params(2);
    a2 = params(3);
    b2 = params(4);
    
    % System of equations
    F = [y - (a1 * x + b1);  % Equation 1: y = a1*x + b1
         y - (a2 * x + b2)]; % Equation 2: y = a2*x + b2
end

% System of inequalities
function G = ineqFunc(vals, params)
    % vals: [x; y]
    % params: [c1; d1; c2; d2]
    x = vals(1);
    y = vals(2);
    c1 = params(1);
    d1 = params(2);
    c2 = params(3);
    d2 = params(4);
    
    % System of inequalities
    G = [(c1 * x + d1) - y;  % Inequality 1: y <= c1*x + d1
         y - (c2 * x + d2)]; % Inequality 2: y >= c2*x + d2
end

% Measured data for x and y (each column is a measurement)
msrd_data = [1.0, 1.1, 0.9;  % Measurements for x
             2.0, 2.2, 1.8]; % Measurements for y

% Error parameters (uncertainties in x and y)
error_params = [0.1; 0.2]; % [error_x; error_y]

% Alpha parameters for semi-parametric reconciliation
alpha_params = [0.5; 0.1; 0.4; 0.2]; % [As1; Ex1; As2; Ex2] for calculMeanAlpha function

% Model parameters for the dependency functions
model_params = [2; 0; 1; 1; 1; 0; 0.5; 1]; % [a1; b1; a2; b2; c1; d1; c2; d2]

% Call the DRsemiparamEqIneqRobust function
reconciled_with_AlphaGramCh = DRsemiparamEqIneqRobust(@eqFunc, @ineqFunc, msrd_data, error_params, alpha_params, model_params);

% Display the reconciled values
disp('Reconciled Values with Alpha Gram-Charlier (Robust):');
disp(reconciled_with_AlphaGramCh)


end

