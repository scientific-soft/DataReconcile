function Example_07_007

% Measured data for x and y (each column is a measurement)
msrd_data = [1.0, 1.1, 0.9;  % Measurements for x
             2.0, 2.2, 1.8]; % Measurements for y

% Error parameters (uncertainties in x and y)
error_params = [0.1; 0.2]; % [error_x; error_y]

% Alpha parameters for semi-parametric reconciliation
alpha_params = [0.5; 0.0; 0.0; 0.2]; % [As1; Ex1; As2; Ex2] for calculMeanAlpha function

% True parameters for the dependency function
true_params = [2; 0; 1; 1]; % [a1; b1; a2; b2]

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

% Call the DRsemiparamEq function
reconciled_with_AlphaGramCh = DRsemiparamEq(@Func, msrd_data, error_params, alpha_params, true_params);

% Display the reconciled values
disp('Reconciled Values with Alpha Gram-Charlier:');
disp(reconciled_with_AlphaGramCh);


end

