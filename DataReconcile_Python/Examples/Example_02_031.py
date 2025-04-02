import numpy as np
from estAccuracyIncreaseByDR import *

# Define the dependency function with two equations
def dep_func(vals, params):
    # vals: [x; y]
    # params: [a1; b1; a2; b2]
    x = vals[0]
    y = vals[1]
    a1 = params[0]
    b1 = params[1]
    a2 = params[2]
    b2 = params[3]
    
    # System of equations
    return np.array([y - (a1 * x + b1),  # Equation 1: y = a1*x + b1
                     y - (a2 * x + b2)]) # Equation 2: y = a2*x + b2

# Values for reconciliation
vals_for_reconc = np.array([1.0, 2.0])  # [x; y]

# Parameters for the dependency function
params_ = np.array([2.0, 0.0, 1.0, 1.0])  # [a1; b1; a2; b2]

# Covariance matrix (uncertainties in x and y)
cov_matrix = np.array([[0.1, 0.05],  # Covariance matrix for x and y
                       [0.05, 0.2]])

# Choose the mode for reconciliation
ChosenMode = 'LS'  # or 'WLS'

# Call the function
accuracy_increase_ratio, variances_of_DR_result = estAccuracyIncreaseByDR(dep_func, ChosenMode, vals_for_reconc, params_, cov_matrix)

# Display the results
print('Accuracy Increase Ratio:')
print(accuracy_increase_ratio)

print('Variances of DR Result:')
print(variances_of_DR_result)
