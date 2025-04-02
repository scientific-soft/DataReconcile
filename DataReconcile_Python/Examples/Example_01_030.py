import numpy as np
#from estAccuracyIncreaseByDR import estAccuracyIncreaseByDR
from estAccuracyIncreaseByDR import *

# Define the dependency function
def dep_func(vals, params):
    # vals: [x; y]
    # params: [a; b]
    x = vals[0]
    y = vals[1]
    a = params[0]
    b = params[1]
    
    return np.array([y - (a * x + b)])  # Linear dependency: y = a*x + b

# Values for reconciliation
vals_for_reconc = np.array([1.0, 2.0])  # [x; y]

# Parameters for the dependency function
params_ = np.array([2.0, 0.0])  # [a; b]

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
