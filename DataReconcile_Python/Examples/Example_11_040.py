import numpy as np
from DRparamEqRobust import *

# Define a dependency model function
def dependency_model(mu, params):
    """
    Example dependency model function.
    This function should describe the relationship between measured quantities.
    
    Parameters:
        mu (ndarray): Array of unknown parameters.
        params (ndarray): Constant model parameters.
    
    Returns:
        ndarray: Model output.
    """
    # Example: A simple linear model
    res = [mu[0]*params[0] - mu[1],
           mu[0]  - mu[1]/params[0]]
    return res

# Define measured data (msrd_data)
msrd_data = np.array([
    [1.0, 1.1, 0.9],
    [1.0, 1.1, 0.9]])  # Example measured data

# Define error parameters (error_params)
error_params = np.array([0.1, 0.1])  # Example error parameters

# Define constant model parameters (params_)
params_ = np.array([2.0])  # Example constant parameters

# Call the GaussDataReconcilRobust function to reconcile the measured data
reconciled_data = DRparamEqRobust(dependency_model, msrd_data, error_params, params_)

# Print the reconciled data
print("Reconciled Data:", reconciled_data)
