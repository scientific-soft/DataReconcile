import numpy as np
from DRparamEq import *

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
    res = [mu[0]*params[0] - mu[1]]
    return res

# Define measured data (msrd_data)
msrd_data = np.array([
    [2.0, 2.1, 1.9],
    [2.0, 2.1, 1.9]])  # Example measured data

# Define error parameters (error_params)
error_params = np.array([0.1, 0.1])  # Example error parameters

# Define constant model parameters (params_)
params_ = np.array([2.0])  # Example constant parameters

# Call the DRparamEq function to reconcile the measured data
reconciled_data = DRparamEq(dependency_model, msrd_data, error_params, params_)

# Print the reconciled data
print("Reconciled Data:", reconciled_data)