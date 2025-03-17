import numpy as np
from numpy import imag
from scipy.optimize import fsolve
from scipy.stats import median_abs_deviation

def gauss_system_to_solve(mu_and_l, depend_func, msrd_data, vars_, params_):
    """ args=(depend_func, msrd_data, error_params, params_)
    Solves for the gauss system to find zeros.

    Parameters:
        mu_and_l (ndarray): Array of unknown parameters (mu and lambda).
        depend_func (function): Dependency model function.
        msrd_data (ndarray): Measured data (n x xNum array).
        vars_ (ndarray): Error parameters, should not contain zeros.
        params_ (ndarray or list): Constant dependency model parameters.

    Returns:
        ndarray: A vector of values that need to be zero (mustbezeros).
    """
            
    if msrd_data.ndim == 1:
        xNum = msrd_data.size
        n = 1
    elif msrd_data.ndim == 2:
        xNum = msrd_data.shape[0]
        n = msrd_data.shape[1]
    else:
        raise ValueError(
                "Only 2-dimensional arrays msrd_data are supported.")
            
    mustbezeros = np.zeros(xNum, dtype=np.complex128)  # Preallocate zeros

    for j in range(xNum):
        if vars_[j] == 0:
            raise ValueError(
                "None of the error parameters should be 0. Note that all constant "
                "dependency model parameters need to be in the "
                "'params_' array.")
        else:
            # Imaginary perturbation for numerical derivative
            imagx = np.zeros(xNum, dtype=np.complex128)
            imagx[j] = 1j * (mu_and_l[j] * 10 ** (-100) + 10 ** (-101))
            
            # Numerical derivative approximation using imaginary perturbation
            df_dx = imag(depend_func(mu_and_l[:xNum] + imagx, params_)) / imag(imagx[j])

            # Update mustbezeros
            mustbezeros[j] = (mu_and_l[j] / vars_[j]
                              - np.mean(msrd_data[j,:]) / vars_[j]
                              + np.sum(mu_and_l[xNum:] * df_dx) / n)

    # Evaluate the model residuals
    model_res = depend_func(mu_and_l[:xNum], params_)
    mustbezeros = np.append(mustbezeros, model_res)

    return np.real(mustbezeros)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def DRparamEqRobust(depend_func, msrd_data, error_params, params_):
    """
    Reconciles measured data

    Parameters:
        depend_func (function): Function describing relationships between measured quantities.
                         It can return either a single equation or a system of equations.
        msrd_data (ndarray): Vector of measured data.
        error_params (ndarray): A priori error variances/limits.
        params_ (ndarray): Known  model parameters.

    Returns:
        ndarray: Reconciled measured data after applying the Alpha Gram-Charlier method.
    """

    mad_values = median_abs_deviation(msrd_data, axis=1, scale=1)
    error_params = mad_values * 1.483


    data_init_points = np.mean(msrd_data, axis=1)
    # Step 1: Compute l (a zero vector of the same length as depend_func's output)
    l = np.zeros(len(depend_func(data_init_points, params_)))
    
    # Step 2: Extend the initial guess (mu_and_l_start) by appending l to msrd_data
    mu_and_l_start = np.hstack((data_init_points, l))
    
    # Solve the system of equations using fsolve
    mu_and_l_res = fsolve(gauss_system_to_solve, mu_and_l_start, args=(depend_func, msrd_data, error_params, params_))
    
    # Step 4: Extract the reconciled measured data (excluding 'l' values)
    reconciled_ = mu_and_l_res[:len(msrd_data)]
    
    return reconciled_

