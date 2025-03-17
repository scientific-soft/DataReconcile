import numpy as np
from numpy import imag
from scipy.optimize import fsolve
from scipy.stats import median_abs_deviation, iqr

def calcul_mean_alpha(mj, msrd_data, alpha_params, sj):
    """
    Calculates the mean alpha value based on the given inputs.

    Parameters:
        mj (float or ndarray): Mean value.
        msrd_data (ndarray): Measured data array.
        alpha_params (list or tuple): Alpha parameters [As, Ex].
        sj (float or ndarray): Standard deviation or scaling factor.

    Returns:
        float: The calculated mean alpha value.
    """
    # Extract alpha parameters
    As = alpha_params[0]
    Ex = alpha_params[1]
    
    # Compute tji
    tji = (msrd_data - mj) / sj
    
    # Compute d_alpha_d_mu
    d_alpha_d_mu = (Ex - 3) * (0.5 * tji - (1 / 6) * tji**3) + As * (0.5 - 0.5 * tji**2)
    
    # Compute alpha_plus_one
    alpha_plus_one = ((Ex - 3) * ((1 / 24) * tji**4 + 1 / 8 - 0.25 * tji**2) 
                      + As * ((1 / 6) * tji**3 - 0.5 * tji) + 1)
    
    # Compute y
    y = d_alpha_d_mu / alpha_plus_one
    
    # Compute alpha_mean
    alpha_mean = (1 / sj) * np.mean(y)
    
    return alpha_mean

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def alpha_system_to_solve(mu_and_l, depend_func, msrd_data, vars_, alpha_params, params_):
#def alpha_system_to_solve(mu_and_l, args):
    """
    depend_func = args[0]
    msrd_data = args[1]
    vars_ = args[2]
    alpha_params = args[3]
    params_ = args[4]
    """
    """ args=(depend_func, msrd_data, error_params, alpha_params, params_)
    Solves for the Alpha system to find zeros.

    Parameters:
        mu_and_l (ndarray): Array of unknown parameters (mu and lambda).
        depend_func (function): Dependency model function.
        msrd_data (ndarray): Measured data (n x xNum array).
        vars_ (ndarray): Error parameters, should not contain zeros.
        alpha_params (ndarray or list): Alpha parameters.
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

            if len(alpha_params) == 2:
                # Simplified alpha_params case
                mean_alpha = calcul_mean_alpha(mu_and_l[j], msrd_data[j,:], alpha_params, vars_[j])
            else:
                # Case where alpha_params depend on j
                alpha_slice = alpha_params[2 * j:2 * j + 2]
                mean_alpha = calcul_mean_alpha(mu_and_l[j], msrd_data[j,:], alpha_slice, vars_[j])

            # Update mustbezeros
            mustbezeros[j] = (mu_and_l[j] / vars_[j]
                              - np.mean(msrd_data[j,:]) / vars_[j]
                              - mean_alpha
                              + np.sum(mu_and_l[xNum:] * df_dx) / n)

    # Evaluate the model residuals
    model_res = depend_func(mu_and_l[:xNum], params_)
    mustbezeros = np.append(mustbezeros, model_res)

    return np.real(mustbezeros)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def DRsemiparamEqRobust(depend_func, msrd_data, error_params, alpha_params, params_):
    """
    Reconciles measured data using the Alpha Gram-Charlier method.

    Parameters:
        depend_func (function): Function describing relationships between measured quantities.
                         It can return either a single equation or a system of equations.
        msrd_data (ndarray): Vector of measured data.
        error_params (ndarray): A priori error variances/limits.
        alpha_params (list or ndarray): Alpha function parameters (As, Ex) or (As1, Ex1, ..., AsN, ExN).
        params_ (ndarray): Known model parameters.

    Returns:
        ndarray: Reconciled measured data after applying the Alpha Gram-Charlier method.
    """

    # Compute MAD and adjust error_params
    mad_values = median_abs_deviation(msrd_data, axis=1, scale=1)
    error_params = mad_values * 1.483

    data_init_points = np.mean(msrd_data, axis=1)

    # Step 1: Compute l (a zero vector of the same length as depend_func's output)
    l = np.zeros(len(depend_func(data_init_points, params_)))
    
    # Step 2: Extend the initial guess (mu_and_l_start) by appending l to msrd_data
    mu_and_l_start = np.hstack((data_init_points, l))
    
    # Solve the system of equations using fsolve
    mu_and_l_res = fsolve(alpha_system_to_solve, mu_and_l_start, args=(depend_func, msrd_data, error_params, alpha_params, params_))
    
    # Step 4: Extract the reconciled measured data (excluding 'l' values)
    reconciled_with_alpha_gram_ch = mu_and_l_res[:len(msrd_data)]
    
    return reconciled_with_alpha_gram_ch


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
