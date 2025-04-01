import numpy as np
from numpy import imag
from scipy.optimize import fsolve

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

def alpha_system_to_solve(mu_and_l, eqies_model, ineqies_model, msrd_data, vars_, alpha_params, params_, xNum, n, eqNum, ineqNum):
#def alpha_system_to_solve(mu_and_l, args):
    """
    eqies_model = args[0]
    msrd_data = args[1]
    vars_ = args[2]
    alpha_params = args[3]
    params_ = args[4]
    """
    """ args=(eqies_model, msrd_data, error_params, alpha_params, params_)
    Solves for the Alpha system to find zeros.

    Parameters:
        mu_and_l (ndarray): Array of unknown parameters (mu and lambda).
        eqies_model (function): Dependency model function.
        msrd_data (ndarray): Measured data (n x xNum array).
        vars_ (ndarray): Error parameters, should not contain zeros.
        alpha_params (ndarray or list): Alpha parameters.
        params_ (ndarray or list): Constant dependency model parameters.

    Returns:
        ndarray: A vector of values that need to be zero (mustbezeros).
    """
            
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
            df_dx_eqies = imag(eqies_model(mu_and_l[:xNum] + imagx, params_)) / imag(imagx[j])
            df_dx_ineqies = imag(ineqies_model(mu_and_l[:xNum] + imagx, params_)) / imag(imagx[j])

            df_dx = np.hstack((df_dx_eqies, df_dx_ineqies))
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
                              + np.sum(mu_and_l[xNum:(xNum+eqNum+ineqNum)] * df_dx) / n)

    # Evaluate the model residuals
    eqies_model_res = eqies_model(mu_and_l[:xNum], params_)
    mustbezeros = np.append(mustbezeros, eqies_model_res)

    ineqies_model_res = ineqies_model(mu_and_l[:xNum], params_)
    mustbezeros = np.append(mustbezeros, ineqies_model_res)

    mustbezeros = np.append(mustbezeros, ineqies_model_res*mu_and_l[(xNum+eqNum):(xNum+eqNum+ineqNum)])

    return np.real(mustbezeros)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def DRsemiparamEqIneq(eqies_model, ineqies_model, msrd_data, error_params, alpha_params, params_):
    """
    Reconciles measured data using the Alpha Gram-Charlier method.

    Parameters:
        eqies_model (function): Function describing relationships between measured quantities.
                         It can return either a single equation or a system of equations.
        msrd_data (ndarray): Vector of measured data.
        error_params (ndarray): A priori error variances/limits.
        alpha_params (list or ndarray): Alpha function parameters (As, Ex) or (As1, Ex1, ..., AsN, ExN).
        params_ (ndarray): Known model parameters.

    Returns:
        ndarray: Reconciled measured data after applying the Alpha Gram-Charlier method.
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

    data_init_points = np.mean(msrd_data, axis=1)
    eqNum = len(eqies_model(data_init_points, params_))
    ineqNum = len(ineqies_model(data_init_points, params_))

    # Step 1: Compute l (a zero vector of the same length as depend_func's output)
    l_eqies = np.zeros(len(eqies_model(data_init_points, params_)))
    l_ineqies = np.zeros(len(ineqies_model(data_init_points, params_)))
    l_ineqies_muly_by_l = np.zeros(len(ineqies_model(data_init_points, params_)))

    l = np.hstack((l_eqies, l_ineqies, l_ineqies_muly_by_l))
    
    # Step 2: Extend the initial guess (mu_and_l_start) by appending l to msrd_data
    mu_and_l_start = np.hstack((data_init_points, l))
    
    # Solve the system of equations using fsolve
    mu_and_l_res = fsolve(alpha_system_to_solve, mu_and_l_start, args=(eqies_model, ineqies_model, msrd_data, error_params, alpha_params, params_, xNum, n, eqNum, ineqNum))
    
    # Step 4: Extract the reconciled measured data (excluding 'l' values)
    reconciled_with_alpha_gram_ch = mu_and_l_res[:len(msrd_data)]
    
    return reconciled_with_alpha_gram_ch

