import numpy as np
from numpy import imag
from scipy.optimize import fsolve

def gauss_system_to_solve(mu_and_l, eqies_model, ineqies_model, msrd_data, vars_, params_, xNum, n, eqNum, ineqNum):
    """ args=(eqies_model, msrd_data, error_params, params_)
    Solves for the gauss system to find zeros.

    Parameters:
        mu_and_l (ndarray): Array of unknown parameters (mu and lambda).
        eqies_model (function): Dependency model function.
        msrd_data (ndarray): Measured data (n x xNum array).
        vars_ (ndarray): Error parameters, should not contain zeros.
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
            df_dx_ineqies = imag(eqies_model(mu_and_l[:xNum] + imagx, params_)) / imag(imagx[j])



            df_dx = np.hstack((df_dx_eqies, df_dx_ineqies))
            # Update mustbezeros
            mustbezeros[j] = (mu_and_l[j] / vars_[j]
                              - np.mean(msrd_data[j,:]) / vars_[j]
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

def DRparamEqIneq(eqies_model, ineqies_model, msrd_data, error_params, params_):
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

    # Step 1: Compute l (a zero vector of the same length as eqies_model's output)
    l_eqies = np.zeros(len(eqies_model(data_init_points, params_)))
    l_ineqies = np.zeros(len(ineqies_model(data_init_points, params_)))
    l_ineqies_muly_by_l = np.zeros(len(ineqies_model(data_init_points, params_)))

    l = np.hstack((l_eqies, l_ineqies, l_ineqies_muly_by_l))

    # Step 2: Extend the initial guess (mu_and_l_start) by appending l to msrd_data
    mu_and_l_start = np.hstack((data_init_points, l))
    
    # Solve the system of equations using fsolve
    mu_and_l_res = fsolve(gauss_system_to_solve, mu_and_l_start, args=(eqies_model, ineqies_model, msrd_data, error_params, params_, xNum, n, eqNum, ineqNum))
    
    # Step 4: Extract the reconciled measured data (excluding 'l' values)
    reconciled_ = mu_and_l_res[:len(data_init_points)]
    
    return reconciled_

   
