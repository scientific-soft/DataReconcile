import numpy as np
from numpy import imag
from scipy.optimize import fsolve

def kernel(x):
    """
    Computes the kernel value for the input x using a Gaussian kernel.
    Parameters:
        x (float or ndarray): Input value(s) for the kernel function.

    Returns:
        float or ndarray: Kernel value(s) computed as exp(-x^2 / 2) / sqrt(2 * pi).
    """
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def kernels_system_to_solve(mu_and_l, eqies_model, ineqies_model, msrd_data, model_params, bandwidths, msrd_vars, xNum, n, eqNum, ineqNum):
    """
    Solves a system of equations using kernel-based estimation.

    Parameters:
        mu_and_l (ndarray): Array of unknown parameters (mu and lambda).
        eqies_model, ineqies_model (function): Dependency model function.
        msrd_data (ndarray): Measured data (n x xNum array).
        model_params (ndarray or list): model parameters.
        bandwidths (ndarray or float): Bandwidth parameter(s).
        msrd_vars (ndarray): Variance of measured data for each dimension.

    Returns:
        ndarray: A vector of values that need to be zero (mustbezeros).
    """
    mustbezeros = np.zeros(xNum, dtype=np.complex128)

    # Handle scalar bandwidth case
    if bandwidths.size == 1:
        bandwidths_ = np.full(xNum, bandwidths)
    else:
        bandwidths_ = bandwidths

    for j in range(xNum):
        mean_msrd = np.mean(msrd_data[j, :])
        s_msrd = msrd_vars[j]
        res = np.zeros(n)

        for i in range(n):
            # Compute kernel weights
            denumerator = kernel(((msrd_data[j, :] - mean_msrd) / s_msrd 
                                  - (msrd_data[j, i] - mu_and_l[j]) / s_msrd) / bandwidths_[j])
                
            numerator = np.sum(msrd_data[j, :] * denumerator)
            if np.sum(denumerator) == 0:
                res[i] = 0
            else:
                res[i] = numerator / np.sum(denumerator)

        mean_via_kerns = np.mean(res)

        # Numerical derivative using small imaginary perturbation
        imagx = np.zeros(xNum, dtype=np.complex128)
        imagx[j] = 1j * (mu_and_l[j] * 10 ** (-100) + 10 ** (-101))

        df_dx_eqies = imag(eqies_model(mu_and_l[:xNum] + imagx, model_params)) / imag(imagx[j])
        df_dx_ineqies = imag(ineqies_model(mu_and_l[:xNum] + imagx, model_params)) / imag(imagx[j])

        df_dx = np.hstack((df_dx_eqies, df_dx_ineqies))

        # Update mustbezeros
        mustbezeros[j] = (mean_msrd - mean_msrd - mean_via_kerns + mu_and_l[j]
                          + bandwidths_[j]**2 * (1 / n) * s_msrd**2 * np.sum(df_dx * mu_and_l[xNum:(xNum+eqNum+ineqNum)]))

    # Evaluate the model residuals
    eqies_model_res = eqies_model(mu_and_l[:xNum], model_params)
    mustbezeros = np.append(mustbezeros, eqies_model_res)

    ineqies_model_res = ineqies_model(mu_and_l[:xNum], model_params)
    mustbezeros = np.append(mustbezeros, ineqies_model_res)

    mustbezeros = np.append(mustbezeros, ineqies_model_res*mu_and_l[(xNum+eqNum):(xNum+eqNum+ineqNum)])

    return np.real(mustbezeros)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def DRnonparamEqIneq(eqies_model, ineqies_model, msrd_data, model_params, prior_vars, bandwidths):
    """
    Reconciles measured data using kernel-based estimation.

    Parameters:
        eqies_model, ineqies_model (functions): Functions describing dependencies between reconciled quantities.
        msrd_data (ndarray): Vector of measured results (n x xNum array).
        model_params (ndarray): Known model parameters.
        prior_vars (ndarray): A priori uncertainty parameters for reconciled quantities.
        bandwidths (ndarray): Vector of bandwidth values.

    Returns:
        ndarray: Reconciled measured data.
    """

    xNum = msrd_data.shape[0]
    n = msrd_data.shape[1]

    data_init_points = np.mean(msrd_data, axis=1)
    eqNum = len(eqies_model(data_init_points, model_params))
    ineqNum = len(ineqies_model(data_init_points, model_params))

    # Step 1: Compute l (a zero vector of the same length as func's output)
    l_eqies = np.zeros(len(eqies_model(data_init_points, model_params)))
    l_ineqies = np.zeros(len(ineqies_model(data_init_points, model_params)))
    l_ineqies_muly_by_l = np.zeros(len(ineqies_model(data_init_points, model_params)))

    l = np.hstack((l_eqies, l_ineqies, l_ineqies_muly_by_l))

    # Step 2: Initialize mu_and_l_start with the mean of msrd_data
    mu_and_l_start = np.hstack((data_init_points, l))  # Append zeros for l

    # Step 3: Define the system of equations and solve using fsolve
    def kernels_system_wrapper(mu_and_l):
        """
        Wrapper function for KernelsSystemToSolve to include all required parameters.
        """
        return kernels_system_to_solve(mu_and_l, eqies_model, ineqies_model, msrd_data, model_params, bandwidths, prior_vars, xNum, n, eqNum, ineqNum)

    # Solve the system of equations using fsolve
    mu_and_l_est = fsolve(kernels_system_wrapper, mu_and_l_start)

    # Step 4: Extract the reconciled measured data (excluding 'l' values)
    reconciled_with_kernels = mu_and_l_est[:len(mu_and_l_start) - len(l)]

    return reconciled_with_kernels

