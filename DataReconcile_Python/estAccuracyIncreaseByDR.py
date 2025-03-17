import numpy as np

def get_imag_numeric_derivative(func, params, params_, deriv_param_num, equation_num):
    alpha = np.zeros_like(params, dtype = float)
    
    alpha[deriv_param_num] = params[deriv_param_num] * 1e-3 + 1e-10
    #df = [func(params + alpha * 1j, params_) - func(params, params_)]
    df = np.subtract(func(params + alpha * 1j, params_), func(params, params_))
    if alpha[deriv_param_num] == 0:
        df_dx = 0
    else:
        df_dx = np.imag(df[equation_num]) / alpha[deriv_param_num]
    return df_dx

def estAccuracyIncreaseByDR(dep_func, ChosenMode, vals_for_reconc, params_, cov_matrix):
    num_equations = len(dep_func(vals_for_reconc, params_))
    num_vals = len(vals_for_reconc)

    variances_of_DR_result = np.zeros(num_vals)
    accuracy_increase_ratio = np.zeros(num_vals)
    dF_dxj = np.zeros(num_equations)
    Jacobian = np.zeros((num_equations, num_vals - 1))

    for j in range(num_vals):
        # nums_without_j_th = list(range(1, j)) + list(range(j + 1, num_vals))
        nums_without_j_th = list(range(0, j)) + list(range(j + 1, num_vals))
        tmp_mx = cov_matrix.copy()
        tmp_mx = np.delete(tmp_mx, j, axis=0)
        tmp_mx = np.delete(tmp_mx, j, axis=1)
        cov_matrix_without_j_th = tmp_mx
        for i in range(num_equations):
            dF_dxj[i] = get_imag_numeric_derivative(dep_func, vals_for_reconc, params_, j, i)
            for j_ in range(num_vals - 1):
                Jacobian[i][j_] = get_imag_numeric_derivative(dep_func, vals_for_reconc, params_, nums_without_j_th[j_], i)
        if ChosenMode == 'LS':
            linear_est_of_indirect_measurement_variance = np.abs(dF_dxj @ Jacobian) @ cov_matrix_without_j_th @ np.abs(Jacobian.T @ dF_dxj) / ((dF_dxj @ dF_dxj) ** 2)
        elif ChosenMode == 'WLS':
            var_of_delta_x_j = np.zeros(num_equations)
            weights_for_WLS = np.zeros(num_equations)
            for j_ in range(num_equations):
                sum_df_j_dz = np.sum(Jacobian[j_, :] ** 2 * np.diag(cov_matrix_without_j_th))
                var_of_delta_x_j[j_] = dF_dxj[j_] ** (-2) * sum_df_j_dz
            for j_ in range(num_equations):
                tmp_w = var_of_delta_x_j[j_] / var_of_delta_x_j
                tmp_w[np.isnan(tmp_w)] = 1
                weights_for_WLS[j_] = 1 / np.sum(tmp_w)
            linear_est_of_indirect_measurement_variance = np.square(weights_for_WLS) @ var_of_delta_x_j
        else:
            print('Unknown mode for linear estimation of variance of system solution')
        if linear_est_of_indirect_measurement_variance == 0:
            print('The true value of {}-th measurand can be derived from given dependency equations. No reconciliation is needed for its measurement result.'.format(j))
        accuracy_increase_ratio[j] = np.sqrt(1 + cov_matrix[j][j] / linear_est_of_indirect_measurement_variance)
        variances_of_DR_result[j] = cov_matrix[j][j] / accuracy_increase_ratio[j] ** 2

    return accuracy_increase_ratio, variances_of_DR_result