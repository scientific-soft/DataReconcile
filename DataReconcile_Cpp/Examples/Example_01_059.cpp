#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

// Complex vector type
typedef Matrix<complex<double>, Dynamic, 1> VectorXcd;

// Function to compute numerical derivative using complex step method
double get_imag_numeric_derivative(
    const std::function<VectorXd(const VectorXd&, const VectorXd&)>& func,
    const VectorXd& params,
    const VectorXd& params_,
    int deriv_param_num,
    int equation_num) {

    VectorXd alpha = VectorXd::Zero(params.size());
    alpha(deriv_param_num) = params(deriv_param_num) * 1e-3 + 1e-10;

    // Original function value
    VectorXd f_original = func(params, params_);

    // Perturb the real parameter
    VectorXd perturbed_params = params;
    perturbed_params(deriv_param_num) += alpha(deriv_param_num);

    // Compute perturbed function value
    VectorXd f_perturbed_real = func(perturbed_params, params_);

    // Approximate imaginary part using finite difference
    double df_imag = (f_perturbed_real(equation_num) - f_original(equation_num)) / alpha(deriv_param_num);

    return df_imag;
}

pair<VectorXd, VectorXd> estAccuracyIncreaseByDR(
    const function<VectorXd(const VectorXd&, const VectorXd&)>& dep_func,
    const string& ChosenMode,
    const VectorXd& vals_for_reconc,
    const VectorXd& params_,
    const MatrixXd& cov_matrix) {

    int num_equations = dep_func(vals_for_reconc, params_).size();
    int num_vals = vals_for_reconc.size();

    VectorXd variances_of_DR_result(num_vals);
    VectorXd accuracy_increase_ratio(num_vals);
    VectorXd dF_dxj(num_equations);
    MatrixXd Jacobian(num_equations, num_vals - 1);

    for (int j = 0; j < num_vals; ++j) {
        // Create indices without j
        vector<int> nums_without_j_th;
        for (int k = 0; k < num_vals; ++k) {
            if (k != j) nums_without_j_th.push_back(k);
        }

        // Create covariance matrix without j-th row/column
        MatrixXd cov_matrix_without_j_th(num_vals - 1, num_vals - 1);
        int row_idx = 0;
        for (int r = 0; r < num_vals; ++r) {
            if (r == j) continue;
            int col_idx = 0;
            for (int c = 0; c < num_vals; ++c) {
                if (c == j) continue;
                cov_matrix_without_j_th(row_idx, col_idx) = cov_matrix(r, c);
                col_idx++;
            }
            row_idx++;
        }

        // Compute Jacobian and derivatives
        for (int i = 0; i < num_equations; ++i) {
            dF_dxj(i) = get_imag_numeric_derivative(dep_func, vals_for_reconc, params_, j, i);

            for (int j_ = 0; j_ < num_vals - 1; ++j_) {
                Jacobian(i, j_) = get_imag_numeric_derivative(
                    dep_func, vals_for_reconc, params_, nums_without_j_th[j_], i);
            }
        }

        double linear_est_of_indirect_measurement_variance = 0;

        if (ChosenMode == "LS") {
            // Least Squares estimation - FIXED
            MatrixXd temp = dF_dxj.transpose() * Jacobian;
            MatrixXd temp2 = temp * cov_matrix_without_j_th * temp.transpose();
            linear_est_of_indirect_measurement_variance =
                temp2(0, 0) / pow(dF_dxj.squaredNorm(), 2);
        }
        else if (ChosenMode == "WLS") {
            // Weighted Least Squares estimation
            VectorXd var_of_delta_x_j(num_equations);
            VectorXd weights_for_WLS(num_equations);

            for (int j_ = 0; j_ < num_equations; ++j_) {
                double sum_df_j_dz = 0;
                for (int k = 0; k < num_vals - 1; ++k) {
                    sum_df_j_dz += pow(Jacobian(j_, k), 2) * cov_matrix_without_j_th(k, k);
                }
                var_of_delta_x_j(j_) = pow(dF_dxj(j_), -2) * sum_df_j_dz;
            }

            for (int j_ = 0; j_ < num_equations; ++j_) {
                VectorXd tmp_w = var_of_delta_x_j.array() / var_of_delta_x_j(j_);
                for (int k = 0; k < tmp_w.size(); ++k) {
                    if (isnan(tmp_w(k))) tmp_w(k) = 1;
                }
                weights_for_WLS(j_) = 1.0 / tmp_w.sum();
            }

            linear_est_of_indirect_measurement_variance =
                weights_for_WLS.array().square().matrix().dot(var_of_delta_x_j);
        }
        else {
            cerr << "Unknown mode for linear estimation of variance of system solution" << endl;
            return make_pair(VectorXd(), VectorXd());
        }

        if (linear_est_of_indirect_measurement_variance == 0) {
            cout << "The true value of " << j << "-th measurand can be derived from given dependency equations. "
                << "No reconciliation is needed for its measurement result." << endl;
        }

        accuracy_increase_ratio(j) = sqrt(1 + cov_matrix(j, j) / linear_est_of_indirect_measurement_variance);
        variances_of_DR_result(j) = cov_matrix(j, j) / pow(accuracy_increase_ratio(j), 2);
    }

    return make_pair(accuracy_increase_ratio, variances_of_DR_result);
}

// Example dependency function
VectorXd example_dependency_func(const VectorXd& vals, const VectorXd& params) {
    VectorXd result(3);
    double x = vals(0), y = vals(1), z = vals(2);
    double a = params(0), b = params(1);

    result << x - a * y + b * z,   // Equation 1
        y - 2 * x + z,     // Equation 2
        z - x - y;       // Equation 3
    return result;
}

int main() {
    // Current measured values (to be reconciled)
    VectorXd vals_for_reconc(3);
    vals_for_reconc << 1.2, 0.9, 2.1;

    // Parameters for the dependency function
    VectorXd params_(2);
    params_ << 0.5, 1.2;

    // Covariance matrix of the measurements
    MatrixXd cov_matrix(3, 3);
    cov_matrix << 0.1, 0.01, 0.02,
        0.01, 0.15, 0.01,
        0.02, 0.01, 0.2;

    // Choose reconciliation mode
    string ChosenMode = "LS";

    // Calculate accuracy increase and variances after reconciliation
    auto result = estAccuracyIncreaseByDR(
        example_dependency_func,
        ChosenMode,
        vals_for_reconc,
        params_,
        cov_matrix
    );

    cout << "Accuracy increase ratios:\n" << result.first.transpose() << endl;
    cout << "Variances after reconciliation:\n" << result.second.transpose() << endl;

    // Interpretation
    for (int i = 0; i < result.first.size(); ++i) {
        cout << "Measurement " << i + 1 << " will be " << result.first(i)
            << " times more accurate after reconciliation" << endl;
    }

    return 0;
}
