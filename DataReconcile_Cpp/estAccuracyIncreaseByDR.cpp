#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Function to compute the imaginary part of the numerical derivative
double get_imag_numeric_derivative(const function<VectorXcd(const VectorXd&, const VectorXd&)>& func,
    const VectorXd& params, const VectorXd& params_,
    int deriv_param_num, int equation_num) {
    VectorXd alpha = VectorXd::Zero(params.size());
    alpha(deriv_param_num) = params(deriv_param_num) * 1e-3 + 1e-10;

    VectorXcd df = func(params + alpha * complex<double>(0, 1), params_) - func(params, params_);

    if (alpha(deriv_param_num) == 0) {
        return 0;
    }
    else {
        return imag(df(equation_num)) / alpha(deriv_param_num);
    }
}

// Function to estimate accuracy increase by data reconciliation
pair<VectorXd, VectorXd> estAccuracyIncreaseByDR(const function<VectorXd(const VectorXd&, const VectorXd&)>& dep_func,
    const string& ChosenMode, const VectorXd& vals_for_reconc,
    const VectorXd& params_, const MatrixXd& cov_matrix) {
    int num_equations = dep_func(vals_for_reconc, params_).size();
    int num_vals = vals_for_reconc.size();

    VectorXd variances_of_DR_result = VectorXd::Zero(num_vals);
    VectorXd accuracy_increase_ratio = VectorXd::Zero(num_vals);
    VectorXd dF_dxj = VectorXd::Zero(num_equations);
    MatrixXd Jacobian = MatrixXd::Zero(num_equations, num_vals - 1);

    for (int j = 0; j < num_vals; ++j) {
        vector<int> nums_without_j_th;
        for (int k = 0; k < num_vals; ++k) {
            if (k != j) {
                nums_without_j_th.push_back(k);
            }
        }

        MatrixXd tmp_mx = cov_matrix;
        tmp_mx.block(j, 0, num_vals - 1, num_vals) = tmp_mx.block(j + 1, 0, num_vals - 1, num_vals);
        tmp_mx.block(0, j, num_vals, num_vals - 1) = tmp_mx.block(0, j + 1, num_vals, num_vals - 1);
        tmp_mx.conservativeResize(num_vals - 1, num_vals - 1);
        MatrixXd cov_matrix_without_j_th = tmp_mx;

        for (int i = 0; i < num_equations; ++i) {
            dF_dxj(i) = get_imag_numeric_derivative([&](const VectorXd& x, const VectorXd& p) {
                return dep_func(x, p).cast<complex<double>>();
                }, vals_for_reconc, params_, j, i);

            for (int j_ = 0; j_ < num_vals - 1; ++j_) {
                Jacobian(i, j_) = get_imag_numeric_derivative([&](const VectorXd& x, const VectorXd& p) {
                    return dep_func(x, p).cast<complex<double>>();
                    }, vals_for_reconc, params_, nums_without_j_th[j_], i);
            }
        }

        double linear_est_of_indirect_measurement_variance;
        if (ChosenMode == "LS") {
            linear_est_of_indirect_measurement_variance = abs((dF_dxj.transpose() * Jacobian) *
                cov_matrix_without_j_th *
                (Jacobian.transpose() * dF_dxj) /
                pow(dF_dxj.squaredNorm(), 2);
        }
        else if (ChosenMode == "WLS") {
            VectorXd var_of_delta_x_j = VectorXd::Zero(num_equations);
            VectorXd weights_for_WLS = VectorXd::Zero(num_equations);

            for (int j_ = 0; j_ < num_equations; ++j_) {
                double sum_df_j_dz = (Jacobian.row(j_).array().square() * cov_matrix_without_j_th.diagonal().array()).sum();
                var_of_delta_x_j(j_) = pow(dF_dxj(j_), -2) * sum_df_j_dz;
            }

            for (int j_ = 0; j_ < num_equations; ++j_) {
                VectorXd tmp_w = var_of_delta_x_j.array() / var_of_delta_x_j(j_);
                tmp_w = tmp_w.array().isNaN().select(1, tmp_w);
                weights_for_WLS(j_) = 1.0 / tmp_w.sum();
            }

            linear_est_of_indirect_measurement_variance = weights_for_WLS.array().square().matrix().dot(var_of_delta_x_j);
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