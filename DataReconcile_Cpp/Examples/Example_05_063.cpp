#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <numeric>

using namespace std;
using namespace Eigen;

// Function to calculate mean alpha value
double calcul_mean_alpha(double mj, const VectorXd& msrd_data, const VectorXd& alpha_params, double sj) {
    // Extract alpha parameters
    double As = alpha_params(0);
    double Ex = alpha_params(1);

    // Compute tji
    VectorXd tji = (msrd_data.array() - mj) / sj;

    // Compute d_alpha_d_mu
    VectorXd d_alpha_d_mu = (Ex - 3) * (0.5 * tji.array() - (1.0 / 6.0) * tji.array().pow(3))
        + As * (0.5 - 0.5 * tji.array().pow(2));

    // Compute alpha_plus_one
    VectorXd alpha_plus_one = ((Ex - 3) * ((1.0 / 24.0) * tji.array().pow(4) + 1.0 / 8.0 - 0.25 * tji.array().pow(2))
        + As * ((1.0 / 6.0) * tji.array().pow(3) - 0.5 * tji.array()) + 1).matrix();

    // Compute y
    VectorXd y = d_alpha_d_mu.array() / alpha_plus_one.array();

    // Compute alpha_mean
    double alpha_mean = (1.0 / sj) * y.mean();

    return alpha_mean;
}

// Function to solve the Alpha system
VectorXd alpha_system_to_solve(const VectorXd& mu_and_l,
    const function<VectorXd(const VectorXd&, const VectorXd&)>& depend_func,
    const MatrixXd& msrd_data,
    const VectorXd& vars_,
    const MatrixXd& alpha_params,
    const VectorXd& params_) {
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();

    if (xNum == 0 || n == 0) {
        throw invalid_argument("Measured data must be a non-empty 2D array.");
    }

    VectorXd mustbezeros(xNum + depend_func(mu_and_l.head(xNum), params_).size());
    mustbezeros.setZero();

    for (int j = 0; j < xNum; ++j) {
        if (vars_[j] == 0) {
            throw invalid_argument("None of the error parameters should be 0.");
        }
        else {
            // Compute mean_alpha based on alpha_params
            VectorXd alpha_slice;
            if (alpha_params.cols() == 2) {
                // Simplified alpha_params case
                alpha_slice = alpha_params.row(0);
            }
            else {
                // Case where alpha_params depend on j
                alpha_slice = alpha_params.row(j);
            }
            double mean_alpha = calcul_mean_alpha(mu_and_l[j], msrd_data.row(j), alpha_slice, vars_[j]);

            // Update mustbezeros
            mustbezeros[j] = (mu_and_l[j] / vars_[j]
                - msrd_data.row(j).mean() / vars_[j]
                - mean_alpha);
        }
    }

    // Evaluate the model residuals
    VectorXd model_res = depend_func(mu_and_l.head(xNum), params_);
    mustbezeros.tail(model_res.size()) = model_res;

    return mustbezeros;
}

// Functor for the system of equations
struct AlphaSystemFunctor {
    const function<VectorXd(const VectorXd&, const VectorXd&)>& depend_func;
    const MatrixXd& msrd_data;
    const VectorXd& vars_;
    const MatrixXd& alpha_params;
    const VectorXd& params_;
    int xNum;

    AlphaSystemFunctor(const function<VectorXd(const VectorXd&, const VectorXd&)>& df,
        const MatrixXd& data,
        const VectorXd& vars,
        const MatrixXd& alpha,
        const VectorXd& params)
        : depend_func(df), msrd_data(data), vars_(vars), alpha_params(alpha), params_(params), xNum(data.rows()) {}

    // Compute the residuals
    int operator()(const VectorXd& mu_and_l, VectorXd& fvec) const {
        VectorXd residuals = alpha_system_to_solve(mu_and_l, depend_func, msrd_data, vars_, alpha_params, params_);
        fvec = residuals;
        return 0;
    }

    // Compute the Jacobian (optional but recommended for performance)
    int df(const VectorXd& mu_and_l, MatrixXd& fjac) const {
        const double epsilon = 1e-8; // Small perturbation for numerical differentiation
        VectorXd fvec(values());
        VectorXd mu_and_l_perturbed = mu_and_l;

        // Compute the Jacobian using finite differences
        for (int j = 0; j < mu_and_l.size(); ++j) {
            double temp = mu_and_l(j);
            mu_and_l_perturbed(j) = temp + epsilon;
            (*this)(mu_and_l_perturbed, fvec);
            VectorXd fvec_plus = fvec;

            mu_and_l_perturbed(j) = temp - epsilon;
            (*this)(mu_and_l_perturbed, fvec);
            VectorXd fvec_minus = fvec;

            fjac.col(j) = (fvec_plus - fvec_minus) / (2 * epsilon); // Central difference
            mu_and_l_perturbed(j) = temp; // Restore the original value
        }
        return 0;
    }

    int inputs() const { return xNum + depend_func(VectorXd::Zero(xNum), params_).size(); }
    int values() const { return xNum + depend_func(VectorXd::Zero(xNum), params_).size(); }
};

// Function to reconcile measured data
VectorXd DRsemiparamEq(const function<VectorXd(const VectorXd&, const VectorXd&)>& depend_func,
    const MatrixXd& msrd_data,
    vector<double>& error_params,
    const MatrixXd& alpha_params,
    const vector<double>& params_) {
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();

    if (xNum == 0 || n == 0) {
        throw invalid_argument("Measured data must be a non-empty 2D array.");
    }

    // -----------------------------------------
    int er_par_num = error_params.size();
    for (int i = 0; i < er_par_num; ++i) {
        error_params[i] = error_params[i] * 10.0;
    }
    // -----------------------------------------

    // Convert error_params and params_ to VectorXd
    VectorXd vars_ = Map<const VectorXd>(error_params.data(), error_params.size());
    VectorXd params = Map<const VectorXd>(params_.data(), params_.size());

    // Step 1: Compute l (a zero vector of the same length as depend_func's output)
    VectorXd l = depend_func(VectorXd::Zero(xNum), params);

    // Step 2: Extend the initial guess (mu_and_l_start) by appending l to msrd_data
    VectorXd mu_and_l_start(xNum + l.size());
    mu_and_l_start.head(xNum) = msrd_data.col(0); // Use the first column as the initial guess
    mu_and_l_start.tail(l.size()) = l;

    // Step 3: Solve the system of equations using Eigen's LevenbergMarquardt solver
    AlphaSystemFunctor functor(depend_func, msrd_data, vars_, alpha_params, params);
    LevenbergMarquardt<AlphaSystemFunctor> lm(functor);
    lm.minimize(mu_and_l_start);

    // Step 4: Extract the reconciled measured data (excluding 'l' values)
    VectorXd reconciled_with_alpha_gram_ch = mu_and_l_start.head(xNum);

    return reconciled_with_alpha_gram_ch;
}

// Example dependency function
VectorXd example_depend_func(const VectorXd& mu, const VectorXd& params) {
    //VectorXd result(mu.size());
    //for (int i = 0; i < mu.size(); ++i) {
    //    result[i] = params[0] * mu[i] + params[1];
    //}
    VectorXd result(2);
    result[0] = mu[0] - mu[1] * params[0];
    result[1] = mu[0] - mu[1] * params[0];
    return result;
}

int main() {
    try {
        // Example usage
        int measured_var_n = 2;
        int n_of_measurements = 2;
        MatrixXd msrd_data(measured_var_n, n_of_measurements);  // Example (measured_var_n x n_of_measurements) matrix
        msrd_data << 1.8, 2.2,
            0.9, 1.1;

        vector<double> error_params = { 0.1, 0.1 }; // Example error parameters
        vector<double> params = { 2.0 }; // Example model parameters

        // Example alpha parameters (As, Ex) or (As1, Ex1, ..., AsN, ExN)
        MatrixXd alpha_params(2, measured_var_n);
        alpha_params << 0.1, 3.0,
            0.1, 3.0;

        VectorXd reconciled_data = DRsemiparamEq(example_depend_func, msrd_data, error_params, alpha_params, params);

        cout << "Reconciled Data:" << endl;
        cout << reconciled_data.transpose() << endl;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
