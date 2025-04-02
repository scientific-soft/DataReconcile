#include <iostream>
#include <vector>
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <stdexcept>
#include <functional>

using namespace std;
using namespace Eigen;

double median_absolute_deviation(const VectorXd& data, double scale = 1.483) {
    vector<double> abs_deviation;
    double median = data[data.size() / 2]; // Simplified median calculation; assumes sorted input
    for (double value : data) {
        abs_deviation.push_back(fabs(value - median));
    }
    sort(abs_deviation.begin(), abs_deviation.end());
    return abs_deviation[abs_deviation.size() / 2] * scale;
}

// Function to compute the Gaussian kernel value
double kernel(double x) {
    return exp(-x * x / 2) / sqrt(2 * M_PI);
}

// Function to solve the kernel-based system of equations
VectorXd kernels_system_to_solve(const VectorXd& mu_and_l,
    const function<VectorXd(const VectorXd&, const VectorXd&)>& depend_func,
    const MatrixXd& msrd_data,
    const VectorXd& model_params,
    const VectorXd& bandwidths,
    const VectorXd& msrd_vars) {
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();
    VectorXd mustbezeros(xNum + depend_func(mu_and_l.head(xNum), model_params).size());
    mustbezeros.setZero();

    // Handle scalar bandwidth case
    VectorXd bandwidths_;
    if (bandwidths.size() == 1) {
        bandwidths_ = VectorXd::Constant(xNum, bandwidths(0));
    }
    else {
        bandwidths_ = bandwidths;
    }

    for (int j = 0; j < xNum; ++j) {
        double mean_msrd = msrd_data.row(j).mean();
        double s_msrd = msrd_vars(j);
        VectorXd res(n);
        for (int i = 0; i < n; ++i) {
            double kernel_arg;
            double denumerator_k;
            double denumerator_sum = 0.0;
            double numerator_k;
            double numerator_sum = 0.0;
            for (int k = 0; k < n; ++k) {
                kernel_arg = ((msrd_data(j, k) - mean_msrd) / s_msrd
                    - (msrd_data(j, i) - mu_and_l[j]) / s_msrd) / bandwidths_(j);
                denumerator_k = kernel(kernel_arg);
                denumerator_sum = denumerator_sum + denumerator_k;
                numerator_k = msrd_data(j, k) * denumerator_k;
                numerator_sum = numerator_sum + numerator_k;
            }

            if (denumerator_sum == 0) {
                res(i) = 0;
            }
            else {
                res(i) = numerator_sum / denumerator_sum;
            }

        }
        double mean_via_kerns = res.mean();

        // Numerical derivative using small imaginary perturbation
        VectorXcd imagx = VectorXcd::Zero(xNum);
        imagx[j] = std::complex<double>(0.0, mu_and_l[j] * pow(10.0, -100) + pow(10.0, -101));

        VectorXcd mu_and_l_complex = mu_and_l.head(xNum).cast<std::complex<double>>() + imagx;
        VectorXd mu_and_l_real = mu_and_l_complex.real().cast<double>();
        VectorXd df_result = depend_func(mu_and_l_real, model_params);

        VectorXd df_dx = df_result.cast<std::complex<double>>().array().imag() / imagx.array().imag()[j];

        // Update mustbezeros
        VectorXd elem_wize_multy_res = df_dx.array() * mu_and_l.tail(mu_and_l.size() - xNum).array();

        mustbezeros[j] = (mean_msrd - mean_msrd - mean_via_kerns + mu_and_l[j]
            + bandwidths_(j) * bandwidths_(j) * (1.0 / n) * s_msrd * s_msrd * elem_wize_multy_res.sum());
    }

    // Evaluate the model residuals
    VectorXd model_res = depend_func(mu_and_l.head(xNum), model_params);
    mustbezeros.tail(model_res.size()) = model_res;

    return mustbezeros;
}

// Functor for the system of equations
struct KernelSystemFunctor {
    const function<VectorXd(const VectorXd&, const VectorXd&)>& depend_func;
    const MatrixXd& msrd_data;
    const VectorXd& model_params;
    const VectorXd& bandwidths;
    const VectorXd& msrd_vars;
    int xNum;

    KernelSystemFunctor(const function<VectorXd(const VectorXd&, const VectorXd&)>& df,
        const MatrixXd& data,
        const VectorXd& params,
        const VectorXd& bw,
        const VectorXd& vars)
        : depend_func(df), msrd_data(data), model_params(params), bandwidths(bw), msrd_vars(vars), xNum(data.rows()) {}

    // Compute the residuals
    int operator()(const VectorXd& mu_and_l, VectorXd& fvec) const {
        VectorXd residuals = kernels_system_to_solve(mu_and_l, depend_func, msrd_data, model_params, bandwidths, msrd_vars);
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

            fjac.row(j) = (fvec_plus - fvec_minus) / (2 * epsilon); // Central difference
            mu_and_l_perturbed(j) = temp; // Restore the original value
        }
        return 0;
    }

    int inputs() const { return xNum + depend_func(VectorXd::Zero(xNum), model_params).size(); }
    int values() const { return xNum + depend_func(VectorXd::Zero(xNum), model_params).size(); }
};

// Function to reconcile measured data using kernel-based estimation
VectorXd DRnonparamEqRobust(const function<VectorXd(const VectorXd&, const VectorXd&)>& func,
    const MatrixXd& msrd_data,
    const VectorXd& model_params,
    VectorXd& prior_vars,
    const VectorXd& bandwidths) {
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();

    // Compute MAD and adjust error_params
    for (size_t j = 0; j < xNum; ++j) {
        VectorXd column(n);
        for (size_t i = 0; i < n; ++i) {
            column(i) = msrd_data(j, i);
        }
        sort(column.begin(), column.end());
        prior_vars[j] = median_absolute_deviation(column) * 10.0;
    }

    // Step 1: Compute l (a zero vector of the same length as func's output)
    VectorXd l = func(msrd_data.rowwise().mean(), model_params);

    // Step 2: Initialize mu_and_l_start with the mean of msrd_data
    VectorXd mu_and_l_start(xNum + l.size());
    mu_and_l_start.head(xNum) = msrd_data.rowwise().mean();
    mu_and_l_start.tail(l.size()) = l;

    // Step 3: Define the system of equations and solve using Eigen's LevenbergMarquardt solver
    KernelSystemFunctor functor(func, msrd_data, model_params, bandwidths, prior_vars);
    LevenbergMarquardt<KernelSystemFunctor> lm(functor);
    lm.minimize(mu_and_l_start);

    // Step 4: Extract the reconciled measured data (excluding 'l' values)
    VectorXd reconciled_with_kernels = mu_and_l_start.head(xNum);

    return reconciled_with_kernels;
}

// Example dependency function
VectorXd example_depend_func(const VectorXd& mu, const VectorXd& params) {
    VectorXd result(2);
    result[0] = params[0] * mu[0] - mu[1];
    result[1] = mu[1] - params[0] * mu[0];
    return result;
}

int main() {
    try {
        // Example usage
        int N_of_quantities = 2;
        int n_of_measurements = 3;
        MatrixXd msrd_data(N_of_quantities, n_of_measurements); // Example [rows]x[cols] matrix
        msrd_data << 0.9, 1.1, 1.0, // cols - measurement results of the same quantity
            1.8, 2.2, 2.0;          // rows - different quantities

        VectorXd model_params(1); // Example model parameters
        model_params << 2.0;
        VectorXd prior_vars(2);// VectorXd::Ones(2); // Example prior variances
        prior_vars << 1.0, 2.0;
        VectorXd bandwidths(2); //VectorXd::Ones(2);//Constant(2, 0.5); // Example bandwidths
        bandwidths(0) = 1.0;
        bandwidths(1) = 1.0;

        // Pass the example_depend_func as a parameter to kernel_data_reconcil_robust
        VectorXd reconciled_data = DRnonparamEqRobust(example_depend_func, msrd_data, model_params, prior_vars, bandwidths);

        cout << "Reconciled Data:" << endl;
        cout << reconciled_data.transpose() << endl;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
