#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <iostream>
#include <cmath>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace Eigen;

typedef std::complex<double> Complex;
typedef Matrix<Complex, Dynamic, 1> VectorXc;
typedef Matrix<double, Dynamic, 1> VectorXd;
typedef Matrix<Complex, Dynamic, Dynamic> MatrixXc;

// Function to compute the Gaussian kernel
double kernel(double x) {
    return std::exp(-x * x / 2.0) / std::sqrt(2.0 * M_PI);
}

// Function to solve the system of equations using kernel-based estimation
VectorXd kernels_system_to_solve(const VectorXd& mu_and_l, const std::function<VectorXc(const VectorXc&, const VectorXd&)>& eqies_model,
    const std::function<VectorXc(const VectorXc&, const VectorXd&)>& ineqies_model, const MatrixXd& msrd_data,
    const VectorXd& model_params, const VectorXd& bandwidths, const VectorXd& msrd_vars, int xNum, int n, int eqNum, int ineqNum) {
    VectorXc mustbezeros = VectorXc::Zero(xNum);

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
        VectorXd res = VectorXd::Zero(n);

        for (int i = 0; i < n; ++i) {
            // Compute kernel weights
            VectorXd denumerator = (msrd_data.row(j).array() - mean_msrd) / s_msrd - (msrd_data(j, i) - mu_and_l(j)) / s_msrd;
            denumerator = denumerator.unaryExpr([&](double x) { return kernel(x / bandwidths_(j)); });

            double numerator = (msrd_data.row(j).array() * denumerator.array()).sum();
            if (denumerator.sum() == 0) {
                res(i) = 0;
            }
            else {
                res(i) = numerator / denumerator.sum();
            }
        }

        double mean_via_kerns = res.mean();

        // Numerical derivative using small imaginary perturbation
        VectorXc imagx = VectorXc::Zero(xNum);
        imagx(j) = Complex(0, 1) * (mu_and_l(j) * 1e-100 + 1e-101);

        VectorXc df_dx_eqies = eqies_model(mu_and_l.head(xNum).cast<Complex>() + imagx, model_params).imag() / imagx(j).imag();
        VectorXc df_dx_ineqies = ineqies_model(mu_and_l.head(xNum).cast<Complex>() + imagx, model_params).imag() / imagx(j).imag();

        VectorXc df_dx(df_dx_eqies.size() + df_dx_ineqies.size());
        df_dx << df_dx_eqies, df_dx_ineqies;

        // Update mustbezeros
        mustbezeros(j) = (mean_msrd - mean_msrd - mean_via_kerns + mu_and_l(j) +
            bandwidths_(j) * bandwidths_(j) * (1.0 / n) * s_msrd * s_msrd * df_dx.dot(mu_and_l.segment(xNum, eqNum + ineqNum).cast<Complex>()));
    }

    // Evaluate the model residuals
    VectorXc eqies_model_res = eqies_model(mu_and_l.head(xNum).cast<Complex>(), model_params);
    VectorXc ineqies_model_res = ineqies_model(mu_and_l.head(xNum).cast<Complex>(), model_params);

    VectorXd mustbezeros_real = mustbezeros.real();
    VectorXd eqies_model_res_real = eqies_model_res.real();
    VectorXd ineqies_model_res_real = ineqies_model_res.real();

    VectorXd result(mustbezeros_real.size() + eqies_model_res_real.size() + ineqies_model_res_real.size());
    result << mustbezeros_real, eqies_model_res_real, ineqies_model_res_real;

    VectorXd ineqies_mul = ineqies_model_res_real.cwiseProduct(mu_and_l.segment(xNum + eqNum, ineqNum));
    result.conservativeResize(result.size() + ineqies_mul.size());
    result.tail(ineqies_mul.size()) = ineqies_mul;

    return result;
}

// Functor for the Levenberg-Marquardt algorithm
struct KernelsSystemFunctor {
    KernelsSystemFunctor(const std::function<VectorXc(const VectorXc&, const VectorXd&)>& eqies_model,
        const std::function<VectorXc(const VectorXc&, const VectorXd&)>& ineqies_model,
        const MatrixXd& msrd_data, const VectorXd& model_params, const VectorXd& bandwidths,
        const VectorXd& msrd_vars, int xNum, int n, int eqNum, int ineqNum)
        : eqies_model(eqies_model), ineqies_model(ineqies_model), msrd_data(msrd_data),
        model_params(model_params), bandwidths(bandwidths), msrd_vars(msrd_vars),
        xNum(xNum), n(n), eqNum(eqNum), ineqNum(ineqNum) {}

    // Operator to compute the residuals
    int operator()(const VectorXd& mu_and_l, VectorXd& fvec) const {
        fvec = kernels_system_to_solve(mu_and_l, eqies_model, ineqies_model, msrd_data, model_params, bandwidths, msrd_vars, xNum, n, eqNum, ineqNum);
        return 0;
    }

    // Method to compute the Jacobian
    int df(const VectorXd& mu_and_l, MatrixXd& fjac) const {
        int m = fvec.size(); // Number of equations
        int p = mu_and_l.size(); // Number of variables

        fjac.resize(m, p);
        fjac.setZero();

        // Compute the Jacobian numerically using finite differences
        const double eps = 1e-8; // Small perturbation for finite differences
        VectorXd mu_and_l_perturbed = mu_and_l;

        for (int j = 0; j < p; ++j) {
            mu_and_l_perturbed(j) += eps;
            VectorXd fvec_perturbed = kernels_system_to_solve(mu_and_l_perturbed, eqies_model, ineqies_model, msrd_data, model_params, bandwidths, msrd_vars, xNum, n, eqNum, ineqNum);
            fjac.col(j) = (fvec_perturbed - fvec) / eps;
            mu_and_l_perturbed(j) = mu_and_l(j); // Reset the perturbation
        }

        return 0;
    }

    int inputs() const { return xNum + eqNum + ineqNum; }
    int values() const { return xNum + eqNum + ineqNum; }

    std::function<VectorXc(const VectorXc&, const VectorXd&)> eqies_model;
    std::function<VectorXc(const VectorXc&, const VectorXd&)> ineqies_model;
    MatrixXd msrd_data;
    VectorXd model_params;
    VectorXd bandwidths;
    VectorXd msrd_vars;
    int xNum, n, eqNum, ineqNum;
    mutable VectorXd fvec; // Store the residuals for use in df
};

// Function to reconcile measured data using kernel-based estimation
VectorXd DRnonparamEqIneq(const std::function<VectorXc(const VectorXc&, const VectorXd&)>& eqies_model,
    const std::function<VectorXc(const VectorXc&, const VectorXd&)>& ineqies_model,
    const MatrixXd& msrd_data, const VectorXd& model_params, const VectorXd& prior_vars, const VectorXd& bandwidths) {
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();

    VectorXd data_init_points = msrd_data.rowwise().mean();
    int eqNum = eqies_model(data_init_points.cast<Complex>(), model_params).size();
    int ineqNum = ineqies_model(data_init_points.cast<Complex>(), model_params).size();

    // Step 1: Compute l (a zero vector of the same length as func's output)
    VectorXd l_eqies = VectorXd::Zero(eqNum);
    VectorXd l_ineqies = VectorXd::Zero(ineqNum);
    VectorXd l_ineqies_mul_by_l = VectorXd::Zero(ineqNum);

    VectorXd l(eqNum + ineqNum + ineqNum);
    l << l_eqies, l_ineqies, l_ineqies_mul_by_l;

    // Step 2: Initialize mu_and_l_start with the mean of msrd_data
    VectorXd mu_and_l_start(xNum + eqNum + ineqNum + ineqNum);
    mu_and_l_start << data_init_points, l;

    // Step 3: Define the system of equations and solve using Levenberg-Marquardt
    KernelsSystemFunctor functor(eqies_model, ineqies_model, msrd_data, model_params, bandwidths, prior_vars, xNum, n, eqNum, ineqNum);
    LevenbergMarquardt<KernelsSystemFunctor> lm(functor);

    // Set the initial guess for the optimizer
    lm.parameters.maxfev = 1000; // Maximum number of function evaluations
    lm.parameters.xtol = 1.0e-6; // Tolerance for the solution

    // Minimize the function
    int status = lm.minimize(mu_and_l_start);

    //if (status != LevenbergMarquardtSpace::Status::RelativeReductionTooSmall && status != LevenbergMarquardtSpace::Status::CosinusTooSmall) {
    //    throw std::runtime_error("Optimization failed to converge.");
    //}

    // Step 4: Extract the reconciled measured data (excluding 'l' values)
    VectorXd reconciled_with_kernels = mu_and_l_start.head(xNum);

    return reconciled_with_kernels;
}

// Define a dependency model function
VectorXc eqies_model(const VectorXc& mu, const VectorXd& params) {
    // Example: A simple linear model
    VectorXc res(2);
    res << mu(0) * params(0) - mu(1),
        mu(0) - mu(1) / params(0);
    return res;
}

// Define a dependency model function
VectorXc ineqies_model(const VectorXc& mu, const VectorXd& params) {
    // Example: A simple linear model
    VectorXc res(2);
    res << mu(0) * params(0) - mu(1),
        mu(0) - mu(1) / params(0);
    return res;
}

int main() {
    // Define measured data (msrd_data)
    MatrixXd msrd_data(2, 3);
    msrd_data << 1.0, 1.1, 0.9,
        1.0, 1.1, 0.9;

    // Define error parameters (error_params)
    VectorXd error_params(2);
    error_params << 0.1, 0.1;

    // Define constant model parameters (params_)
    VectorXd model_params(1);
    model_params << 2.0;

    // Define bandwidths
    VectorXd bandwidths(2);
    bandwidths << 0.5, 0.5;

    // Call the kernel_data_reconcil function to reconcile the measured data
    VectorXd reconciled_data = DRnonparamEqIneq(eqies_model, ineqies_model, msrd_data, model_params, error_params, bandwidths);

    // Print the reconciled data
    std::cout << "Reconciled Data:\n" << reconciled_data << std::endl;

    return 0;
}
