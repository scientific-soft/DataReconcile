#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Eigen;

typedef std::complex<double> Complex;
typedef Matrix<Complex, Dynamic, 1> VectorXc;
typedef Matrix<double, Dynamic, 1> VectorXd;
typedef Matrix<Complex, Dynamic, Dynamic> MatrixXc;

// Function to compute the median absolute deviation (MAD)
VectorXd median_abs_deviation(const MatrixXd& data, int axis = 0) {
    VectorXd mad_values(data.rows());
    for (int i = 0; i < data.rows(); ++i) {
        VectorXd row = data.row(i);
        double median = row.size() % 2 == 0 ?
            (row(row.size() / 2 - 1) + row(row.size() / 2)) / 2.0 :
            row(row.size() / 2);
        VectorXd abs_dev = (row.array() - median).abs();
        std::sort(abs_dev.data(), abs_dev.data() + abs_dev.size());
        double mad = abs_dev.size() % 2 == 0 ?
            (abs_dev(abs_dev.size() / 2 - 1) + abs_dev(abs_dev.size() / 2)) / 2.0 :
            abs_dev(abs_dev.size() / 2);
        mad_values(i) = mad * 1.483; // Scale factor for consistency with standard deviation
    }
    return mad_values;
}

// Function to calculate the mean alpha value
double calcul_mean_alpha(double mj, const VectorXd& msrd_data, const VectorXd& alpha_params, double sj) {
    // Extract alpha parameters
    double As = alpha_params(0);
    double Ex = alpha_params(1);

    // Compute tji
    VectorXd tji = (msrd_data.array() - mj) / sj;

    // Compute d_alpha_d_mu
    VectorXd d_alpha_d_mu = (Ex - 3) * (0.5 * tji.array() - (1.0 / 6.0) * tji.array().pow(3)) + As * (0.5 - 0.5 * tji.array().pow(2));

    // Compute alpha_plus_one
    VectorXd alpha_plus_one = (Ex - 3) * ((1.0 / 24.0) * tji.array().pow(4) + 1.0 / 8.0 - 0.25 * tji.array().pow(2))
        + As * ((1.0 / 6.0) * tji.array().pow(3) - 0.5 * tji.array()) + 1;

    // Compute y
    VectorXd y = d_alpha_d_mu.array() / alpha_plus_one.array();

    // Compute alpha_mean
    double alpha_mean = (1.0 / sj) * y.mean();

    return alpha_mean;
}

// Function to compute the system of equations to solve
VectorXd alpha_system_to_solve(const VectorXd& mu_and_l, const std::function<VectorXc(const VectorXc&, const VectorXd&)>& eqies_model,
    const std::function<VectorXc(const VectorXc&, const VectorXd&)>& ineqies_model, const MatrixXd& msrd_data,
    const VectorXd& vars_, const VectorXd& alpha_params, const VectorXd& params_, int xNum, int n, int eqNum, int ineqNum) {
    VectorXc mustbezeros = VectorXc::Zero(xNum);

    for (int j = 0; j < xNum; ++j) {
        if (vars_(j) == 0) {
            throw std::invalid_argument("None of the error parameters should be 0.");
        }
        else {
            // Imaginary perturbation for numerical derivative
            VectorXc imagx = VectorXc::Zero(xNum);
            imagx(j) = Complex(0, 1) * (mu_and_l(j) * 1e-100 + 1e-101);

            // Numerical derivative approximation using imaginary perturbation
            VectorXc df_dx_eqies = eqies_model(mu_and_l.head(xNum).cast<Complex>() + imagx, params_).imag() / imagx(j).imag();
            VectorXc df_dx_ineqies = ineqies_model(mu_and_l.head(xNum).cast<Complex>() + imagx, params_).imag() / imagx(j).imag();

            VectorXc df_dx(df_dx_eqies.size() + df_dx_ineqies.size());
            df_dx << df_dx_eqies, df_dx_ineqies;

            double mean_alpha;
            if (alpha_params.size() == 2) {
                // Simplified alpha_params case
                mean_alpha = calcul_mean_alpha(mu_and_l(j), msrd_data.row(j).transpose(), alpha_params, vars_(j));
            }
            else {
                // Case where alpha_params depend on j
                VectorXd alpha_slice = alpha_params.segment(2 * j, 2);
                mean_alpha = calcul_mean_alpha(mu_and_l(j), msrd_data.row(j).transpose(), alpha_slice, vars_(j));
            }

            // Update mustbezeros
            mustbezeros(j) = (mu_and_l(j) / Complex(vars_(j)) - Complex(msrd_data.row(j).mean()) / Complex(vars_(j)) - mean_alpha +
                mu_and_l.segment(xNum, eqNum + ineqNum).cast<Complex>().dot(df_dx) / Complex(n));
        }
    }

    // Evaluate the model residuals
    VectorXc eqies_model_res = eqies_model(mu_and_l.head(xNum).cast<Complex>(), params_);
    VectorXc ineqies_model_res = ineqies_model(mu_and_l.head(xNum).cast<Complex>(), params_);

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
struct AlphaSystemFunctor {
    AlphaSystemFunctor(const std::function<VectorXc(const VectorXc&, const VectorXd&)>& eqies_model,
        const std::function<VectorXc(const VectorXc&, const VectorXd&)>& ineqies_model,
        const MatrixXd& msrd_data, const VectorXd& vars_, const VectorXd& alpha_params,
        const VectorXd& params_, int xNum, int n, int eqNum, int ineqNum)
        : eqies_model(eqies_model), ineqies_model(ineqies_model), msrd_data(msrd_data),
        vars_(vars_), alpha_params(alpha_params), params_(params_), xNum(xNum), n(n), eqNum(eqNum), ineqNum(ineqNum) {}

    // Operator to compute the residuals
    int operator()(const VectorXd& mu_and_l, VectorXd& fvec) const {
        fvec = alpha_system_to_solve(mu_and_l, eqies_model, ineqies_model, msrd_data, vars_, alpha_params, params_, xNum, n, eqNum, ineqNum);
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
            VectorXd fvec_perturbed = alpha_system_to_solve(mu_and_l_perturbed, eqies_model, ineqies_model, msrd_data, vars_, alpha_params, params_, xNum, n, eqNum, ineqNum);
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
    VectorXd vars_;
    VectorXd alpha_params;
    VectorXd params_;
    int xNum, n, eqNum, ineqNum;
    mutable VectorXd fvec; // Store the residuals for use in df
};

// Function to reconcile measured data using robust estimation
VectorXd DRsemiparamEqIneqRobust(const std::function<VectorXc(const VectorXc&, const VectorXd&)>& eqies_model,
    const std::function<VectorXc(const VectorXc&, const VectorXd&)>& ineqies_model,
    const MatrixXd& msrd_data, const VectorXd& error_params, const VectorXd& alpha_params, const VectorXd& params_) {
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();

    // Compute MAD and adjust error_params
    VectorXd mad_values = median_abs_deviation(msrd_data);
    VectorXd robust_error_params = mad_values * 1.483;

    VectorXd data_init_points = msrd_data.rowwise().mean();
    int eqNum = eqies_model(data_init_points.cast<Complex>(), params_).size();
    int ineqNum = ineqies_model(data_init_points.cast<Complex>(), params_).size();

    // Step 1: Compute l (a zero vector of the same length as eqies_model's and ineqies_model's output)
    VectorXd l_eqies = VectorXd::Zero(eqNum);
    VectorXd l_ineqies = VectorXd::Zero(ineqNum);
    VectorXd l_ineqies_mul_by_l = VectorXd::Zero(ineqNum);

    VectorXd l(eqNum + ineqNum + ineqNum);
    l << l_eqies, l_ineqies, l_ineqies_mul_by_l;

    // Step 2: Extend the initial guess (mu_and_l_start) by appending l to data_init_points
    VectorXd mu_and_l_start(xNum + eqNum + ineqNum + ineqNum);
    mu_and_l_start << data_init_points, l;

    // Step 3: Solve the system of equations using Levenberg-Marquardt
    AlphaSystemFunctor functor(eqies_model, ineqies_model, msrd_data, robust_error_params, alpha_params, params_, xNum, n, eqNum, ineqNum);
    LevenbergMarquardt<AlphaSystemFunctor> lm(functor);

    // Set the initial guess for the optimizer
    lm.parameters.maxfev = 1000; // Maximum number of function evaluations
    lm.parameters.xtol = 1.0e-6; // Tolerance for the solution

    // Minimize the function
    int status = lm.minimize(mu_and_l_start);

    //if (status != LevenbergMarquardtSpace::Status::RelativeReductionTooSmall && status != LevenbergMarquardtSpace::Status::CosinusTooSmall) {
    //    throw std::runtime_error("Optimization failed to converge.");
    //}

    // Step 4: Extract the reconciled measured data (excluding 'l' values)
    VectorXd reconciled_with_alpha_gram_ch = mu_and_l_start.head(xNum);

    return reconciled_with_alpha_gram_ch;
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

    // Define alpha parameters (alpha_params)
    VectorXd alpha_params(4);
    alpha_params << 0.1, 3.1, 0.1, 3.1;

    // Define constant model parameters (params_)
    VectorXd params_(1);
    params_ << 2.0;

    // Call the GramChDataReconcileRobust function to reconcile the measured data
    VectorXd reconciled_data = DRsemiparamEqIneqRobust(eqies_model, ineqies_model, msrd_data, error_params, alpha_params, params_);

    // Print the reconciled data
    std::cout << "Reconciled Data:\n" << reconciled_data << std::endl;

    return 0;
}
