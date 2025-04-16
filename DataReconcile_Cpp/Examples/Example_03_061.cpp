#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace Eigen;

// Function signature for the dependency model function
typedef VectorXd(*DependFunc)(const VectorXd&, const VectorXd&);

// Function to solve the Alpha system
VectorXd alpha_system_to_solve(const VectorXd& mu_and_l,
    DependFunc depend_func,
    const MatrixXd& msrd_data,
    const VectorXd& vars_,
    const VectorXd& params_) {
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();

    if (xNum == 0 || n == 0) {
        throw invalid_argument("Measured data must be a non-empty 2D array.");
    }

    VectorXd mustbezeros(xNum);
    mustbezeros.setZero();

    for (int j = 0; j < xNum; ++j) {
        if (vars_(j) == 0) {
            throw invalid_argument("None of the error parameters should be 0.");
        }
        else {
            //  // Imaginary perturbation for numerical derivative
            //   VectorXcd imagx = VectorXcd::Zero(xNum);
            //   imagx(j) = complex<double>(0.0, mu_and_l(j) * pow(10.0, -100) + pow(10.0, -101));

            //   // Create perturbed input for numerical derivative
            //   VectorXd mu_and_l_perturbed = mu_and_l;
            //   mu_and_l_perturbed += imagx.real();


               // ---------------
               // Numerical derivative using small imaginary perturbation
            VectorXcd imagx = VectorXcd::Zero(xNum);
            imagx[j] = std::complex<double>(0.0, mu_and_l[j] * pow(10.0, -100) + pow(10.0, -101));

            VectorXcd mu_and_l_complex = mu_and_l.head(xNum).cast<std::complex<double>>() + imagx;
            VectorXd mu_and_l_real = mu_and_l_complex.real().cast<double>();
            VectorXd df_result = depend_func(mu_and_l_real, params_);

            VectorXd df_dx = df_result.cast<std::complex<double>>().array().imag() / imagx.array().imag()[j];

            // Update mustbezeros
            VectorXd elem_wize_multy_res = df_dx.array() * mu_and_l.tail(mu_and_l.size() - xNum).array();
            // ---------------


            //// Numerical derivative approximation using imaginary perturbation
            //VectorXd df_dx_result = depend_func(mu_and_l_perturbed, params_);
            //complex<double> df_dx = df_dx_result(j).imag() / imagx(j).imag();

            // Update mustbezeros
            double mean_msrd_data_j = msrd_data.row(j).mean();
            mustbezeros(j) = mu_and_l(j) / vars_(j)
                - mean_msrd_data_j / vars_(j) + elem_wize_multy_res.sum() / n;
        }
    }

    // Evaluate the model residuals
    VectorXd model_res = depend_func(mu_and_l.head(xNum), params_);
    mustbezeros.conservativeResize(mustbezeros.size() + model_res.size());
    mustbezeros.tail(model_res.size()) = model_res;

    return mustbezeros;
}

// Functor for the system of equations
struct AlphaSystemFunctor {
    DependFunc depend_func;
    const MatrixXd& msrd_data;
    const VectorXd& vars_;
    const VectorXd& params_;
    int xNum;

    AlphaSystemFunctor(DependFunc df, const MatrixXd& data, const VectorXd& vars, const VectorXd& params)
        : depend_func(df), msrd_data(data), vars_(vars), params_(params), xNum(data.rows()) {}

    // Compute the residuals
    int operator()(const VectorXd& mu_and_l, VectorXd& fvec) const {
        VectorXd residuals = alpha_system_to_solve(mu_and_l, depend_func, msrd_data, vars_, params_);
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
VectorXd DRparamEq(DependFunc depend_func,
    const MatrixXd& msrd_data,
    VectorXd& error_params,
    const VectorXd& params_) {
    int er_par_num = error_params.size();
    int xNum = msrd_data.rows();
    int n = msrd_data.cols();

    // -----------------------------------------
    error_params = error_params * 10.0;
    // -----------------------------------------

    if (xNum == 0 || n == 0) {
        throw invalid_argument("Measured data must be a non-empty 2D array.");
    }

    // Step 1: Compute l (a zero vector of the same length as depend_func's output)
    VectorXd l = depend_func(VectorXd::Zero(xNum), params_);

    // Step 2: Extend the initial guess (mu_and_l_start) by appending l to msrd_data
    VectorXd mu_and_l_start(xNum + l.size());
    mu_and_l_start.head(xNum) = msrd_data.col(0); // Assuming first column as initial guess
    mu_and_l_start.tail(l.size()) = l;

    // Step 3: Solve the system of equations using Eigen's LevenbergMarquardt solver
    AlphaSystemFunctor functor(depend_func, msrd_data, error_params, params_);
    LevenbergMarquardt<AlphaSystemFunctor> lm(functor);
    lm.minimize(mu_and_l_start);

    // Step 4: Extract the reconciled measured data (excluding 'l' values)
    VectorXd reconciled = mu_and_l_start.head(xNum);

    return reconciled;
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
        int n_of_measurements = 3;
        MatrixXd msrd_data(measured_var_n, n_of_measurements);  // Example (measured_var_n x n_of_measurements) matrix
        msrd_data << 9.0, 10.0, 11.0,
            11.0, 10.0, 9.0;

        VectorXd error_params(2); // length is (measured_var_n)
        error_params << 0.1, 0.1;
        VectorXd params(1);
        params << 2.0;


        // Example alpha parameters (As, Ex) or (As1, Ex1, ..., AsN, ExN)
        MatrixXd alpha_params(2, measured_var_n);
        alpha_params << 0.1, 3.0,
            0.1, 3.0;


        VectorXd reconciled_data = DRparamEq(example_depend_func, msrd_data, error_params, params);

        cout << "Reconciled Data:" << endl;
        cout << reconciled_data.transpose() << endl;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
