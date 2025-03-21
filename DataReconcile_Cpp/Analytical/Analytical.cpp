#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <complex>

using namespace Eigen;
using namespace std;

// Аналитическая модель решения задачи Data Reconciliation
VectorXd AnalyticalModel(const VectorXd& x, const MatrixXd& A_y, const VectorXd& b_y, const VectorXd& c_y, double r2_y) {
    // Гауссова модель распределения погрешностей.
    
    // x - вектор уточняемых результатов измерений,
    // A_y - система линейных ограничений на уравнения связи для результатов уточнения, by - вектор правых частей этих уравнений: A_y*y=b_y;
    //     т.к. y=x+dx, то A_y*(x+dx)=b_y. Следовательно, A_y*dx=b_y-A_y*x. Значит, в модели Ефремова-Козлова А=A_y, b=b_y-A_y*x.
    // с_y и r2_y - совокупное ограничение неравенством (y-c_y)'*(y-c_y)<r2_y. Т.к. y=x+dx, то (x+dx-c_y)'*(x+dx-c_y)<r2_y. 
    // Следовательно, (dx-(c_y-x))'*(dx-(c_y-x))<r2_y. Значит, в модели Ефремова-Козлова c=c_y-x, r2=r2_y.
    // y - результат уточнения
    
    VectorXd x_col = x;  // принудительное приведение к вектор-столбцу
    VectorXd b_y_col = b_y;
    VectorXd c_y_col = c_y;

    MatrixXd A = A_y;
    VectorXd b = A_y * x_col - b_y_col;
    VectorXd c = x_col - c_y_col;
    double r2 = r2_y;

    // Проверки:
    int n = x.size();
    if (c.size() != n) {
        throw invalid_argument("Размерность векторов х и с должны совпадать.");
    }
    if (A.cols() != n) {
        throw invalid_argument("Число столбцов в матрице А должно совпадать с числом элементов в х.");
    }
    if (A.rows() != b.size()) {
        throw invalid_argument("Число строк в матрице А должно совпадать с числом элементов в b.");
    }

    MatrixXd P;
    try {
        P = A.transpose() * (A * A.transpose()).inverse();
    } catch (const exception& e) {
        throw invalid_argument("Матрица A*A^T вырождена, невозможно вычислить обратную матрицу");
    }

    MatrixXd D = MatrixXd::Identity(n, n) - P * A;

    double a = (b.transpose() * P.transpose() * P * b
                - b.transpose() * P.transpose() * c
                - c.transpose() * P * b
                + c.transpose() * c
                - c.transpose() * D * c
                - r2).value();
    double h = 2 * a;
    // У Ефремова-Козлова: double d = (a + c.transpose() * D * c).value();
    double d = (c.transpose() * D * c).value();

    // Решение квадратного уравнения
    double discriminant = h * h - 4 * a * d;
    double root1 = (-h + sqrt(discriminant)) / (2 * a);
    double root2 = (-h - sqrt(discriminant)) / (2 * a);
    VectorXd roots(2);
    roots << root1, root2;

    MatrixXd dx(n, 2);
    for (int i = 0; i < 2; ++i) {
        double lbd = roots(i);
        if (abs(lbd) < 1e-10 || abs(lbd + 1) < 1e-10) {
            dx.col(i) = P * b;
        } else {
            dx.col(i) = P * b + lbd * (D * c) / (1 + lbd);
        }
    }

    // Выбор решения с минимальной нормой
    int idx = 0;
    double min_norm = dx.col(0).squaredNorm();
    if (dx.col(1).squaredNorm() < min_norm) {
        idx = 1;
    }
    VectorXd y = x_col - dx.col(idx);

    return y;
}

VectorXd GetUncertainty(const VectorXd& x, const VectorXd& dx, const MatrixXd& A_y, const VectorXd& b_y, const VectorXd& c_y, double r2_y) {
    // Все обозначения - те же, что и для AnalyticalModel,
    // dx - массив-вектор cell неопределенностей, формализованных по одному из типов.
    int n = dx.size();
    // Оценка частных производных
    double alpha = 1e-100;
    MatrixXd dy_dx = MatrixXd::Zero(n, n);

    for (int i = 0; i < n; ++i) {
        VectorXcd d = VectorXcd::Zero(n);
        d(i) = complex<double>(0, alpha);  // Добавление мнимой части
        VectorXcd perturbed = AnalyticalModel(x.cast<complex<double>>() + d, A_y, b_y, c_y, r2_y);
        dy_dx.col(i) = perturbed.imag() / alpha;
    }

    // Цикл по операциям сложения и умножения по типу неопределенности
    VectorXd dy = VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {
        dy(i) = abs(dy_dx(i, 0)) * dx(0) + abs(dy_dx(i, 1)) * dx(1);
        for (int j = 2; j < n; ++j) {
            dy(i) += abs(dy_dx(i, j)) * dx(j);
        }
    }

    return dy;
}

int main() {
    // Пример использования:
    // x = [x1;x2]; результат измерения x = [1.0; 0.9; 0.9];
    // линейная связь вида 2*x1+3*x2=5; 1*x1+1*x2+1*x3=3
    // групповое ограничение: недопустимо R2 при отклонении от столбца [1.0;1.0;1.0], большее, чем 0.2.

    VectorXd x(3);
    x << 1.0, 0.9, 0.9;

    MatrixXd A(2, 3);
    A << 2, 3, 0,
         1, 1, 1;

    VectorXd b(2);
    b << 5, 3;

    VectorXd c(3);
    c << 1.0, 1.0, 1.0;

    double r2 = 0.2;

    // Уточненный результат Data Reconciliation
    VectorXd y = AnalyticalModel(x, A, b, c, r2);

    VectorXd dx(3);
    dx << 0.1, 0.1, 0.1;  // пределы абсолютной погрешности (FuzzyInterval)
    VectorXd dy = GetUncertainty(x, dx, A, b, c, r2);

    cout << "Уточненные значения: " << y.transpose() << endl;
 
    return 0;
}