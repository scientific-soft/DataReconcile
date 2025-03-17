#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>  // Для работы с матрицами и векторами

using namespace std;
using namespace Eigen;

// Класс для переменных типа Fuzzy (нечеткие переменные)
class Fuzzy {
public:
    // Конструктор
    Fuzzy(const VectorXd& universe, const VectorXd& membership) {
        // Проверка входных переменных
        if (universe.size() != membership.size()) {
            throw invalid_argument("Поля Universe и Membership должны быть одинаковой длины.");
        }
        for (int i = 0; i < membership.size(); ++i) {
            if (membership(i) < 0 || membership(i) > 1) {
                throw invalid_argument("Значения функции принадлежности должны быть от 0 до 1.");
            }
        }

        // Порядок следования значений в поле universe - по возрастанию
        VectorXd sorted_universe = universe;
        VectorXd sorted_membership = membership;
        vector<size_t> idx(universe.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&sorted_universe](size_t i1, size_t i2) { return sorted_universe(i1) < sorted_universe(i2); });

        this->universe.resize(universe.size());
        this->membership.resize(membership.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            this->universe(i) = sorted_universe(idx[i]);
            this->membership(i) = sorted_membership(idx[i]);
        }
    }

    // Сложение с использованием принципа Заде (правило max-min)
    Fuzzy operator+(const Fuzzy& other) const {
        // Создание результирующей переменной (поле Universe)
        double min_z = universe(0) + other.universe(0);
        double max_z = universe(universe.size() - 1) + other.universe(other.universe.size() - 1);
        double z_step = min(universe(1) - universe(0), other.universe(1) - other.universe(0));
        VectorXd z_universe = VectorXd::LinSpaced(static_cast<int>((max_z - min_z) / z_step) + 1, min_z, max_z);

        // Выделение памяти под функцию принадлежности
        VectorXd z_membership = VectorXd::Zero(z_universe.size());

        // Принцип Заде (композиция)
        for (int i = 0; i < z_universe.size(); ++i) {
            double z = z_universe(i);

            // Все возможные пары значений (x,y), для которых x+y=z.
            VectorXd x_vals = universe;
            VectorXd y_vals = z - x_vals;

            // Определение значений функции принадлежности для значений y_vals в B
            VectorXd y_membership = interp1(other.universe, other.membership, y_vals);

            // Вычисление минимакса от функции принадлежности
            VectorXd combined = x_vals.array().min(y_membership.array());
            z_membership(i) = combined.maxCoeff();
        }

        return Fuzzy(z_universe, z_membership);
    }

    // Умножение на коэффициент
    Fuzzy operator*(double k) const {
        VectorXd z_membership = membership;
        VectorXd z_universe = k * universe;
        if (k < 0) {
            reverse(z_universe.data(), z_universe.data() + z_universe.size());
            reverse(z_membership.data(), z_membership.data() + z_membership.size());
        }
        return Fuzzy(z_universe, z_membership);
    }

    // Вычисление вложенного (α-cut) интервала
    Vector2d alphaCut(double alpha) const {
        // Валидация значения α
        if (alpha < 0 || alpha > 1) {
            throw invalid_argument("Уровень значимости должен быть числом в интервале [0, 1].");
        }

        // Значения индексов, когда Membership >= alpha
        vector<size_t> above;
        for (size_t i = 0; i < membership.size(); ++i) {
            if (membership(i) >= alpha) {
                above.push_back(i);
            }
        }

        if (above.empty()) {
            return {NAN, NAN};
        }

        // Левая граница
        size_t first_idx = above.front();
        double left;
        if (first_idx == 0) {
            left = universe(0);
        } else {
            // Линейная интерполяция
            double x_prev = universe(first_idx - 1);
            double x_curr = universe(first_idx);
            double mu_prev = membership(first_idx - 1);
            double mu_curr = membership(first_idx);
            left = x_prev + (alpha - mu_prev) * (x_curr - x_prev) / (mu_curr - mu_prev);
        }

        // Правая граница
        size_t last_idx = above.back();
        double right;
        if (last_idx == universe.size() - 1) {
            right = universe(universe.size() - 1);
        } else {
            // Линейная интерполяция
            double x_curr = universe(last_idx);
            double x_next = universe(last_idx + 1);
            double mu_curr = membership(last_idx);
            double mu_next = membership(last_idx + 1);
            right = x_curr + (alpha - mu_curr) * (x_next - x_curr) / (mu_next - mu_curr);
        }

        return {left, right};
    }

    // Определение величины возможного среднеквадратического отклонения
    Vector2d getStd() const {
        // Определение значений функции принадлежности для уровня α-cut, равного 0.05.
        Vector2d int0 = alphaCut(0.05);
        Vector2d int1 = alphaCut(1 - numeric_limits<double>::epsilon());

        if (isnan(int1(0)) || isnan(int1(1))) {
            return {0, 0.25 * (int0(1) - int0(0)};
        } else {
            double val1 = 0.5 * (int1(0) - int0(0));
            double val2 = 0.5 * (int0(1) - int1(1));
            return {0, (val1 + val2) / 2};
        }
    }

    // Отображение функции принадлежности на графике
    void plot() const {
        // Здесь должна быть реализована функция для построения графиков
        // В данном примере опущена для упрощения
        cout << "График функции принадлежности" << endl;
    }

    // Преобразование в PBox
    PBox Fuzzy2PBox(int numPoints = -1) const {
        if (numPoints == -1) {
            numPoints = universe.size();
        }
        VectorXd x_vals = VectorXd::LinSpaced(numPoints, universe.minCoeff(), universe.maxCoeff());
        VectorXd lowerCDF = VectorXd::Zero(numPoints);
        VectorXd upperCDF = VectorXd::Zero(numPoints);

        for (int i = 0; i < numPoints; ++i) {
            double x = x_vals(i);
            // Нижняя граница CDF = 1 - Possibility(X > x)
            double bound = 1 - (membership.array() * (universe.array() > x).cast<double>()).maxCoeff();
            lowerCDF(i) = isnan(bound) ? 1 : bound;

            // Верхняя граница CDF = 1 - Necessity(X > x)
            bound = (membership.array() * (universe.array() <= x).cast<double>()).maxCoeff();
            upperCDF(i) = isnan(bound) ? 0 : bound;
        }

        return PBox(x_vals, lowerCDF, upperCDF);
    }

    // Преобразование в Hist
    Hist Fuzzy2Hist(int numBins = -1) const {
        if (numBins == -1) {
            numBins = universe.size();
        }
        VectorXd edges = VectorXd::LinSpaced(numBins + 1, universe.minCoeff(), universe.maxCoeff());
        VectorXd lowerPDF = VectorXd::Zero(numBins);
        VectorXd upperPDF = VectorXd::Zero(numBins);

        for (int i = 0; i < numBins; ++i) {
            double left = edges(i);
            double right = edges(i + 1);

            // Определение полос
            VectorXd in_bin = (universe.array() >= left && universe.array() <= right).cast<double>();

            if (in_bin.sum() > 0) {
                lowerPDF(i) = in_bin.mean() * (membership.array() * in_bin.array()).minCoeff();
                upperPDF(i) = in_bin.mean() * (membership.array() * in_bin.array()).maxCoeff();
            }
        }

        // Валидация
        if (lowerPDF.sum() > 1) {
            throw runtime_error("Оценка нижних границ вероятностей попадания в полосы содержит ошибку.");
        }

        // Оценка значения PDF
        VectorXd bin_widths = edges.tail(numBins) - edges.head(numBins);
        lowerPDF = lowerPDF.array() / bin_widths.array();
        upperPDF = upperPDF.array() / bin_widths.array();

        return Hist(edges, lowerPDF, upperPDF);
    }

    // Преобразование в DempsterShafer
    DempsterShafer Fuzzy2DempsterShafer(int numAlpha = -1) const {
        if (numAlpha == -1) {
            numAlpha = universe.size();
        }
        VectorXd alphaLevels = VectorXd::LinSpaced(numAlpha + 1, 0, 1).tail(numAlpha);
        MatrixXd intervals(numAlpha, 2);
        VectorXd masses = alphaLevels.tail(numAlpha) - alphaLevels.head(numAlpha);

        for (int i = 0; i < numAlpha; ++i) {
            intervals.row(i) = alphaCut(alphaLevels(i));
        }

        return DempsterShafer(intervals, masses);
    }

    // Преобразование в FuzzyInterval
    FuzzyInterval Fuzzy2FuzzyInterval(int numAlpha = -1) const {
        if (numAlpha == -1) {
            numAlpha = universe.size();
        }
        VectorXd alphaLevels = VectorXd::LinSpaced(numAlpha, 0, 1).reverse();
        MatrixXd intervals(numAlpha, 2);

        for (int i = 0; i < numAlpha; ++i) {
            intervals.row(i) = alphaCut(alphaLevels(i));
        }

        return FuzzyInterval(alphaLevels, intervals);
    }

private:
    VectorXd universe;      // Значения носителя (вектор 1xN)
    VectorXd membership;    // Значения функции принадлежности (вектор 1xN, значения из [0,1])

    // Функция для линейной интерполяции
    VectorXd interp1(const VectorXd& x, const VectorXd& y, const VectorXd& xi) const {
        VectorXd yi(xi.size());
        for (int i = 0; i < xi.size(); ++i) {
            if (xi(i) < x(0)) {
                yi(i) = y(0);
            } else if (xi(i) > x(x.size() - 1)) {
                yi(i) = y(y.size() - 1);
            } else {
                int idx = 0;
                while (x(idx) < xi(i)) {
                    ++idx;
                }
                double x0 = x(idx - 1);
                double x1 = x(idx);
                double y0 = y(idx - 1);
                double y1 = y(idx);
                yi(i) = y0 + (y1 - y0) * (xi(i) - x0) / (x1 - x0);
            }
        }
        return yi;
    }
};

// Пример использования
int main() {
    try {
        // Пример создания объекта Fuzzy
        VectorXd universe(6);
        universe << -1, -0.5, 0, 0.5, 1, 1.5;
        VectorXd membership(6);
        membership << 0, 0.2, 0.7, 1.0, 0.5, 0.0;

        Fuzzy fuzzy(universe, membership);

        // Пример использования методов
        Vector2d stdCint = fuzzy.getStd();
        cout << "Среднеквадратическое отклонение: " << stdCint.transpose() << endl;

        // Построение графика
        fuzzy.plot();
    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
    }

    return 0;
}