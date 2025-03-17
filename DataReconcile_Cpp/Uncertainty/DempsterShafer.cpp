#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>  // Для работы с матрицами и векторами

using namespace std;
using namespace Eigen;

// Класс для Dempster-Shafer (структура Демпстера-Шафера)
class DempsterShafer {
public:
    // Конструктор
    DempsterShafer(const MatrixXd& intervals, const VectorXd& masses) {
        // Проверка входных аргументов
        if (intervals.cols() != 2) {
            throw invalid_argument("Поле Intervals должно быть матрицей размера N×2.");
        }
        for (int i = 0; i < intervals.rows(); ++i) {
            if (intervals(i, 0) > intervals(i, 1)) {
                throw invalid_argument("В поле Interval левые границы интервалов должны быть меньше правых границ.");
            }
        }
        if (intervals.rows() != masses.size()) {
            throw invalid_argument("Количество фокальных элементов должно совпадать с переданным числом масс.");
        }
        for (int i = 0; i < masses.size(); ++i) {
            if (masses(i) < 0 || masses(i) > 1) {
                throw invalid_argument("Значения в поле Masses должны быть от 0 до 1.");
            }
        }
        double threshold = 1e-2;
        if (abs(masses.sum() - 1) > threshold) {
            throw invalid_argument("Сумма масс должна быть равна 1 (текущая сумма: " + to_string(masses.sum()) + ").");
        }

        this->intervals = intervals;
        this->masses = masses;
    }

    // Сложение по правилу Демпстера (нормализированное)
    DempsterShafer operator+(const DempsterShafer& other) const {
        // Проверка типов операндов
        if (typeid(*this) != typeid(other)) {
            throw invalid_argument("Оба операнда должны быть объектами типа DempsterShafer.");
        }

        // Вычисление всех возможных сумм границ интервалов
        MatrixXd sum_intervals(intervals.rows() * other.intervals.rows(), 2);
        VectorXd comb_masses(intervals.rows() * other.intervals.rows());

        int idx = 0;
        for (int i = 0; i < intervals.rows(); ++i) {
            for (int j = 0; j < other.intervals.rows(); ++j) {
                sum_intervals.row(idx) = intervals.row(i) + other.intervals.row(j);
                comb_masses(idx) = masses(i) * other.masses(j);
                ++idx;
            }
        }

        // Во избежание комбинаторного взрыва, объединяем близкие границы
        MatrixXd unique_int;
        VectorXd sum_masses;
        // Здесь должна быть реализована функция uniquetol и accumarray (аналоги MATLAB)
        // В данном примере опущена для упрощения

        // Нормализация после агрегации
        if (sum_masses.size() < comb_masses.size()) {
            sum_masses /= sum_masses.sum();
        }

        return DempsterShafer(unique_int, sum_masses);
    }

    // Умножение на константу
    DempsterShafer operator*(double k) const {
        MatrixXd z_intervals = k * intervals;
        if (k < 0) {
            z_intervals.col(0).swap(z_intervals.col(1));
        }
        return DempsterShafer(z_intervals, masses);
    }

    // Определение величины возможного среднеквадратического отклонения
    VectorXd getStd() const {
        int n = intervals.rows();
        VectorXd stdCint(2);

        // 1. Минимальное значение среднеквадратического отклонения
        VectorXd midpoints = intervals.rowwise().mean();
        double mean_mid = masses.dot(midpoints);
        double between_var = (masses.array() * (midpoints.array() - mean_mid).square()).sum();
        stdCint(0) = sqrt(between_var);

        // 2. Максимальное значение дисперсии
        double within_var = (masses.array() * (intervals.col(1) - intervals.col(0)).square() / 4).sum();
        stdCint(1) = sqrt(within_var + between_var);

        return stdCint;
    }

    // Построение графика фокальных элементов с массами
    void plot() const {
        // Здесь должна быть реализована функция для построения графиков
        // В данном примере опущена для упрощения
        cout << "График фокальных элементов с массами" << endl;
    }

    // Преобразование в PBox
    PBox DempsterShafer2PBox(int numPoints = -1) const {
        if (numPoints == -1) {
            numPoints = masses.size();
        }
        // Дискретизация носителя
        VectorXd x_vals = VectorXd::LinSpaced(numPoints, intervals.minCoeff(), intervals.maxCoeff());
        MatrixXd borders(numPoints, 2);

        for (int i = 0; i < numPoints; ++i) {
            double x = x_vals(i);
            // belief (нижняя граница CDF)
            borders(i, 0) = (masses.array() * (intervals.col(0).array() <= x).cast<double>()).sum();
            // plausibility (верхняя граница CDF)
            borders(numPoints - i - 1, 1) = (masses.array() * (intervals.col(1).array() >= x).cast<double>()).sum();
        }

        VectorXd lowerCDF = borders.col(0).cwiseMin(borders.col(1));
        VectorXd upperCDF = borders.col(0).cwiseMax(borders.col(1));

        return PBox(x_vals, lowerCDF, upperCDF);
    }

    // Преобразование в Hist
    Hist DempsterShafer2Hist(int numBins = -1) const {
        if (numBins == -1) {
            numBins = intervals.rows();
        }
        // Определение границ полос гистограммы
        VectorXd all_points(intervals.rows() * 2);
        all_points << intervals.col(0), intervals.col(1);
        VectorXd edges = VectorXd::LinSpaced(numBins + 1, all_points.minCoeff(), all_points.maxCoeff());

        VectorXd lowerPDF(numBins);
        VectorXd upperPDF(numBins);

        for (int i = 0; i < numBins; ++i) {
            double left = edges(i);
            double right = edges(i + 1);

            // Оценка границ вероятностей
            lowerPDF(i) = (masses.array() * (intervals.col(0).array() >= left && intervals.col(1).array() <= right).cast<double>()).sum();
            upperPDF(i) = (masses.array() * (intervals.col(0).array() <= right && intervals.col(1).array() >= left).cast<double>()).sum();
        }

        // Нормализация по длине полос
        VectorXd bin_widths = edges.tail(numBins) - edges.head(numBins);
        lowerPDF = lowerPDF.array() / bin_widths.array();
        upperPDF = upperPDF.array() / bin_widths.array();

        return Hist(edges, lowerPDF, upperPDF);
    }

    // Преобразование в Fuzzy
    Fuzzy DempsterShafer2Fuzzy(int numPoints = -1) const {
        if (numPoints == -1) {
            numPoints = masses.size();
        }
        // Дискретизация носителя
        VectorXd x_vals = VectorXd::LinSpaced(numPoints, intervals.minCoeff(), intervals.maxCoeff());
        VectorXd membership(numPoints);

        // Вычисление plausibility для каждой точки
        for (int i = 0; i < numPoints; ++i) {
            double x = x_vals(i);
            membership(i) = (masses.array() * (intervals.col(0).array() <= x && intervals.col(1).array() >= x).cast<double>()).sum();
        }

        return Fuzzy(x_vals, membership / membership.maxCoeff());
    }

    // Преобразование в FuzzyInterval
    FuzzyInterval DempsterShafer2FuzzyInterval() const {
        // Используем значения масс в качестве уровней значимости
        VectorXd sortedMasses = masses;
        sort(sortedMasses.data(), sortedMasses.data() + sortedMasses.size(), greater<double>());
        VectorXd alphaLevels = sortedMasses.cumsum();

        // Вычисление вложенных интервалов
        MatrixXd sorted_intervals = intervals;
        sort(sorted_intervals.data(), sorted_intervals.data() + sorted_intervals.size());

        for (int i = 1; i < sorted_intervals.rows(); ++i) {
            sorted_intervals(i, 0) = min(sorted_intervals(i - 1, 0), sorted_intervals(i, 0));
            sorted_intervals(i, 1) = max(sorted_intervals(i - 1, 1), sorted_intervals(i, 1));
        }

        return FuzzyInterval(alphaLevels.reverse(), sorted_intervals);
    }

private:
    MatrixXd intervals;  // Массив интервалов для фокальных элементов [N×2]
    VectorXd masses;     // Соответствующие им массы [N×1]
    static constexpr double tolerance_threshold = 0.2;  // Порог для объединения близких границ
};

// Пример использования
int main() {
    try {
        // Пример создания объекта DempsterShafer
        MatrixXd intervals(4, 2);
        intervals << -1, -0.5,
                     -0.5, 0,
                     0, 1,
                     1, 1.5;
        VectorXd masses(4);
        masses << 0.2, 0.3, 0.4, 0.1;

        DempsterShafer ds(intervals, masses);

        // Пример использования методов
        VectorXd stdCint = ds.getStd();
        cout << "Среднеквадратическое отклонение (приближенное): " << stdCint.transpose() << endl;

        // Построение графика
        ds.plot();
    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
    }

    return 0;
}