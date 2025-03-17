#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <numeric>
#include <iomanip>

// Вспомогательные функции для работы с векторами
template <typename T>
std::vector<T> diff(const std::vector<T>& vec) {
    std::vector<T> result(vec.size() - 1);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = vec[i + 1] - vec[i];
    }
    return result;
}

template <typename T>
std::vector<T> cummax(const std::vector<T>& vec) {
    std::vector<T> result(vec.size());
    T current_max = vec[0];
    for (size_t i = 0; i < vec.size(); ++i) {
        current_max = std::max(current_max, vec[i]);
        result[i] = current_max;
    }
    return result;
}

template <typename T>
std::vector<T> linspace(T start, T end, size_t num) {
    std::vector<T> result(num);
    T step = (end - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// Вспомогательная функция для линейной интерполяции
double interp1(const std::vector<double>& x, const std::vector<double>& y, double xi, const std::string& method, double extrap) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Размеры x и y должны совпадать.");
    }
    if (xi < x.front()) return extrap;
    if (xi > x.back()) return extrap;

    auto it = std::lower_bound(x.begin(), x.end(), xi);
    size_t idx = std::distance(x.begin(), it);

    if (idx == 0) return y[0];
    if (idx == x.size()) return y.back();

    double x0 = x[idx - 1];
    double x1 = x[idx];
    double y0 = y[idx - 1];
    double y1 = y[idx];

    return y0 + (y1 - y0) * (xi - x0) / (x1 - x0);
}

// Класс PBox (дискретизированный p-box)
class PBox {
public:
    std::vector<double> x;           // Сетка дискретизации значений аргумента cdf
    std::vector<double> lowerCDF;    // Нижние границы p-box
    std::vector<double> upperCDF;    // Верхние границы p-box

    // Конструктор
    PBox(const std::vector<double>& x, const std::vector<double>& lowerCDF, const std::vector<double>& upperCDF) {
        // Проверка размерностей входных данных
        if (x.size() != lowerCDF.size() || x.size() != upperCDF.size()) {
            throw std::invalid_argument("x, lowerCDF и upperCDF должны быть векторами одинакового размера.");
        }
        for (size_t i = 0; i < lowerCDF.size(); ++i) {
            if (lowerCDF[i] > upperCDF[i]) {
                throw std::invalid_argument("Нижние границы p-box не должны превышать верхние границы p-box для всех значений аргументов.");
            }
            if (i > 0 && (lowerCDF[i] < lowerCDF[i - 1] || upperCDF[i] < upperCDF[i - 1])) {
                throw std::invalid_argument("Функции распределения не могут убывать.");
            }
            if (lowerCDF[i] < 0 || upperCDF[i] < 0) {
                throw std::invalid_argument("Значения функции распределения не могут быть меньше 0.");
            }
            if (lowerCDF[i] > 1 || upperCDF[i] > 1) {
                throw std::invalid_argument("Значения функции распределения не могут быть больше 1.");
            }
        }

        this->x = x;
        this->lowerCDF = lowerCDF;
        this->upperCDF = upperCDF;
    }

    // Оператор сложения двух p-boxes
    PBox operator+(const PBox& other) const {
        // Определение сетки значений аргумента x для результата операции
        double min_z = x.front() + other.x.front();
        double max_z = x.back() + other.x.back();
        std::vector<double> z_values = linspace(min_z, max_z, std::max(x.size(), other.x.size()));

        std::vector<double> lowerZ(z_values.size(), 0.0);
        std::vector<double> upperZ(z_values.size(), 0.0);

        for (size_t i = 0; i < z_values.size(); ++i) {
            double z = z_values[i];
            // Для всех значений из x вычисляем разности y = z - x
            std::vector<double> y_B;
            for (double x_A : x) {
                y_B.push_back(z - x_A);
            }

            // Оценка значения cdf для B в точках y_B
            std::vector<double> F_B_lower, F_B_upper;
            for (double y : y_B) {
                F_B_lower.push_back(interp1(other.x, other.lowerCDF, y, "linear", 0.0));
                F_B_upper.push_back(interp1(other.x, other.upperCDF, y, "linear", 1.0));
            }

            // Вычисление нижней границы для Z в точке z
            std::vector<double> temp_lower;
            for (size_t j = 0; j < lowerCDF.size(); ++j) {
                temp_lower.push_back(lowerCDF[j] + F_B_lower[j] - 1);
            }
            lowerZ[i] = std::max(0.0, *std::max_element(temp_lower.begin(), temp_lower.end()));

            // Вычисление верхней границы для Z в точке z
            std::vector<double> temp_upper;
            for (size_t j = 0; j < upperCDF.size(); ++j) {
                temp_upper.push_back(upperCDF[j] + F_B_upper[j]);
            }
            upperZ[i] = std::min(1.0, *std::min_element(temp_upper.begin(), temp_upper.end()));
        }

        // Проверка, что функция распределения не убывает
        lowerZ = cummax(lowerZ);
        upperZ = cummax(upperZ);

        // Приведение к [0, 1]
        for (double& val : lowerZ) val = std::max(val, 0.0);
        for (double& val : upperZ) val = std::min(val, 1.0);

        return PBox(z_values, lowerZ, upperZ);
    }

    // Оператор умножения на коэффициент
    PBox operator*(double k) const {
        std::vector<double> z_values;
        for (double val : x) {
            z_values.push_back(k * val);
        }

        std::vector<double> lowerZ, upperZ;
        if (k >= 0) {
            lowerZ = lowerCDF;
            upperZ = upperCDF;
        } else {
            z_values = std::vector<double>(z_values.rbegin(), z_values.rend());
            lowerZ.resize(upperCDF.size());
            upperZ.resize(lowerCDF.size());
            std::transform(upperCDF.rbegin(), upperCDF.rend(), lowerZ.begin(), [](double val) { return 1.0 - val; });
            std::transform(lowerCDF.rbegin(), lowerCDF.rend(), upperZ.begin(), [](double val) { return 1.0 - val; });
        }

        return PBox(z_values, lowerZ, upperZ);
    }

    // Определение величины возможного среднеквадратического отклонения
    std::pair<double, double> getStd(const std::string& method) const {
        size_t n = x.size();

        size_t ind_left_lo = 0;
        for (size_t i = 0; i < n; ++i) {
            if (lowerCDF[i] < 0.1) ind_left_lo = i;
        }
        ind_left_lo = std::min(ind_left_lo + 1, n - 1);

        size_t ind_left_hi = 0;
        for (size_t i = 0; i < n; ++i) {
            if (upperCDF[i] < 0.1) ind_left_hi = i;
        }
        ind_left_hi = std::min(ind_left_hi + 1, n - 1);

        size_t ind_right_lo = 0;
        for (size_t i = 0; i < n; ++i) {
            if (lowerCDF[i] < 0.9) ind_right_lo = i;
        }
        ind_right_lo = std::min(ind_right_lo + 1, n - 1);

        size_t ind_right_hi = 0;
        for (size_t i = 0; i < n; ++i) {
            if (upperCDF[i] < 0.9) ind_right_hi = i;
        }
        ind_right_hi = std::min(ind_right_hi + 1, n - 1);

        std::pair<double, double> stdCint;
        stdCint.first = std::max(0.25 * (x[ind_right_hi] - x[ind_left_lo]), 0.0);
        stdCint.second = 0.25 * (x[ind_right_lo] - x[ind_left_hi]);

        return stdCint;
    }

    // Преобразование в Hist
    Hist PBox2Hist(size_t numBins = 0) const {
        if (numBins == 0) numBins = x.size();

        // Дискретизация носителя
        double x_min = x.front();
        double x_max = x.back();
        std::vector<double> edges = linspace(x_min, x_max, numBins + 1);

        std::vector<double> lowerPDF(numBins, 0.0);
        std::vector<double> upperPDF(numBins, 0.0);

        for (size_t i = 0; i < numBins; ++i) {
            double left = edges[i];
            double right = edges[i + 1];

            // Вычисление значений CDF в границах полос гистограммы
            size_t idx_left = std::distance(x.begin(), std::lower_bound(x.begin(), x.end(), left));
            size_t idx_right = std::distance(x.begin(), std::lower_bound(x.begin(), x.end(), right));

            // Валидация
            idx_left = std::min(idx_left, x.size() - 1);
            idx_right = std::min(idx_right, x.size() - 1);

            // Вычисление границ для вероятности
            double p_lower = std::max(lowerCDF[idx_right] - upperCDF[idx_left], 0.0);
            double p_upper = std::min(upperCDF[idx_right] - lowerCDF[idx_left], 1.0);

            // Определение границ для PDF
            double bin_width = right - left;
            lowerPDF[i] = p_lower / bin_width;
            upperPDF[i] = p_upper / bin_width;
        }

        return Hist(edges, lowerPDF, upperPDF);
    }

    // Преобразование в DempsterShafer
    DempsterShafer PBox2DempsterShafer(size_t numFocal = 0) const {
        if (numFocal == 0) numFocal = x.size();

        // Дискретизация в фокальные элементы
        std::vector<double> x_vals = linspace(x.front(), x.back(), numFocal + 1);
        std::vector<std::pair<double, double>> intervals;
        for (size_t i = 0; i < numFocal; ++i) {
            intervals.emplace_back(x_vals[i], x_vals[i + 1]);
        }

        std::vector<double> prob_low = linspace(lowerCDF.front(), lowerCDF.back(), numFocal + 1);
        std::vector<double> prob_high = linspace(upperCDF.front(), upperCDF.back(), numFocal + 1);
        std::vector<double> masses = diff(prob_low);
        for (size_t i = 0; i < masses.size(); ++i) {
            masses[i] = 0.5 * (masses[i] + diff(prob_high)[i]);
        }

        // Нормализация масс
        double sum_masses = std::accumulate(masses.begin(), masses.end(), 0.0);
        for (double& mass : masses) mass /= sum_masses;

        return DempsterShafer(intervals, masses);
    }

    // Преобразование в Fuzzy
    Fuzzy PBox2Fuzzy(size_t numPoints = 0) const {
        if (numPoints == 0) numPoints = x.size();

        // Дискретизация носителя x
        std::vector<double> x_vals = linspace(x.front(), x.back(), numPoints);
        std::vector<double> membership(x_vals.size(), 0.0);

        // Вычисление значений функции принадлежности как (1 - CDF) (мера возможности)
        for (size_t i = 0; i < x_vals.size(); ++i) {
            double lowerCDF_interp = interp1(x, lowerCDF, x_vals[i], "linear", 0.0);
            double upperCDF_interp = interp1(x, upperCDF, x_vals[i], "linear", 1.0);
            if (upperCDF_interp > 1 - lowerCDF_interp) {
                membership[i] = 1 - lowerCDF_interp;
            } else {
                membership[i] = upperCDF_interp;
            }
        }

        // Валидация
        double max_membership = *std::max_element(membership.begin(), membership.end());
        for (double& val : membership) val /= max_membership;

        return Fuzzy(x_vals, membership);
    }

    // Преобразование в FuzzyInterval
    FuzzyInterval PBox2FuzzyInterval(size_t numAlpha = 0) const {
        if (numAlpha == 0) numAlpha = x.size();

        // Определение уровней значимости
        std::vector<double> alphaLevels = linspace(0.0, 1.0, numAlpha);
        std::reverse(alphaLevels.begin(), alphaLevels.end());

        std::vector<std::pair<double, double>> intervals(numAlpha);

        for (size_t i = 0; i < numAlpha; ++i) {
            double alpha = alphaLevels[i];
            double lower_bound = interp1(upperCDF, x, 1 - alpha, "linear", x.front());
            double upper_bound = interp1(lowerCDF, x, 1 - alpha, "linear", x.back());
            intervals[i] = {lower_bound, upper_bound};
        }

        // Валидация структуры вложенных интервалов
        for (size_t i = 1; i < numAlpha; ++i) {
            intervals[i].first = std::min(intervals[i - 1].first, intervals[i].first);
            intervals[i].second = std::max(intervals[i - 1].second, intervals[i].second);
        }

        return FuzzyInterval(alphaLevels, intervals);
    }
};

int main() {
    // Пример использования
    std::vector<double> x = {0, 1, 2, 3};
    std::vector<double> lowerCDF = {0.0, 0.1, 0.4, 1.0};
    std::vector<double> upperCDF = {0.0, 0.3, 0.7, 1.0};

    PBox pbox(x, lowerCDF, upperCDF);
    PBox pbox2 = pbox * 2.0;

    auto stdDev = pbox.getStd("method");
    std::cout << "Среднеквадратическое отклонение: (" << stdDev.first << ", " << stdDev.second << ")" << std::endl;

    return 0;
}