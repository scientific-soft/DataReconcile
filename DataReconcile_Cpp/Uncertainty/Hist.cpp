#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <numeric>
#include <utility>
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
std::vector<T> cumsum(const std::vector<T>& vec) {
    std::vector<T> result(vec.size());
    std::partial_sum(vec.begin(), vec.end(), result.begin());
    return result;
}


// Класс Hist (дискретизированная гистограмма, pdf)
class Hist {
public:
    std::vector<double> x;           // Границы полос гистограммы
    std::vector<double> lowerPDF;    // Нижние границы высоты полосы гистограммы
    std::vector<double> upperPDF;    // Верхние границы высоты полосы гистограммы

    static constexpr double tolerance_threshold = 0.1; // Порог толерантности

    // Конструктор
    Hist(const std::vector<double>& x, const std::vector<double>& lowerPDF, const std::vector<double>& upperPDF) {
        // Проверка размерностей входных данных
        if (x.size() < 2) {
            throw std::invalid_argument("В гистограмме должна быть хотя бы одна полоса.");
        }
        if (x.size() != lowerPDF.size() + 1 || x.size() != upperPDF.size() + 1) {
            throw std::invalid_argument("lowerPDF и upperPDF должны быть векторами одинакового размера, x иметь на одно значение больше.");
        }
        for (size_t i = 0; i < lowerPDF.size(); ++i) {
            if (lowerPDF[i] > upperPDF[i]) {
                throw std::invalid_argument("Нижние границы полос гистограммы не должны превышать верхние границы гистограммы для всех значений аргументов.");
            }
            if (lowerPDF[i] < 0 || upperPDF[i] < 0) {
                throw std::invalid_argument("Значения высоты полос гистограммы не могут быть меньше 0.");
            }
        }

        this->x = x;
        this->lowerPDF = lowerPDF;
        this->upperPDF = upperPDF;

        // Сильная проверка условия нормировки
        double threshold = 1e-2;
        double meanPDF = 0.0;
        for (size_t i = 0; i < lowerPDF.size(); ++i) {
            meanPDF += (lowerPDF[i] + upperPDF[i]) / 2.0 * (x[i + 1] - x[i]);
        }
        if (std::abs(meanPDF - 1.0) > threshold) {
            std::cerr << "Предупреждение: Площадь гистограммы не равна 1.0 при условии симметричности границ значений PDF." << std::endl;
        }

        // Слабая проверка условия нормировки
        double lowerSum = 0.0, upperSum = 0.0;
        for (size_t i = 0; i < lowerPDF.size(); ++i) {
            lowerSum += lowerPDF[i] * (x[i + 1] - x[i]);
            upperSum += upperPDF[i] * (x[i + 1] - x[i]);
        }
        if (lowerSum > 1.0 || upperSum < 1.0) {
            std::cerr << "Предупреждение: Площадь гистограммы не равна 1.0." << std::endl;
        }

        // Проверка на превышение вероятности
        for (size_t i = 0; i < upperPDF.size(); ++i) {
            if (upperPDF[i] * (x[i + 1] - x[i]) > 1.0) {
                std::cerr << "Предупреждение: Площади полос гистограммы не могут быть больше 1.0." << std::endl;
                break;
            }
        }
    }

    // Оператор сложения двух гистограмм
    Hist operator+(const Hist& other) const {
        // Генерация всех возможных значений суммы границ полос гистограмм
        std::vector<double> all_edges;
        for (double a : x) {
            for (double b : other.x) {
                all_edges.push_back(a + b);
            }
        }

        // Устранение дубликатов и сортировка
        std::sort(all_edges.begin(), all_edges.end());
        auto last = std::unique(all_edges.begin(), all_edges.end(), [](double a, double b) {
            return std::abs(a - b) < tolerance_threshold;
        });
        all_edges.erase(last, all_edges.end());

        std::vector<double> lower_pdf(all_edges.size() - 1, 0.0);
        std::vector<double> upper_pdf(all_edges.size() - 1, 0.0);

        // Вычисление вероятностей для каждой полосы
        for (size_t k = 0; k < all_edges.size() - 1; ++k) {
            double z_min = all_edges[k];
            double z_max = all_edges[k + 1];

            double total_lower = 0.0;
            double total_upper = 0.0;

            for (size_t i = 0; i < x.size() - 1; ++i) {
                for (size_t j = 0; j < other.x.size() - 1; ++j) {
                    double sum_min = x[i] + other.x[j];
                    double sum_max = x[i + 1] + other.x[j + 1];

                    double overlap_min = std::max(z_min, sum_min);
                    double overlap_max = std::min(z_max, sum_max);

                    if (overlap_max > overlap_min) {
                        double fraction = (overlap_max - overlap_min) / (sum_max - sum_min);
                        total_lower += lowerPDF[i] * other.lowerPDF[j] * fraction;
                        total_upper += upperPDF[i] * other.upperPDF[j] * fraction;
                    }
                }
            }

            double bin_width = all_edges[k + 1] - all_edges[k];
            lower_pdf[k] = total_lower / bin_width;
            upper_pdf[k] = total_upper / bin_width;
        }

        return Hist(all_edges, lower_pdf, upper_pdf);
    }

    // Оператор умножения на коэффициент
    Hist operator*(double k) const {
        std::vector<double> z_values;
        for (double val : x) {
            z_values.push_back(k * val);
        }

        std::vector<double> lowerZ, upperZ;
        if (k >= 0) {
            for (double val : lowerPDF) {
                lowerZ.push_back(val / k);
            }
            for (double val : upperPDF) {
                upperZ.push_back(val / k);
            }
        } else {
            for (auto it = lowerPDF.rbegin(); it != lowerPDF.rend(); ++it) {
                lowerZ.push_back(-(*it) / k);
            }
            for (auto it = upperPDF.rbegin(); it != upperPDF.rend(); ++it) {
                upperZ.push_back(-(*it) / k);
            }
        }

        return Hist(z_values, lowerZ, upperZ);
    }

    // Вычисление среднеквадратического отклонения
    std::pair<double, double> getStd(const std::string& method) const {
        std::vector<double> binEdges = x;
        size_t nBins = binEdges.size() - 1;
        std::vector<double> binWidths = diff(binEdges);
        std::vector<double> binMids(nBins);
        for (size_t i = 0; i < nBins; ++i) {
            binMids[i] = 0.5 * (binEdges[i] + binEdges[i + 1]);
        }

        std::vector<double> lowerProbs = lowerPDF;
        std::vector<double> upperProbs = upperPDF;
        for (size_t i = 0; i < nBins; ++i) {
            lowerProbs[i] *= binWidths[i];
            upperProbs[i] *= binWidths[i];
        }

        double m1 = 0.0;
        for (size_t i = 0; i < nBins; ++i) {
            m1 += (lowerProbs[i] + upperProbs[i]) / 2.0 * binMids[i];
        }

        std::pair<double, double> varCint;
        varCint.first = 0.0;
        varCint.second = 0.0;
        for (size_t i = 0; i < nBins; ++i) {
            varCint.first += lowerProbs[i] * std::pow(binMids[i] - m1, 2) + (1.0 / 12.0) * lowerProbs[i] * std::pow(binWidths[i], 2);
            varCint.second += upperProbs[i] * std::pow(binMids[i] - m1, 2) + (1.0 / 12.0) * upperProbs[i] * std::pow(binWidths[i], 2);
        }

        return std::make_pair(std::sqrt(varCint.first), std::sqrt(varCint.second));
    }

    // Преобразование в PBox
    PBox Hist2PBox() const {
        std::vector<double> lowerProbs = lowerPDF;
        std::vector<double> upperProbs = upperPDF;
        for (size_t i = 0; i < lowerProbs.size(); ++i) {
            lowerProbs[i] *= (x[i + 1] - x[i]);
            upperProbs[i] *= (x[i + 1] - x[i]);
        }

        std::vector<double> lowerCDF = cumsum(lowerProbs);
        std::vector<double> upperCDF = cumsum(upperProbs);

        lowerCDF.insert(lowerCDF.begin(), 0.0);
        upperCDF.insert(upperCDF.begin(), 0.0);

        for (double& val : lowerCDF) val = std::min(val, 1.0);
        for (double& val : upperCDF) val = std::min(val, 1.0);

        if (lowerCDF.back() < 1.0) {
            lowerCDF.push_back(1.0);
            upperCDF.push_back(1.0);
        }

        return PBox(x, lowerCDF, upperCDF);
    }

    // Преобразование в DempsterShafer
    DempsterShafer Hist2DempsterShafer() const {
        std::vector<std::pair<double, double>> intervals;
        for (size_t i = 0; i < x.size() - 1; ++i) {
            intervals.emplace_back(x[i], x[i + 1]);
        }

        std::vector<double> masses(lowerPDF.size());
        for (size_t i = 0; i < masses.size(); ++i) {
            masses[i] = (lowerPDF[i] + upperPDF[i]) / 2.0 * (x[i + 1] - x[i]);
        }

        double sum_masses = std::accumulate(masses.begin(), masses.end(), 0.0);
        for (double& mass : masses) mass /= sum_masses;

        return DempsterShafer(intervals, masses);
    }

    // Преобразование в Fuzzy
    Fuzzy Hist2Fuzzy(size_t numUniverse = 0) const {
        if (numUniverse == 0) numUniverse = x.size() - 1;

        std::vector<double> bin_centers;
        for (size_t i = 0; i < x.size() - 1; ++i) {
            bin_centers.push_back(0.5 * (x[i] + x[i + 1]));
        }

        std::vector<double> x_universe = linspace(x.front(), x.back(), numUniverse);
        std::vector<double> membership(x_universe.size(), 0.0);

        for (size_t i = 0; i < x_universe.size(); ++i) {
            for (size_t j = 0; j < bin_centers.size(); ++j) {
                if (x_universe[i] >= x[j] && x_universe[i] <= x[j + 1]) {
                    membership[i] = upperPDF[j] * (x[j + 1] - x[j]);
                    break;
                }
            }
        }

        return Fuzzy(x_universe, membership);
    }

    // Преобразование в FuzzyInterval
    FuzzyInterval Hist2FuzzyInterval(size_t numAlpha = 0) const {
        if (numAlpha == 0) numAlpha = x.size() - 1;

        std::vector<double> alpha_levels = linspace(0.0, 1.0, numAlpha);
        std::reverse(alpha_levels.begin(), alpha_levels.end());

        std::vector<std::pair<double, double>> intervals(numAlpha);
        for (size_t i = 0; i < numAlpha; ++i) {
            double alpha = alpha_levels[i];
            bool valid = false;
            for (size_t j = 0; j < upperPDF.size(); ++j) {
                if (upperPDF[j] >= alpha) {
                    intervals[i].first = x[j];
                    intervals[i].second = x[j + 1];
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                intervals[i] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
            }
        }

        return FuzzyInterval(alpha_levels, intervals);
    }

private:
    // Вспомогательная функция для генерации линейного пространства
    std::vector<double> linspace(double start, double end, size_t num) const {
        std::vector<double> result(num);
        double step = (end - start) / (num - 1);
        for (size_t i = 0; i < num; ++i) {
            result[i] = start + i * step;
        }
        return result;
    }
};

int main() {
    // Пример использования
    std::vector<double> x = {0, 1, 2, 3};
    std::vector<double> lowerPDF = {0.1, 0.2, 0.3};
    std::vector<double> upperPDF = {0.2, 0.3, 0.4};

    Hist hist(x, lowerPDF, upperPDF);
    Hist hist2 = hist * 2.0;

    auto stdDev = hist.getStd("method");
    std::cout << "Среднеквадратическое отклонение: (" << stdDev.first << ", " << stdDev.second << ")" << std::endl;

    return 0;
}