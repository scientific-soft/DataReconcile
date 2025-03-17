#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <numeric>
#include <random>
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
std::vector<T> linspace(T start, T end, size_t num) {
    std::vector<T> result(num);
    T step = (end - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// Вспомогательная функция для генерации случайных чисел
int randi(int n, std::mt19937& gen) {
    std::uniform_int_distribution<> dis(0, n - 1);
    return dis(gen);
}

// Вспомогательная функция для вычисления квантилей
template <typename T>
T quantile(const std::vector<T>& data, double p) {
    if (data.empty()) throw std::invalid_argument("Данные не могут быть пустыми.");
    if (p < 0 || p > 1) throw std::invalid_argument("Квантиль должен быть в диапазоне [0, 1].");

    std::vector<T> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    double index = p * (sorted_data.size() - 1);
    size_t lower_index = static_cast<size_t>(std::floor(index));
    size_t upper_index = static_cast<size_t>(std::ceil(index));

    if (lower_index == upper_index) {
        return sorted_data[lower_index];
    }

    double weight = index - lower_index;
    return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight;
}

// Класс Sample (одномерная выборка)
class Sample {
public:
    std::vector<double> x;  // Значения в выборке

    // Конструктор
    Sample(const std::vector<double>& x) {
        if (x.empty()) {
            throw std::invalid_argument("В выборке должно быть хотя бы одно значение.");
        }
        for (double val : x) {
            if (std::isnan(val)) {
                throw std::invalid_argument("Выборка не может содержать значения типа NaN.");
            }
        }
        this->x = x;
    }

    // Оператор сложения двух выборок
    Sample operator+(const Sample& other) const {
        size_t n1 = x.size();
        size_t n2 = other.x.size();
        std::vector<double> z_values;

        if (n1 == n2) {
            for (size_t i = 0; i < n1; ++i) {
                z_values.push_back(x[i] + other.x[i]);
            }
        } else if (n1 < n2) {
            z_values = other.x;
            for (size_t i = 0; i < n1; ++i) {
                z_values[i] += x[i];
            }
            std::mt19937 gen(std::random_device{}());
            for (size_t i = n1; i < n2; ++i) {
                z_values[i] += x[randi(n1, gen)];
            }
        } else {
            z_values = x;
            for (size_t i = 0; i < n2; ++i) {
                z_values[i] += other.x[i];
            }
            std::mt19937 gen(std::random_device{}());
            for (size_t i = n2; i < n1; ++i) {
                z_values[i] += other.x[randi(n2, gen)];
            }
        }

        return Sample(z_values);
    }

    // Оператор умножения на коэффициент
    Sample operator*(double k) const {
        std::vector<double> z_values;
        for (double val : x) {
            z_values.push_back(k * val);
        }
        return Sample(z_values);
    }

    // Вычисление среднеквадратического отклонения
    std::pair<double, double> getStd(const std::string& method = "approximate") const {
        size_t n = x.size();
        if (method == "approximate") {
            std::vector<double> sorted_x = x;
            std::sort(sorted_x.begin(), sorted_x.end());

            double q[] = {0.05, 0.95};
            double left[2], right[2];
            for (int i = 0; i < 2; ++i) {
                double k1 = std::max(std::floor(n * q[i] - std::sqrt(n * q[i] * (1 - q[i])) * 1.96), 1.0);
                double k2 = std::min(std::ceil(n * q[i] + std::sqrt(n * q[i] * (1 - q[i])) * 1.96), static_cast<double>(n));
                left[i] = sorted_x[static_cast<size_t>(k1) - 1];
                right[i] = sorted_x[static_cast<size_t>(k2) - 1];
            }
            return std::make_pair(0.25 * (left[1] - right[0]), 0.25 * (right[1] - left[0]));
        } else if (method == "accurate") {
            double s = std::sqrt(std::accumulate(x.begin(), x.end(), 0.0, [](double acc, double val) {
                return acc + val * val;
            }) / n - std::pow(std::accumulate(x.begin(), x.end(), 0.0) / n, 2));

            if (n < 10) {
                size_t M = 100;
                std::vector<double> buf(M);
                std::mt19937 gen(std::random_device{}());
                for (size_t i = 0; i < M; ++i) {
                    std::vector<double> sample;
                    for (size_t j = 0; j < n; ++j) {
                        sample.push_back(x[randi(n, gen)]);
                    }
                    double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / n;
                    buf[i] = std::accumulate(sample.begin(), sample.end(), 0.0, [mean](double acc, double val) {
                        return acc + (val - mean) * (val - mean);
                    }) / n;
                }
                std::sort(buf.begin(), buf.end());
                return std::make_pair(std::sqrt(quantile(buf, 0.05)), std::sqrt(quantile(buf, 0.95)));
            } else {
                double chi2_low = 129;
                double chi2_high = 74; 
                return std::make_pair(s * std::sqrt(n - 1) / std::sqrt(chi2_high), s * std::sqrt(n - 1) / std::sqrt(chi2_low));
            }
        } else {
            throw std::invalid_argument("Аргумент method не задан. Допустимые значения: approximate, accurate.");
        }
    }

    // Преобразование в PBox
    PBox Sample2PBox() const {
        std::vector<double> sorted_x = x;
        std::sort(sorted_x.begin(), sorted_x.end());

        std::vector<double> X = sorted_x;
        std::vector<double> Flo(X.size()), Fup(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            Flo[i] = static_cast<double>(i) / X.size();
            Fup[i] = static_cast<double>(i + 1) / X.size();
        }

        return PBox(X, Flo, Fup);
    }

    // Преобразование в FuzzyInterval
    FuzzyInterval Sample2FuzzyInterval(size_t numAlpha = 0) const {
        if (numAlpha == 0) numAlpha = 5;

        std::vector<double> alphaLevels = linspace(0.0, 1.0, numAlpha);
        std::reverse(alphaLevels.begin(), alphaLevels.end());

        std::vector<std::pair<double, double>> intervals(numAlpha);
        for (size_t i = 0; i < numAlpha; ++i) {
            double alpha = alphaLevels[i];
            double p_low = (1 - alpha) / 2;
            double p_high = (1 + alpha) / 2;
            intervals[i] = {quantile(x, p_low), quantile(x, p_high)};
        }

        return FuzzyInterval(alphaLevels, intervals);
    }
};

int main() {
    // Пример использования
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    Sample sample(x);

    auto stdDev = sample.getStd("approximate");
    std::cout << "Среднеквадратическое отклонение: (" << stdDev.first << ", " << stdDev.second << ")" << std::endl;

    return 0;
}