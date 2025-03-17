#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>  // Для работы с матрицами и векторами

using namespace std;
using namespace Eigen;

// Класс для переменных типа FuzzyInterval (нечеткий интервал)
class FuzzyInterval {
public:
    // Конструктор
    FuzzyInterval(const VectorXd& alphaLevels, const MatrixXd& intervals) {
        // Проверка входных аргументов
        if (intervals.cols() != 2) {
            throw invalid_argument("Границы вложенных интервалов должны задаваться строкой из двух значений.");
        }
        if (alphaLevels.size() != intervals.rows()) {
            throw invalid_argument("Количество уровней alpha-cut должно соответствовать числу вложенных интервалов.");
        }
        for (int i = 0; i < alphaLevels.size(); ++i) {
            if (alphaLevels(i) < 0 || alphaLevels(i) > 1) {
                throw invalid_argument("Уровни alpha-cut должны иметь значения в пределах от 0 до 1.");
            }
        }
        for (int i = 0; i < intervals.rows(); ++i) {
            if (intervals(i, 0) > intervals(i, 1)) {
                throw invalid_argument("Правые границы вложенных интервалов должны быть меньше левых границ.");
            }
        }

        // Сортировка по значениям alpha-cut
        vector<size_t> idx(alphaLevels.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&alphaLevels](size_t i1, size_t i2) { return alphaLevels(i1) > alphaLevels(i2); });

        this->alphaLevels.resize(alphaLevels.size());
        this->intervals.resize(intervals.rows(), intervals.cols());
        for (size_t i = 0; i < idx.size(); ++i) {
            this->alphaLevels(i) = alphaLevels(idx[i]);
            this->intervals.row(i) = intervals.row(idx[i]);
        }

        // Проверка вложенного характера интервалов Intervals:
        // с уменьшением α-cut вложенные интервалы должны расширяться.
        VectorXd sortedLeft = this->intervals.col(0);
        VectorXd sortedRight = this->intervals.col(1);
        sort(sortedLeft.data(), sortedLeft.data() + sortedLeft.size(), greater<double>());
        sort(sortedRight.data(), sortedRight.data() + sortedRight.size());

        if ((this->intervals.col(0) - sortedLeft).norm() > 1e-6 || (this->intervals.col(1) - sortedRight).norm() > 1e-6) {
            throw invalid_argument("Вложенность интервалов в поле Intervals нарушена.");
        }
    }

    // Определение величины возможного среднеквадратического отклонения (согласно теореме Крейновича-Тао).
    Vector2d getStd() const {
        VectorXd widths = intervals.col(1) - intervals.col(0);  // Ширина интервалов

        double alpha = 0.05;
        int ind1 = -1, ind2 = -1;

        for (int i = 0; i < alphaLevels.size(); ++i) {
            if (abs(alphaLevels(i) - alpha) < 1e-6) {
                ind1 = i;
            }
            if (abs(alphaLevels(i) - 1.0) < 1e-6) {
                ind2 = i;
            }
        }

        Vector2d stdCint;

        if (ind1 != -1 && ind2 != -1) {
            stdCint << 0, 0.25 * (widths(ind1) - widths(ind2));
        } else {
            ind1 = -1;
            for (int i = 0; i < alphaLevels.size(); ++i) {
                if (alphaLevels(i) > alpha) {
                    ind1 = i;
                }
            }

            if (ind1 == -1) {
                stdCint << 0, 0.25 * widths(0);
            } else if (ind1 == 0) {
                stdCint << 0, 0.25 * widths(ind1);
            } else {
                // Линейная интерполяция
                double width = (alpha - alphaLevels(ind1 - 1)) * (widths(ind1) - widths(ind1 - 1)) / (alphaLevels(ind1) - alphaLevels(ind1 - 1)) + widths(ind1 - 1);
                stdCint << 0, 0.25 * width;
            }

            if (ind2 != -1) {
                stdCint(1) -= 0.25 * widths(ind2);
            }
        }

        return stdCint;
    }

    // Сложение с использованием принципа Заде (правило max-min)
    FuzzyInterval operator+(const FuzzyInterval& other) const {
        // Проверка типов операндов
        if (typeid(*this) != typeid(other)) {
            throw invalid_argument("Оба операнда должны быть объектами типа FuzzyInterval.");
        }

        // Комбинирование и сортировка уровней значимости
        VectorXd combinedAlpha = VectorXd::Zero(alphaLevels.size() + other.alphaLevels.size());
        combinedAlpha << alphaLevels, other.alphaLevels;
        sort(combinedAlpha.data(), combinedAlpha.data() + combinedAlpha.size(), greater<double>());
        auto last = unique(combinedAlpha.data(), combinedAlpha.data() + combinedAlpha.size());
        combinedAlpha.conservativeResize(last - combinedAlpha.data());

        // Вычисление интервалов на каждом уровне значимости
        MatrixXd sumIntervals(combinedAlpha.size(), 2);
        for (int i = 0; i < combinedAlpha.size(); ++i) {
            double alpha = combinedAlpha(i);
            Vector2d intA = getIntervalAtAlpha(alpha);
            Vector2d intB = other.getIntervalAtAlpha(alpha);
            sumIntervals.row(i) << intA(0) + intB(0), intA(1) + intB(1);
        }

        return FuzzyInterval(combinedAlpha, sumIntervals);
    }

    // Вложенный интервал на заданном уровне значимости (с интерполяцией)
    Vector2d getIntervalAtAlpha(double alpha) const {
        if (alpha < 0 || alpha > 1) {
            throw invalid_argument("Значение alpha должно быть от 0 до 1.");
        }

        // Граничные значения α
        int idxHigh = -1, idxLow = -1;
        for (int i = 0; i < alphaLevels.size(); ++i) {
            if (alphaLevels(i) >= alpha) {
                idxHigh = i;
            }
            if (alphaLevels(i) <= alpha) {
                idxLow = i;
                break;
            }
        }

        Vector2d interval;

        if (idxHigh == -1) {
            interval << intervals(0, 0), intervals(0, 1);
        } else if (idxLow == -1) {
            interval << intervals(intervals.rows() - 1, 0), intervals(intervals.rows() - 1, 1);
        } else if (idxHigh == idxLow) {  // Точное соответствие
            interval << intervals(idxHigh, 0), intervals(idxHigh, 1);
        } else {  // Линейная интерполяция
            double alphaHigh = alphaLevels(idxHigh);
            double alphaLow = alphaLevels(idxLow);
            double weight = (alpha - alphaLow) / (alphaHigh - alphaLow);

            // Интерполяция левых границ
            double leftHigh = intervals(idxHigh, 0);
            double leftLow = intervals(idxLow, 0);
            double left = leftHigh + (leftLow - leftHigh) * (1 - weight);

            // Интерполяция правых границ
            double rightHigh = intervals(idxHigh, 1);
            double rightLow = intervals(idxLow, 1);
            double right = rightHigh + (rightLow - rightHigh) * weight;

            interval << left, right;
        }

        return interval;
    }

    // Умножение на коэффициент
    FuzzyInterval operator*(double k) const {
        MatrixXd z_intervals = k * intervals;
        if (k < 0) {
            z_intervals.col(0).swap(z_intervals.col(1));
        }
        return FuzzyInterval(alphaLevels, z_intervals);
    }

    // Графическое отображение функции принадлежности нечеткого интервала
    void plot() const {
        // Здесь должна быть реализована функция для построения графиков
        // В данном примере опущена для упрощения
        cout << "График нечеткого интервала" << endl;
    }

    // Преобразование в PBox
    PBox FuzzyInterval2PBox(int numPoints = -1) const {
        if (numPoints == -1) {
            numPoints = alphaLevels.size();
        }
        VectorXd x_vals = VectorXd::LinSpaced(numPoints, intervals.minCoeff(), intervals.maxCoeff());
        VectorXd lowerCDF = VectorXd::Zero(numPoints);
        VectorXd upperCDF = VectorXd::Zero(numPoints);

        for (int i = 0; i < numPoints; ++i) {
            double x = x_vals(i);
            double bound = (alphaLevels.array() * (intervals.col(0).array() < x).cast<double>()).maxCoeff();
            upperCDF(i) = isnan(bound) ? 0 : min(bound, 1.0);

            bound = 1 - (alphaLevels.array() * (intervals.col(1).array() > x).cast<double>()).maxCoeff();
            lowerCDF(i) = isnan(bound) ? 1 : max(bound, 0.0);
        }

        return PBox(x_vals, lowerCDF, upperCDF);
    }

    // Преобразование в Hist
    Hist FuzzyInterval2Hist(int numBins = -1) const {
        if (numBins == -1) {
            numBins = alphaLevels.size();
        }
        VectorXd edges = VectorXd::LinSpaced(numBins + 1, intervals.minCoeff(), intervals.maxCoeff());
        VectorXd lowerPDF = VectorXd::Zero(numBins);
        VectorXd upperPDF = VectorXd::Zero(numBins);

        for (int i = 0; i < numBins; ++i) {
            double left = edges(i);
            double right = edges(i + 1);

            // Определение максимального уровня значимости, при котором интервал содержит данную полосу
            VectorXd containing = (intervals.col(0).array() <= left && intervals.col(1).array() >= right).cast<double>();
            double max_alpha = (alphaLevels.array() * containing.array()).maxCoeff();

            // Границы PDF пропорционально уровню значимости
            lowerPDF(i) = 0;  // Консервативная нижняя граница
            upperPDF(i) = max_alpha;
        }

        // Определение границ значений PDF
        VectorXd bin_widths = edges.tail(numBins) - edges.head(numBins);
        double total_upper = (upperPDF.array() * bin_widths.array()).sum();
        upperPDF = upperPDF / total_upper;

        return Hist(edges, lowerPDF, upperPDF);
    }

    // Преобразование в DempsterShafer
    DempsterShafer FuzzyInterval2DempsterShafer(const string& method = "fractional") const {
        if (method == "fractional") {
            // Вычисление масс из разностей значений уровня значимости
            VectorXd masses = -0.5 * (alphaLevels.tail(alphaLevels.size() - 1) - alphaLevels.head(alphaLevels.size() - 1));
            masses = VectorXd::Zero(masses.size() * 2);
            masses << masses, masses;

            MatrixXd intervals = MatrixXd::Zero(masses.size(), 2);
            if (abs(alphaLevels(0) - 1.0) < 1e-6) {
                masses = VectorXd::Zero(masses.size() + 1);
                masses << 1.0, masses;
                intervals = MatrixXd::Zero(masses.size(), 2);
                intervals.row(0) << this->intervals(0, 0), this->intervals(0, 1);
            }

            for (int i = 1; i < this->intervals.rows(); ++i) {
                intervals.row(i) << this->intervals(i, 0), this->intervals(i - 1, 0);
                intervals.row(i + this->intervals.rows() - 1) << this->intervals(i - 1, 1), this->intervals(i, 1);
            }

            // Удаление интервалов с нулевой массой
            VectorXd valid = (masses.array() > 0).cast<double>();
            masses = masses.array() * valid.array();
            intervals = intervals.array().colwise() * valid.array();

            return DempsterShafer(intervals, masses / masses.sum());
        } else if (method == "nested") {
            VectorXd masses = - (alphaLevels.tail(alphaLevels.size() - 1) - alphaLevels.head(alphaLevels.size() - 1));
            VectorXd valid = (masses.array() > 0).cast<double>();
            masses = masses.array() * valid.array();
            return DempsterShafer(intervals, masses / masses.sum());
        } else {
            throw invalid_argument("Поле method может быть равно только fractional или nested.");
        }
    }

    // Преобразование в Fuzzy
    Fuzzy FuzzyInterval2Fuzzy(int numPoints = -1) const {
        if (numPoints == -1) {
            numPoints = alphaLevels.size();
        }
        VectorXd x_vals = VectorXd::LinSpaced(numPoints, intervals.minCoeff(), intervals.maxCoeff());
        VectorXd membership = VectorXd::Zero(numPoints);

        // Оценка уровня значимости как максимального alpha, при котором вложенный интервал содержит x
        for (int i = 0; i < numPoints; ++i) {
            double x = x_vals(i);
            VectorXd contains_x = (intervals.col(0).array() <= x && intervals.col(1).array() >= x).cast<double>();
            membership(i) = (alphaLevels.array() * contains_x.array()).maxCoeff();
        }

        return Fuzzy(x_vals, membership);
    }

private:
    VectorXd alphaLevels;  // Уровни α-cut (вектор 1×N)
    MatrixXd intervals;    // Соответствующие вложенные интервалы (матрица N×2)
};

// Пример использования
int main() {
    try {
        // Пример создания объекта FuzzyInterval
        VectorXd alphaLevels(4);
        alphaLevels << 1.0, 0.9, 0.5, 0.1;
        MatrixXd intervals(4, 2);
        intervals << -4, 4,
                     -3, 3,
                     -2, 2,
                     -1, 1;

        FuzzyInterval fi(alphaLevels, intervals);

        // Пример использования методов
        Vector2d stdCint = fi.getStd();
        cout << "Среднеквадратическое отклонение: " << stdCint.transpose() << endl;

        // Построение графика
        fi.plot();
    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
    }

    return 0;
}