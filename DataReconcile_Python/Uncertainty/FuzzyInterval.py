import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Преобразование разных типов представления неточных данных к формату,
# унифицированному для дальнейшего применения Data Reconciliation.

# Класс для переменных типа FuzzyInterval (нечеткий интервал)
class FuzzyInterval:
    def __init__(self, alphaLevels, intervals):
        # Проверка входных аргументов
        if intervals.shape[1] != 2:
            raise ValueError('Границы вложенных интервалов должны задаваться строкой из двух значений.')
        if len(alphaLevels) != intervals.shape[0]:
            raise ValueError('Количество уровней alpha-cut должно соответствовать числу вложенных интервалов.')
        if np.any((alphaLevels < 0-1e-6) | (alphaLevels > 1+1e-6)):
            raise ValueError('Уровни alpha-cut должны иметь значения в пределах от 0 до 1.')
        if np.any(intervals[:, 0] > intervals[:, 1]):
            raise ValueError('Правые границы вложенных интервалов должны быть меньше левых границ.')

        # Сортировка по значениям alpha-cut
        idx = np.argsort(alphaLevels)[::-1]
        self.AlphaLevels = np.array(alphaLevels)[idx]
        self.Intervals = np.array(intervals)[idx]

        # # Проверка вложенного характера интервалов Intervals:
        # # с уменьшением α-cut вложенные интервалы должны расширяться.
        # if np.any(self.Intervals[:, 0] != np.sort(self.Intervals[:, 0])[::-1]) or \
        #    np.any(self.Intervals[:, 1] != np.sort(self.Intervals[:, 1])):
        #     raise ValueError('Вложенность интервалов в поле Intervals нарушена.')

    # Определение величины возможного среднеквадратического отклонения (согласно теореме Крейновича-Тао).
    def getStd(self, method='approximate'):
        int = self.Intervals  # Переменная для краткости
        a = self.AlphaLevels  # Переменная для краткости
        widths = int[:, 1] - int[:, 0]  # Ширина интервалов

        if method == 'approximate':
            alpha = 0.05
            ind1 = np.where(a == alpha)[0]
            ind2 = np.where(a == 1)[0]
            if ind1.size > 0 and ind2.size > 0:
                stdCint = np.array([0, 0.25 * (widths[ind1[0]] - widths[ind2[0]])])
            else:
                ind1 = np.where(a > alpha)[0]
                if ind1.size == 0:
                    stdCint = np.array([0, 0.25 * widths[0]])
                else:
                    ind1 = ind1[-1]
                    if ind1 == 0:
                        stdCint = np.array([0, 0.25 * widths[ind1]])
                    else:
                        # Линейная интерполяция
                        width = (alpha - a[ind1 - 1]) * (widths[ind1] - widths[ind1 - 1]) / (a[ind1] - a[ind1 - 1]) + widths[ind1 - 1]
                        stdCint = np.array([0, 0.25 * width])
                    if ind2.size > 0:
                        stdCint[1] -= 0.25 * widths[ind2[0]]
        elif method == 'accurate':
            # Дискретизация носителя, используя границы вложенных интервалов
            breakpoints = np.unique(np.concatenate([self.Intervals[:, 0], self.Intervals[:, 1]]))
            nBins = len(breakpoints) - 1
            binEdges = np.sort(breakpoints)

            # Определение ограничений на вероятности из α-cuts
            A = []
            b = []
            for i in range(len(self.AlphaLevels)):
                alpha = self.AlphaLevels[i]
                a_int = self.Intervals[i, 0]
                b_int = self.Intervals[i, 1]
                inBin = (binEdges[:-1] >= a_int) & (binEdges[1:] <= b_int)

                # Добавление ограничения: sum(p(inBin)) >= 1 - alpha
                A.append(-inBin.astype(float))
                b.append(-(1 - alpha))

            A = np.array(A)
            b = np.array(b)

            # Добавление ограничений на сумму вероятностей
            Aeq = np.ones((1, nBins))
            beq = np.array([1])
            # Добавление ограничения на неотрицательность вероятностей
            lb = np.zeros(nBins)
            ub = None

            # Параметры промежутков
            binMids = (binEdges[:-1] + binEdges[1:]) / 2

            # 1. Минимальное значение дисперсии
            def objFun(p):
                return np.dot(p, binMids**2) - np.dot(p, binMids)**2

            p0 = np.ones(nBins) / nBins
            res = minimize(objFun, p0, constraints={'type': 'ineq', 'fun': lambda p: b - np.dot(A, p)},
                          bounds=[(0, None)] * nBins, options={'maxiter': 10000, 'disp': False})
            minVar = objFun(res.x)

            # 2. Максимальное значение дисперсии
            def objFunMax(p):
                return -(np.dot(p, binMids**2) - np.dot(p, binMids)**2)

            res = minimize(objFunMax, p0, constraints={'type': 'ineq', 'fun': lambda p: b - np.dot(A, p)},
                          bounds=[(0, None)] * nBins, options={'maxiter': 10000, 'disp': False})
            maxVar = -objFunMax(res.x)

            stdCint = np.sqrt([minVar, maxVar])
        else:
            raise ValueError('Аргумент method не задан. Допустимые значения: approximate, accurate.')

        return stdCint

    # Сложение с использованием принципа Заде (правило max-min)
    def __add__(self, other):
        if not isinstance(other, FuzzyInterval):
            raise TypeError('Оба операнда должны быть объектами типа FuzzyInterval.')

        # Комбинирование и сортировка уровней значимости
        combinedAlpha = np.unique(np.concatenate([self.AlphaLevels, other.AlphaLevels]))
        combinedAlpha = np.sort(combinedAlpha)[::-1]

        # Вычисление интервалов на каждом уровне значимости
        sumIntervals = np.zeros((len(combinedAlpha), 2))
        for i, alpha in enumerate(combinedAlpha):
            intA = self.getIntervalAtAlpha(alpha)
            intB = other.getIntervalAtAlpha(alpha)
            # Суммирование интервалов по интервальной арифметике
            sumIntervals[i] = [intA[0] + intB[0], intA[1] + intB[1]]

        return FuzzyInterval(combinedAlpha, sumIntervals)

    # Вложенный интервал на заданном уровне значимости (с интерполяцией)
    def getIntervalAtAlpha(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError('Значение alpha должно быть от 0 до 1.')

        # Граничные значения α
        idxHigh = np.where(self.AlphaLevels >= alpha)[0]
        idxLow = np.where(self.AlphaLevels <= alpha)[0]

        if idxHigh.size == 0:
            return self.Intervals[0]
        elif idxLow.size == 0:
            return self.Intervals[-1]
        elif idxHigh[-1] == idxLow[0]:  # Точное соответствие
            return self.Intervals[idxHigh[-1]]
        else:  # Линейная интерполяция
            idxHigh = idxHigh[-1]
            idxLow = idxLow[0]
            alphaHigh = self.AlphaLevels[idxHigh]
            alphaLow = self.AlphaLevels[idxLow]
            weight = (alpha - alphaLow) / (alphaHigh - alphaLow)

            # Интерполяция левых границ
            leftHigh = self.Intervals[idxHigh, 0]
            leftLow = self.Intervals[idxLow, 0]
            left = leftHigh + (leftLow - leftHigh) * (1 - weight)

            # Интерполяция правых границ
            rightHigh = self.Intervals[idxHigh, 1]
            rightLow = self.Intervals[idxLow, 1]
            right = rightHigh + (rightLow - rightHigh) * weight

            return np.array([left, right])

    # Умножение на коэффициент
    def __rmul__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError('Операция умножения задана только для перемножения с коэффициентом.')
        z_alphalevels = self.AlphaLevels
        z_intervals = k * self.Intervals
        if k < 0:
            z_intervals = z_intervals[:, [1, 0]]
        return FuzzyInterval(z_alphalevels, z_intervals)

    # Графическое отображение функции принадлежности нечеткого интервала
    def plot(self):
        plt.figure()

        # Отображение α-cuts
        for i in range(len(self.AlphaLevels)):
            a = self.Intervals[i, 0]
            b = self.Intervals[i, 1]
            alpha = self.AlphaLevels[i]
            plt.plot([a, b], [alpha, alpha], 'b-', linewidth=1.5)

            # Отображение вертикальных соединительных линий
            if i < len(self.AlphaLevels) - 1:
                next_a = self.Intervals[i + 1, 0]
                next_b = self.Intervals[i + 1, 1]
                plt.plot([a, next_a], [alpha, self.AlphaLevels[i + 1]], 'b--')
                plt.plot([b, next_b], [alpha, self.AlphaLevels[i + 1]], 'b--')

        plt.xlabel('Значение')
        plt.ylabel('α')
        plt.title('Нечеткий интервал.')
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.show()

    # Преобразование в PBox
    def FuzzyInterval2PBox(self, numPoints=None):
        if numPoints is None:
            numPoints = len(self.AlphaLevels)
        x_vals = np.linspace(np.min(self.Intervals), np.max(self.Intervals), numPoints)
        lowerCDF = np.zeros_like(x_vals)
        upperCDF = np.zeros_like(x_vals)

        for i, x in enumerate(x_vals):
            bound = np.max(self.AlphaLevels[self.Intervals[:, 0] < x], initial=0)
            upperCDF[i] = min(bound, 1) if bound.size > 0 else 0
            bound = 1 - np.max(self.AlphaLevels[self.Intervals[:, 1] > x], initial=0)
            lowerCDF[i] = max(bound, 0) if bound.size > 0 else 1

        return PBox(x_vals, lowerCDF, upperCDF)

    # Преобразование в Hist
    def FuzzyInterval2Hist(self, numBins=None):
        if numBins is None:
            numBins = len(self.AlphaLevels)
        # Определение границ полос гистограммы
        edges = np.linspace(np.min(self.Intervals), np.max(self.Intervals), numBins + 1)

        lowerPDF = np.zeros(numBins)
        upperPDF = np.zeros(numBins)

        for i in range(numBins):
            left = edges[i]
            right = edges[i + 1]

            # Определение максимального уровня значимости, при котором интервал содержит данную полосу
            containing = (self.Intervals[:, 0] <= left) & (self.Intervals[:, 1] >= right)
            if np.any(containing):
                max_alpha = np.max(self.AlphaLevels[containing])
            else:
                max_alpha = 0

            # Границы PDF пропорционально уровню значимости
            lowerPDF[i] = 0  # Консервативная нижняя граница
            upperPDF[i] = max_alpha

        # Определение границ значений PDF
        bin_widths = np.diff(edges)
        total_upper = np.sum(upperPDF * bin_widths)
        upperPDF = upperPDF / total_upper

        return Hist(edges, lowerPDF, upperPDF)

    # Преобразование в DempsterShafer
    def FuzzyInterval2DempsterShafer(self, method='fractional'):
        if method == 'fractional':
            # Вычисление масс из разностей значений уровня значимости
            masses = -0.5 * np.diff(np.concatenate([[0], self.AlphaLevels]))
            masses = np.concatenate([masses, masses])
            intervals = []
            if self.AlphaLevels[0] == 1:
                masses = np.concatenate([[1.0], masses])
                intervals = [self.Intervals[0]]
            intervals = np.concatenate([intervals, self.Intervals[1:, 0].reshape(-1, 1), self.Intervals[:-1, 0].reshape(-1, 1)], axis=1)
            intervals = np.concatenate([intervals, self.Intervals[:-1, 1].reshape(-1, 1), self.Intervals[1:, 1].reshape(-1, 1)], axis=1)
            # Удаление интервалов с нулевой массой
            valid = masses > 0
            ds = DempsterShafer(intervals[valid], masses[valid] / np.sum(masses[valid]))
        elif method == 'nested':
            masses = -np.diff(np.concatenate([[0], self.AlphaLevels]))
            # Удаление интервалов с нулевой массой
            valid = masses > 0
            ds = DempsterShafer(self.Intervals[valid], masses[valid] / np.sum(masses[valid]))
        else:
            raise ValueError('Поле method может быть равно только fractional или nested.')

        return ds

    # Преобразование в Fuzzy
    def FuzzyInterval2Fuzzy(self, numPoints=None):
        if numPoints is None:
            numPoints = len(self.AlphaLevels)
        # Дискретизация носителя
        x_vals = np.linspace(np.min(self.Intervals), np.max(self.Intervals), numPoints)
        membership = np.zeros_like(x_vals)

        # Оценка уровня значимости как максимального alpha, при котором вложенный интервал содержит x
        for i, x in enumerate(x_vals):
            contains_x = (self.Intervals[:, 0] <= x) & (self.Intervals[:, 1] >= x)
            if np.any(contains_x):
                membership[i] = np.max(self.AlphaLevels[contains_x])

        return Fuzzy(x_vals, membership)
