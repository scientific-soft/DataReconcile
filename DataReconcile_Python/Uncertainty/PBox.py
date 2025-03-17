import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Преобразование разных типов представления неточных данных к формату,
# унифицированному для дальнейшего применения Data Reconciliation.

# Класс для p-box (область возможных значений для функции распределения, cdf)
class PBox:  # дискретизированный p-box
    def __init__(self, x, lowerCDF, upperCDF):
        # проверка размерностей входных данных
        if len(x) != len(lowerCDF) or len(x) != len(upperCDF):
            raise ValueError('x, lowerCDF и upperCDF должны быть векторами одинакового размера.')
        if any(l > u+1e-6 for l, u in zip(lowerCDF, upperCDF)):
            raise ValueError('Нижние границы p-box не должны превышать верхние границы p-box для всех значений аргументов.')
        if any(np.diff(lowerCDF) < 0) or any(np.diff(upperCDF) < 0):
            raise ValueError('Функции распределения не могут убывать.')
        if any(l < 0 for l in lowerCDF) or any(u < 0 for u in upperCDF):
            raise ValueError('Значения функции распределения не могут быть меньше 0.')
        if any(l > 1 for l in lowerCDF) or any(u > 1 for u in upperCDF):
            raise ValueError('Значения функции распределения не могут быть больше 1.')
        self.x = np.array(x).flatten()  # принудительный перевод в вектор-строку
        self.lowerCDF = np.array(lowerCDF).flatten()  # принудительный перевод в вектор-строку
        self.upperCDF = np.array(upperCDF).flatten()  # принудительный перевод в вектор-строку

    # оператор сложения двух p-boxes (операнды: Z = A + B)
    def __add__(self, other):
        if not isinstance(other, PBox):
            raise TypeError('Оба операнда должны быть объектами типа PBox.')
        # определение сетки значений аргумента x для результата операции
        min_z = min(self.x) + min(other.x)
        max_z = max(self.x) + max(other.x)
        z_values = np.linspace(min_z, max_z, max(len(self.x), len(other.x)))

        lowerZ = np.zeros_like(z_values)
        upperZ = np.zeros_like(z_values)

        for i, z in enumerate(z_values):
            # для всех значений их self.x вычисляем разности y=z-x
            x_A = self.x
            y_B = z - x_A

            # оценка значения cdf для B в точках y_B
            F_B_lower = interp1d(other.x, other.lowerCDF, kind='linear', fill_value=0, bounds_error=False)(y_B)
            F_B_lower = np.clip(F_B_lower, 0, 1)  # приведение к [0,1]
            F_B_upper = interp1d(other.x, other.upperCDF, kind='linear', fill_value=1, bounds_error=False)(y_B)
            F_B_upper = np.clip(F_B_upper, 0, 1)  # приведение к [0,1]

            # вычисление нижней границы для Z в точке z
            temp_lower = self.lowerCDF + F_B_lower - 1
            lowerZ[i] = max(np.maximum(temp_lower, 0))

            # вычисление верхней границы для Z в точке z
            temp_upper = self.upperCDF + F_B_upper
            upperZ[i] = min(np.minimum(temp_upper, 1))

        # проверка, что функция распределения не убывает
        lowerZ = np.maximum.accumulate(lowerZ)
        upperZ = np.maximum.accumulate(upperZ)

        # приведение к [0,1]
        lowerZ = np.clip(lowerZ, 0, 1)
        upperZ = np.clip(upperZ, 0, 1)

        return PBox(z_values, lowerZ, upperZ)

    def __rmul__(self, k):
        if not isinstance(k, (int, float)) or not isinstance(self, PBox):
            raise TypeError('Операция умножения задана только для перемножения с коэффициентом.')
        z_values = k * self.x
        if k >= 0:
            lowerZ = self.lowerCDF
            upperZ = self.upperCDF
        else:
            z_values = z_values[::-1]
            lowerZ = 1 - self.upperCDF[::-1]
            upperZ = 1 - self.lowerCDF[::-1]
        return PBox(z_values, lowerZ, upperZ)

    # построение графика p-box
    def plot(self):
        plt.figure()
        plt.step(self.x, self.lowerCDF, 'b', linewidth=1.5, where='post')
        plt.step(self.x, self.upperCDF, 'b', linewidth=1.5, where='post')
        plt.xlabel('x')
        plt.ylabel('CDF')
        plt.legend(['нижняя граница P-Box', 'верхняя граница P-Box'], loc='lower right')
        plt.title('Границы P-box')
        plt.grid(True)
        plt.show()

    # определение величины возможного среднеквадратического отклонения.
    def getStd(self, method='approximate'):
        n = len(self.x)

        # приближенный вариант: согласно теореме Крейновича-Тао.
        if method == 'approximate':
            ind_left_lo = np.where(self.lowerCDF < 0.1)[0][-1] + 1 if np.any(self.lowerCDF < 0.1) else 0
            ind_left_hi = np.where(self.upperCDF < 0.1)[0][-1] + 1 if np.any(self.upperCDF < 0.1) else 0
            ind_right_lo = np.where(self.lowerCDF < 0.9)[0][-1] + 1 if np.any(self.lowerCDF < 0.9) else 0
            ind_right_hi = np.where(self.upperCDF < 0.9)[0][-1] + 1 if np.any(self.upperCDF < 0.9) else 0

            stdCint = np.array([np.nan, np.nan])
            stdCint[0] = max(0.25 * (self.x[ind_right_hi] - self.x[ind_left_lo]), 0)
            stdCint[1] = 0.25 * (self.x[ind_right_lo] - self.x[ind_left_hi])

        elif method == 'accurate':
            # переменные для сокращения записей
            x = self.x
            lowerCDF = self.lowerCDF
            upperCDF = self.upperCDF

            # дискретизация в интервалы
            nIntervals = len(x) - 1
            p_lower = np.zeros(nIntervals)
            p_upper = np.zeros(nIntervals)

            # вычисление границ интервалов вероятности
            for i in range(nIntervals):
                p_lower[i] = max(lowerCDF[i + 1] - upperCDF[i], 0)
                p_upper[i] = upperCDF[i + 1] - lowerCDF[i]

            # проверки согласованности
            if np.sum(p_lower) > 1 or np.sum(p_upper) < 1:
                raise ValueError('Несогласованный p-box: не включает ни одного распределения.')

            # вычисление середин интервал для оценки дисперсии
            x_mid = 0.5 * (x[:-1] + x[1:])

            # оптимизация
            # 1. минимальное значение дисперсии
            def objFun(p):
                return np.dot(p, x_mid**2) - np.dot(p, x_mid)**2

            p0 = p_lower / np.sum(p_lower) if np.sum(p_lower) != 0 else p_lower
            bounds = [(l, u) for l, u in zip(p_lower, p_upper)]
            constraints = {'type': 'eq', 'fun': lambda p: np.sum(p) - 1}
            res = minimize(objFun, p0, bounds=bounds, constraints=constraints, method='SLSQP')
            minVar = objFun(res.x)

            # 2. максимальное значение дисперсии
            def objFunMax(p):
                return -(np.dot(p, x_mid**2) - np.dot(p, x_mid)**2)

            p0 = p_upper / np.sum(p_upper)
            res = minimize(objFunMax, p0, bounds=bounds, constraints=constraints, method='SLSQP')
            maxVar = -objFunMax(res.x)

            stdCint = np.sqrt([minVar, maxVar])

        else:
            raise ValueError('Аргумент method не задан. Допустимые значения: approximate, accurate.')

        return stdCint

    # преобразование в Hist
    def PBox2Hist(self, numBins=None):
        if numBins is None:
            numBins = len(self.x)
        # дискретизация носителя
        x_min = min(self.x)
        x_max = max(self.x)
        edges = np.linspace(x_min, x_max, numBins + 1)

        lowerPDF = np.zeros(numBins)
        upperPDF = np.zeros(numBins)

        for i in range(numBins):
            left = edges[i]
            right = edges[i + 1]

            # вычисление значений CDF в границах полос гистограммы
            idx_left = np.searchsorted(self.x, left, side='right') - 1
            idx_right = np.searchsorted(self.x, right, side='right') - 1

            # валидация
            idx_left = max(0, min(idx_left, len(self.x) - 1))
            idx_right = max(0, min(idx_right, len(self.x) - 1))

            # вычисления границ для вероятности
            p_lower = max(self.lowerCDF[idx_right] - self.upperCDF[idx_left], 0)
            p_upper = min(self.upperCDF[idx_right] - self.lowerCDF[idx_left], 1)

            # определение границ для PDF
            bin_width = right - left
            lowerPDF[i] = p_lower / bin_width
            upperPDF[i] = p_upper / bin_width

        return Hist(edges, lowerPDF, upperPDF)

    # преобразование в DempsterShafer
    def PBox2DempsterShafer(self, numFocal=None):
        if numFocal is None:
            numFocal = len(self.x)
        # дискретизация в фокальные элементы
        x_vals = np.linspace(min(self.x), max(self.x), numFocal + 1)
        intervals = np.column_stack((x_vals[:-1], x_vals[1:]))
        prob_low = self.lowerCDF[np.round(np.linspace(0, len(self.x) - 1, numFocal + 1)).astype(int)]
        prob_high = self.upperCDF[np.round(np.linspace(0, len(self.x) - 1, numFocal + 1)).astype(int)]
        masses = np.diff(0.5 * (prob_low + prob_high))

        # нормализация масс
        masses = masses / np.sum(masses)
        return DempsterShafer(intervals, masses)

    # преобразование в Fuzzy
    def PBox2Fuzzy(self, numPoints=None):
        if numPoints is None:
            numPoints = len(self.x)
        # дискретизация носителя х
        x_vals = np.linspace(min(self.x), max(self.x), numPoints)
        membership = np.full_like(x_vals, np.nan)

        # вычисление значений функции принадлежности как (1-CDF) (мера возможности)
        lowerCDF_interp = interp1d(self.x, self.lowerCDF, kind='linear', fill_value='extrap')(x_vals)
        upperCDF_interp = interp1d(self.x, self.upperCDF, kind='linear', fill_value='extrap')(x_vals)
        ind1 = upperCDF_interp > 1 - lowerCDF_interp
        ind2 = upperCDF_interp <= 1 - lowerCDF_interp
        membership[ind1] = 1 - lowerCDF_interp[ind1]
        membership[ind2] = upperCDF_interp[ind2]

        # валидация
        membership = membership / np.max(membership)  # приведение к [0,1].
        return Fuzzy(x_vals, membership)

    # преобразование в FuzzyInterval
    def PBox2FuzzyInterval(self, numAlpha=None):
        if numAlpha is None:
            numAlpha = len(self.x)
        # определение уровней значимости
        alphaLevels = np.linspace(0, 1, numAlpha)
        alphaLevels = np.sort(alphaLevels)[::-1]
        intervals = np.zeros((numAlpha, 2))

        n = len(self.x)
        for i, alpha in enumerate(alphaLevels):
            # отыскание интервала, такого, что Lower CDF <= (1 - alpha) <= Upper CDF
            try:
               lower_bound = interp1d(self.upperCDF, self.x, kind='linear', fill_value='extrap')(1 - alpha)
            except:
               lower_bound = self.x[0]
            try:
               upper_bound = interp1d(self.lowerCDF, self.x, kind='linear', fill_value='extrap')(1 - alpha)
            except:
               upper_bound = self.x[-1]
            intervals[i, :] = [lower_bound, upper_bound]

        # Валидация структуры вложенных интервалов
        for i in range(1, numAlpha):
            intervals[i, 0] = min(intervals[i - 1, 0], intervals[i, 0])
            intervals[i, 1] = max(intervals[i - 1, 1], intervals[i, 1])

        return FuzzyInterval(alphaLevels, intervals)