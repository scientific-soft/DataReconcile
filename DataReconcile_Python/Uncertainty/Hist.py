import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Преобразование разных типов представления неточных данных к формату,
# унифицированному для дальнейшего применения Data Reconciliation.

# Класс для Hist (дискретизированная гистограмма, pdf)
class Hist:  # дискретизированная гистограмма
    tolerance_threshold = 0.1  # Константа для объединения близких границ

    def __init__(self, x, lowerPDF, upperPDF):
        # Проверка размерностей входных данных
        if len(x) < 2:
            raise ValueError('В гистограмме должна быть хотя бы одна полоса.')
        if len(x) != len(lowerPDF) + 1 or len(x) != len(upperPDF) + 1:
            raise ValueError('lowerPDF и upperPDF должны быть векторами одинакового размера, x иметь на одно значение больше.')
        if np.any(lowerPDF > upperPDF):
            raise ValueError('Нижние границы полос гистограммы не должны превышать верхние границы гистограммы для всех значений аргументов.')
        if np.any(lowerPDF < 0) or np.any(upperPDF < 0):
            raise ValueError('Значения высоты полос гистограммы не могут быть меньше 0.')

        self.x = np.array(x).flatten()  # Принудительный перевод в вектор-строку
        self.lowerPDF = np.array(lowerPDF).flatten()  # Принудительный перевод в вектор-строку
        self.upperPDF = np.array(upperPDF).flatten()  # Принудительный перевод в вектор-строку

        # Сильная проверка условия нормировки: в предположении, что границы вероятности определены асимптотическим доверительным интервалом
        threshold = 1e-2
        if abs(np.sum(np.mean([self.lowerPDF, self.upperPDF], axis=0) * np.diff(self.x)) - 1) > threshold:
            print('Предупреждение: Площадь гистограммы не равна 1.0 при условии симметричности границ значений PDF.')

        # Слабая проверка условия нормировки: хотя бы при каком-то сочетании значений PDF нормировка должна выполняться
        if (np.sum(self.lowerPDF * np.diff(self.x)) > 1) or (np.sum(self.upperPDF * np.diff(self.x)) < 1):
            print('Предупреждение: Площадь гистограммы не равна 1.0.')

        # Значения вероятностей не должны быть больше единицы
        if np.any(self.upperPDF * np.diff(self.x) > 1):
            print('Предупреждение: Площади полос гистограммы не могут быть больше 1.0.')

    # Оператор сложения двух гистограмм (операнды: Z = A + B)
    def __add__(self, other):
        if not isinstance(other, Hist):
            raise TypeError('Оба операнда должны быть объектами типа Hist.')

        # Генерация всех возможных значений суммы границ полос гистограмм
        A_edges, B_edges = np.meshgrid(self.x, other.x)
        all_edges = (A_edges + B_edges).flatten()

        # Во избежание комбинаторного взрыва, объединяем близкие границы
        sum_edges = np.unique(np.sort(all_edges))

        lower_pdf = np.zeros(len(sum_edges) - 1)
        upper_pdf = np.zeros(len(sum_edges) - 1)

        # Определение вероятностей, соответствующих попаданию в полосы
        A_lowerProb = self.lowerPDF * np.diff(self.x)
        A_upperProb = self.upperPDF * np.diff(self.x)
        B_lowerProb = other.lowerPDF * np.diff(other.x)
        B_upperProb = other.upperPDF * np.diff(other.x)

        # Вычисление пределов вероятности для каждой полосы итоговой гистограммы
        for k in range(len(sum_edges) - 1):
            z_min = sum_edges[k]
            z_max = sum_edges[k + 1]

            total_lower = 0
            total_upper = 0

            # Цикл по всем возможным парам полос гистограмм-операндов
            for i in range(len(self.x) - 1):
                for j in range(len(other.x) - 1):
                    # Определение возможных сумм для границ операндов
                    sum_min = self.x[i] + other.x[j]
                    sum_max = self.x[i + 1] + other.x[j + 1]

                    # Вычисление доли пересечения (в стандартном предположении равномерного распределения внутри полосы)
                    overlap_min = max(z_min, sum_min)
                    overlap_max = min(z_max, sum_max)

                    if overlap_max > overlap_min:
                        # Величина пересечения
                        fraction = (overlap_max - overlap_min) / (sum_max - sum_min)

                        # Вклад в вероятность попадания в полосу
                        total_lower += A_lowerProb[i] * B_lowerProb[j] * fraction
                        total_upper += A_upperProb[i] * B_upperProb[j] * fraction

            # Перевод в плотность вероятности
            bin_width = sum_edges[k + 1] - sum_edges[k]
            lower_pdf[k] = total_lower / bin_width
            upper_pdf[k] = total_upper / bin_width

        # Валидность границ
        if np.any(lower_pdf > upper_pdf):
            raise ValueError('Среди нижних границ PDF есть значения, превышающие верхние границы PDF.')

        return Hist(sum_edges, lower_pdf, upper_pdf)

    # Умножение на коэффициент
    def __rmul__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError('Операция умножения задана только для перемножения с коэффициентом.')
        z_values = k * self.x
        if k >= 0:
            lowerZ = self.lowerPDF / k
            upperZ = self.upperPDF / k
        else:
            lowerZ = -self.lowerPDF[::-1] / k
            upperZ = -self.upperPDF[::-1] / k
        return Hist(z_values, lowerZ, upperZ)

    # Построение графика гистограммы
    def plot(self):
        plt.figure()
        n = len(self.x)
        for i in range(1, n):  # Цикл по полосам гистограммы
            plt.plot([self.x[i - 1], self.x[i - 1], self.x[i], self.x[i]], [0, self.upperPDF[i - 1], self.upperPDF[i - 1], 0], 'b', linewidth=1.5)
            plt.plot([self.x[i - 1], self.x[i - 1], self.x[i], self.x[i]], [0, self.lowerPDF[i - 1], self.lowerPDF[i - 1], 0], 'r', linewidth=1.5)
        plt.xlabel('x')
        plt.ylabel('PDF')
        plt.legend(['верхняя граница Hist', 'нижняя граница Hist'], loc='lower right')
        plt.title('Гистограмма Берлинта')
        plt.grid(True)
        plt.show()

    # Определение величины возможного среднеквадратического отклонения
    def getStd(self, method='approximate'):
        # Характеристики полос гистограммы
        binEdges = self.x
        nBins = len(binEdges) - 1
        binWidths = np.diff(binEdges)
        binMids = 0.5 * (binEdges[:-1] + binEdges[1:])

        # Границы возможной вероятности на полосу
        lowerProbs = self.lowerPDF * binWidths
        upperProbs = self.upperPDF * binWidths

        # Валидация получившихся границ вероятности
        if np.sum(lowerProbs) > 1 or np.sum(upperProbs) < 1:
            raise ValueError('Границы высот полос гистограммы заданы с ошибками.')

        if method == 'approximate':
            m1 = np.sum(np.mean([upperProbs, lowerProbs], axis=0) * binMids)

            varCint = np.array([np.nan, np.nan])
            varCint[0] = np.sum(lowerProbs * ((binMids - m1) ** 2)) + (1 / 12) * np.sum(lowerProbs * (binWidths ** 2))
            varCint[1] = np.sum(upperProbs * ((binMids - m1) ** 2)) + (1 / 12) * np.sum(upperProbs * (binWidths ** 2))
            stdCint = np.sqrt(varCint)

        elif method == 'accurate':
            # Задание условий оптимизации
            Aeq = np.ones((1, nBins))
            beq = np.array([1])
            lb = lowerProbs
            ub = upperProbs

            # Оптимизация
            options = {'maxiter': 10000, 'disp': False}
            p0 = (lowerProbs + upperProbs) / 2  # Начальное приближение

            # 1. Минимальное значение дисперсии
            def objFun(p):
                return np.dot(p, binMids ** 2) - np.dot(p, binMids) ** 2

            res = minimize(objFun, p0, constraints={'type': 'eq', 'fun': lambda p: np.dot(Aeq, p) - beq},
                          bounds=[(lb[i], ub[i]) for i in range(nBins)], options=options)
            minVar = objFun(res.x)

            # 2. Максимальное значение дисперсии
            def objFunMax(p):
                return -(np.dot(p, binMids ** 2) - np.dot(p, binMids) ** 2)

            res = minimize(objFunMax, p0, constraints={'type': 'eq', 'fun': lambda p: np.dot(Aeq, p) - beq},
                          bounds=[(lb[i], ub[i]) for i in range(nBins)], options=options)
            maxVar = -objFunMax(res.x)

            stdCint = np.sqrt([minVar, maxVar])

        else:
            raise ValueError('Аргумент method не задан. Допустимые значения: approximate, accurate.')

        return stdCint

    # Преобразование в PBox
    def Hist2PBox(self):
        # Вычисление границ CDF
        lowerProbs = self.lowerPDF * np.diff(self.x)
        upperProbs = self.upperPDF * np.diff(self.x)

        lowerCDF = np.concatenate([[0], np.cumsum(lowerProbs)])
        upperCDF = np.concatenate([[0], np.cumsum(upperProbs)])

        # Нормализация для обеспечения верных границ для CDF
        lowerCDF[lowerCDF > 1] = 1.0
        upperCDF[upperCDF > 1] = 1.0

        if lowerCDF[-1] < 1:
            lowerCDF = np.concatenate([lowerCDF, [1.0]])
            upperCDF = np.concatenate([upperCDF, [1.0]])

        return PBox(np.concatenate([self.x, [self.x[-1] + np.finfo(float).eps]]), lowerCDF, upperCDF)

    # Преобразование в DempsterShafer
    def Hist2DempsterShafer(self):
        # Создание фокальных элементов из полос гистограммы
        intervals = np.column_stack([self.x[:-1], self.x[1:]])

        # Вычисление масс
        lowerMasses = self.lowerPDF * np.diff(self.x)
        upperMasses = self.upperPDF * np.diff(self.x)

        # Создание структуры Демпстера-Шафера
        masses = (lowerMasses + upperMasses) / 2  # Аппроксимация средним
        masses = masses / np.sum(masses)  # Нормализация

        return DempsterShafer(intervals, masses)

    # Преобразование в Fuzzy
    def Hist2Fuzzy(self, numUniverse=None):
        if numUniverse is None:
            numUniverse = len(self.x) - 1
        bin_centers = 0.5 * (self.x[:-1] + self.x[1:])
        # Определение функции принадлежности по верхним границам PDF
        x_universe = np.linspace(np.min(self.x), np.max(self.x), numUniverse)
        membership = np.interp(x_universe, bin_centers, self.upperPDF * np.diff(self.x), left=0, right=0)

        return Fuzzy(x_universe, membership)

    # Преобразование в FuzzyInterval
    def Hist2FuzzyInterval(self, numAlpha=None):
        if numAlpha is None:
            numAlpha = len(self.x) - 1
        alpha_levels = np.linspace(0, 1, numAlpha)
        alpha_levels = np.sort(alpha_levels)[::-1]

        intervals = np.zeros((numAlpha, 2))
        for i, alpha in enumerate(alpha_levels):
            valid_bins = self.upperPDF >= alpha
            if np.any(valid_bins):
                intervals[i, 0] = self.x[np.where(valid_bins)[0][0]]
                intervals[i, 1] = self.x[np.where(valid_bins)[0][-1] + 1]
            else:
                intervals[i] = [np.nan, np.nan]

        return FuzzyInterval(alpha_levels, intervals)