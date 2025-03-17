import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Преобразование разных типов представления неточных данных к формату,
# унифицированному для дальнейшего применения Data Reconciliation.

# Класс для переменных типа Fuzzy (нечеткие переменные)
class Fuzzy:
    def __init__(self, universe, membership):
        # Проверка входных переменных
        universe = np.array(universe).flatten()  # перевод в вектор-строку
        membership = np.array(membership).flatten()  # перевод в вектор-строку

        if len(universe) != len(membership):
            raise ValueError('Поля Universe и Membership должны быть одинаковой длины.')
        if np.any(membership < 0) or np.any(membership > 1):
            raise ValueError('Значения функции принадлежности должны быть от 0 до 1.')

        # Порядок следования значений в поле universe - по возрастанию
        idx = np.argsort(universe)
        self.Universe = universe[idx]
        self.Membership = membership[idx]

    # Сложение с использованием принципа Заде (правило max-min)
    def __add__(self, other):
        # Создание результирующей переменной (поле Universe)
        min_z = self.Universe[0] + other.Universe[0]
        max_z = self.Universe[-1] + other.Universe[-1]
        z_step = min(np.diff(self.Universe[:2]), np.diff(other.Universe[:2]))
        z_universe = np.arange(min_z, max_z + z_step, z_step)

        # Выделение памяти под функцию принадлежности
        z_membership = np.zeros_like(z_universe)

        # Принцип Заде (композиция)
        for i, z in enumerate(z_universe):
            # Все возможные пары значений (x, y), для которых x + y = z
            x_vals = self.Universe
            y_vals = z - x_vals

            # Определение значений функции принадлежности для значений y_vals в B
            y_membership = interp1d(other.Universe, other.Membership, kind='linear', fill_value=0, bounds_error=False)(y_vals)

            # Вычисление минимакса от функции принадлежности
            combined = np.minimum(self.Membership, y_membership)
            z_membership[i] = np.max(combined)

        return Fuzzy(z_universe, z_membership)

    # Умножение на коэффициент
    def __rmul__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError('Операция умножения задана только для перемножения с коэффициентом.')
        z_membership = self.Membership
        z_universe = k * self.Universe
        if k < 0:
            z_universe = z_universe[::-1]
            z_membership = z_membership[::-1]
        return Fuzzy(z_universe, z_membership)

    # Вычисление вложенного (α-cut) интервала
    def alphaCut(self, alpha):
        # Валидация значения α
        if alpha < 0 or alpha > 1:
            raise ValueError('Уровень значимости должен быть числом в интервале [0, 1].')

        x = self.Universe  # Переменная для краткости
        mu = self.Membership  # Переменная для краткости

        # Значения индексов, когда Membership >= alpha
        above = mu >= alpha
        if not np.any(above):
            return np.array([np.nan, np.nan])

        # Левая граница
        first_idx = np.where(above)[0][0]
        if first_idx == 0:
            left = x[0]
        else:
            # Линейная интерполяция
            x_prev = x[first_idx - 1]
            x_curr = x[first_idx]
            mu_prev = mu[first_idx - 1]
            mu_curr = mu[first_idx]
            left = x_prev + (alpha - mu_prev) * (x_curr - x_prev) / (mu_curr - mu_prev)

        # Правая граница
        last_idx = np.where(above)[0][-1]
        if last_idx == len(x) - 1:
            right = x[-1]
        else:
            # Линейная интерполяция
            x_curr = x[last_idx]
            x_next = x[last_idx + 1]
            mu_curr = mu[last_idx]
            mu_next = mu[last_idx + 1]
            right = x_curr + (alpha - mu_curr) * (x_next - x_curr) / (mu_next - mu_curr)

        return np.array([left, right])

    def getStd(self, method='approximate'):
        if method == 'approximate':
            # Определение значений функции принадлежности для уровня α-cut, равного 0.05
            int0 = self.alphaCut(0.05)
            int1 = self.alphaCut(1 - np.finfo(float).eps)
            if np.any(np.isnan(int1)):
                stdCint = np.array([0, 0.25 * (int0[1] - int0[0])])
            else:
                val1 = 0.5 * (int1[0] - int0[0])
                val2 = 0.5 * (int0[1] - int1[1])
                stdCint = np.array([0, np.mean([val1, val2])])
        elif method == 'accurate':
            # Дискретизация уровней значимости
            alphaLevels = np.unique(np.concatenate([[0], self.Membership, [1]]))
            alphaLevels = np.sort(alphaLevels)[::-1]

            # Дискретизация поля Universe
            x = self.Universe
            nBins = len(x) - 1
            binEdges = x
            binMids = (x[:-1] + x[1:]) / 2

            # Матрицы ограничений для оптимизации
            A = []
            B = []

            # Построение ограничений из α-cuts
            for alpha in alphaLevels:
                interval = self.alphaCut(alpha)
                if np.any(np.isnan(interval)):
                    continue
                a, b = interval
                inBin = (binEdges[:-1] >= a) & (binEdges[1:] <= b)
                # Добавление ограничения: sum(p(inBin)) >= 1 - alpha
                A.append(-inBin.astype(float))
                B.append(-(1 - alpha))

            A = np.array(A)
            B = np.array(B)

            # Ограничение на нормировочное свойство
            Aeq = np.ones((1, nBins))
            Beq = np.array([1])

            # Неотрицательность вероятностей
            lb = np.zeros(nBins)
            ub = None

            # Оптимизация
            options = {'disp': False, 'maxiter': 10000}

            # 1. Минимальное значение дисперсии
            def objFun(p):
                return np.dot(p, binMids**2) - np.dot(p, binMids)**2

            p0 = np.ones(nBins) / nBins
            res = minimize(objFun, p0, constraints={'type': 'ineq', 'fun': lambda p: B - np.dot(A, p)},
                          bounds=[(0, None)] * nBins, options=options)
            minVar = objFun(res.x)
            if abs(minVar) < 1e-6:  # Устранение ошибок округления
                minVar = 0

            # 2. Максимальное значение дисперсии
            def objFunMax(p):
                return -(np.dot(p, binMids**2) - np.dot(p, binMids)**2)

            res = minimize(objFunMax, p0, constraints={'type': 'ineq', 'fun': lambda p: B - np.dot(A, p)},
                          bounds=[(0, None)] * nBins, options=options)
            maxVar = -objFunMax(res.x)

            stdCint = np.sqrt([minVar, maxVar])
        else:
            raise ValueError('Аргумент method не задан. Допустимые значения: approximate, accurate.')

        return stdCint

    # Отображение функции принадлежности на графике
    def plot(self):
        plt.figure()
        plt.fill_between(self.Universe, self.Membership, color=[0.8, 0.9, 1], edgecolor='b', linewidth=2)
        plt.xlabel('Значения переменной')
        plt.ylabel('Функция принадлежности')
        plt.title('Нечеткая переменная')
        plt.ylim([0, 1.1])
        plt.grid(True)
        plt.show()

    # Преобразование в PBox
    def Fuzzy2PBox(self, numPoints=None):
        if numPoints is None:
            numPoints = len(self.Universe)
        x_vals = np.linspace(np.min(self.Universe), np.max(self.Universe), numPoints)
        lowerCDF = np.zeros_like(x_vals)
        upperCDF = np.zeros_like(x_vals)

        for i, x in enumerate(x_vals):
            # Нижняя граница CDF = 1 - Possibility(X > x)
            bound = 1 - np.max(self.Membership[self.Universe > x], initial=0)
            lowerCDF[i] = bound if not np.isnan(bound) else 1
            # Верхняя граница CDF = 1 - Necessity(X > x)
            bound = np.max(self.Membership[self.Universe <= x], initial=0)
            upperCDF[i] = bound if not np.isnan(bound) else 0

        return PBox(x_vals, lowerCDF, upperCDF)

    # Преобразование в Hist
    def Fuzzy2Hist(self, numBins=None):
        if numBins is None:
            numBins = len(self.Universe)
        edges = np.linspace(np.min(self.Universe), np.max(self.Universe), numBins + 1)
        lowerProb = np.zeros(numBins)
        upperProb = np.zeros(numBins)

        for i in range(numBins):
            left = edges[i]
            right = edges[i + 1]
            # Определение полос
            in_bin = (self.Universe >= left) & (self.Universe <= right)
            if np.any(in_bin):
                lowerProb[i] = np.mean(in_bin) * np.min(self.Membership[in_bin])
                upperProb[i] = np.mean(in_bin) * np.max(self.Membership[in_bin])

        # Валидация
        if np.sum(lowerProb) > 1:
            raise ValueError('Оценка нижних границ вероятностей попадания в полосы содержит ошибку.')

        # Оценка значения PDF
        lowerPDF = lowerProb / np.diff(edges)
        upperPDF = upperProb / np.diff(edges)

        return Hist(edges, lowerPDF, upperPDF)

    # Преобразование в DempsterShafer
    def Fuzzy2DempsterShafer(self, numAlpha=None):
        if numAlpha is None:
            numAlpha = len(self.Universe)
        alphaLevels = np.linspace(0, 1, numAlpha + 1)
        alphaLevels = alphaLevels[1:]  # Исключаем ноль

        intervals = np.zeros((len(alphaLevels), 2))
        masses = np.diff(np.concatenate([[0], alphaLevels]))

        for i, alpha in enumerate(alphaLevels):
            intervals[i] = self.alphaCut(alpha)

        return DempsterShafer(intervals, masses)

    # Преобразование в FuzzyInterval
    def Fuzzy2FuzzyInterval(self, numAlpha=None):
        if numAlpha is None:
            numAlpha = len(self.Universe)
        alphaLevels = np.linspace(0, 1, numAlpha)
        alphaLevels = np.sort(alphaLevels)[::-1]

        intervals = np.zeros((len(alphaLevels), 2))
        for i, alpha in enumerate(alphaLevels):
            intervals[i] = self.alphaCut(alpha)

        return FuzzyInterval(alphaLevels, intervals)